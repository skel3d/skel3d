import os
import sys
import glob
import json
import math
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image
from pathlib import Path
from einops import rearrange
from torchvision import transforms
from .objaverse import ObjaverseDataset


from utils.vis_camera import vis_points_images_cameras

class ObjaverseSkelGraphDataset(ObjaverseDataset):

    def __init__(self,
                 root_dir = '',
                 cfg = None,
                 debug = False,
                 warp_img = False,
                 ) -> None:
        super().__init__(root_dir=root_dir, cfg=cfg, debug=debug)
        self.warp_img = warp_img

    def read_samples(self):
        json_file = 'val.json' if self.validation else 'train.json'
        # load the file name
        with open(os.path.join(self.root_dir, json_file)) as f:
            self.paths = json.load(f)
            print('============= length of dataset time %d =============' % len(self.paths))

    def load_joints_bones(self, path):
        joints = np.load(path + '/joints.npy')
        bones = np.load(path + '/bones.npy')
        return joints, bones
    
    def project_joints(self, joints, RT_mtx):
        # Project joints to 2D using the camera matrix
        joints_homogeneous = np.hstack((joints, np.ones((joints.shape[0], 1))))
        joints_2d_homogeneous = RT_mtx @ joints_homogeneous.T
        joints_2d = joints_2d_homogeneous[:2, :] / joints_2d_homogeneous[2, :]
        return joints_2d.T
        
    
    def pre_data(self, paths, views):
        '''
        load the data for given filename 
        '''
        color = [1., 1., 1., 1.]
        skels = []
        imgs = []
        w2cs = []
        joint_list = []
        bone_list = []
        intrinsics = []
        index = range(views) if self.validation else torch.randperm(views)

        frame_path = os.path.dirname(paths[0])
        joints, bones = self.load_joints_bones(frame_path)

        joints_homogeneous = torch.tensor(np.hstack((joints, np.ones((joints.shape[0], 1)))), dtype=torch.float32)  # shape (N, 4)

        # # find the closer views
        for i in index[:self.load_view]:
            skel = self.process_img(self.load_im(paths[i], color)).unsqueeze(0)
            skels.append(skel)
            img = self.process_img(self.load_im(paths[i].replace('_skel.png', '.png'), color)).unsqueeze(0)
            imgs.append(img)           
            
            w2c_gl = np.load(paths[i].replace('_skel.png', '.npy'))
            w2cs.append(w2c_gl)
            focal = .5 / np.tan(.5 * 0.8575560450553894)
            intrinsic = np.array([[focal, 0.0, 1.0 / 2.0],
                                        [0.0, focal, 1.0 / 2.0],
                                        [0.0, 0.0, 1.0]])
            intrinsics.append(intrinsic)
        
        imgs = torch.cat(imgs)
        skels = torch.cat(skels)
        intrinsics = torch.tensor(np.array(intrinsics)).to(imgs)


        w2cs = torch.tensor(np.array(w2cs)).to(imgs)
        w2cs_gl = torch.eye(4).unsqueeze(0).repeat(imgs.size(0),1,1)
        w2cs_gl[:,:3,:] = w2cs
        # camera poses in .npy files are in OpenGL convention: 
        #     x right, y up, z into the camera (backward),
        # need to transform to COLMAP / OpenCV:
        #     x right, y down, z away from the camera (forward)
        w2cs = torch.einsum('nj, bjm-> bnm', self.opengl_to_colmap, w2cs_gl)
        c2ws = torch.linalg.inv(w2cs)
        camera_centers = c2ws[:, :3, 3].clone()
        # fix the distance of the source camera to the object / world center
        assert torch.norm(camera_centers[0]) > 1e-5
        translation_scaling_factor = 2.0 / torch.norm(camera_centers[0])
        w2cs[:, :3, 3] *= translation_scaling_factor
        c2ws[:, :3, 3] *= translation_scaling_factor
        camera_centers *= translation_scaling_factor

        # Transform joints to camera coordinates
        joints_camera = (w2cs @ joints_homogeneous.T).permute(0, 2, 1)
        # Project joints to 2D image coordinates
        joints_2d = (intrinsics @ joints_camera[:, :, :3].transpose(1, 2)).transpose(1, 2)

        # Normalize by the third (z) coordinate
        joints_2d = joints_2d[:, :, :2] / joints_2d[:, :, 2:3]

        embedding_size = 256 if self.warp_img else 32
        max_joints = 300
        joints_2d = joints_2d * embedding_size

        # Cast to int
        joints_2d = joints_2d.int()

        # take the joints for the first image       
        joints_first = joints_2d[0]   
        # Get the indeces of the joints that are outside the image
        outside = (joints_first[:, 0] < 0) | (joints_first[:, 0] >= embedding_size) | (joints_first[:, 1] < 0) | (joints_first[:, 1] >= embedding_size)
        # Remove the joints that are outside the image
        joints_2d = joints_2d[:, ~outside, :]

        # Clamp the other joints to the image size
        joints_2d[:, :, 0] = torch.clamp(joints_2d[:, :, 0], 0, embedding_size - 1)
        joints_2d[:, :, 1] = torch.clamp(joints_2d[:, :, 1], 0, embedding_size - 1)

        # Limit the number of joints to 300 by removing random joints
        if joints_2d.size(1) > max_joints:
            indices = torch.randperm(joints_2d.size(1))[:300]
            joints_2d = joints_2d[:, indices, :]
        # Extend the to J fix  300 repeat the last joint
        joints_2d = torch.cat([joints_2d, joints_2d[:, -1:, :].repeat(1, max_joints - joints_2d.size(1), 1)], dim=1)

        if joints_2d.size(1) != max_joints:
            joints_2d = torch.zeros((joints_2d.size(0), max_joints, 2), dtype=joints_2d.dtype, device=joints_2d.device) 


        if self.warp_img:
            skels = torch.zeros_like(skels)
            tgt_ix = range(views).tolist()
            skels.permute(1, 2, 0)[:, joints_2d[tgt_ix, :, 1], joints_2d[tgt_ix, :, 0]] = imgs[0].permute(1, 2, 0)[joints_2d[0, :, 1], joints_2d[0, :, 0]]


            

        return imgs, skels, w2cs, c2ws, intrinsics, joints_2d
    
    def __getitem__(self, index):
        # load the rendered images

        
        obj = self.paths[index]
        sample_path = os.path.join(self.root_dir, obj)

        #frame_paths = [os.path.join(o.path, obj) for o in os.scandir(sample_path) if o.is_dir()]
        frame_paths = [o.path for o in os.scandir(sample_path) if o.is_dir()]

        indices = np.random.choice(len(frame_paths), size=1, replace=False)
        frame_path = frame_paths[indices[0]]
        paths = glob.glob(frame_path + '/*_skel.png')
        views = len(paths)
        imgs, skels, w2cs, c2ws, intrinsics, joints_2d = self.pre_data(paths, views)
        

        # debug the camera system for debugging
        if self.debug:
            import pdb
            pdb.set_trace()
            intrinsics[:, 0, :] *=256
            intrinsics[:, 1, :] *=256
            vis_points_images_cameras(w2cs, intrinsics, imgs, frustum_size=0.5, filename=sample_path.split('/')[-1] + 'camemra_ori.html')

        data = {
                'images': imgs,
                'skeletons': skels,
                'joints_2d': joints_2d,
                'w2cs': w2cs,
                'c2ws': c2ws,
                'intrinsics': intrinsics,
                'filename': sample_path.split('/')[-2]  + sample_path.split('/')[-1]
        }
        return data