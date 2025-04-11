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
from .objaverse_skel import ObjaverseSkelDataset


from utils.vis_camera import vis_points_images_cameras

class ObjaverseSkelTimeDataset(ObjaverseSkelDataset):

    
    def pre_data(self, paths1, paths2, views):
        '''
        load the data for given filename 
        '''
        color = [1., 1., 1., 1.]
        skels = []
        imgs = []
        w2cs = []
        intrinsics = []
        index = range(views) if self.validation else torch.randperm(views)

        path = np.random.choice(paths1)
        skel = self.process_img(self.load_im(path, color)).unsqueeze(0)
        skels.append(skel)
        img = self.process_img(self.load_im(path.replace('_skel.png', '.png'), color)).unsqueeze(0)
        imgs.append(img)
        w2c_gl = np.load(path.replace('_skel.png', '.npy'))
        w2cs.append(w2c_gl)
        focal = .5 / np.tan(.5 * 0.8575560450553894)
        intrinsics.append(np.array([[focal, 0.0, 1.0 / 2.0],
                                    [0.0, focal, 1.0 / 2.0],
                                    [0.0, 0.0, 1.0]]))

        # # find the closer views
        for i in index[:self.load_view-1]:
            skel = self.process_img(self.load_im(paths2[i], color)).unsqueeze(0)
            skels.append(skel)
            img = self.process_img(self.load_im(paths2[i].replace('_skel.png', '.png'), color)).unsqueeze(0)
            imgs.append(img)
            w2c_gl = np.load(paths2[i].replace('_skel.png', '.npy'))
            w2cs.append(w2c_gl)
            focal = .5 / np.tan(.5 * 0.8575560450553894)
            intrinsics.append(np.array([[focal, 0.0, 1.0 / 2.0],
                                        [0.0, focal, 1.0 / 2.0],
                                        [0.0, 0.0, 1.0]]))
        
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
        return imgs, skels, w2cs, c2ws, intrinsics
    
    def __getitem__(self, index):
        # load the rendered images
        
        
        obj = self.paths[index]
        sample_path = os.path.join(self.root_dir, obj)

        frame_paths = [os.path.join(o.path, obj) for o in os.scandir(sample_path) if o.is_dir()]

        # pick two different elements randomly
        if len(frame_paths) > 1:
            indices = np.random.choice(len(frame_paths), size=2, replace=False)
            frame1_path = frame_paths[indices[0]]
            frame2_path = frame_paths[indices[1]]
        else:
            frame1_path = frame_paths[0]
            frame2_path = frame_paths[0]

        paths1 = glob.glob(frame1_path + '/*_skel.png')
        paths2 = glob.glob(frame2_path + '/*_skel.png')

        views = len(paths2)
        imgs, skels, w2cs, c2ws, intrinsics = self.pre_data(paths1, paths2, views)
        
        
        assert imgs.shape == (8, 256, 256, 3), "Actual shape: {} {}".format(imgs.shape, obj)
        assert skels.shape == (8, 256, 256, 3), "Actual shape: {} {}".format(skels.shape, obj)
        assert w2cs.shape == (8, 4, 4), "Actual shape: {} {}".format(w2cs.shape, obj)
        assert c2ws.shape == (8, 4, 4), "Actual shape: {} {}".format(c2ws.shape, obj)
        assert intrinsics.shape == (8, 3, 3), "Actual shape: {} {}".format(intrinsics.shape, obj)
        


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
                'w2cs': w2cs,
                'c2ws': c2ws,
                'intrinsics': intrinsics,
                'filename': sample_path.split('/')[-2]  + sample_path.split('/')[-1]
        }
        return data