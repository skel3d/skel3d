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

class ObjaverseSkelDataset(ObjaverseDataset):

    def __init__(self,
                 root_dir = '',
                 cfg = None,
                 debug = False,
                 bw = False,
                 ) -> None:
        super().__init__(root_dir=root_dir, cfg=cfg, debug=debug)
        self.bw = bw

    def read_samples(self):
        json_file = 'val.json' if self.validation else 'train.json'
        # load the file name
        with open(os.path.join(self.root_dir, json_file)) as f:
            self.paths = json.load(f)
            print('============= length of dataset time %d =============' % len(self.paths))
        
    
    def pre_data(self, paths, views):
        '''
        load the data for given filename 
        '''
        color = [1., 1., 1., 1.]
        skels = []
        imgs = []
        w2cs = []
        intrinsics = []
        index = range(views) if self.validation else torch.randperm(views)

        # # find the closer views
        for i in index[:self.load_view]:
            skel = self.process_img(self.load_im(paths[i], color)).unsqueeze(0)
            if self.bw:
                skel = skel.mean(dim=1, keepdim=True) 
                skel = (skel > 0.5).float()
                
            skels.append(skel)
            img = self.process_img(self.load_im(paths[i].replace('_skel.png', '.png'), color)).unsqueeze(0)
            imgs.append(img)
            w2c_gl = np.load(paths[i].replace('_skel.png', '.npy'))
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

        #frame_paths = [os.path.join(o.path, obj) for o in os.scandir(sample_path) if o.is_dir()]
        frame_paths = [o.path for o in os.scandir(sample_path) if o.is_dir()]

        indices = np.random.choice(len(frame_paths), size=1, replace=False)
        frame_path = frame_paths[indices[0]]
        paths = glob.glob(frame_path + '/*_skel.png')
        views = len(paths)
        imgs, skels, w2cs, c2ws, intrinsics = self.pre_data(paths, views)
        

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