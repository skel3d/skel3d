import os
import sys
import glob
import json
import math
import torch
import torchvision
import numpy as np



from PIL import Image
from pathlib import Path
from einops import rearrange
from torchvision import transforms
from datasets.objaverse_hdf5 import ObjaverseHDF5Dataset
import argparse
import h5py
from tqdm import tqdm

def sort_joints_by_length_and_degree_normalized(joint_coords, bones, eps=1e-8):
    """
    Sorts joints by an importance measure derived from:
      - sum of connected bone lengths (normalized to [0,1])
      - connection degree (normalized to [0,1])
    
    Args:
        joint_coords (torch.Tensor): 
            Shape (B, num_joints, 3), where:
              B = batch size
              num_joints = number of joints
              3 = (x, y, z)
        bones (list or torch.Tensor): 
            A list of edges representing the skeleton's adjacency.
            Each element is (j1, j2) denoting a bone between joints j1 and j2.
            This adjacency is assumed to be the same for the entire batch.
        eps (float, optional):
            Small constant to avoid division by zero in normalization.
    
    Returns:
        sorted_indices (torch.LongTensor):
            Shape (B, num_joints), where each row is the sorted ordering
            of joints for that sample in descending order of importance.
    """

    num_joints, _ = joint_coords.shape
    
    # Convert bones to a Python list of edges if needed
    if isinstance(bones, torch.Tensor):
        bones = bones.tolist()
    
    # Build adjacency list: adjacency[j] => list of joints connected to j
    adjacency = [[] for _ in range(num_joints)]
    for j1, j2 in bones:
        adjacency[j1].append(j2)
        adjacency[j2].append(j1)
    
    # We'll store sorted indices for each sample
    all_sorted_indices = []
    
    coords = joint_coords  # shape: (num_joints, 3)
    
    # Compute degree and sum of bone lengths
    degree = torch.zeros(num_joints, dtype=torch.float, device=coords.device)
    sum_length = torch.zeros(num_joints, dtype=torch.float, device=coords.device)
    
    for j in range(num_joints):
        connected_joints = adjacency[j]
        degree[j] = len(connected_joints)
        
        total_len = 0.0
        for cj in connected_joints:
            dist = torch.norm(coords[j] - coords[cj], p=2)
            total_len += dist
        
        sum_length[j] = total_len
    
    # ----------------- NORMALIZE -----------------
    # 1. Normalize sum_length to [0,1]
    min_sum_len, max_sum_len = sum_length.min(), sum_length.max()
    if (max_sum_len - min_sum_len) < eps:
        # Edge case: all sum_length are identical or no meaningful variation
        # We can just set them all to 0.5 or 0.0
        sum_length_norm = torch.zeros_like(sum_length)
    else:
        sum_length_norm = (sum_length - min_sum_len) / (max_sum_len - min_sum_len + eps)
    
    # 2. Normalize degree to [0,1]
    min_deg, max_deg = degree.min(), degree.max()
    if (max_deg - min_deg) < eps:
        # Edge case: if all degrees are the same
        degree_norm = torch.zeros_like(degree)
    else:
        degree_norm = (degree - min_deg) / (max_deg - min_deg + eps)
    
    # ----------------- IMPORTANCE -----------------
    # For simplicity, let's just add them
    # or you can use importance = alpha * sum_length_norm + beta * degree_norm
    importance = sum_length_norm + degree_norm
    
    # ----------------- SORT -----------------
    sorted_idx = torch.argsort(importance, descending=True)
    all_sorted_indices.append(sorted_idx)
    
    sorted_indices = torch.stack(all_sorted_indices, dim=0)
    return sorted_indices



    


class ObjaverseHDF5CoordDataset(ObjaverseHDF5Dataset):
    def __init__(self,
                 root_dir = '',
                 cfg = None,
                 debug = False,
                 ) -> None:
        super().__init__(root_dir=root_dir, cfg=cfg, debug=debug)

        
        self.image_size = 256
        self.used_joints = 40

        

        

        self.intrinsic = torch.tensor([[ 2.1875,  0.0000,  0.0000,  0.0000],
                                        [ 0.0000,  2.1875,  0.0000,  0.0000],
                                        [ 0.0000,  0.0000, -1.0020, -0.2002],
                                        [ 0.0000,  0.0000, -1.0000,  0.0000]], dtype=torch.float32)
        
        
        

     

    

    def project_joints_to_2d(self, joints, extrinsic):
        # Convert joints to homogeneous coordinates
        location_h = torch.cat((joints, torch.ones(joints.shape[0], 1)), dim=1)
        
        # Camera coordinates
        cam_coord_t = torch.matmul(extrinsic, location_h.T).T  # shape: (N, 3)
        cam_coord_h_t = torch.cat([cam_coord_t, torch.ones(cam_coord_t.shape[0], 1)], dim=1)  # shape: (N, 4)
        # NDC
        ndc_t = torch.matmul(self.intrinsic, cam_coord_h_t.T).T
        ndc_t = ndc_t / ndc_t[:, 3].unsqueeze(1)
        x_ndc, y_ndc, z_ndc, _ = ndc_t.T
        image_width = self.image_size
        image_height = self.image_size
        # Pixel coordinates
        pixel_x_t = (x_ndc + 1.0) * (image_width / 2.0)
        pixel_y_t = (1.0 - y_ndc) * (image_height / 2.0)
        
        return torch.stack((pixel_x_t, pixel_y_t), dim=1)
    
    def pre_data(self, data, ref_data=None):
        '''
        load the data for given filename 
        '''        
        imgs = []
        #skels = []
        w2cs = []
        intrinsics = []

        view_ids = [d.split('_')[0] for d in data.keys() if d.endswith('_image')]
        views = len(view_ids)

        index = range(views) if self.validation else torch.randperm(views)
        
        bones = torch.tensor(np.array(data['bones']))
        joints = torch.tensor(data['joints']).float()

        if ref_data is not None:
            ref_joints = torch.tensor(ref_data['joints']).float()
            ref_view_ids = [d.split('_')[0] for d in ref_data.keys() if d.endswith('_image')]

        if bones.shape[0] > 0:
            sorted_ids = sort_joints_by_length_and_degree_normalized(joints, bones)            
        else:
            sorted_ids = torch.arange(joints.shape[0]).unsqueeze(0)

        joints_2ds = []
        # # find the closer views
        for e, i in enumerate(index[:self.load_view]):
            if e == 0 and ref_data is not None:
                img = self.process_img(self.load_im_hdf5(ref_data[f'{ref_view_ids[i]}_image'])).unsqueeze(0)
                w2c_gl = ref_data[f'{ref_view_ids[i]}_camera']
                joints_2d = self.project_joints_to_2d(ref_joints, extrinsic=torch.tensor(w2c_gl).float())
            else:
                img = self.process_img(self.load_im_hdf5(data[f'{view_ids[i]}_image'])).unsqueeze(0)
                w2c_gl = data[f'{view_ids[i]}_camera']
                joints_2d = self.project_joints_to_2d(joints, extrinsic=torch.tensor(w2c_gl).float()) 
            
            imgs.append(img)
            joints_2d = joints_2d[sorted_ids[0, :self.used_joints]]
            joints_2ds.append(joints_2d)
                        
            w2cs.append(w2c_gl)
            focal = .5 / np.tan(.5 * 0.8575560450553894)
            intrinsics.append(np.array([[focal, 0.0, 1.0 / 2.0],
                                        [0.0, focal, 1.0 / 2.0],
                                        [0.0, 0.0, 1.0]]))           


            
        imgs = torch.cat(imgs)
        skels = torch.zeros_like(imgs)

        #skels = torch.cat(skels)
        joints_2ds = torch.stack(joints_2ds)

        # take the joints for the first image       
        joints_first = joints_2ds[0]   
        # Get the indeces of the joints that are outside the image
        outside = (joints_first[:, 0] < 0) | (joints_first[:, 0] >= self.image_size) | (joints_first[:, 1] < 0) | (joints_first[:, 1] >= self.image_size)
        # Remove the joints that are outside the image
        joints_2ds = joints_2ds[:, ~outside, :]

        # Clamp the other joints to the image size
        joints_2ds[:, :, 0] = torch.clamp(joints_2ds[:, :, 0], 0, self.image_size - 1)
        joints_2ds[:, :, 1] = torch.clamp(joints_2ds[:, :, 1], 0, self.image_size - 1)

        joints_2ds = torch.cat([joints_2ds, joints_2ds[:, -1:, :].repeat(1, self.used_joints - joints_2ds.size(1), 1)], dim=1)

        if joints_2d.size(1) != self.used_joints:
            joints_2d = torch.zeros((joints_2ds.size(0), self.used_joints, 2), dtype=joints_2ds.dtype, device=joints_2ds.device) 

        


        x = joints_2ds[:, :, 0].long().reshape(-1)
        y = joints_2ds[:, :, 1].long().reshape(-1)

        view_indices = torch.arange(len(joints_2ds)).unsqueeze(1).expand(-1, joints_2ds.shape[1]).reshape(-1)
        src_view_indices = torch.ones_like(view_indices) * 0

        src_x = joints_2ds[0, :, 0].long().repeat(len(joints_2ds))
        src_y = joints_2ds[0, :, 1].long().repeat(len(joints_2ds))

        skels[view_indices, y, x] = imgs[src_view_indices, src_y, src_x]


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
    

    