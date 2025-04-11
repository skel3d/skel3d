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
from torch.utils.data import Dataset
import argparse
import h5py
from tqdm import tqdm




class ObjaverseHDF5Dataset(Dataset):
    def __init__(self,
                 root_dir = '',
                 cfg = None,
                 debug = False,
                 ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)        

        self.total_view = cfg.total_view
        self.load_view = cfg.load_view
        self.debug = debug
        self.validation = cfg.validation

        

        self.read_samples()

        self.opengl_to_colmap = torch.tensor([[  1,  0,  0,  0],
                                              [  0, -1,  0,  0],
                                              [  0,  0, -1,  0],
                                              [  0,  0,  0,  1]], dtype=torch.float32)
        
        

        #transforms.Lambda(rearrange_func)

        image_transforms = [
            torchvision.transforms.Resize(256),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
            transforms.Lambda(self.rearrange_func)
        ]
        self.image_transforms = transforms.Compose(image_transforms)

    def read_samples(self):

        hdf5_file = os.path.join(self.root_dir, 'val.hdf5' if self.validation else 'train.hdf5')        
        self.hdf5_file_path = hdf5_file
        self.h5f = None

        self.object_names = self._load_object_names()
        exlude = ['bac0fce4bc37425fb3a4596d02b95514',
                  '13262e8213d648f8926d718e5796e4c7',
                  '291d73dd9a1746c1947836bce1f446ab',
                  '30ea70178d624b87a39553108db61bca',
                  '63cb3f33ad7f4b739811409d802f921f',
                  '9c5317ed6f78424cb1e772d46308261d',
                  '8ddd8d6309244f629b5ee3eed72d94a4']
        self.object_names = [obj for obj in self.object_names if obj not in exlude]
        

    def _load_object_names(self):
        with h5py.File(self.hdf5_file_path, 'r') as f:
            return list(f.keys())

    def rearrange_func(self, x):
            return rearrange(x * 2. - 1., 'c h w -> h w c')

    def __len__(self):
        return len(self.object_names)
    

    def __enter__(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.hdf5_file_path, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5f is not None:
            self.h5f.close()
            self.h5f = None


    def load_im_hdf5(self, img):
        '''
        replace background pixel with random color in rendering
        '''        
        color = [1., 1., 1., 1.]   
        img = np.array(img)
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img
    
    def pre_data(self, data):
        '''
        load the data for given filename 
        '''        
        imgs = []
        skels = []
        w2cs = []
        intrinsics = []

        view_ids = [d.split('_')[0] for d in data.keys() if d.endswith('_image')]
        views = len(view_ids)
        index = range(views) if self.validation else torch.randperm(views)
        # # find the closer views
        for i in index[:self.load_view]:
            img = self.process_img(self.load_im_hdf5(data[f'{view_ids[i]}_image'])).unsqueeze(0)
            imgs.append(img)
            skel = self.process_img(self.load_im_hdf5(data[f'{view_ids[i]}_skeleton'])).unsqueeze(0)
            skels.append(skel)
            w2c_gl = data[f'{view_ids[i]}_camera']
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
        # Ensure the HDF5 file is open
        if self.h5f is None:
            self.h5f = h5py.File(self.hdf5_file_path, 'r')
        

        try:
            object_group = self.h5f[self.object_names[index]]
            frame_name =  np.random.choice(list(object_group.keys()))
            data = object_group[frame_name]
            imgs, skels, w2cs, c2ws, intrinsics = self.pre_data(data)

            if len(data['joints']) == 0:
                print('error in loading data idjdjdjdjdj', flush=True)
                print('filename:', self.object_names[index], flush=True)
            
            #Check the sapes of the data
            if imgs.shape[0] != self.load_view or skels.shape[0] != self.load_view or w2cs.shape[0] != self.load_view or c2ws.shape[0] != self.load_view or intrinsics.shape[0] != self.load_view:
                print('error in loading data', flush=True)
                print('filename:', self.object_names[index], flush=True)
                print('imgs.shape', imgs.shape, flush=True)                

                frame_name =  list(object_group.keys())[0]
                data = object_group[frame_name]
                imgs, skels, w2cs, c2ws, intrinsics = self.pre_data(data)
            
        except:
            print('error in loading data', flush=True)
            print('filename:', self.object_names[index], flush=True)
            frame_name =  list(object_group.keys())[0]# np.random.choice(list(object_group.keys()))
            data = object_group[frame_name]
            imgs, skels, w2cs, c2ws, intrinsics = self.pre_data(data)
        # debug the camera system for debugging
        if self.debug:
            asd
            import pdb
            from utils.vis_camera import vis_points_images_cameras


            pdb.set_trace()
            intrinsics[:, 0, :] *=256
            intrinsics[:, 1, :] *=256
            vis_points_images_cameras(w2cs, intrinsics, imgs, frustum_size=0.5, filename=filename.split('/')[-1] + 'camemra_ori.html')

        data = {
                'images': imgs,
                'skeletons': skels,
                'w2cs': w2cs,
                'c2ws': c2ws,
                'intrinsics': intrinsics,
                'filename': self.object_names[index]
        }
        return data

    def process_img(self, img):
        # Convert the image data to a PIL Image before processing
        img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        return self.image_transforms(img)
    

def main():

    parser = argparse.ArgumentParser(description='Test ObjaverseHDF5Dataset')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--total_view', type=int, default=12, help='Total number of views')
    parser.add_argument('--load_view', type=int, default=12, help='Number of views to load')
    parser.add_argument('--validation', action='store_true', help='Use validation set')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    cfg = type('Config', (object,), {
        'total_view': args.total_view,
        'load_view': args.load_view,
        'validation': args.validation
    })

    dataset = ObjaverseHDF5Dataset(root_dir=args.root_dir, cfg=cfg, debug=args.debug)

    for i in tqdm(range(len(dataset)), desc="Processing dataset"):
        data = dataset[i]

        assert 'images' in data
        #print(data['images'].shape, data['skeletons'].shape, data['w2cs'].shape, data['c2ws'].shape, data['intrinsics'].shape, data['filename'])

    print('Done')

if __name__ == '__main__':
    main()