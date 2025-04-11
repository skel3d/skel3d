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
from datasets.objaverse_hdf5_coord import ObjaverseHDF5CoordDataset
import argparse
import h5py
from tqdm import tqdm

class ObjaverseHDF5TimeDataset(ObjaverseHDF5CoordDataset):
    def __getitem__(self, index):
        # Ensure the HDF5 file is open
        if self.h5f is None:
            self.h5f = h5py.File(self.hdf5_file_path, 'r')
        

        try:
            object_group = self.h5f[self.object_names[index]]
            # Chose two random frames            
            frame_names =  np.random.choice(list(object_group.keys()), 2)

            # concatenate one of the view from the first frame and the rest from the second frame
            data = object_group[frame_names[0]]
            ref_data = object_group[frame_names[1]]

            imgs, skels, w2cs, c2ws, intrinsics = self.pre_data(data, ref_data=ref_data)

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
        
        data = {
                'images': imgs,
                'skeletons': skels,
                'w2cs': w2cs,
                'c2ws': c2ws,
                'intrinsics': intrinsics,
                'filename': self.object_names[index]
        }
        return data

   