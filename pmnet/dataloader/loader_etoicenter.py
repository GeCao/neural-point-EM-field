from __future__ import print_function, division
import os
import torch
import torch.nn.functional as F
import h5py
import pandas as pd
import cv2
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
warnings.filterwarnings("ignore")


class PMnetEtoiCenter(Dataset):
    def __init__(self,
                 is_train: bool = True,
                 dir_dataset="sionna_etoicenter_shadowing_fastfading/",               
                 transform= transforms.ToTensor()):
        
        self.dir_dataset = dir_dataset
        self.transform = transform
        
        # load all of the dataset at first
        filename = "train.h5" if is_train else "validation.h5"
        data = h5py.File(os.path.join(self.dir_dataset, filename))
        self.ch = np.array(data["gain"])[..., 0]  # [T, H, W]
        self.rx = np.array(data["rx"])  # [T, H, W, dim=3]
        self.tx = np.array(data["tx"])  # [T, dim=3]
        if is_train:
            self.ch = self.ch[0::6]
            self.rx = self.rx[0::6]
            self.tx = self.tx[0::6]
        T, H, W, _ = self.rx.shape
        data.close()
        
        # Normalization:
        n_tx = self.ch.shape[0]
        mean_ch = np.mean(self.ch.reshape(n_tx, -1), axis=1).reshape(-1, 1, 1)  # [T,]
        min_ch = np.min(self.ch.reshape(n_tx, -1), axis=1).reshape(-1, 1, 1)  # [T,]
        inf_mask = np.abs(self.ch) < 1e-3
        self.ch[inf_mask] = -1000
        max_ch = np.max(self.ch.reshape(n_tx, -1), axis=1).reshape(-1, 1, 1)  # [T,]
        self.ch[inf_mask] = 0
        self.ch = (self.ch - min_ch) / (max_ch - min_ch)  # [0, 1]
        self.ch[inf_mask] = 0
        
        # prepare map
        data = h5py.File(os.path.join(self.dir_dataset, "map.h5"))
        self.maps = np.array(data["map"])  # [1, H, W]
        data.close()
        
        # prepare tx map
        self.tx_maps = np.zeros_like(self.ch)  # [T, H, W]
        AABB_min = self.rx.reshape(n_tx, -1, 3).min(axis=1)[:, 0:2]  # [T, 2]
        AABB_max = self.rx.reshape(n_tx, -1, 3).max(axis=1)[:, 0:2]  # [T, 2]
        AABB_spacing = AABB_max - AABB_min
        AABB_spacing[..., 0] = AABB_spacing[..., 0] / (W - 1)
        AABB_spacing[..., 1] = AABB_spacing[..., 1] / (H - 1)
        tx_positions = np.asarray((self.tx[:, 0:2] - AABB_min) / AABB_spacing, dtype=np.int32)  # [T, 2]
        x_ = tx_positions[:, 0].tolist()  # [T,]
        y_ = tx_positions[:, 1].tolist()  # [T,]
        batch = np.linspace(0, n_tx - 1, n_tx)
        batch = np.asarray(batch, dtype=np.int32).tolist()
        self.tx_maps[(batch, y_, x_)] = 1.0
        
        # cv2.imwrite("/home/gecao2/homework/ACEM/pmnet/tx_map.png", cv2.flip(self.tx_maps[0] * 255, 0))
        # cv2.imwrite("/home/gecao2/homework/ACEM/pmnet/map.png", cv2.flip(self.maps[0] * 255, 0))
        # cv2.imwrite("/home/gecao2/homework/ACEM/pmnet/pmap.png", cv2.flip(self.ch[0] * 255, 0))
        # print(self.tx_maps.max(), self.tx_maps.min(), self.tx_maps.mean())
        # print(self.maps.max(), self.maps.min(), self.maps.mean())
        # exit(0)
        
        # Pad on x from 30 -> 32
        pad = (1, 1)
        self.ch = torch.from_numpy(self.ch).to(torch.float32)
        self.maps = torch.from_numpy(self.maps).to(torch.float32)
        self.tx_maps = torch.from_numpy(self.tx_maps).to(torch.float32)
        
        self.ch = F.pad(self.ch, pad=pad, mode="replicate")
        self.maps = F.pad(self.maps, pad=pad, mode="constant")
        self.tx_maps = F.pad(self.tx_maps, pad=pad, mode="constant")
        
        # print(self.ch.min(), self.ch.max())
        # print(self.ch.shape, self.rx.shape, self.tx.shape)
        # exit(0)

    def __len__(self):
        return self.ch.shape[0]
    
    def __getitem__(self, idx):

        #Load city map
        image_buildings = self.maps[0]
        
        #Load Tx (transmitter):
        image_Tx = self.tx_maps[idx]

        #Load Rx (reciever): (not used in our training)

        #Load Power:
        image_power = self.ch[idx]    

        inputs=torch.stack([image_buildings, image_Tx], dim=2)

        # if self.transform:
        #     inputs = self.transform(inputs).type(torch.float32)
        #     image_power = self.transform(image_power).type(torch.float32)
        inputs = inputs.permute((2, 0, 1))
        image_power = image_power.unsqueeze(-3)

        return [inputs , image_power]

