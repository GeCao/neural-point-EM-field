import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from src.EM.scenes import NeuralScene
from src.EM.renderer import PointLightFieldRenderer
from src.EM.managers import AbstractManager
from src.EM.losses import ChannelLoss


class PointLFEMModel(object):
    def __init__(
        self,
        scene: NeuralScene,
        opt,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs
    ) -> None:
        super(PointLFEMModel, self).__init__(*args, **kwargs)

        self.opt = opt
        self.device = device
        self.dtype = dtype

        self.scene = scene
        self.renderer = PointLightFieldRenderer(
            scene=scene, device=self.device, dtype=self.dtype
        )

        self.train_dataloader = DataLoader(
            scene,
            shuffle=True,
            batch_size=self.opt["batch_size"],
            num_workers=self.opt["num_workers"],
            drop_last=True,
        )
        self.test_dataloader = DataLoader(
            scene,
            shuffle=False,
            batch_size=self.opt["batch_size"],
            num_workers=self.opt["num_workers"],
            drop_last=False,
        )
        self.loss = ChannelLoss()
        self.optimizer = Adam(
            [p for p in self.renderer.parameters() if p.requires_grad == True],
            lr=self.opt["lr"],
        )

    def train_on_scene(self, scene: NeuralScene, epoch: int):
        self.renderer.train()  # Set train flags as true

        loss_list = []
        for ray_info, pts_info, gt_channels in tqdm(self.train_dataloader):
            # Typically [B, n_rays,            (3+1+1+1+1)] - ray_info
            # Typically [B, n_rays, K_closest, (3+1+1+1+1)] - pts_info
            # Typically [B, n_rays,            (3+2+2+1)  ] - gt_channels
            # points, distance, walk, pitch, azimuth in pts_info
            start_time = time.time()
            predicted_channels = self.renderer.forward_on_batch(ray_info, pts_info)
            end_time = time.time()

            loss = self.loss(predicted_channels, gt_channels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())

        return loss_list
