import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from src.EM.scenes import NeuralScene
from src.EM.renderer import PointLightFieldRenderer
from src.EM.managers import AbstractManager, SceneDataSet
from src.EM.losses import ChannelLoss
from src.EM.utils import TrainType


class PointLFEMModel(object):
    def __init__(
        self,
        scene: NeuralScene,
        opt,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs,
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
            SceneDataSet(scene=scene, train_type=int(TrainType.TRAIN)),
            shuffle=False,
            batch_size=self.opt["batch_size"],
            num_workers=self.opt["num_workers"],
            drop_last=True,
        )
        self.test_dataloader = DataLoader(
            SceneDataSet(scene=scene, train_type=int(TrainType.TEST)),
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

        self.scene.InfoLog("Point Light Field Model fully prepared")

    def train_on_scene(self, epoch: int) -> torch.Tensor:
        self.renderer.train()  # Set train flags as true

        loss_list = []
        for x, ray_info, pts_info, K_closest_indices, gt_channels in tqdm(
            self.train_dataloader
        ):
            # Typically [B,         n_pts,     3          ] - x
            # Typically [B, n_rays,            (3+3      )] - ray_info
            # Typically [B, n_rays, K_closest, (3+1+1+1+1)] - pts_info
            # Typically [B, n_rays, K_closest, 1          ] - K_closest_indices
            # Typically [B, n_rays,            (3+2+2+1)  ] - gt_channels
            # points, distance, walk, pitch, azimuth in pts_info
            start_time = time.time()
            Batch_size, n_rays, K_closest = (
                K_closest_indices.shape[0],
                K_closest_indices.shape[1],
                K_closest_indices.shape[2],
            )
            K_closest_mask = (
                torch.linspace(
                    0, Batch_size - 1, Batch_size, device=x.device, dtype=torch.int32
                )
                .reshape(Batch_size, 1, 1)
                .repeat(1, n_rays, K_closest)
                .flatten()
                .cpu()
                .tolist(),
                K_closest_indices.flatten().cpu().tolist(),
            )
            predicted_channels = self.renderer.forward_on_batch(
                x, ray_info, pts_info, K_closest_mask
            )
            end_time = time.time()

            loss = self.loss(predicted_channels, gt_channels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scene.InfoLog(f"epoch = {epoch}, loss = {loss.item()}")
            # self.scene.log_manager.WriterAddScalar("loss", loss, epoch)

            loss_list.append(loss.item())

        test_loss_list = []
        with torch.no_grad():
            for x, ray_info, pts_info, K_closest_indices, gt_channels in tqdm(
                self.test_dataloader
            ):
                Batch_size, n_rays, K_closest = (
                    K_closest_indices.shape[0],
                    K_closest_indices.shape[1],
                    K_closest_indices.shape[2],
                )
                K_closest_mask = (
                    torch.linspace(
                        0,
                        Batch_size - 1,
                        Batch_size,
                        device=x.device,
                        dtype=torch.int32,
                    )
                    .reshape(Batch_size, 1, 1)
                    .repeat(1, n_rays, K_closest)
                    .flatten()
                    .cpu()
                    .tolist(),
                    K_closest_indices.flatten().cpu().tolist(),
                )
                predicted_channels = self.renderer.forward_on_batch(
                    x, ray_info, pts_info, K_closest_mask
                )
                test_loss = self.loss(predicted_channels, gt_channels)
                test_loss_list.append(loss.item())

        return np.array(loss_list).mean(), np.array(test_loss_list).mean()

    def GetOptimizer(self):
        return self.optimizer

    def GetRenderer(self):
        return self.renderer
