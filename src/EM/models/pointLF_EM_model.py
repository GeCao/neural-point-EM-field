import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from tqdm import tqdm

from src.EM.scenes import NeuralScene
from src.EM.renderer import PointLightFieldRenderer
from src.EM.managers import AbstractManager, SceneDataSet
from src.EM.losses import ChannelLoss
from src.EM.utils import TrainType, LearnTarget


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

        self.learn_target = int(LearnTarget.RECEIVER_GAIN)
        if "coverage_map" in opt["data_set"] or "cm" in opt["data_set"]:
            self.learn_target = int(LearnTarget.COVERAGE_MAP)

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
        self.validation_dataloader = DataLoader(
            SceneDataSet(scene=scene, train_type=int(TrainType.VALIDATION)),
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
        # self.optimizer = SGD(
        #     [p for p in self.renderer.parameters() if p.requires_grad == True],
        #     lr=self.opt["lr"],
        # )

        self.scene.InfoLog("Point Light Field Model fully prepared")

    def train_on_scene(self, epoch: int) -> float:
        self.renderer.train()  # Set train flags as true

        loss_list = []
        for (
            x,
            ray_info,
            pts_info,
            K_closest_indices,
            valid_rays,
            sky_mask,
            tx_info,
            gt_ch,
        ) in self.train_dataloader:
            # Typically [B,         n_pts,     3          ] - x
            # Typically [B, n_rays,            (3+3      )] - ray_info
            # Typically [B, n_rays, K_closest, (1+1+1+1  )] - pts_info
            # Typically [B, n_rays,            1          ] - valid_rays
            # Typically [B, n_rays, K_closest, 1          ] - K_closest_indices
            # Typically [B, n_rays,            (3+2+2+1)  ] - gt_ch
            # points, distance, proj_distance, pitch, azimuth in pts_info
            start_time = time.time()
            Batch_size, n_rays, K_closest = K_closest_indices.shape[0:3]
            # Prepare all of the dataset
            # valid_rays = valid_rays.flatten()  # [B*n_rays,], dtype=bool
            # ray_info = ray_info.reshape(-1, *ray_info.shape[2:])[valid_rays]
            # pts_info = pts_info.reshape(-1, *pts_info.shape[2:])[valid_rays]
            # gt_ch = gt_ch.reshape(-1, *gt_ch.shape[2:])[valid_rays]
            # K_closest_indices = K_closest_indices.reshape(
            #     -1, *K_closest_indices.shape[2:]
            # )[valid_rays]
            K_closest_mask = (
                torch.linspace(
                    0,
                    Batch_size - 1,
                    Batch_size,
                    device=torch.device("cpu"),
                    dtype=torch.int32,
                )
                .reshape(Batch_size, 1, 1)
                .repeat(1, n_rays, K_closest)
                .flatten()
                .tolist(),
                K_closest_indices.flatten().cpu().tolist(),
            )  # ([B*n_rays*K_closest], [B*n_rays*K_closest])
            predicted_ch = self.renderer.forward_on_batch(
                x, ray_info, pts_info, K_closest_mask, sky_mask, tx_info
            )
            end_time = time.time()

            loss = self.loss(predicted_ch, gt_ch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scene.InfoLog(f"epoch = {epoch}, loss = {loss.item()}")
            # self.scene.log_manager.WriterAddScalar("loss", loss, epoch)

            loss_list.append(loss.item())

        mean_loss = np.array(loss_list).mean()
        return mean_loss

    def validation_on_scene(self) -> List[torch.Tensor]:
        self.renderer.train()  # Set train flags as true

        with torch.no_grad():
            test_loss_list = []
            predicted_gains = torch.zeros((0, 1)).to(self.device).to(self.dtype)
            gt_gains = torch.zeros((0, 1)).to(self.device).to(self.dtype)
            rx_pos = torch.zeros((0, 3)).to(self.device).to(self.dtype)
            for (
                x,
                ray_info,
                pts_info,
                K_closest_indices,
                valid_rays,
                sky_mask,
                tx_info,
                gt_ch,
            ) in self.validation_dataloader:
                Batch_size, n_rays, K_closest = K_closest_indices.shape[0:3]
                # Prepare all of the dataset
                # valid_rays = valid_rays.flatten()  # [B*n_rays,], dtype=bool
                # ray_info = ray_info.reshape(-1, *ray_info.shape[2:])[valid_rays]
                # pts_info = pts_info.reshape(-1, *pts_info.shape[2:])[valid_rays]
                # gt_ch = gt_ch.reshape(-1, *gt_ch.shape[2:])[valid_rays]
                # K_closest_indices = K_closest_indices.reshape(
                #     -1, *K_closest_indices.shape[2:]
                # )[valid_rays]
                K_closest_mask = (
                    torch.linspace(
                        0,
                        Batch_size - 1,
                        Batch_size,
                        device=torch.device("cpu"),
                        dtype=torch.int32,
                    )
                    .reshape(Batch_size, 1, 1)
                    .repeat(1, n_rays, K_closest)
                    .flatten()
                    .tolist(),
                    K_closest_indices.flatten().cpu().tolist(),
                )
                predicted_ch = self.renderer.forward_on_batch(
                    x, ray_info, pts_info, K_closest_mask, sky_mask, tx_info
                )
                test_loss = self.loss(predicted_ch, gt_ch)
                test_loss_list.append(test_loss.item())

                valid_rays = valid_rays.flatten()  # [B*n_rays,], dtype=bool
                ray_info = ray_info.reshape(-1, *ray_info.shape[2:])[valid_rays]
                predicted_ch = predicted_ch.reshape(-1, *predicted_ch.shape[2:])[
                    valid_rays
                ]
                gt_ch = gt_ch.reshape(-1, *gt_ch.shape[2:])[valid_rays]

                predicted_gains = torch.cat(
                    (
                        predicted_gains,
                        predicted_ch.view(-1, predicted_ch.shape[-1])[:, 0:1],
                    ),
                    dim=0,
                )
                gt_gains = torch.cat(
                    (
                        gt_gains,
                        gt_ch.view(-1, gt_ch.shape[-1])[:, 0:1],
                    ),
                    dim=0,
                )
                rx_pos = torch.cat(
                    (
                        rx_pos,
                        ray_info.view(-1, ray_info.shape[-1])[:, 0:3],
                    ),
                    dim=0,
                )

        mean_test_loss = np.array(test_loss_list).mean()
        return [mean_test_loss, rx_pos, predicted_gains, gt_gains]

    def GetOptimizer(self):
        return self.optimizer

    def GetRenderer(self):
        return self.renderer
