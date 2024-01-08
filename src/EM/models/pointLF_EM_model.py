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

        if opt["is_training"]:
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
            ray_o,
            probe_indices,
            rx_to_probe_and_tx_info,
            probe_to_tx_info,
            gt_ch,
        ) in self.train_dataloader:
            # Typically [B, 1,       3        ] - ray_o
            # Typically [B, n_ray,            ] - probe_indices
            # Typically [B, n_ray+1, (3+1+1+1)] - rx_probetx_info
            # Typically [B, n_ray,   (3+1+1+1)] - probe_tx_info
            # Typically [B,          1        ] - gt_ch
            # points, distance, proj_distance, pitch, azimuth in pts_info
            start_time = time.time()
            Batch_size, n_rays = probe_indices.shape[0:2]
            env_idx = 0
            [
                probe_to_pts_indices,
                probe_to_pts_info,
            ] = self.scene.ray_sampler.GetProbePtsIndicesAndInfo(
                env_idx=env_idx, train_type=int(TrainType.TRAIN)
            )  # [n_probes, K_closet, 3]
            n_probes, K_closest, _ = probe_to_pts_info.shape[-3:]
            probe_mask = (
                torch.linspace(
                    0,
                    Batch_size - 1,
                    Batch_size,
                    device=torch.device("cpu"),
                    dtype=torch.int32,
                )
                .reshape(Batch_size, 1)
                .repeat(1, n_rays)
                .flatten()
                .tolist(),
                probe_indices.flatten().cpu().tolist(),
            )  # ([B*n_rays], [B*n_rays])
            probe_to_pts_indices = probe_to_pts_indices.view(n_probes, K_closest)
            probe_to_pts_indices = probe_to_pts_indices[
                probe_indices.flatten().cpu().tolist()
            ].view(
                Batch_size, n_rays, K_closest
            )  # [B, n_rays, K_closest]
            probe_to_pts_and_tx_info = torch.cat(
                (
                    probe_to_pts_info[probe_indices.flatten().cpu().tolist()].view(
                        Batch_size, n_rays, K_closest, 6
                    ),
                    probe_to_tx_info.unsqueeze(-2),
                ),
                dim=-2,
            )  # [B, n_rays, K_closest+1, 6]
            x = self.scene.GetPointCloud(env_index=0)  # TODO: [n_pts, 3]
            light_probe_pos = self.scene.GetLightProbePosition()
            predicted_ch = self.renderer.forward_on_batch(
                x,
                light_probe_pos,
                probe_mask,
                rx_to_probe_and_tx_info,
                probe_to_pts_indices,
                probe_to_pts_and_tx_info,
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
                ray_o,
                probe_indices,
                rx_to_probe_and_tx_info,
                probe_to_tx_info,
                gt_ch,
            ) in self.validation_dataloader:
                Batch_size, n_rays = probe_indices.shape[0:2]
                env_idx = 0
                assert len(self.validation_dataloader.dataset.validation_names) == 1
                [
                    probe_to_pts_indices,
                    probe_to_pts_info,
                ] = self.scene.ray_sampler.GetProbePtsIndicesAndInfo(
                    env_idx=env_idx,
                    train_type=int(TrainType.VALIDATION),
                    validation_name=self.validation_dataloader.dataset.validation_names[
                        0
                    ],
                )  # [n_probes, K_closet, 3]
                n_probes, K_closest, _ = probe_to_pts_info.shape[-3:]
                probe_mask = (
                    torch.linspace(
                        0,
                        Batch_size - 1,
                        Batch_size,
                        device=torch.device("cpu"),
                        dtype=torch.int32,
                    )
                    .reshape(Batch_size, 1)
                    .repeat(1, n_rays)
                    .flatten()
                    .tolist(),
                    probe_indices.flatten().cpu().tolist(),
                )  # ([B*n_rays], [B*n_rays])
                probe_to_pts_indices = probe_to_pts_indices.view(n_probes, K_closest)
                probe_to_pts_indices = probe_to_pts_indices[
                    probe_indices.flatten().cpu().tolist()
                ].view(
                    Batch_size, n_rays, K_closest
                )  # [B, n_rays, K_closest]
                probe_to_pts_and_tx_info = torch.cat(
                    (
                        probe_to_pts_info[probe_indices.flatten().cpu().tolist()].view(
                            Batch_size, n_rays, K_closest, 6
                        ),
                        probe_to_tx_info.unsqueeze(-2),
                    ),
                    dim=-2,
                )  # [B, n_rays, K_closest+1, 6]
                x = self.scene.GetPointCloud(env_index=0)  # TODO: [n_pts, 3]
                light_probe_pos = self.scene.GetLightProbePosition()
                predicted_ch = self.renderer.forward_on_batch(
                    x,
                    light_probe_pos,
                    probe_mask,
                    rx_to_probe_and_tx_info,
                    probe_to_pts_indices,
                    probe_to_pts_and_tx_info,
                )
                test_loss = self.loss(predicted_ch, gt_ch)
                test_loss_list.append(test_loss.item())

                predicted_gains = torch.cat((predicted_gains, predicted_ch), dim=0)
                gt_gains = torch.cat((gt_gains, gt_ch), dim=0)
                rx_pos = torch.cat((rx_pos, ray_o[:, 0, 0:3]), dim=0)
                # TODO: tx-idx = 3 only
                tx_pos = self.scene.GetTransmitter(
                    transmitter_idx=0,
                    train_type=int(TrainType.VALIDATION),
                    validation_name=self.scene.validation_target[0],
                ).GetSourceLocation()

        mean_test_loss = np.array(test_loss_list).mean()
        return [mean_test_loss, tx_pos, rx_pos, predicted_gains, gt_gains]

    def GetOptimizer(self):
        return self.optimizer

    def GetRenderer(self):
        return self.renderer
