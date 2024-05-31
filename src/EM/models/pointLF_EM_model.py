import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

from src.EM.scenes import NeuralScene
from src.EM.renderer import PointLightFieldRenderer
from src.EM.managers import AbstractManager, SceneDataSet
from src.EM.losses import ChannelLoss
from src.EM.utils import TrainType, LearnTarget, ScaleAABB


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

        batch_size = self.opt["batch_size"]
        num_workers = self.opt["num_workers"]
        if opt["is_training"]:
            self.train_dataloader = DataLoader(
                SceneDataSet(
                    scene=scene, train_type=int(TrainType.TRAIN), batch_size=batch_size
                ),
                shuffle=False,
                batch_size=1,
                num_workers=num_workers,
                drop_last=True,
            )
            self.test_dataloader = DataLoader(
                SceneDataSet(
                    scene=scene, train_type=int(TrainType.TEST), batch_size=batch_size
                ),
                shuffle=False,
                batch_size=1,
                num_workers=num_workers,
                drop_last=False,
            )
        self.validation_dataloader = DataLoader(
            SceneDataSet(
                scene=scene, train_type=int(TrainType.VALIDATION), batch_size=batch_size
            ),
            shuffle=False,
            batch_size=1,
            num_workers=num_workers,
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
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)

        self.scene.InfoLog("Point Light Field Model fully prepared")

    def train_on_scene(self, epoch: int) -> float:
        self.renderer.train()  # Set train flags as true

        loss_list = []
        env_idx = 0
        pts = self.scene.GetPointCloud(env_index=0)  # TODO: [n_pts, 3]
        for dataset in self.train_dataloader:
            dataset = [val.squeeze(0) for val in dataset]

            gt_ch = dataset[4]
            interactions = None if len(dataset) <= 5 else dataset[5]
            valid_index = None
            if interactions is not None:
                valid_index = interactions[:, :, 0, 0:1] >= -1e-5
            else:
                valid_index = gt_ch.abs() > 0.01
            if valid_index.sum() == 0:
                continue

            if self.scene.is_ablation():
                ray_o, pts_indices, hit_sky, rx_to_pts_and_tx_info = dataset[0:4]
                # Typically [B, 1,            3        ] - ray_o
                # Typically [B, n_ray,   K,            ] - pts_indices
                # Typically [B, n_ray,   K,            ] - hit_sky
                # Typically [B, n_ray,   K+1, (3+1+1+1)] - rx_ptstx_info
                pts_mask = pts_indices.flatten().cpu().tolist()  # [B*n_rays*K]
                predicted_ch = self.renderer.forward_on_batch_ablation(
                    pts, hit_sky, pts_mask, rx_to_pts_and_tx_info
                )
            else:
                ray_o, probe_indices, rx_to_probe_and_tx_info, probe_to_tx_info = (
                    dataset[0:4]
                )
                # Typically [B, 1,        3        ] - ray_o
                # Typically [B, n_ray,             ] - probe_indices
                # Typically [B, n_ray+1,  (3+1+1+1)] - rx_to_probe_and_tx_info
                # Typically [n_probes, 1, (3+1+1+1)] - probe_to_tx_info
                # Typically [B,           1        ] - gt_ch
                [
                    probe_to_pts_mask,
                    probe_to_pts_info,
                ] = self.scene.ray_sampler.GetProbePtsIndicesAndInfo(
                    env_idx=env_idx, train_type=int(TrainType.TRAIN)
                )  # [n_probes, K_closet, 6]
                probe_to_pts_and_tx_info = torch.cat(
                    (probe_to_pts_info, probe_to_tx_info), dim=-2
                )  # [n_probes, K_closest+1, 6]
                probe_mask = probe_indices.flatten().cpu().tolist()  # [B*n_rays]
                probe_to_pts_mask = (
                    probe_to_pts_mask.flatten().cpu().tolist()
                )  # [n_probes*K_closest]
                predicted_ch = self.renderer.forward_on_batch(
                    pts,
                    ray_o,
                    rx_to_probe_and_tx_info,
                    probe_mask,
                    probe_to_pts_mask,
                    probe_to_pts_and_tx_info,
                )

            loss = self.loss(predicted_ch, gt_ch, valid_index)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scene.InfoLog(f"epoch = {epoch}, loss = {loss.item()}")
            # self.scene.log_manager.WriterAddScalar("loss", loss, epoch)

            loss_list.append(loss.item())

        mean_loss = np.array(loss_list).mean()
        self.scheduler.step()
        return mean_loss

    def validation_on_scene(self, num_vis: int = 4) -> List[torch.Tensor]:
        # self.renderer.train()  # Set train flags as true
        validation_name = "default"
        num_env = self.validation_dataloader.dataset.num_envs[validation_name]
        num_tx = self.validation_dataloader.dataset.num_tx[validation_name]

        with torch.no_grad():
            test_loss_list = []
            env_idx = 0
            pts = self.scene.GetPointCloud(env_index=0)  # TODO: [n_pts, 3]

            tx_positions = torch.zeros((0, 3), dtype=self.dtype).to(torch.device("cpu"))
            rx_positions = torch.zeros((0, 3)).to(self.device).to(self.dtype)
            output_pred = None
            output_gt = None
            # batch_idx = [0, 1] + (
            #     np.random.choice(num_env * num_tx - 2, num_vis - 2, replace=False) + 2
            # ).tolist()
            batch_idx = [0, 1, 2, 3]

            for dataset in self.validation_dataloader:
                dataset = [val.squeeze(0) for val in dataset]

                gt_ch = dataset[4]
                interactions = None if len(dataset) <= 5 else dataset[5]
                valid_index = None
                if interactions is not None:
                    valid_index = interactions[:, :, 0, 0:1] >= -1e-5
                else:
                    valid_index = gt_ch.abs() > 0.01
                if valid_index.sum() == 0:
                    continue

                if self.scene.is_ablation():
                    ray_o, pts_indices, hit_sky, rx_to_pts_and_tx_info = dataset[0:4]

                    ray_d = rx_to_pts_and_tx_info[:, 0, -1, 0:3]
                    ray_dist = rx_to_pts_and_tx_info[:, 0, -1, 3:4]
                    tx_pos = (ray_o[:, 0, 0:3] + ray_d * ray_dist).cpu()  # [B, dim=3]

                    pts_mask = pts_indices.flatten().cpu().tolist()  # [B*n_rays*K]
                    predicted_ch = self.renderer.forward_on_batch_ablation(
                        pts, hit_sky, pts_mask, rx_to_pts_and_tx_info
                    )
                else:
                    (
                        ray_o,
                        probe_indices,
                        rx_to_probe_and_tx_info,
                        probe_to_tx_info,
                    ) = dataset[0:4]

                    ray_d = rx_to_probe_and_tx_info[:, -1, 0:3]
                    ray_dist = rx_to_probe_and_tx_info[:, -1, 3:4]
                    tx_pos = (ray_o[:, 0, 0:3] + ray_d * ray_dist).cpu()  # [B, dim=3]

                    [
                        probe_to_pts_mask,
                        probe_to_pts_info,
                    ] = self.scene.ray_sampler.GetProbePtsIndicesAndInfo(
                        env_idx=env_idx,
                        train_type=int(TrainType.VALIDATION),
                        validation_name=validation_name,
                    )  # [n_probes, K_closet, 6]
                    probe_to_pts_and_tx_info = torch.cat(
                        (probe_to_pts_info, probe_to_tx_info), dim=-2
                    )  # [n_probes, K_closest+1, 6]
                    probe_mask = probe_indices.flatten().cpu().tolist()  # [B*n_rays]
                    probe_to_pts_mask = (
                        probe_to_pts_mask.flatten().cpu().tolist()
                    )  # [n_probes*K_closest]
                    predicted_ch = self.renderer.forward_on_batch(
                        pts,
                        ray_o,
                        rx_to_probe_and_tx_info,
                        probe_mask,
                        probe_to_pts_mask,
                        probe_to_pts_and_tx_info,
                    )

                test_loss = self.loss(predicted_ch, gt_ch, valid_index=valid_index)
                test_loss_list.append(test_loss.item())

                if len(predicted_ch.shape) == 3:
                    # Gain-LOS
                    # is_LOS = (
                    #     interactions[:, :, 0:1, 0].abs()
                    #     + interactions[:, :, 1:2, 0].abs()
                    #     == 0
                    # ).to(torch.int32) + (interactions[:, :, 0:1, 0] == 0) * (
                    #     interactions[:, :, 2:3, 0] == 0
                    # ) * (
                    #     (interactions[:, :, 1:2, 0] - 1).abs() == 1
                    # ) > 0
                    # predicted_ch = predicted_ch[:, :, 0:1]
                    # gt_ch = gt_ch[:, :, 0:1]
                    # predicted_ch[~is_LOS] = 0.0
                    # gt_ch[~is_LOS] = 0.0
                    # predicted_ch = predicted_ch.sum(dim=1)
                    # gt_ch = gt_ch.sum(dim=1)

                    # pred_mask = predicted_ch[..., 0:1].abs() < 1e-5
                    # predicted_ch[..., 0:1][pred_mask] = -10000
                    # predicted_ch = predicted_ch[..., 0:1].max(dim=-2)[0]
                    # gt_mask = gt_ch[..., 0:1].abs() < 1e-5
                    # gt_ch[..., 0:1][gt_mask] = -10000
                    # gt_ch = gt_ch[..., 0:1].max(dim=-2)[0]

                    # Gain
                    # Avoid double counting
                    # n_rays = interactions.shape[1]
                    # mark_as_throw = torch.zeros(
                    #     (Batch_size, 1, 3), dtype=torch.bool
                    # ).to(
                    #     gt_ch.device
                    # )  # [B, 0, ch]
                    # for j in range(1, n_rays):
                    #     point_mse = (
                    #         interactions[:, j : (j + 1), :, 0:3]
                    #         - interactions[:, 0:j, :, 0:3]
                    #     ).norm(dim=-1).sum(dim=2) / j
                    #     # [B, j]
                    #     point_mse = point_mse.min(dim=1)[0].reshape(Batch_size)
                    #     double_counted = (
                    #         (point_mse < 0.1).reshape(Batch_size, 1, 1).repeat(1, 1, 3)
                    #     )  # [B, 1, ch]
                    #     mark_as_throw = torch.cat(
                    #         (mark_as_throw, double_counted), dim=1
                    #     )

                    # freq = 3.5
                    # k = 2.0 * np.pi * freq / 0.3
                    # directivity = torch.pow(10.0, predicted_ch[:, :, 0:1] / 10.0)
                    # directivity[predicted_ch[:, :, 0:1].abs() < 1e-5] = 0.0
                    # phase = gt_ch[:, :, 1:2] + k * 0.3 * gt_ch[:, :, 2:3]
                    # U = 1.0 * directivity / (4.0 * np.pi)
                    # E = U * torch.cos(phase)
                    # gt_dir = torch.stack(
                    #     (
                    #         torch.cos(gt_ch[..., 3]) * torch.sin(gt_ch[..., 4]),
                    #         torch.sin(gt_ch[..., 3]) * torch.sin(gt_ch[..., 4]),
                    #         torch.cos(gt_ch[..., 4]),
                    #     ),
                    #     dim=-1,
                    # )
                    # E = E * gt_dir  # [B, n_rays, 3]
                    # E[mark_as_throw] = 0.0
                    # E = E.sum(dim=1).norm(dim=-1, keepdim=True)
                    # U = E
                    # directivity = 4.0 * np.pi * U / 1.0
                    # predicted_ch = 10.0 * torch.log10(directivity)

                    # directivity = torch.pow(10.0, gt_ch[:, :, 0:1] / 10.0)
                    # directivity[gt_ch[:, :, 0:1].abs() < 1e-5] = 0.0
                    # phase = gt_ch[:, :, 1:2] + k * 0.3 * gt_ch[:, :, 2:3]
                    # U = 1.0 * directivity / (4.0 * np.pi)
                    # E = U * torch.cos(phase)
                    # gt_dir = torch.stack(
                    #     (
                    #         torch.cos(gt_ch[..., 3]) * torch.sin(gt_ch[..., 4]),
                    #         torch.sin(gt_ch[..., 3]) * torch.sin(gt_ch[..., 4]),
                    #         torch.cos(gt_ch[..., 4]),
                    #     ),
                    #     dim=-1,
                    # )
                    # E = E * gt_dir  # [B, n_rays, 3]
                    # E[mark_as_throw] = 0.0
                    # E = E.sum(dim=1).norm(dim=-1, keepdim=True)
                    # U = E
                    # directivity = 4.0 * np.pi * U / 1.0
                    # gt_ch = 10.0 * torch.log10(directivity)
                    # gt_ch[gt_ch.isinf()] = 0

                    # time-LOS
                    # is_LOS = (
                    #     interactions[:, :, 0:1, 0].abs()
                    #     + interactions[:, :, 1:2, 0].abs()
                    #     == 0
                    # )
                    # predicted_ch = predicted_ch[:, :, 2:3]
                    # gt_ch = gt_ch[:, :, 2:3]
                    # predicted_ch[~is_LOS] = 0.0
                    # gt_ch[~is_LOS] = 0.0
                    # predicted_ch = predicted_ch.sum(dim=1)
                    # gt_ch = gt_ch.sum(dim=1)
                    # time
                    predicted_ch = predicted_ch[:, :, 2:3].mean(dim=1)
                    gt_ch = gt_ch[:, :, 2:3].mean(dim=1)

                    # azimuth
                    # predicted_ch = torch.cos(predicted_ch[:, :, 3:4]).mean(dim=1)
                    # gt_ch = torch.cos(gt_ch[:, :, 3:4]).mean(dim=1)
                    # elevation
                    # predicted_ch = torch.cos(predicted_ch[:, :, 4:5]).mean(dim=1)
                    # gt_ch = torch.cos(gt_ch[:, :, 4:5]).mean(dim=1)
                elif predicted_ch.shape[-1] == 4:
                    predicted_ch = predicted_ch[..., 0:1]
                    gt_ch = gt_ch[..., 0:1]

                rx_positions = torch.cat((rx_positions, ray_o[:, 0, 0:3]), dim=0)
                tx_positions = torch.cat((tx_positions, tx_pos), dim=0)
                if output_pred is None:
                    output_pred = predicted_ch.cpu()
                    output_gt = gt_ch.cpu()
                else:
                    output_pred = torch.cat((output_pred, predicted_ch.cpu()), dim=0)
                    output_gt = torch.cat((output_gt, gt_ch.cpu()), dim=0)

        mean_test_loss = np.array(test_loss_list).mean()
        H, W = self.scene.H, self.scene.W
        output_pred = output_pred.reshape(num_env * num_tx, H, W, 1)
        output_gt = output_gt.reshape(num_env * num_tx, H, W, 1)
        tx_positions = tx_positions.reshape(num_env * num_tx, H * W, 3)
        if num_vis is not None:
            output_pred = output_pred[batch_idx]
            output_gt = output_gt[batch_idx]
            tx_positions = tx_positions[batch_idx, 0, :]
        else:
            tx_positions = tx_positions[:, 0, :]
        output_pred = output_pred.permute((0, 3, 1, 2))
        output_gt = output_gt.permute((0, 3, 1, 2))
        return [mean_test_loss, tx_positions, rx_positions, output_pred, output_gt]

    def GetOptimizer(self):
        return self.optimizer

    def GetRenderer(self):
        return self.renderer
