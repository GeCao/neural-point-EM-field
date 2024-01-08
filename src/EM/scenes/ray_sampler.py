from typing import Any, List
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer.implicit.utils import RayBundle

from src.EM.scenes import AbstractScene, Camera
from src.EM.utils import TrainType


class RayAABBIntersection(nn.Module):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs,
    ) -> None:
        super(RayAABBIntersection, self).__init__(*args, **kwargs)
        self.device = device
        self.dtype = dtype
        self.AABB_min = torch.Tensor([-1.0, -1.0, -1.0]).to(dtype).to(device)
        self.AABB_max = torch.Tensor([1.0, 1.0, 1.0]).to(dtype).to(device)

    def forward(self, ray_o: torch.Tensor, ray_d: torch.Tensor) -> List[torch.Tensor]:
        """See reference from https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms"""
        # Make everything flat
        assert len(ray_d.shape) == 3
        assert len(ray_o.shape) == 3
        n_rays = ray_o.shape[1]

        inv_d = torch.reciprocal(ray_d.reshape(-1, 3))
        t1 = (self.AABB_min - ray_o.reshape(-1, 3)) * inv_d
        t2 = (self.AABB_min - ray_o.reshape(-1, 3)) * inv_d

        t_min = torch.minimum(t1, t2)
        t_max = torch.maximum(t1, t2)

        t_near, _ = t_min.max(dim=-1, keepdim=False)
        t_far, _ = t_max.min(dim=-1, keepdim=False)

        intersection_mask = (t_far >= 0) * (t_far >= t_near)
        (intersection_index,) = torch.where(intersection_mask)
        if not len(intersection_index) == 0:
            # Got some rays intersected.
            z_in = t_near[intersection_index]
            z_out = t_far[intersection_index]

            intersection_index = (
                intersection_index // n_rays,
                intersection_index % n_rays,
            )  # (batch_index, numray_index)

            return [z_in, z_out, intersection_index]
        else:
            return [None, None, None]


class RaySampler(nn.Module):
    def __init__(
        self,
        K_closest: int,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs,
    ) -> None:
        super(RaySampler, self).__init__(*args, **kwargs)
        self.K_closest = K_closest
        self.device = device
        self.dtype = dtype

        self.nodes = {}

    @staticmethod
    def squareToUniformCylinder(r1: float, r2: float) -> List[float]:
        fai = 2.0 * math.pi * r2
        return [math.cos(fai), math.sin(fai), 2 * r1 - 1]

    @staticmethod
    def squareToUniformSphere(r1: float, r2: float) -> List[float]:
        Cylinder_res = RaySampler.squareToUniformCylinder(r1, r2)
        r = math.sqrt(1 - Cylinder_res[2] * Cylinder_res[2])
        return [r * Cylinder_res[0], r * Cylinder_res[1], Cylinder_res[2]]

    def Initialization(
        self,
        scene: AbstractScene,
        env_idx: int = 0,
        validation_name: str = None,
        train_type: int = 0,
    ):
        # Prepare everything:
        n_rays = self.K_closest
        K_closest = self.K_closest
        ch, rx, tx = scene.GetData(train_type, validation_name=validation_name)
        light_probe_pos = scene.GetLightProbePosition().view(
            1, -1, 3
        )  # [1, n_probes, 3]
        pts = scene.GetPointCloud(env_index=env_idx)

        self.nodes[env_idx]
        self.nodes[env_idx][train_type] = {}

        F_, T_, R_, _ = rx.shape
        FTR = F_ * T_ * R_
        n_probes = light_probe_pos.shape[-2]
        eps = 1e-5

        # 1. Ray info
        scene.InfoLog("Data Preparing: ray_o & ray_d")
        ray_o = rx.detach()  # [F, T, R, 3]
        ray_o = ray_o.reshape(FTR, 3).unsqueeze(1)  # [FTR, 1, 3]
        rx_probe_distance = (ray_o - light_probe_pos).norm(dim=-1)  # [FTR, n_probes,]
        nearest_probe_distance, nearest_indices = torch.topk(
            input=rx_probe_distance, k=n_rays, largest=False, dim=-1
        )  # [F*T*R, n_rays]
        nearest_light_probe_r = (
            light_probe_pos.repeat(FTR, 1, 1)
            .view(FTR * n_probes, 3)[nearest_indices.cpu().flatten().tolist()]
            .reshape(FTR, n_rays, 3)
            - ray_o
        )  # [FTR, n_rays, 3]
        ray_d = F.normalize(nearest_light_probe_r, dim=-1)  # [FTR, n_rays, 3]
        self.nodes[env_idx][train_type]["ray_o"] = ray_o.cpu()
        self.nodes[env_idx][train_type]["ray_d"] = ray_d.cpu()

        tx_rx_r = (tx.unsqueeze(2) - rx).reshape(FTR, 3)  # [FTR, 3]
        tx_rx_dir = F.normalize(tx_rx_r, dim=-1)  # [FTR, 3]
        tx_rx_dist = tx_rx_r.norm(dim=-1).unsqueeze(-1)  # [FTR, 1]
        tx_rx_elevation = torch.acos(tx_rx_r[..., 2:3] / (tx_rx_dist + eps))  # [FTR, 1]
        tx_rx_azimuth = torch.acos(
            tx_rx_r[..., 0:1] / (tx_rx_dist * torch.sin(tx_rx_elevation) + eps)
        )  # [FTR, 1] -> clamp(0, PI)
        tx_rx_azimuth = torch.where(
            tx_rx_r[..., 1:2] < 0.0, 2.0 * math.pi - tx_rx_azimuth, tx_rx_azimuth
        )  # [FTR, 1] -> clamp(0, 2PI)
        # # TODO: Check this log10
        # tx_rx_dist = torch.where(
        #     tx_rx_dist < 0.001, 1000 * torch.ones_like(tx_rx_dist), 1.0 / tx_rx_dist
        # )
        # tx_rx_dist = 10.0 * torch.log10(tx_rx_dist)
        self.nodes[env_idx][train_type]["tx_rx_dir"] = tx_rx_dir.cpu()
        self.nodes[env_idx][train_type]["tx_rx_distance"] = tx_rx_dist.cpu()
        self.nodes[env_idx][train_type]["tx_rx_elevation"] = tx_rx_elevation.cpu()
        self.nodes[env_idx][train_type]["tx_rx_azimuth"] = tx_rx_azimuth.cpu()

        # 2. rx - Light Probe info
        scene.InfoLog("Data Preparing: rx - Light probe information")
        nearest_probe_distance = nearest_probe_distance.unsqueeze(-1)
        nearest_probe_elevation = torch.acos(
            nearest_light_probe_r[..., 2:3] / (nearest_probe_distance + eps)
        )
        nearest_probe_azimuth = torch.acos(
            nearest_light_probe_r[..., 0:1]
            / (nearest_probe_distance * torch.sin(nearest_probe_elevation) + eps)
        )  # [0, PI]
        nearest_probe_azimuth = torch.where(
            nearest_light_probe_r[..., 1:2] < 0.0,
            2.0 * math.pi - nearest_probe_azimuth,
            nearest_probe_azimuth,
        )  # [0, 2PI]

        self.nodes[env_idx][train_type]["nearest_indices"] = nearest_indices.cpu()
        self.nodes[env_idx][train_type][
            "nearest_probe_distance"
        ] = nearest_probe_distance.cpu()
        self.nodes[env_idx][train_type][
            "nearest_probe_elevation"
        ] = nearest_probe_elevation.cpu()
        self.nodes[env_idx][train_type][
            "nearest_probe_azimuth"
        ] = nearest_probe_azimuth.cpu()

        # 3.1 pts(tx) - Light Probe info
        scene.InfoLog("Data Preparing: pts - Light probe information")
        light_probe_pos = light_probe_pos.view(-1, 1, 3)  # [n_probes, 1, 3]
        pts = pts.view(1, -1, 3)  # [1, n_pts, 3]
        light_probe_pts_distance = (light_probe_pos - pts).norm(
            dim=-1
        )  # [n_probes, n_pts,]
        nearest_probe_pts_distance, nearest_probe_pts_indices = torch.topk(
            input=light_probe_pts_distance, k=K_closest, largest=False, dim=-1
        )  # [n_probes, K_closest,]
        index = (
            torch.linspace(
                0, n_probes - 1, n_probes, device=torch.device("cpu"), dtype=torch.int32
            )
            .unsqueeze(-1)
            .repeat(1, K_closest)
            .flatten()
            .tolist(),
            nearest_probe_pts_indices.cpu().flatten().tolist(),
        )
        nearest_probe_pts_distance = nearest_probe_pts_distance.unsqueeze(-1)
        nearest_probe_pts_r = (
            pts.repeat(n_probes, 1, 1)[index].reshape(n_probes, K_closest, 3)
            - light_probe_pos
        )  # [n_probe, K_closest, 3]
        nearest_probe_pts_dir = F.normalize(
            nearest_probe_pts_r, dim=-1
        )  # [n_probe, K_closest, 3]
        nearest_probe_pts_elevation = torch.acos(
            nearest_probe_pts_r[..., 2:3] / (nearest_probe_pts_distance + eps)
        )  # [n_probe, K_closest, 1]
        nearest_probe_pts_azimuth = torch.acos(
            nearest_probe_pts_r[..., 0:1]
            / (
                nearest_probe_pts_distance * torch.sin(nearest_probe_pts_elevation)
                + eps
            )
        )  # [n_probe, K_closest, 1] -> clamp(0, PI)
        nearest_probe_pts_azimuth = torch.where(
            nearest_probe_pts_r[..., 1:2] < 0.0,
            2.0 * math.pi - nearest_probe_pts_azimuth,
            nearest_probe_pts_azimuth,
        )  # [n_probe, K_closest, 1] -> clamp(0, 2PI)
        self.nodes[env_idx][train_type][
            "nearest_probe_pts_indices"
        ] = nearest_probe_pts_indices.cpu()
        self.nodes[env_idx][train_type][
            "nearest_probe_pts_dir"
        ] = nearest_probe_pts_dir.cpu()
        self.nodes[env_idx][train_type][
            "nearest_probe_pts_distance"
        ] = nearest_probe_pts_distance.cpu()
        self.nodes[env_idx][train_type][
            "nearest_probe_pts_elevation"
        ] = nearest_probe_pts_elevation.cpu()
        self.nodes[env_idx][train_type][
            "nearest_probe_pts_azimuth"
        ] = nearest_probe_pts_azimuth.cpu()
        # 3.2 Add tx information into this wrap
        scene.InfoLog("Data Preparing: tx - Light probe information")
        # tx:= [F, T, dim=3]
        tx = tx.view(1, T_, 3)  # [1, T, 3]
        light_probe_tx_r = tx - light_probe_pos  # [n_probes, T, 3]
        light_probe_tx_dir = F.normalize(light_probe_tx_r, dim=-1)  # [n_probes, T, 3]
        light_probe_tx_distance = light_probe_tx_r.norm(dim=-1)  # [n_probes, T,]
        light_probe_tx_distance = light_probe_tx_distance.unsqueeze(-1)
        light_probe_tx_elevation = torch.acos(
            light_probe_tx_r[..., 2:3] / (light_probe_tx_distance + eps)
        )  # [n_probe, T, 1]
        light_probe_tx_azimuth = torch.acos(
            light_probe_tx_r[..., 0:1]
            / (light_probe_tx_distance * torch.sin(light_probe_tx_elevation) + eps)
        )  # [n_probe, T, 1] -> clamp(0, PI)
        light_probe_tx_azimuth = torch.acos(
            light_probe_tx_r[..., 0:1]
            / (light_probe_tx_distance * torch.sin(light_probe_tx_elevation) + eps)
        )  # [n_probe, T, 1] -> clamp(0, PI)
        self.nodes[env_idx][train_type]["light_probe_tx_dir"] = light_probe_tx_dir.cpu()
        self.nodes[env_idx][train_type][
            "light_probe_tx_distance"
        ] = light_probe_tx_distance.cpu()
        self.nodes[env_idx][train_type][
            "light_probe_tx_elevation"
        ] = light_probe_tx_elevation.cpu()
        self.nodes[env_idx][train_type][
            "light_probe_tx_azimuth"
        ] = light_probe_tx_azimuth.cpu()

        # 4. ground truth
        scene.InfoLog("Data Preparing: Ground truth")
        self.nodes[env_idx][train_type]["gt_channels"] = (
            ch[..., 0:1].reshape(-1, 1).cpu()
        )
        scene.InfoLog("Data Preparing: Finished!")

    def GetProbePtsIndicesAndInfo(
        self, env_idx: int, validation_name: str = None, train_type: int = 0
    ) -> List[torch.Tensor]:
        device = self.device
        nearest_probe_pts_indices = self.nodes[env_idx][train_type][
            "nearest_probe_pts_indices"
        ]  # [n_probes, K_closest,]

        nearest_probe_pts_dir = self.nodes[env_idx][train_type][
            "nearest_probe_pts_dir"
        ].to(device)
        nearest_probe_pts_distance = self.nodes[env_idx][train_type][
            "nearest_probe_pts_distance"
        ].to(device)
        nearest_probe_pts_elevation = self.nodes[env_idx][train_type][
            "nearest_probe_pts_elevation"
        ].to(device)
        nearest_probe_pts_azimuth = self.nodes[env_idx][train_type][
            "nearest_probe_pts_azimuth"
        ].to(device)
        probe_pts_info = torch.cat(
            (
                nearest_probe_pts_dir,
                nearest_probe_pts_distance,
                nearest_probe_pts_azimuth,
                nearest_probe_pts_elevation,
            ),
            dim=-1,
        )  # [n_probes, K_closest, 6]

        return [nearest_probe_pts_indices, probe_pts_info]

    def forward(
        self,
        scene: AbstractScene,
        env_idx: int,
        tx_idx: int,
        rx_idx: int,
        validation_name: str = None,
        train_type: int = 0,
    ) -> List[torch.Tensor]:
        """
        Args:
            idx   (int): index for sampling from dataloader
            scene (NeuralScene): a scene with points clouds and train/test/validation data
            train_type (int): Train(0)/Test(1)/Validation(2), see enum TrainType from utils
            env_idx    (int): None if all various scenes included, or indicate a specific scene only.
                              Typically, None for train, int for evaluation (Test/Validation)
            tx_idx     (int): None if all various scenes included, or indicate a specific scene only.
                              Typically, None for train, int for evaluation (Test/Validation)

        Returns:
            List[torch.Tensor]:
                points_clouds: [num_pts, dim=3]---------- All of the original points from this scene
                ray_info:      [n_rays, 6]--------------- ray_o + ray_d
                points_info:   [n_rays, K_closest, 4]---- 4 = distance(1) + proj_distance(1) + azimuth(1) + pitch(1)
                topK_indices:  [n_rays, K_closest, 1]---- points_clouds[(linspace, topK_indices.flatten().tolist())] = selected_points
                gt_channels:   [n_rays, D=3]------------- ground truth of wireless channels
        """
        if env_idx not in self.nodes:
            scene.InfoLog(f"env_idx {env_idx} not recorded, Start recording")
            self.nodes[env_idx] = {}
            self.Initialization(
                scene=scene,
                env_idx=env_idx,
                validation_name=validation_name,
                train_type=train_type,
            )

        if train_type not in self.nodes[env_idx]:
            scene.InfoLog(
                f"train_type {train_type} not recorded in env {env_idx}, Start recording"
            )
            self.Initialization(
                scene=scene,
                env_idx=env_idx,
                validation_name=validation_name,
                train_type=train_type,
            )

        F_ = scene.GetNumEnvs(train_type=train_type)
        T_ = scene.GetNumTransmitters(train_type=train_type)
        R_ = scene.GetNumReceivers(train_type=train_type)
        if train_type == int(TrainType.VALIDATION):
            F_ = F_[validation_name]
            T_ = T_[validation_name]
            R_ = R_[validation_name]

        device = self.device
        ftr_idx = env_idx * (T_ * R_) + tx_idx * R_ + rx_idx
        ray_o = self.nodes[env_idx][train_type]["ray_o"][ftr_idx].to(
            device
        )  # [n_rays, 3]
        ray_d = self.nodes[env_idx][train_type]["ray_d"][ftr_idx].to(
            device
        )  # [n_rays, 3]
        nearest_indices = self.nodes[env_idx][train_type]["nearest_indices"][
            ftr_idx
        ]  # [n_ray,]
        nearest_probe_distance = self.nodes[env_idx][train_type][
            "nearest_probe_distance"
        ][ftr_idx].to(device)
        nearest_probe_elevation = self.nodes[env_idx][train_type][
            "nearest_probe_elevation"
        ][ftr_idx].to(device)
        nearest_probe_azimuth = self.nodes[env_idx][train_type][
            "nearest_probe_azimuth"
        ][ftr_idx].to(device)
        rx_probe_info = torch.cat(
            (
                ray_d,
                nearest_probe_distance,
                nearest_probe_azimuth,
                nearest_probe_elevation,
            ),
            dim=-1,
        )  # [n_rays, 6]
        tx_rx_dir = self.nodes[env_idx][train_type]["tx_rx_dir"][ftr_idx].to(
            device
        )  # [3]
        tx_rx_distance = self.nodes[env_idx][train_type]["tx_rx_distance"][ftr_idx].to(
            device
        )  # [1]
        tx_rx_elevation = self.nodes[env_idx][train_type]["tx_rx_elevation"][
            ftr_idx
        ].to(
            device
        )  # [1]
        tx_rx_azimuth = self.nodes[env_idx][train_type]["tx_rx_azimuth"][ftr_idx].to(
            device
        )  # [1]
        rx_tx_info = torch.cat(
            (
                tx_rx_dir,
                tx_rx_distance,
                tx_rx_azimuth,
                tx_rx_elevation,
            ),
            dim=-1,
        ).view(
            1, 6
        )  # [1, 6]
        rx_probetx_info = torch.cat((rx_probe_info, rx_tx_info), dim=0)  # [n_rays+1, 6]

        light_probe_tx_dir = self.nodes[env_idx][train_type]["light_probe_tx_dir"][
            :, tx_idx, :
        ].to(
            device
        )  # [n_probes, 3]
        light_probe_tx_distance = self.nodes[env_idx][train_type][
            "light_probe_tx_distance"
        ][:, tx_idx, :].to(
            device
        )  # [n_probes, 1]
        light_probe_tx_elevation = self.nodes[env_idx][train_type][
            "light_probe_tx_elevation"
        ][:, tx_idx, :].to(
            device
        )  # [n_probes, 1]
        light_probe_tx_azimuth = self.nodes[env_idx][train_type][
            "light_probe_tx_azimuth"
        ][:, tx_idx, :].to(
            device
        )  # [n_probes, 1]
        probe_tx_info = torch.cat(
            (
                light_probe_tx_dir,
                light_probe_tx_distance,
                light_probe_tx_azimuth,
                light_probe_tx_elevation,
            ),
            dim=-1,
        )  # [n_probes, 3]
        probe_tx_info = probe_tx_info[
            nearest_indices.cpu().flatten().tolist()
        ]  # [n_rays, 3]

        gt_channels = self.nodes[env_idx][train_type]["gt_channels"][ftr_idx].to(device)
        return (
            ray_o,
            nearest_indices,
            rx_probetx_info,
            probe_tx_info,
            gt_channels,
        )
