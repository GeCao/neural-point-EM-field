from typing import Any, List
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.EM.scenes import AbstractScene, Camera
from src.EM.utils import TrainType


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

    @staticmethod
    def GetDistAzimuthElevationFromVector(
        r: torch.Tensor, eps: float = 1e-5
    ) -> torch.Tensor:
        assert r.shape[-1] == 3
        dir = F.normalize(r, dim=-1)  # [..., 3]
        dist = r.norm(dim=-1, keepdim=True)  # [..., 1]
        elevation = torch.acos(dir[..., 2:3].clamp(-1, 1))  # [..., 1]
        sin_elevation_inv = torch.where(
            torch.sin(elevation).abs() < eps,
            torch.zeros_like(elevation),
            1.0 / torch.sin(elevation),
        )
        azimuth = torch.acos(
            (dir[..., 0:1] * sin_elevation_inv).clamp(-1, 1)
        )  # [..., 1] -> clamp(0, PI)
        azimuth = torch.where(
            dir[..., 1:2] < 0, 2.0 * math.pi - azimuth, azimuth
        )  # [..., 1] -> clamp(0, 2PI)
        info = torch.cat((dir, dist, azimuth, elevation), dim=-1)  # [..., 6]
        return info

    def Initialization(
        self,
        scene: AbstractScene,
        env_idx: int = 0,
        validation_name: str = None,
        train_type: int = 0,
    ):
        # Prepare everything:
        is_ablation = scene.is_ablation()
        n_rays = 8  # self.K_closest
        K_closest = self.K_closest
        ch, rx, tx = scene.GetData(train_type, validation_name=validation_name)[0:3]
        interactions = scene.GetInterections(
            train_type, validation_name=validation_name
        )
        pts = scene.GetPointCloud(env_index=env_idx)
        n_pts = pts.shape[-2]
        if not is_ablation:
            light_probe_pos = scene.GetLightProbePosition().view(
                1, -1, 3
            )  # [1, n_probes, 3]
            n_probes = light_probe_pos.shape[-2]

        self.nodes[env_idx][train_type] = {}

        F_, T_, R_, _ = rx.shape
        FTR = F_ * T_ * R_
        eps = 1e-5

        gain_only = scene.gain_only
        if not gain_only:
            _, _, _, n_ch, n_rays = ch.shape

        ch = ch.cpu()
        rx = rx.cpu()
        tx = tx.cpu()
        pts = pts.cpu()
        if not is_ablation:
            light_probe_pos = light_probe_pos.cpu()

        if is_ablation:
            # 1. Ray info
            scene.InfoLog("Data Preparing: ray_o & ray_d")
            ray_o = rx.detach()  # [F, T, R, 3]
            ray_o = ray_o.reshape(FTR, 3).unsqueeze(1)  # [FTR, 1, 3]
            pts = pts.view(1, -1, 3)
            n_pts = pts.shape[-2]
            if gain_only:
                ray_d = torch.from_numpy(
                    np.array(
                        [
                            RaySampler.squareToUniformSphere(
                                random.random(), random.random()
                            )
                            for i in range(n_rays)
                        ]
                    )
                ).to(
                    ch.dtype
                )  # [n_rays, 3]
                ray_d = ray_d.reshape(1, n_rays, 3).repeat(
                    FTR, 1, 1
                )  # [FTR, n_rays, 3]
            else:
                # departure (3, 4) from tx
                # arrival   (5, 6) to rx
                ray_d_azimuth = ch[..., 5, :].reshape(FTR, n_rays, 1)
                ray_d_elevation = ch[..., 6, :].reshape(FTR, n_rays, 1)
                ray_d = -torch.cat(
                    (
                        torch.cos(ray_d_azimuth) * torch.sin(ray_d_elevation),
                        torch.sin(ray_d_azimuth) * torch.sin(ray_d_elevation),
                        torch.cos(ray_d_elevation),
                    ),
                    dim=-1,
                )  # [FTR, n_rays, 3]

            rx_pts_cosphi = (
                ray_d[:, :, None, :]
                * F.normalize(pts[None, :, :, :] - ray_o[:, :, None, :], dim=-1)
            ).sum(
                dim=-1
            )  # [FTR, n_rays, n_pts]
            rx_pts_sinphi = torch.sqrt(1.0 - rx_pts_cosphi * rx_pts_cosphi)
            rx_pts_proj_distance = rx_pts_sinphi * (
                pts[None, :, :, :] - ray_o[:, :, None, :]
            ).norm(
                dim=-1
            )  # [FTR, n_rays, n_pts]
            rx_pts_proj_distance[rx_pts_cosphi < 0.866] = (
                100000000.0  # Delete back-side  # fov = 30 degree
            )
            nearest_pts_proj_distance, nearest_indices = torch.topk(
                input=rx_pts_proj_distance, k=K_closest, largest=False, dim=-1
            )  # [F*T*R, n_rays, K]
            hit_sky = (nearest_pts_proj_distance >= 100000000 - 1).to(torch.bool)
            index = nearest_indices.cpu().flatten().tolist()
            nearest_pts_r = (
                pts.reshape(-1, 3)[index].reshape(FTR, n_rays, K_closest, 3)
                - ray_o[:, :, None, :]
            )  # [FTR, n_rays, K, 3]
            nearest_pts_distance = nearest_pts_r.norm(dim=-1)  # [FTR, n_rays, K]
            self.nodes[env_idx][train_type]["ray_o"] = ray_o.to(self.device)
            self.nodes[env_idx][train_type]["ray_d"] = ray_d.to(self.device)
            self.nodes[env_idx][train_type]["hit_sky"] = hit_sky.to(self.device)

            tx_rx_r = (tx.unsqueeze(2) - rx).reshape(FTR, 1, 3)  # [FTR, 1, 3]
            tx_rx_info = self.GetDistAzimuthElevationFromVector(r=tx_rx_r, eps=eps)
            self.nodes[env_idx][train_type]["tx_rx_info"] = tx_rx_info.to(self.device)

            # 2. rx - pts info
            scene.InfoLog("Data Preparing: rx - pts information")
            nearest_pts_distance = nearest_pts_distance.unsqueeze(-1)
            rx_nearest_pts_r = ray_d.unsqueeze(-2) * nearest_pts_distance
            rx_nearest_pts_info = self.GetDistAzimuthElevationFromVector(
                r=rx_nearest_pts_r, eps=eps
            )

            self.nodes[env_idx][train_type]["nearest_indices"] = nearest_indices.cpu()
            self.nodes[env_idx][train_type]["rx_nearest_pts_info"] = (
                rx_nearest_pts_info.to(self.device)
            )

            tx_rx_r = (tx.unsqueeze(2) - rx).reshape(FTR, 1, 3)  # [FTR, 3]
            rx_tx_info = self.GetDistAzimuthElevationFromVector(r=tx_rx_r, eps=eps)
            self.nodes[env_idx][train_type]["rx_tx_info"] = rx_tx_info.to(self.device)

            # 3. ground truth
            scene.InfoLog("Data Preparing: Ground truth")
            if gain_only:
                self.nodes[env_idx][train_type]["gt_channels"] = ch.reshape(
                    -1, ch.shape[-1]
                ).to(self.device)
            else:
                self.nodes[env_idx][train_type]["gt_channels"] = (
                    ch.reshape(FTR, n_ch, n_rays)
                    .transpose(1, 2)[..., 0:5]
                    .to(self.device)
                )
                # elevation inverse
                self.nodes[env_idx][train_type]["gt_channels"][..., 4] = (
                    math.pi - self.nodes[env_idx][train_type]["gt_channels"][..., 4]
                )
                # azimuth inverse
                self.nodes[env_idx][train_type]["gt_channels"][..., 3] = torch.where(
                    self.nodes[env_idx][train_type]["gt_channels"][..., 3] >= math.pi,
                    self.nodes[env_idx][train_type]["gt_channels"][..., 3] - math.pi,
                    self.nodes[env_idx][train_type]["gt_channels"][..., 3] + math.pi,
                )

            # 5. interactions
            scene.InfoLog("Data Preparing: Interactions")
            self.nodes[env_idx][train_type]["interactions"] = (
                None
                if interactions is None
                else (interactions.reshape(FTR, n_rays, -1, 4).cpu())
            )
            scene.InfoLog("Data Preparing: Finished!")
            return

        # 1. Ray info
        scene.InfoLog("Data Preparing: ray_o & ray_d")
        ray_o = rx.detach()  # [F, T, R, 3]
        ray_o = ray_o.reshape(FTR, 3).unsqueeze(1)  # [FTR, 1, 3]
        if gain_only:
            rx_probe_distance = (ray_o - light_probe_pos).norm(
                dim=-1
            )  # [FTR, n_probes,]
            nearest_probe_distance, nearest_indices = torch.topk(
                input=rx_probe_distance, k=n_rays, largest=False, dim=-1
            )  # [F*T*R, n_rays]
            index = nearest_indices.cpu().flatten().tolist()
            nearest_probe_r = (
                light_probe_pos.reshape(n_probes, 3)[index].reshape(FTR, n_rays, 3)
                - ray_o
            )  # [FTR, n_rays, 3]
            ray_d = F.normalize(nearest_probe_r, dim=-1)  # [FTR, n_rays, 3]
            self.nodes[env_idx][train_type]["ray_o"] = ray_o.to(self.device)
            self.nodes[env_idx][train_type]["ray_d"] = ray_d.to(self.device)
        else:
            # departure (3, 4) from tx
            # arrival   (5, 6) to rx
            ray_d_azimuth = ch[..., 5, :].reshape(FTR, n_rays, 1)
            ray_d_elevation = ch[..., 6, :].reshape(FTR, n_rays, 1)
            ray_d = -torch.cat(
                (
                    torch.cos(ray_d_azimuth) * torch.sin(ray_d_elevation),
                    torch.sin(ray_d_azimuth) * torch.sin(ray_d_elevation),
                    torch.cos(ray_d_elevation),
                ),
                dim=-1,
            )  # [FTR, n_rays, 3]
            probe_cosphi = (
                ray_d[:, :, None, :]
                * F.normalize(
                    light_probe_pos[None, :, :, :] - ray_o[:, :, None, :], dim=-1
                )
            ).sum(
                dim=-1
            )  # [FTR, n_rays, n_probes]
            probe_sinphi = torch.sqrt(1.0 - probe_cosphi * probe_cosphi)
            probe_proj_distance = probe_sinphi * (
                light_probe_pos[None, :, :, :] - ray_o[:, :, None, :]
            ).norm(
                dim=-1
            )  # [FTR, n_rays, n_probes]
            probe_proj_distance[probe_cosphi < 0] = 100000000.0  # Delete back-side
            nearest_probe_proj_distance, nearest_indices = torch.topk(
                input=probe_proj_distance, k=1, largest=False, dim=-1
            )  # [F*T*R, n_rays, 1]
            index = nearest_indices.cpu().flatten().tolist()
            nearest_probe_r = (
                light_probe_pos.reshape(n_probes, 3)[index].reshape(FTR, n_rays, 3)
                - ray_o
            )  # [FTR, n_rays, 3]
            nearest_probe_distance = nearest_probe_r.norm(dim=-1)  # [FTR, n_rays]
            self.nodes[env_idx][train_type]["ray_o"] = ray_o.to(self.device)
            self.nodes[env_idx][train_type]["ray_d"] = ray_d.to(self.device)

        tx_rx_r = (tx.unsqueeze(2) - rx).reshape(FTR, 1, 3)  # [FTR, 3]
        rx_tx_info = self.GetDistAzimuthElevationFromVector(r=tx_rx_r, eps=eps)
        self.nodes[env_idx][train_type]["rx_tx_info"] = rx_tx_info.to(self.device)

        # 2. rx - Light Probe info
        scene.InfoLog("Data Preparing: rx - Light probe information")
        nearest_probe_distance = nearest_probe_distance.unsqueeze(-1)
        nearest_probe_r = nearest_probe_distance * ray_d
        rx_probe_info = self.GetDistAzimuthElevationFromVector(
            r=nearest_probe_r, eps=eps
        )

        self.nodes[env_idx][train_type]["nearest_indices"] = nearest_indices.cpu()
        self.nodes[env_idx][train_type]["rx_probe_info"] = rx_probe_info.to(self.device)

        # 3.1 pts(tx) - Light Probe info
        scene.InfoLog("Data Preparing: pts - Light probe information")
        light_probe_pos = light_probe_pos.view(-1, 1, 3)  # [n_probes, 1, 3]
        pts = pts.view(1, -1, 3)  # [1, n_pts, 3]
        probe_pts_distance = (light_probe_pos - pts).norm(dim=-1)  # [n_probes, n_pts,]
        probe_nearest_pts_distance, probe_nearest_pts_indices = torch.topk(
            input=probe_pts_distance, k=K_closest, largest=False, dim=-1
        )  # [n_probes, K_closest,]
        index = probe_nearest_pts_indices.cpu().flatten().tolist()
        probe_nearest_pts_r = (
            pts.reshape(n_pts, 3)[index].reshape(n_probes, K_closest, 3)
            - light_probe_pos
        )  # [n_probe, K_closest, 3]
        probe_nearest_pts_info = self.GetDistAzimuthElevationFromVector(
            r=probe_nearest_pts_r, eps=eps
        )  # [n_probes, K_closest, 6]
        self.nodes[env_idx][train_type][
            "probe_nearest_pts_indices"
        ] = probe_nearest_pts_indices.cpu()
        self.nodes[env_idx][train_type]["probe_nearest_pts_info"] = (
            probe_nearest_pts_info.to(self.device)
        )
        # 3.2 Add tx information into this wrap
        scene.InfoLog("Data Preparing: tx - Light probe information")
        # tx:= [F, T, dim=3]
        tx = tx.view(1, T_, 3)  # [1, T, 3]
        probe_tx_r = tx - light_probe_pos  # [n_probes, T, 3]
        probe_tx_info = self.GetDistAzimuthElevationFromVector(r=probe_tx_r, eps=eps)
        self.nodes[env_idx][train_type]["probe_tx_info"] = probe_tx_info.to(self.device)

        # 4. ground truth
        scene.InfoLog("Data Preparing: Ground truth")
        if gain_only:
            self.nodes[env_idx][train_type]["gt_channels"] = ch.reshape(
                -1, ch.shape[-1]
            ).to(self.device)
        else:
            self.nodes[env_idx][train_type]["gt_channels"] = (
                ch.reshape(FTR, n_ch, n_rays).transpose(1, 2)[..., 0:5].to(self.device)
            )
            # elevation inverse
            self.nodes[env_idx][train_type]["gt_channels"][..., 4] = (
                math.pi - self.nodes[env_idx][train_type]["gt_channels"][..., 4]
            )
            # azimuth inverse
            self.nodes[env_idx][train_type]["gt_channels"][..., 3] = torch.where(
                self.nodes[env_idx][train_type]["gt_channels"][..., 3] >= math.pi,
                self.nodes[env_idx][train_type]["gt_channels"][..., 3] - math.pi,
                self.nodes[env_idx][train_type]["gt_channels"][..., 3] + math.pi,
            )

        # 5. interactions
        scene.InfoLog("Data Preparing: Interactions")
        self.nodes[env_idx][train_type]["interactions"] = (
            None
            if interactions is None
            else (interactions.reshape(FTR, n_rays, -1, 4).cpu())
        )
        scene.InfoLog("Data Preparing: Finished!")

    def GetProbePtsIndicesAndInfo(
        self, env_idx: int, validation_name: str = None, train_type: int = 0
    ) -> List[torch.Tensor]:
        device = self.device
        probe_nearest_pts_indices = self.nodes[env_idx][train_type][
            "probe_nearest_pts_indices"
        ]  # [n_probes, K_closest,]

        probe_nearest_pts_info = self.nodes[env_idx][train_type][
            "probe_nearest_pts_info"
        ].to(
            device
        )  # [n_probes, K_closest, 6]

        return [probe_nearest_pts_indices, probe_nearest_pts_info]

    def forward(
        self,
        scene: AbstractScene,
        batch_size: int,
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

        is_ablation = scene.is_ablation()
        F_ = scene.GetNumEnvs(train_type=train_type)
        T_ = scene.GetNumTransmitters(train_type=train_type)
        R_ = scene.GetNumReceivers(train_type=train_type)
        if train_type == int(TrainType.VALIDATION):
            F_ = F_[validation_name]
            T_ = T_[validation_name]
            R_ = R_[validation_name]

        device = self.device
        ftr_idx_start = env_idx * (T_ * R_) + tx_idx * R_ + rx_idx
        rx_idx_end = rx_idx + batch_size
        if rx_idx_end > R_:
            rx_idx_end = R_

        ftr_idx_end = env_idx * (T_ * R_) + tx_idx * R_ + rx_idx_end
        ray_o = self.nodes[env_idx][train_type]["ray_o"][ftr_idx_start:ftr_idx_end].to(
            device
        )  # [n_rays, 3]
        gt_channels = self.nodes[env_idx][train_type]["gt_channels"][
            ftr_idx_start:ftr_idx_end
        ].to(device)
        if self.nodes[env_idx][train_type]["interactions"] is not None:
            interactions = self.nodes[env_idx][train_type]["interactions"][
                ftr_idx_start:ftr_idx_end
            ].to(device)
        else:
            interactions = None

        rx_tx_info = self.nodes[env_idx][train_type]["rx_tx_info"][
            ftr_idx_start:ftr_idx_end
        ].to(
            device
        )  # [B, 1, 6]

        if is_ablation:
            nearest_indices = self.nodes[env_idx][train_type]["nearest_indices"][
                ftr_idx_start:ftr_idx_end
            ]  # [B, n_ray,]
            rx_nearest_pts_info = self.nodes[env_idx][train_type][
                "rx_nearest_pts_info"
            ][ftr_idx_start:ftr_idx_end].to(
                device
            )  # [B, n_rays, K, 6]
            n_rays = rx_nearest_pts_info.shape[-3]
            rx_nearest_pts_tx_info = torch.cat(
                (rx_nearest_pts_info, rx_tx_info.unsqueeze(-3).repeat(1, n_rays, 1, 1)),
                dim=-2,
            )  # [B, n_rays, K+1, 6]

            hit_sky = self.nodes[env_idx][train_type]["hit_sky"][
                ftr_idx_start:ftr_idx_end
            ].to(device)

            results = [
                ray_o,
                nearest_indices,
                hit_sky,
                rx_nearest_pts_tx_info,
                gt_channels,
            ]
            if interactions is not None:
                results = results + [interactions]

            return results

        nearest_indices = self.nodes[env_idx][train_type]["nearest_indices"][
            ftr_idx_start:ftr_idx_end
        ]  # [B, n_ray,]
        batch_size, n_rays = nearest_indices.shape[0:2]
        rx_probe_info = self.nodes[env_idx][train_type]["rx_probe_info"][
            ftr_idx_start:ftr_idx_end
        ].to(
            device
        )  # [B, n_rays, 6]

        rx_probetx_info = torch.cat(
            (rx_probe_info, rx_tx_info), dim=-2
        )  # [B, n_rays+1, 6]

        probe_tx_info = self.nodes[env_idx][train_type]["probe_tx_info"][
            :, tx_idx, :
        ].to(
            device
        )  # [n_probes, 6]
        n_probes = probe_tx_info.shape[0]
        probe_tx_info = probe_tx_info.reshape(n_probes, 1, 6)

        results = [ray_o, nearest_indices, rx_probetx_info, probe_tx_info, gt_channels]
        if interactions is not None:
            results = results + [interactions]

        return results
