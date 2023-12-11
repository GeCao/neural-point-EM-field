from typing import Any, List
import torch
import torch.nn as nn
from pytorch3d.renderer.implicit.utils import RayBundle

from src.EM.scenes import AbstractScene, Camera
from src.EM.utils import TrainType


class RayAABBIntersection(nn.Module):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs
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
        **kwargs
    ) -> None:
        super(RaySampler, self).__init__(*args, **kwargs)
        self.K_closest = K_closest
        self.device = device
        self.dtype = dtype

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
                points_info:   [n_rays, K_closest, 7]---- 7 = position(3) + distance(1) + proj_distance(1) + azimuth(1) + pitch(1)
                topK_indices:  [n_rays, K_closest, 1]---- points_clouds[(linspace, topK_indices.flatten().tolist())] = selected_points
                gt_channels:   [n_rays, D=3]------------- ground truth of wireless channels
        """

        # 1. Generate rays.
        # [F, T, 1, R, D=8, K] channels
        # [F, T, 1, R, 3] rx
        ch, _, rx, _ = scene.GetData(train_type, validation_name=validation_name)
        # Please note the ray is actually all shot from tx
        ray_o = rx
        ray_azimuth = ch[..., 5, :]  # [F, T, 1, R, K]
        ray_elevation = ch[..., 6, :]  # [F, T, 1, R, K]
        ray_d = torch.stack(
            (
                torch.cos(ray_azimuth) * torch.sin(ray_elevation),
                torch.sin(ray_azimuth) * torch.sin(ray_elevation),
                torch.cos(ray_elevation),
            ),
            dim=-1,
        )  # [F, T, 1, R, K, 3]

        n_rays = ray_d.shape[-2]
        ray_o_detach = ray_o.detach()
        ray_d_detach = ray_d.detach()

        # 2. Load related object/ Point clouds firstly
        point_clouds = scene.GetPointCloud(env_index=env_idx).reshape(-1, 3)
        # Before everything, find the intersections

        ray_o_input = ray_o_detach[env_idx, tx_idx, 0, rx_idx, :].view(1, 3)  # [1, 3]
        ray_d_input = ray_d_detach[env_idx, tx_idx, 0, rx_idx, :, :]  # [n_rays, 3]
        (
            topK_indices,
            distance,
            proj_distance,
            azimuth,
            pitch,
            sky_mask,
        ) = scene.GetReceiver(
            transmitter_idx=tx_idx,
            receiver_idx=rx_idx,
            train_type=train_type,
            validation_name=validation_name,
        ).FindKClosest(
            ray_o=ray_o_input,
            ray_d=ray_d_input,
            points=point_clouds,
            K_closest=self.K_closest,
        )  # [n_rays, K_closest, 3(1)]

        # We would also have to get all of the channel ground truth
        gt_channels = torch.transpose(
            ch[env_idx, tx_idx, 0, rx_idx, :, :], -2, -1
        )  # [n_rays, 8]

        ray_info = torch.cat((ray_o_input.repeat(n_rays, 1), ray_d_input), dim=-1)
        points_info = torch.cat((distance, proj_distance, azimuth, pitch), dim=-1)

        eps = 1e-6
        valid_rays = ch[env_idx, tx_idx, 0, rx_idx, 7:8, :].to(
            torch.bool
        )  # [n_rays, 1]
        transmitter = scene.GetTransmitter(
            transmitter_idx=tx_idx,
            train_type=train_type,
            validation_name=validation_name,
        )
        tx_info = transmitter.Decompose(points=point_clouds)

        return (
            point_clouds,
            ray_info,
            points_info,
            topK_indices,
            valid_rays,
            sky_mask,
            tx_info,
            gt_channels,
        )
