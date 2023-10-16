from typing import Any, List
import torch
import torch.nn as nn
from pytorch3d.renderer.implicit.utils import RayBundle

from src.EM.scenes import AbstractScene
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
        self.AABB_min = torch.Tensor([-1.0, -1.0, -1.0], device=device).to(dtype)
        self.AABB_max = torch.Tensor([1.0, 1.0, 1.0], device=device).to(dtype)

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


class RaySampler(object):
    def __init__(
        self,
        K_closest: int,
        scene: AbstractScene,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
    ) -> None:
        self.K_closest = K_closest
        self.scene = scene
        self.device = device
        self.dtype = dtype

        self._local_sampler = RayAABBIntersection(device=device, dtype=dtype)

    def forward(
        self, idx: int, scene: AbstractScene, train_type: int = 0
    ) -> List[torch.Tensor]:
        device = self.device
        dtype = self.dtype

        n_transmitters = scene.GetNumTransmitters()
        n_cameras = scene.GetNumCameras()
        frames = scene.GetFrames()  # List[frames]

        # 1. Generate rays.
        # TODO: Rethink, we got all the rays prepared, so no further sampling
        # Should we add more scenes?
        # [F, T, 1, R, K, I, 4] Intersection
        # [F, T, 1, R, D=8, K] channels
        # TODO: We are only rely one scene for now.
        (
            train_ch,
            train_floor_idx,
            train_interactions,
            train_rx,
            train_tx,
        ) = scene.GetData(train_type).values()[0]
        # TODO: We usually think the first one intersection is rx location, I hope so
        ray_o = train_interactions[..., 0, 1:]  # [F, T, 1, R, K, 3]
        ray_azimuth = train_ch[..., 3, :]  # [F, T, 1, R, K]
        ray_elevation = train_ch[..., 4, :]  # [F, T, 1, R, K]
        ray_d = torch.stack(
            (
                torch.cos(ray_azimuth) * torch.sin(ray_elevation),
                torch.sin(ray_azimuth) * torch.sin(ray_elevation),
                torch.cos(ray_elevation),
            ),
            dim=-1,
        )  # [F, T, 1, R, K, 3]

        n_rays = ray_o.shape[-2]
        ray_o_detach = ray_o.detach()
        ray_d_detach = ray_d.detach()

        # 2. We got many Transmitters, and many Receivers.
        tx_idx = (idx % (n_transmitters * n_cameras)) // n_cameras
        rx_idx = (idx % (n_transmitters * n_cameras)) % n_cameras
        # Load related object/ Point clouds firstly
        point_clouds = frames[rx_idx].GetPointCloud().reshape(-1, 3)
        object_to_world_mat = frames[rx_idx].GetObjectToWorldTransformation()
        # Before everything, find the intersections

        # Get your view frustum firstly
        in_frustum_index = scene.GetCamera(
            transmitter_idx=tx_idx, camera_idx=rx_idx
        ).FrustumCulling(points=point_clouds)

        ray_o_input = ray_o_detach[0, tx_idx, 0, rx_idx, :, :]  # [n_rays, 3]
        ray_d_input = ray_d_detach[0, tx_idx, 0, rx_idx, :, :]  # [n_rays, 3]
        if in_frustum_index is not None:
            (
                topK_index,
                points,
                distance,
                walk,
                azimuth,
                pitch,
            ) = scene.GetCamera(transmitter_idx=tx_idx, camera_idx=rx_idx).FindKClosest(
                ray_o=ray_o_input,
                ray_d=ray_d_input,
                points=point_clouds.view(-1, 3)[in_frustum_index, :],
                K_closest=self.K_closest,
            )  # [n_rays, K_closest, 3(1)]
        else:
            # TODO: Sample from background (Far field)
            pass

        # We would also have to get all of the channel ground truth
        gt_channels = torch.transpose(
            train_ch[0, tx_idx, 0, rx_idx, :, :], -2, -1
        )  # [n_rays, 3]

        ray_info = torch.cat((ray_o_input, ray_d_input), dim=-1)

        return (
            ray_info,
            torch.cat((points, distance, walk, azimuth, pitch), dim=-1),
            gt_channels,
        )
