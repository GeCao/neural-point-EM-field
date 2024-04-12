from typing import Callable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

from src.EM.scenes import NeuralScene
from src.EM.utils import ModuleType, NodeType
from src.EM.renderer import PointLightField, PointLightFieldAblation, PointLightFieldMLP


class PointLightFieldRenderer(nn.Module):
    def __init__(
        self,
        scene: NeuralScene,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs,
    ) -> None:
        super(PointLightFieldRenderer, self).__init__(*args, **kwargs)
        self.device = device
        self.dtype = dtype
        self.scene = scene
        self.light_field_opt = self.scene.scene_opt["lightfield"]

        # Settings
        self.module_type = int(ModuleType.LIGHTFIELD)
        self.ParameterInitialization()

    def ParameterInitialization(self):
        module_name = "background"
        output_ch = self.scene.n_ch if self.scene.gain_only else 6
        if self.scene.is_ablation():
            lightfield = PointLightFieldAblation(
                k_closest=self.light_field_opt["k_closest"],
                n_sample_pts=self.light_field_opt["n_sample_pts"],
                n_pt_features=self.light_field_opt["n_features"],
                feature_encoder=self.light_field_opt["pointfeat_encoder"],
                feature_transform=True,
                lf_architecture={
                    "D": self.light_field_opt.get("D_lf", 4),
                    "W": self.light_field_opt.get("W_lf", 256),
                    "skips": self.light_field_opt.get("skips_lf", []),
                    "modulation": self.light_field_opt.get("layer_modulation", False),
                    "poseEnc": self.light_field_opt.get("ray_encoding", 4),
                },
                new_encoding=self.light_field_opt.get("new_enc", False),
                output_ch=output_ch,
                device=self.device,
                dtype=self.dtype,
            )
        elif self.scene.is_MLP():
            lightfield = PointLightFieldMLP(
                output_ch=output_ch,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            lightfield = PointLightField(
                k_closest=self.light_field_opt["k_closest"],
                n_sample_pts=self.light_field_opt["n_sample_pts"],
                n_pt_features=self.light_field_opt["n_features"],
                feature_encoder=self.light_field_opt["pointfeat_encoder"],
                feature_transform=True,
                lf_architecture={
                    "D": self.light_field_opt.get("D_lf", 4),
                    "W": self.light_field_opt.get("W_lf", 256),
                    "skips": self.light_field_opt.get("skips_lf", []),
                    "modulation": self.light_field_opt.get("layer_modulation", False),
                    "poseEnc": self.light_field_opt.get("ray_encoding", 4),
                },
                new_encoding=self.light_field_opt.get("new_enc", False),
                output_ch=output_ch,
                device=self.device,
                dtype=self.dtype,
            )

        self.add_module(module_name, lightfield)

    def forward_on_batch(
        self,
        pts: torch.Tensor,
        ray_o: torch.Tensor,
        rx_to_probe_and_tx_info: torch.Tensor,
        probe_to_pts_indices: torch.Tensor,
        probe_to_pts_and_tx_info: torch.Tensor,
    ):
        if self.training:
            # Do not over hierachy
            # exert light_field_module
            # x                           [1, n_pts, 3]
            pts = pts.reshape(1, *pts.shape[-2:])

            ray_d = rx_to_probe_and_tx_info[..., 0:3]  # [B, n_rays+1, 3]
            rx_to_probe_and_tx_distance = rx_to_probe_and_tx_info[
                ..., 3:4
            ]  # [B, n_rays+1, 1]
            rx_to_probe_and_tx_azimuth = rx_to_probe_and_tx_info[
                ..., 4:5
            ]  # [B, n_rays+1, 1]
            rx_to_probe_and_tx_elevation = rx_to_probe_and_tx_info[
                ..., 5:6
            ]  # [B, n_rays+1, 1]

            probe_to_pts_and_tx_dir = probe_to_pts_and_tx_info[
                ..., 0:3
            ]  # [B, n_rays, K_closest+1, 3]
            probe_to_pts_and_tx_distance = probe_to_pts_and_tx_info[
                ..., 3:4
            ]  # [B, n_rays, K_closest+1, 1]
            probe_to_pts_and_tx_azimuth = probe_to_pts_and_tx_info[
                ..., 4:5
            ]  # [B, n_rays, K_closest+1, 1]
            probe_to_pts_and_tx_elevation = probe_to_pts_and_tx_info[
                ..., 5:6
            ]  # [B, n_rays, K_closest+1, 1]
            # K_closest_mask                  x[mask] -> Shape[B*n_rays, 3]
            probe_pts_mask = (
                probe_to_pts_indices.cpu().flatten().tolist()
            )  # [B*n_rays*K_closest]
            for module_name, module in self.named_modules():
                if isinstance(module, PointLightField):
                    prediction = module(
                        pts=pts,
                        probe_pts_mask=probe_pts_mask,
                        ray_dirs=ray_d,
                        rx_to_probe_and_tx_distance=rx_to_probe_and_tx_distance,
                        rx_to_probe_and_tx_azimuth=rx_to_probe_and_tx_azimuth,
                        rx_to_probe_and_tx_elevation=rx_to_probe_and_tx_elevation,
                        probe_to_pts_and_tx_dir=probe_to_pts_and_tx_dir,
                        probe_to_pts_and_tx_distance=probe_to_pts_and_tx_distance,
                        probe_to_pts_and_tx_azimuth=probe_to_pts_and_tx_azimuth,
                        probe_to_pts_and_tx_elevation=probe_to_pts_and_tx_elevation,
                    )
                    return prediction
                elif isinstance(module, PointLightFieldMLP):
                    prediction = module(
                        ray_o=ray_o[:, 0, 0:3],
                        rx_to_tx_distance=rx_to_probe_and_tx_distance[:, -1, :],
                        rx_to_tx_azimuth=rx_to_probe_and_tx_azimuth[:, -1, :],
                        rx_to_tx_elevation=rx_to_probe_and_tx_elevation[:, -1, :],
                    )
                    return prediction
                else:
                    pass
                    # self.scene.WarnLog(f"Unexpected module {module_name} loaded.")
        else:
            # Evaluation (Test or Validation)
            pass

    def forward_on_batch_ablation(
        self,
        pts: torch.Tensor,
        hit_sky: torch.Tensor,
        pts_mask: List[int],
        rx_to_pts_and_tx_info: torch.Tensor,
    ):
        if self.training:
            # Do not over hierachy
            # exert light_field_module
            # pts                           [n_pts, 3]
            pts = pts.reshape(1, *pts.shape[-2:])

            ray_d = rx_to_pts_and_tx_info[..., 0:3]  # [B, n_rays, K+1, 3]
            rx_to_pts_and_tx_distance = rx_to_pts_and_tx_info[
                ..., 3:4
            ]  # [B, n_rays, K+1, 1]
            rx_to_pts_and_tx_azimuth = rx_to_pts_and_tx_info[
                ..., 4:5
            ]  # [B, n_rays, K+1, 1]
            rx_to_pts_and_tx_elevation = rx_to_pts_and_tx_info[
                ..., 5:6
            ]  # [B, n_rays, K+1, 1]

            for module_name, module in self.named_modules():
                if isinstance(module, PointLightFieldAblation):
                    prediction = module(
                        pts=pts,
                        hit_sky=hit_sky,
                        pts_mask=pts_mask,
                        ray_dirs=ray_d,
                        rx_to_pts_and_tx_distance=rx_to_pts_and_tx_distance,
                        rx_to_pts_and_tx_azimuth=rx_to_pts_and_tx_azimuth,
                        rx_to_pts_and_tx_elevation=rx_to_pts_and_tx_elevation,
                    )
                    return prediction
                else:
                    pass
                    # self.scene.WarnLog(f"Unexpected module {module_name} loaded.")
