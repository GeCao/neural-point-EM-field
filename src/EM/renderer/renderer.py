from typing import Callable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

from src.EM.scenes import NeuralScene
from src.EM.utils import ModuleType, NodeType
from src.EM.renderer import PointLightField

# from src.EM.renderer import PointLightField


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

        # self._density_noise_std = 0.0
        # self._latent_reg = 1e-7
        # self._chunk_size_test = self.scene.scene_opt
        # self._transient_head = False

    def ParameterInitialization(self):
        module_name = "background"
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
            device=self.device,
            dtype=self.dtype,
        )

        self.add_module(module_name, lightfield)

    def forward_on_batch(
        self,
        x: torch.Tensor,
        ray_info: torch.Tensor,
        pts_info: torch.Tensor,
        K_closest_mask: Tuple[List[int]],
        sky_mask: torch.Tensor,
        tx_info: torch.Tensor,
    ):
        if self.training:
            # Do not over hierachy
            # exert light_field_module
            # x                           [B, n_pts, 3]
            if pts_info.dim() == 3:
                pts_info = pts_info.unsqueeze(0)  # [1, selected_rays, K_closest, 4]
            if ray_info.dim() == 2:
                ray_info = ray_info.unsqueeze(0)  # [1, selected_rays, 3]

            ray_d = ray_info[..., 3:6]  # [B, n_rays, 3]
            pts_distance = pts_info[..., 0:1]  # [B, n_rays, K, 1]
            pts_proj_distance = pts_info[..., 1:2]  # [B, n_rays, K, 1]
            pts_azimuth = pts_info[..., 2:3]  # [B, n_rays, K, 1]
            pts_pitch = pts_info[..., 3:4]  # [B, n_rays, K, 1]
            # K_closest_mask                  x[mask] -> Shape[B*n_rays*n_pts, 3]
            for module_name, module in self.named_modules():
                if isinstance(module, PointLightField):
                    prediction = module(
                        x=x,
                        ray_dirs=ray_d,
                        closest_mask=K_closest_mask,
                        pts_distance=pts_distance,
                        pts_proj_distance=pts_proj_distance,
                        pts_pitch=pts_pitch,
                        pts_azimuth=pts_azimuth,
                        sky_mask=sky_mask,
                        tx_info=tx_info,
                    )
                    return prediction
                else:
                    pass
                    # self.scene.WarnLog(f"Unexpected module {module_name} loaded.")
        else:
            # Evaluation (Test or Validation)
            pass
