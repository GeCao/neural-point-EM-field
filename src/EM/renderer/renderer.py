from typing import Callable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

from src.EM.scenes import NeuralScene
from src.EM.utils import ModuleType, NodeType
from src.pointLF.point_light_field import PointLightField

# from src.EM.renderer import PointLightField


class PointLightFieldRenderer(nn.Module):
    def __init__(
        self,
        scene: NeuralScene,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs
    ) -> None:
        super(PointLightFieldRenderer, self).__init__(*args, **kwargs)
        self.device = device
        self.dtype = dtype
        self.scene = scene
        self.light_field_opt = self.scene.scene_opt["lightfield"]

        self._latent_codes = nn.ParameterDict()
        self._objs_size = nn.ParameterDict()
        self._cameras = nn.ParameterDict()

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
            sky_dome=self.light_field_opt.get("sky_dome", False),
        )

        self.add_module(module_name, lightfield)

    def forward_on_batch(self, ray_info, pts_info):
        if self.training:
            # Do not over hierachy
            # 1. TODO: See LightFieldRenderer for points mask details
            # 2. exert light_field_module
            ray_d = ray_info[..., 3:6]
            pts_positions = pts_info[..., 0:3]
            pts_distance = pts_info[..., 3:4]
            pts_walk = pts_info[..., 4:5]
            pts_azimuth = pts_info[..., 5:6]
            pts_pitch = pts_info[..., 6:7]
            for module_name, module in self.named_modules():
                if isinstance(module, PointLightField):
                    prediction = module(
                        x=pts_positions,
                        ray_dirs=ray_d,
                        closest_mask,
                        x_dist=pts_distance,
                        x_proj=pts_walk,
                        x_pitch=pts_pitch,
                        x_azimuth=pts_azimuth,
                        rgb=closest_rgb,
                        sample_idx=sample_idx,
                    )
                else:
                    self.scene.WarnLog(f"Unexpected module {module_name} loaded.")
        else:
            # Evaluation (Test or Validation)
            pass
