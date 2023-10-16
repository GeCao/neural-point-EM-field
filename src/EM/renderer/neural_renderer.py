import torch
import torch.nn as nn
import torch.nn.functional as F

from src.EM.scenes import AbstractScene
from src.EM.renderer.pointcloud_encoding import MVModel
from src.pointLF.point_light_field import LightFieldNet


# Takes Points, weights  and rays and maps to color
class PointLightField(nn.Module):
    def __init__(
        self,
        k_closest=30,
        n_sample_pts=1000,
        n_pt_features=8,
        feature_encoder="pointnet_segmentation",
        feature_transform=True,
        lf_architecture={
            "D": 4,
            "W": 256,
            "skips": [],
            "modulation": False,
            "poseEnc": 4,
        },
        new_encoding=False,
        sky_dome=False,
    ):
        super(PointLightField, self).__init__()

        self.feat_weighting = None
        self._RGBFeatures = False
        self.n_pt_features = n_pt_features
        self.pre_scale = False
        self.no_feat = False
        self.stored_feature_maps = {}
        self.stored_points_in = None
        layer_modulation = False
        n_feat_in = k_closest
        upscale_feat_maps = False

        if feature_encoder == "multiview" or feature_encoder == "multiview_encoded":
            self._PointFeatures = MVModel(task="cls", backbone="resnet18", feat_size=16)
            self.n_pt_features = 128
            self.pre_scale = True
            self.feat_weighting = "max_pool"
        else:
            raise NotImplementedError(
                f"The feature encode strategy of {feature_encoder} has not been implemented yet."
            )

        self._LightField = LightFieldNet(
            n_feat_in=n_feat_in,
            n_pt_feat=self.n_pt_features,
            D=lf_architecture["D"],
            W=lf_architecture["W"],
            multires=lf_architecture["poseEnc"],
            skips=lf_architecture["skips"],
            layer_modulation=lf_architecture["modulation"],
        )

    def forward(
        self,
        ray_dirs: torch.Tensor,
        pts_points: torch.Tensor,
        pts_distance: torch.Tensor,
        pts_walks: torch.Tensor,
        pts_azimuth: torch.Tensor,
        pts_pitch: torch.Tensor,
    ):
        pts_x = pts_points  # pre_scale_MV(x)
        feat, trans, trans_feat = self._PointFeatures(pts_x, rgb=None)
        if self.feat_weighting == "max_pool":
            feat = torch.max(feat, dim=-2)[0]
        color = self._LightField(ray_dirs, feat)
