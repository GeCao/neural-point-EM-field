import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

from src.EM.scenes import AbstractScene
from src.EM.renderer.pointcloud_encoding import MVModel
from src.EM.utils import ScaleToUintCube, PostProcessFeatures, FeatureWeighting
from src.pointLF.attention_modules import PointFeatureAttention
from src.pointLF.pointcloud_encoding.pointnet_features import PointNetLightFieldEncoder
from src.pointLF.feature_mapping import PositionalEncoding
from src.pointLF.layer import DenseLayer, ModulationLayer


class LightFieldNet(nn.Module):
    def __init__(
        self,
        n_feat_in: int,
        n_pt_feat: int,
        output_ch: int,
        layer_modulation: bool = False,
        D: int = 4,
        W: int = 256,
        multires: int = 6,
        skips: List[int] = [],
    ):
        super(LightFieldNet, self).__init__()

        self.D = D
        self.W = W
        self.n_feat_in = n_feat_in
        self.feat_ch = n_pt_feat
        self.output_ch = output_ch

        self.pose_encoding = PositionalEncoding(multires)

        # self.input_ch = k * (n_pt_feat + 3)

        self.skips = skips

        if not layer_modulation:
            Layer = DenseLayer
            self.input_ch = self.n_feat_in * (n_pt_feat + (multires * 2 * 3 + 3))
        else:
            Layer = ModulationLayer
            self.input_ch = multires * 2 * 3 + 3

        self.layer_modulation = layer_modulation

        self.pts_linears = nn.ModuleList(
            [Layer(self.input_ch, W, z_dim=n_pt_feat)]
            + [
                Layer(W, W, z_dim=n_pt_feat)
                if i not in self.skips
                else Layer(W + self.input_ch, W, z_dim=n_pt_feat)
                for i in range(D - 1)
            ]
        )

        self.ch_linear = DenseLayer(W, output_ch, activate=False)

    def forward(self, ray_dirs, z):
        """
        ray_dirs: [batch_size, N_rays, 3]
        z: Latent encoding of the Light Field [batch_size, N_rays, k, n_feat]
        """
        if z.dim() == 4:
            n_batch, n_rays, n_feat, feat_ch = z.shape
            assert n_feat == self.n_feat_in
            if not self.layer_modulation:
                ray_dirs = ray_dirs[..., None, :].repeat(1, 1, n_feat, 1)
        else:
            n_batch, n_rays, feat_ch = z.shape
            n_feat = 1

        ray_dirs = self.pose_encoding(ray_dirs)

        if not self.layer_modulation:
            inputs = torch.cat([z, ray_dirs], dim=-1).view(n_batch, -1, self.input_ch)
            h = inputs
        else:
            h = ray_dirs.view(n_batch * n_rays, self.input_ch)
            z = z.view(n_batch * n_rays, self.feat_ch)

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h, z=z)
            if i in self.skips:
                h = torch.cat([inputs, h], -1)

        out = self.ch_linear(h, z=z)
        channels = F.leaky_relu(out)  # torch.sigmoid(out)

        return channels.view(n_batch, n_rays, self.output_ch)


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
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
    ):
        super(PointLightField, self).__init__()
        self.device = device
        self.dtype = dtype

        self.feat_weighting = None
        self._RGBFeatures = False
        self.n_pt_features = n_pt_features
        self.pre_scale = False
        self.no_feat = False
        self.stored_feature_maps = {}
        self.stored_points_in = None
        self.new_enc = new_encoding
        layer_modulation = False
        n_feat_in = k_closest
        upscale_feat_maps = False

        if feature_encoder == "multiview" or feature_encoder == "multiview_encoded":
            self._PointFeatures = MVModel(task="cls", backbone="resnet18", feat_size=16)
            self.n_pt_features = 128
            self.pre_scale = True
            self.feat_weighting = int(FeatureWeighting.MAXPOOL)
        elif feature_encoder == 'multiview_attention':
            self._PointFeatures = MVModel(task='cls', backbone='resnet18', feat_size=16)
            self.n_pt_features = 128
            self.key_len = 64
            self.pre_scale = True
            self.feat_weighting = int(FeatureWeighting.ATTENTION)
            self.AttentionModule = PointFeatureAttention(
                feat_dim_in=self.n_pt_features,
                feat_dim_out=self.n_pt_features,
                embeded_dim=256,
                n_att_heads=8,
                kdim=128,
                vdim=128,
                new_encoding=self.new_enc,
            )
            n_feat_in = 1
        elif feature_encoder == 'pointnet_ablation':
            self._PointFeatures = (
                PointNetLightFieldEncoder(
                    k=self.n_pt_features,
                    feature_transform=feature_transform,
                    points_only=False,
                    original=True,
                )
                .to(device)
                .to(dtype)
            )
            self.n_pt_features = 128
            self.key_len = 64
            self.pre_scale = False
            self.feat_weighting = int(FeatureWeighting.ATTENTION)
            self.AttentionModule = (
                PointFeatureAttention(
                    feat_dim_in=self.n_pt_features,
                    feat_dim_out=self.n_pt_features,
                    embeded_dim=256,
                    n_att_heads=8,
                    kdim=128,
                    vdim=128,
                    new_encoding=self.new_enc,
                )
                .to(device)
                .to(dtype)
            )
            n_feat_in = 1
        else:
            raise NotImplementedError(
                f"The feature encode strategy of {feature_encoder} has not been implemented yet."
            )

        self._LightField = (
            LightFieldNet(
                n_feat_in=n_feat_in,
                n_pt_feat=self.n_pt_features,
                output_ch=8,
                D=lf_architecture["D"],
                W=lf_architecture["W"],
                multires=lf_architecture["poseEnc"],
                skips=lf_architecture["skips"],
                layer_modulation=lf_architecture["modulation"],
            )
            .to(device)
            .to(dtype)
        )

    def forward(
        self,
        x: torch.Tensor,
        ray_dirs: torch.Tensor,
        closest_mask: Tuple[List[int]],
        pts_distance: torch.Tensor,
        pts_walks: torch.Tensor,
        pts_azimuth: torch.Tensor,
        pts_pitch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Take ray direction and point information, return rendered wireless channel
        Args:
            x                  (torch.Tensor):                [B, n_pts,             dim=3]
            ray_dirs           (torch.Tensor):                [B, n_rays, K_closest, dim=3]
            closest_mask       (Tuple(List[int], List[int])): x[closest_mask] = [B*n_rays*K_closest, dim=3]
            pts_distance       (torch.Tensor):                [B, n_rays, K_closest, 1]
            pts_walks          (torch.Tensor):                [B, n_rays, K_closest, 1]
            pts_azimuth        (torch.Tensor):                [B, n_rays, K_closest, 1]
            pts_pitch          (torch.Tensor):                [B, n_rays, K_closest, 1]

        Returns:
            torch.Tensor: rendered wireless channels with shape = [B, n_rays, 3] (TODO:)
        """
        batch_size = pts_distance.shape[0]
        n_rays = pts_distance.shape[1]
        K_closest = pts_distance.shape[2]

        # 1. Transform x so that AABB is a unit cube
        if self.pre_scale:
            pts_x = ScaleToUintCube(x)
        else:
            pts_x = x[..., :3].transpose(2, 1)

        # 2. Encode point clouds to feature
        feat, trans, trans_feat = self._PointFeatures(pts_x, rgb=None)

        # 3. TODO: Select K-closest points features
        if self.pre_scale:
            feat = PostProcessFeatures(
                feat=feat,
                scaled_pts=pts_x,
                K_closest_mask=closest_mask,
                K_closest=K_closest,
                feature_extractor=self._PointFeatures,
                img_resolution=trans.shape[-1],
                feature_resolution=feat.shape[-1],
            )  # [B, n_rays, K_closest, maps, n_features]
        else:
            feat = feat[closest_mask].reshape(
                batch_size, n_rays, K_closest, feat.shape[-1]
            )

        # 4. Apply weighting strategy to features
        if self.feat_weighting == int(FeatureWeighting.MAXPOOL):
            feat, _ = torch.max(feat, dim=-2, keepdim=True)
        elif self.feat_weighting == int(FeatureWeighting.ATTENTION):
            if feat.dim() == 5:
                n_feat_per_point = feat.shape[-2]
                feat = torch.sum(feat, dim=-2, keepdim=True) / n_feat_per_point
            else:
                n_feat_per_point = 1
                feat = feat[..., None, :]

            feat, attn_weights = self.AttentionModule(
                directions=ray_dirs,
                features=feat,
                distance=pts_distance.squeeze(-1),
                projected_distance=pts_walks.squeeze(-1),
                pitch=pts_pitch.squeeze(-1),
                azimuth=pts_azimuth.squeeze(-1),
            )
        elif self.feat_weighting == int(FeatureWeighting.SUM):
            n_feat_per_point = feat.shape[-2]
            feat = torch.sum(feat, dim=-2, keepdim=True) / n_feat_per_point
            feat = torch.sum(feat[..., 0, :], dim=-2)  # [B, n_rays, n_features]
        else:
            feat, attn_weights = self.AttentionModule(
                directions=ray_dirs,
                features=feat,
                distance=pts_distance.squeeze(-1),
                projected_distance=pts_walks.squeeze(-1),
                pitch=pts_pitch.squeeze(-1),
                azimuth=pts_azimuth.squeeze(-1),
            )

        # 5. Predict Light field output with pre-baked features and ray directions
        color = self._LightField(ray_dirs, feat)

        return color
