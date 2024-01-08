import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math

from src.EM.scenes import AbstractScene
from src.EM.renderer.pointcloud_encoding import MVModel
from src.EM.utils import ScaleToUintCube, PostProcessFeatures, FeatureWeighting
from src.EM.renderer.pointcloud_encoding import (
    PointFeatureAttention,
    PointNetLightFieldEncoder,
    PositionalEncoding,
    DenseLayer,
    ModulationLayer,
    FeatureDistanceEncoder,
)


class LightFieldNet(nn.Module):
    def __init__(
        self,
        n_frequ_pos_encoding: int,
        n_feat_in: int,
        n_pt_feat: int,
        output_ch: int,
        layer_modulation: bool = False,
        D: int = 4,
        W: int = 256,
        multires: int = 6,
        skips: List[int] = [],
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
    ):
        super(LightFieldNet, self).__init__()

        self.D = D
        self.W = W
        self.n_feat_in = n_feat_in
        self.feat_ch = n_pt_feat
        self.output_ch = output_ch

        self.pose_encoding = PositionalEncoding(multires)

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
                new_encoding=False,
            )
            .to(device)
            .to(dtype)
        )

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
        self.SH_basis_hlp = [
            0.5 / math.sqrt(math.pi),
            math.sqrt(3.0 / (4.0 * math.pi)),
            math.sqrt(3.0 / (4.0 * math.pi)),
            math.sqrt(3.0 / (4.0 * math.pi)),
            0.5 * math.sqrt(15.0 / math.pi),
            0.5 * math.sqrt(15.0 / math.pi),
            0.25 * math.sqrt(5.0 / math.pi),
            0.5 * math.sqrt(15.0 / math.pi),
            0.25 * math.sqrt(15.0 / math.pi),
        ]

    def forward(
        self,
        feat: torch.Tensor,
        probe_mask: Tuple[List[int]],
        ray_d: torch.Tensor,
        rx_to_probe_and_tx_distance: torch.Tensor,
        rx_to_probe_and_tx_azimuth: torch.Tensor,
        rx_to_probe_and_tx_elevation: torch.Tensor,
    ):
        """
        Args:
            feat                             (torch.Tensor):                [B, n_rays, n_feat]
            probe_mask                       (Tuple(List[int], List[int])): x[closest_mask] = [B*n_rays, dim=3]

            ray_d                            (torch.Tensor):                [B, n_rays+1, dim=3]
            rx_to_probe_and_tx_distance      (torch.Tensor):                [B, n_rays+1, 1]
            rx_to_probe_and_tx_azimuth       (torch.Tensor):                [B, n_rays+1, 1]
            rx_to_probe_and_tx_elevation     (torch.Tensor):                [B, n_rays+1, 1]
        """
        n_batch, n_rays, feat_ch = feat.shape
        n_feat = 1
        feat = feat.unsqueeze(-2)  # [B, n_rays, 1, feat_ch]

        z, attn_weights = self.AttentionModule(
            directions=ray_d[:, :, :],
            features=feat,
            pts_distance=rx_to_probe_and_tx_distance[..., :-1, :],
            pts_azimuth=rx_to_probe_and_tx_azimuth[..., :-1, :],
            pts_elevation=rx_to_probe_and_tx_elevation[..., :-1, :],
            tx_distance=rx_to_probe_and_tx_distance[..., -1:, :],
            tx_azimuth=rx_to_probe_and_tx_azimuth[..., -1:, :],
            tx_elevation=rx_to_probe_and_tx_elevation[..., -1:, :],
        )  # [B, n_rays+1, feat_ch]
        ray_info = self.pose_encoding(ray_d)  # [B, 1, ray_ch]

        if not self.layer_modulation:
            inputs = torch.cat([z, ray_info], dim=-1).view(n_batch, -1, self.input_ch)
            h = inputs
        else:
            h = ray_info.view(n_batch * n_rays, self.input_ch)
            z = z.view(n_batch * n_rays, self.feat_ch)

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h, z=z)
            if i in self.skips:
                h = torch.cat([inputs, h], -1)

        out = self.ch_linear(h, z=z)
        out = -F.relu(out)  # torch.sigmoid(out)
        # out = [B, n_rays+1, 9]

        # Try SH-encoder (with n = 3)
        assert self.output_ch == 9
        normal = F.normalize(-ray_d[:, :-1, :], dim=-1)  # [B, n_rays, 3]
        SH_basis = torch.zeros_like(out[:, :-1, :])  # [B, n_rays, 9]
        SH_basis[..., 0] = self.SH_basis_hlp[0]
        SH_basis[..., 1] = self.SH_basis_hlp[1] * normal[..., 2]
        SH_basis[..., 2] = self.SH_basis_hlp[2] * normal[..., 1]
        SH_basis[..., 3] = self.SH_basis_hlp[3] * normal[..., 0]
        SH_basis[..., 4] = self.SH_basis_hlp[4] * normal[..., 0] * normal[..., 2]
        SH_basis[..., 5] = self.SH_basis_hlp[5] * normal[..., 2] * normal[..., 1]
        SH_basis[..., 6] = self.SH_basis_hlp[6] * (
            -normal[..., 0] * normal[..., 0]
            - normal[..., 2] * normal[..., 2]
            + 2 * normal[..., 1] * normal[..., 1]
        )
        SH_basis[..., 7] = self.SH_basis_hlp[7] * normal[..., 1] * normal[..., 0]
        SH_basis[..., 8] = self.SH_basis_hlp[8] * (
            normal[..., 1] * normal[..., 1] - normal[..., 2] * normal[..., 2]
        )

        # n_rays + 1 Decompose:
        #     n_rays: SH_coeff * SH_basis (sample from light probe)
        #     1:      Sum pooling         (sample from LOS)
        out = (out[:, :-1, :] * SH_basis).sum(dim=-1).sum(dim=-1) + out[:, -1, :].sum(
            dim=-1
        )

        return out.view(n_batch, 1)


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
        sky_dome: bool = True,
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
        elif feature_encoder == "multiview_attention":
            self._PointFeatures = MVModel(task="cls", backbone="resnet18", feat_size=16)
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
        elif feature_encoder == "pointnet_ablation":
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
                n_frequ_pos_encoding=4,
                n_feat_in=n_feat_in,
                n_pt_feat=self.n_pt_features,
                output_ch=9,
                D=lf_architecture["D"],
                W=lf_architecture["W"],
                multires=lf_architecture["poseEnc"],
                skips=lf_architecture["skips"],
                layer_modulation=lf_architecture["modulation"],
                device=device,
                dtype=dtype,
            )
            .to(device)
            .to(dtype)
        )

        self.sky_dome = sky_dome
        if self.sky_dome:
            self._sky_latent = (
                nn.Parameter(torch.rand(self.n_pt_features), requires_grad=True)
                .to(device)
                .to(dtype)
            )

    def _add_sky_features(
        self,
        ray_dirs: torch.Tensor,
        sky_mask: torch.Tensor,
        feat: torch.Tensor,
        x_dist: torch.Tensor,
        x_proj: torch.Tensor,
        x_pitch: torch.Tensor,
        x_azimuth: torch.Tensor,
        tx_dist: torch.Tensor,
        tx_azimuth: torch.Tensor,
        tx_elevation: torch.Tensor,
    ):
        """Module for preparing sky feature encoding
        Args:
            ray_dirs           (torch.Tensor):                [B, n_rays, 3]
            sky_mask           (torch.Tensor):                [B, n_rays]
            feat               (torch.Tensor):                [B, n_rays, K_closest, 1, n_feat]
            x_dist             (torch.Tensor):                [B, n_rays, K_closest, 1]
            x_proj             (torch.Tensor):                [B, n_rays, K_closest, 1]
            x_pitch            (torch.Tensor):                [B, n_rays, K_closest, 1]
            x_azimuth          (torch.Tensor):                [B, n_rays, K_closest, 1]
            ...

        Returns:
            sky_rays           (torch.Tensor):                [1, n_sky_rays, 3]
            sky_feat           (torch.Tensor):                [1, n_sky_rays, K_closest+1, 1, n_feat]
            sky_dist           (torch.Tensor):                [1, n_sky_rays, K_closest+1, 1]
            sky_proj           (torch.Tensor):                [1, n_sky_rays, K_closest+1, 1]
            sky_pitch          (torch.Tensor):                [1, n_sky_rays, K_closest+1, 1]
            sky_azimuth        (torch.Tensor):                [1, n_sky_rays, K_closest+1, 1]
            ...
        """
        device = ray_dirs.device
        dtype = ray_dirs.dtype

        sky_rays = ray_dirs[sky_mask].unsqueeze(0)  # [1, n_sky, 3]
        n_sky_rays = sky_rays.shape[-2]

        sky_feat = feat[sky_mask].unsqueeze(0)  # [1, n_sky, K_closest, 1, n_feat]
        sky_feat = torch.cat(
            [
                sky_feat,
                self._sky_latent.expand(
                    list(sky_feat.shape[:2]) + [1] + list(sky_feat.shape[3:])
                ),
            ],
            dim=2,
        )  # [1, n_sky, K_closest+1, 1, n_feat]
        sky_dist = torch.cat(
            [
                x_dist[sky_mask].unsqueeze(0),
                torch.zeros([1, n_sky_rays, 1, 1]).to(device).to(dtype),
            ],
            dim=-2,
        )  # [1, n_sky, K_closest+1, 1]
        sky_proj = torch.cat(
            [
                x_proj[sky_mask].unsqueeze(0),
                torch.zeros([1, n_sky_rays, 1, 1]).to(device).to(dtype),
            ],
            dim=-2,
        )  # [1, n_sky, K_closest+1, 1]
        sky_pitch = torch.cat(
            [
                x_pitch[sky_mask].unsqueeze(0),
                torch.zeros([1, n_sky_rays, 1, 1]).to(device).to(dtype),
            ],
            dim=-2,
        )  # [1, n_sky, K_closest+1, 1]
        sky_azimuth = torch.cat(
            [
                x_azimuth[sky_mask].unsqueeze(0),
                torch.zeros([1, n_sky_rays, 1, 1]).to(device).to(dtype),
            ],
            dim=-2,
        )  # [1, n_sky, K_closest+1, 1]

        sky_tx_dist = torch.cat(
            [
                tx_dist[sky_mask].unsqueeze(0),
                torch.zeros([1, n_sky_rays, 1, 1]).to(device).to(dtype),
            ],
            dim=-2,
        )  # [1, n_sky, K_closest+1, 1]
        sky_tx_azimuth = torch.cat(
            [
                tx_azimuth[sky_mask].unsqueeze(0),
                torch.zeros([1, n_sky_rays, 1, 1]).to(device).to(dtype),
            ],
            dim=-2,
        )  # [1, n_sky, K_closest+1, 1]
        sky_tx_elevation = torch.cat(
            [
                tx_elevation[sky_mask].unsqueeze(0),
                torch.zeros([1, n_sky_rays, 1, 1]).to(device).to(dtype),
            ],
            dim=-2,
        )  # [1, n_sky, K_closest+1, 1]

        return (
            sky_rays,
            sky_feat,
            sky_dist,
            sky_proj,
            sky_pitch,
            sky_azimuth,
            sky_tx_dist,
            sky_tx_azimuth,
            sky_tx_elevation,
        )

    def forward(
        self,
        x: torch.Tensor,
        light_probe_pos: torch.Tensor,
        probe_mask: Tuple[List[int]],
        probe_pts_mask: List[int],
        ray_dirs: torch.Tensor,
        rx_to_probe_and_tx_distance: torch.Tensor,
        rx_to_probe_and_tx_azimuth: torch.Tensor,
        rx_to_probe_and_tx_elevation: torch.Tensor,
        probe_to_pts_and_tx_dir: torch.Tensor,
        probe_to_pts_and_tx_distance: torch.Tensor,
        probe_to_pts_and_tx_azimuth: torch.Tensor,
        probe_to_pts_and_tx_elevation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Take ray direction and point information, return rendered wireless channel
        Args:
            x                                (torch.Tensor):                [1, n_pts, dim=3]
            light_probe_pos                  (torch.Tensor):                [n_probes, dim=3]

            probe_mask                       (Tuple(List[int], List[int])): x[closest_mask] = [B*n_rays, dim=3]
            probe_pts_mask                   (List[int]):                   x[probe_pts_mask] = [B*n_rays*K_closest, ...]

            ray_dirs                         (torch.Tensor):                [B, n_rays+1, dim=3]
            rx_to_probe_and_tx_distance      (torch.Tensor):                [B, n_rays+1, 1]
            rx_to_probe_and_tx_azimuth       (torch.Tensor):                [B, n_rays+1, 1]
            rx_to_probe_and_tx_elevation     (torch.Tensor):                [B, n_rays+1, 1]

            probe_to_pts_and_tx_dir          (torch.Tensor):                [B, n_rays, K_closest+1, dim=3]
            probe_to_pts_and_tx_distance     (torch.Tensor):                [B, n_rays, K_closest+1, 1]
            probe_to_pts_and_tx_azimuth      (torch.Tensor):                [B, n_rays, K_closest+1, 1]
            probe_to_pts_and_tx_elevation    (torch.Tensor):                [B, n_rays, K_closest+1, 1]

        Returns:
            torch.Tensor: rendered wireless channels with shape = [B, n_rays, 3] (TODO:)
        """
        batch_size = probe_to_pts_and_tx_distance.shape[0]
        n_rays = probe_to_pts_and_tx_distance.shape[1]
        K_closest = probe_to_pts_and_tx_distance.shape[2] - 1
        n_feat = self.n_pt_features
        n_pts = x.shape[-2]

        # 1. Transform x so that AABB is a unit cube
        pts_x = x[..., 0:3].transpose(2, 1)  # [B, 3, n_pts]
        # 2. Encode point clouds to feature
        feat, trans, trans_feat = self._PointFeatures(
            pts_x, rgb=None
        )  # [1, n_pts, n_feat] (n_feat=128)

        # 3. Select K-closest points features
        feat = feat.view(n_pts, n_feat)
        feat = feat[probe_pts_mask]  # [B*n_rays*K_closest, feat]
        feat = feat.view(batch_size, n_rays, K_closest, n_feat)

        # 4. Apply weighting strategy to features
        if self.feat_weighting == int(FeatureWeighting.MAXPOOL):
            feat, _ = torch.max(feat, dim=-2, keepdim=True)
        elif self.feat_weighting == int(FeatureWeighting.ATTENTION):
            n_feat_per_point = 1
            feat = feat[..., None, :]  # [B, n_rays, K_closest, 1, n_feat]

            feat, attn_weights = self.AttentionModule(
                directions=probe_to_pts_and_tx_dir[..., -1:, :],
                features=feat,
                pts_distance=probe_to_pts_and_tx_distance[..., :-1, :],
                pts_azimuth=probe_to_pts_and_tx_azimuth[..., :-1, :],
                pts_elevation=probe_to_pts_and_tx_elevation[..., :-1, :],
                tx_distance=probe_to_pts_and_tx_distance[..., -1:, :],
                tx_azimuth=probe_to_pts_and_tx_azimuth[..., -1:, :],
                tx_elevation=probe_to_pts_and_tx_elevation[..., -1:, :],
            )  # [B, n_rays, n_feat]
        elif self.feat_weighting == int(FeatureWeighting.SUM):
            n_feat_per_point = feat.shape[-2]
            feat = torch.sum(feat, dim=-2, keepdim=True) / n_feat_per_point
            feat = torch.sum(feat[..., 0, :], dim=-2)  # [B, n_rays, n_features]
        else:
            feat, attn_weights = self.AttentionModule(
                directions=probe_to_pts_and_tx_dir,
                features=feat,
                pts_distance=probe_to_pts_and_tx_distance[..., :-1, :],
                pts_azimuth=probe_to_pts_and_tx_azimuth[..., :-1, :],
                pts_elevation=probe_to_pts_and_tx_elevation[..., :-1, :],
                tx_distance=probe_to_pts_and_tx_distance[..., -1:, :],
                tx_azimuth=probe_to_pts_and_tx_azimuth[..., -1:, :],
                tx_elevation=probe_to_pts_and_tx_elevation[..., -1:, :],
            )  # [B, n_rays, n_feat]

        # 5. Predict Light field output with pre-baked features and ray directions
        color = self._LightField(
            feat,
            probe_mask,
            ray_dirs,
            rx_to_probe_and_tx_distance,
            rx_to_probe_and_tx_azimuth,
            rx_to_probe_and_tx_elevation,
        )

        return color
