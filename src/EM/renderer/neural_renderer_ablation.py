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


class LightFieldNetAblation(nn.Module):
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
        super(LightFieldNetAblation, self).__init__()

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
                (
                    Layer(W, W, z_dim=n_pt_feat)
                    if i not in self.skips
                    else Layer(W + self.input_ch, W, z_dim=n_pt_feat)
                )
                for i in range(D - 1)
            ]
        )

        self.ch_linear = DenseLayer(W, output_ch, activate=False)

    def forward(self, ray_dirs: torch.Tensor, z: torch.Tensor):
        """
        ray_dirs: [batch_size, N_rays, 3]
        z: Latent encoding of the Light Field [batch_size, N_rays, k, n_feat]
        """
        if z.dim() == 4:
            print("z = ", z.shape)
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
        # channels = -F.relu(out)  # torch.sigmoid(out)
        # channels = torch.cat(
        #     (
        #         out[..., 0:1],
        #         torch.sigmoid(out[..., 1:2]) * (2.0 * math.pi),
        #         out[..., 2:3],
        #         torch.sigmoid(out[..., 3:7]) * (2.0 * math.pi),
        #         torch.sigmoid(out[..., 7:8]),
        #     ),
        #     dim=-1,
        # )

        out = out.view(n_batch, n_rays, self.output_ch)
        if self.output_ch == 1 or self.output_ch == 4:
            out = out.sum(dim=-2)
        else:
            out[..., -3:] = F.normalize(out[..., -3:], dim=-1)
        return out


# Takes Points, weights  and rays and maps to color
class PointLightFieldAblation(nn.Module):
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
        output_ch: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
    ):
        super(PointLightFieldAblation, self).__init__()
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
            LightFieldNetAblation(
                n_feat_in=n_feat_in,
                n_pt_feat=self.n_pt_features,
                output_ch=output_ch,
                D=lf_architecture["D"],
                W=lf_architecture["W"],
                multires=lf_architecture["poseEnc"],
                skips=lf_architecture["skips"],
                layer_modulation=lf_architecture["modulation"],
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
        sky_mask: List[int],
        feat: torch.Tensor,
        x_dist: torch.Tensor,
        x_azimuth: torch.Tensor,
        x_elevation: torch.Tensor,
        tx_dist: torch.Tensor,
        tx_azimuth: torch.Tensor,
        tx_elevation: torch.Tensor,
    ):
        """Module for preparing sky feature encoding
        Args:
            ray_dirs           (torch.Tensor):                [B*n_rays*K_closest, 3]
            sky_mask           (torch.Tensor):                [B*n_rays*K_closest]
            feat               (torch.Tensor):                [B*n_rays*K_closest, 1, n_feat]
            x_dist             (torch.Tensor):                [B*n_rays*K_closest,]
            x_azimuth          (torch.Tensor):                [B*n_rays*K_closest,]
            x_elevation        (torch.Tensor):                [B*n_rays*K_closest,]
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

        sky_feat = feat[sky_mask].unsqueeze(0)  # [1, n_sky, 1, n_feat]
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
        pts: torch.Tensor,
        hit_sky: torch.Tensor,
        pts_mask: List[int],
        ray_dirs: torch.Tensor,
        rx_to_pts_and_tx_distance: torch.Tensor,
        rx_to_pts_and_tx_azimuth: torch.Tensor,
        rx_to_pts_and_tx_elevation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Take ray direction and point information, return rendered wireless channel
        Args:
            pts                              (torch.Tensor):                [1, n_pts, dim=3]
            light_probe_pos                  (torch.Tensor):                [n_probes, dim=3]

            hit_sky                          (torch.Tensor):                [B, n_rays, K_closest]
            pts_mask                         List[int]:       x[pts_mask] = [B*n_rays*K_closest, dim=3]

            ray_dirs                         (torch.Tensor):                [B, n_rays, K_closest+1, dim=3]
            rx_to_pts_and_tx_distance        (torch.Tensor):                [B, n_rays, K_closest+1, 1]
            rx_to_pts_and_tx_azimuth         (torch.Tensor):                [B, n_rays, K_closest+1, 1]
            rx_to_pts_and_tx_elevation       (torch.Tensor):                [B, n_rays, K_closest+1, 1]

        Returns:
            torch.Tensor: rendered wireless channels with shape = [B, n_rays, 3] (TODO:)
        """
        batch_size = rx_to_pts_and_tx_distance.shape[0]
        n_rays = rx_to_pts_and_tx_distance.shape[1]
        K_closest = rx_to_pts_and_tx_distance.shape[2] - 1
        n_feat = self.n_pt_features
        n_pts = pts.shape[-2]

        # 1. Transform x so that AABB is a unit cube
        pts_x = pts[..., 0:3].transpose(2, 1)  # [1, 3, n_pts]
        # 2. Encode point clouds to feature
        feat, trans, trans_feat = self._PointFeatures(
            pts_x, rgb=None
        )  # [1, n_pts, n_feat] (n_feat=128)

        # 3. Select K-closest points features
        feat = feat.view(n_pts, n_feat)
        feat = feat[pts_mask].reshape(
            batch_size, n_rays, K_closest, n_feat
        )  # [B, n_rays, K_closest, n_feat]

        # 4. Apply weighting strategy to features
        if self.feat_weighting == int(FeatureWeighting.MAXPOOL):
            feat, _ = torch.max(feat, dim=-2, keepdim=True)
        elif self.feat_weighting == int(FeatureWeighting.ATTENTION):
            n_feat_per_point = 1

            hit_sky = hit_sky.reshape(batch_size, n_rays, K_closest, 1)
            hit_pts = ~hit_sky

            pts_feat = feat[hit_pts.repeat(1, 1, 1, n_feat)].reshape(
                1, -1, 1, n_feat
            )  # [1, x1, 1, n_feat]
            pts_distance = rx_to_pts_and_tx_distance[..., :-1, :][hit_pts].reshape(
                1, -1, 1
            )
            pts_azimuth = rx_to_pts_and_tx_azimuth[..., :-1, :][hit_pts].reshape(
                1, -1, 1
            )
            pts_elevation = rx_to_pts_and_tx_elevation[..., :-1, :][hit_pts].reshape(
                1, -1, 1
            )

            tx_distance = rx_to_pts_and_tx_distance[..., -1:, :]  # [B, n_rays, 1, 1]
            tx_azimuth = rx_to_pts_and_tx_azimuth[..., -1:, :]  # [B, n_rays, 1, 1]
            tx_elevation = rx_to_pts_and_tx_elevation[..., -1:, :]  # [B, n_rays, 1, 1]

            tx_distance = tx_distance.repeat(1, 1, K_closest, 1)[hit_pts].reshape(
                1, -1, 1
            )
            tx_azimuth = tx_azimuth.repeat(1, 1, K_closest, 1)[hit_pts].reshape(
                1, -1, 1
            )
            tx_elevation = tx_elevation.repeat(1, 1, K_closest, 1)[hit_pts].reshape(
                1, -1, 1
            )

            queries = ray_dirs[..., :-1, :][hit_pts.repeat(1, 1, 1, 3)].reshape(
                1, -1, 3
            )
            pts_feat, attn_weights = self.AttentionModule(
                directions=queries,
                features=pts_feat,
                pts_distance=pts_distance,
                pts_azimuth=pts_azimuth,
                pts_elevation=pts_elevation,
                tx_distance=tx_distance,
                tx_azimuth=tx_azimuth,
                tx_elevation=tx_elevation,
            )  # [1, x1, n_feat]

            feat = torch.zeros(
                (batch_size, n_rays, K_closest, n_feat), dtype=ray_dirs.dtype
            ).to(ray_dirs.device)
            feat[hit_pts.repeat(1, 1, 1, n_feat)] = pts_feat.flatten()

            if hit_sky.sum() > 0:
                sky_feat = feat[hit_sky.repeat(1, 1, 1, n_feat)].reshape(
                    1, -1, 1, n_feat
                )  # [1, x2, 1, n_feat]
                sky_feat = self._sky_latent.expand_as(sky_feat)

                tx_distance = rx_to_pts_and_tx_distance[
                    ..., -1:, :
                ]  # [B, n_rays, 1, 1]
                tx_azimuth = rx_to_pts_and_tx_azimuth[..., -1:, :]  # [B, n_rays, 1, 1]
                tx_elevation = rx_to_pts_and_tx_elevation[
                    ..., -1:, :
                ]  # [B, n_rays, 1, 1]

                tx_distance = tx_distance.repeat(1, 1, K_closest, 1)[hit_sky].reshape(
                    1, -1, 1
                )
                tx_azimuth = tx_azimuth.repeat(1, 1, K_closest, 1)[hit_sky].reshape(
                    1, -1, 1
                )
                tx_elevation = tx_elevation.repeat(1, 1, K_closest, 1)[hit_sky].reshape(
                    1, -1, 1
                )

                sky_distance = torch.zeros_like(tx_distance)
                sky_azimuth = torch.zeros_like(tx_azimuth)
                sky_elevation = torch.zeros_like(tx_elevation)

                queries = ray_dirs[..., :-1, :][hit_sky.repeat(1, 1, 1, 3)].reshape(
                    1, -1, 3
                )
                sky_feat, attn_weights = self.AttentionModule(
                    directions=queries,
                    features=sky_feat,
                    pts_distance=sky_distance,
                    pts_azimuth=sky_azimuth,
                    pts_elevation=sky_elevation,
                    tx_distance=tx_distance,
                    tx_azimuth=tx_azimuth,
                    tx_elevation=tx_elevation,
                )  # [1, x2, n_feat]
                feat[hit_sky.repeat(1, 1, 1, n_feat)] = sky_feat.flatten()

            feat = feat.sum(dim=-2)  # [B, n_rays, n_feat]
        elif self.feat_weighting == int(FeatureWeighting.SUM):
            n_feat_per_point = feat.shape[-2]
            feat = torch.sum(feat, dim=-2, keepdim=True) / n_feat_per_point
            feat = torch.sum(feat[..., 0, :], dim=-2)  # [B, n_rays, n_features]
        else:
            feat, attn_weights = self.AttentionModule(
                directions=ray_dirs[..., :-1, :],
                features=feat,
                pts_distance=rx_to_pts_and_tx_distance[..., :-1, :],
                pts_azimuth=rx_to_pts_and_tx_azimuth[..., :-1, :],
                pts_elevation=rx_to_pts_and_tx_elevation[..., :-1, :],
                tx_distance=rx_to_pts_and_tx_distance[..., -1:, :],
                tx_azimuth=rx_to_pts_and_tx_azimuth[..., -1:, :],
                tx_elevation=rx_to_pts_and_tx_elevation[..., -1:, :],
            )  # [B, n_rays, n_feat]

        # 5. Predict Light field output with pre-baked features and ray directions
        color = self._LightField(ray_dirs[..., -1, :], feat)

        return color
