import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math

from src.EM.scenes import AbstractScene
from src.EM.renderer.pointcloud_encoding import MVModel
from src.EM.utils import PostProcessFeatures, FeatureWeighting
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
                feat_dim_out=self.n_pt_features * self.output_ch,
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
                (
                    Layer(W, W, z_dim=n_pt_feat)
                    if i not in self.skips
                    else Layer(W + self.input_ch, W, z_dim=n_pt_feat)
                )
                for i in range(D - 1)
            ]
        )

        self.n_SH = 9
        self.ch_linear = DenseLayer(W, self.n_SH, activate=False)
        self.SH_basis_hlp = [
            0.5 / math.sqrt(math.pi),
            0.5 * math.sqrt(1.5 / math.pi),
            0.5 * math.sqrt(1.5 / math.pi),
            0.5 * math.sqrt(1.5 / math.pi),
            0.25 * math.sqrt(7.5 / math.pi),
            0.5 * math.sqrt(7.5 / math.pi),
            0.25 * math.sqrt(5.0 / math.pi),
            0.5 * math.sqrt(7.5 / math.pi),
            0.25 * math.sqrt(7.5 / math.pi),
        ]

    def forward(
        self,
        feat: torch.Tensor,
        ray_d: torch.Tensor,
        rx_to_probe_and_tx_distance: torch.Tensor,
        rx_to_probe_and_tx_azimuth: torch.Tensor,
        rx_to_probe_and_tx_elevation: torch.Tensor,
    ):
        """
        Args:
            feat                             (torch.Tensor):                [B, n_rays, n_feat]

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
        )  # [B, n_rays+1, feat_ch * n_ch]
        z = z.reshape(n_batch, n_rays + 1, feat_ch, self.output_ch)
        z = z.permute((0, 3, 1, 2)).reshape(
            n_batch * self.output_ch, n_rays + 1, feat_ch
        )
        ray_info = self.pose_encoding(ray_d)  # [B, 1, ray_ch]
        ray_info = ray_info.unsqueeze(1).repeat(1, self.output_ch, 1, 1)
        ray_info = ray_info.reshape(n_batch * self.output_ch, *ray_info.shape[2:])

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
        out = out.reshape(n_batch, self.output_ch, n_rays + 1, self.n_SH)
        out = out.permute((0, 2, 1, 3))  # [B, n_rays+1, n_ch, 9]

        # Try SH-encoder (with n = 3)
        assert self.n_SH == 9
        normal = F.normalize(-ray_d[:, :-1, None, :], dim=-1)  # [B, n_rays, ch, 3]
        SH_basis = torch.zeros_like(out[:, :-1, :, :])  # [B, n_rays, ch, 9]
        SH_basis[..., 0] = self.SH_basis_hlp[0]
        SH_basis[..., 1] = self.SH_basis_hlp[1] * normal[..., 0]
        SH_basis[..., 2] = self.SH_basis_hlp[2] * normal[..., 2]
        SH_basis[..., 3] = self.SH_basis_hlp[3] * normal[..., 1]
        SH_basis[..., 4] = self.SH_basis_hlp[4] * (
            normal[..., 0] * normal[..., 0] - normal[..., 1] * normal[..., 1]
        )
        SH_basis[..., 5] = self.SH_basis_hlp[5] * normal[..., 0] * normal[..., 2]
        SH_basis[..., 6] = self.SH_basis_hlp[6] * (
            3 * normal[..., 2] * normal[..., 2] - 1
        )
        SH_basis[..., 7] = self.SH_basis_hlp[7] * normal[..., 1] * normal[..., 2]
        SH_basis[..., 8] = self.SH_basis_hlp[8] * (2 * normal[..., 0] * normal[..., 1])

        # n_rays + 1 Decompose:
        #     n_rays: SH_coeff * SH_basis (sample from light probe)
        #     1:      Sum pooling         (sample from LOS)

        if self.output_ch == 1 or self.output_ch == 4:
            # Only take gain as output
            out = (out[:, :-1, :, :] * SH_basis).sum(dim=-1).sum(dim=1) + out[
                :, -1, :, :
            ].sum(
                dim=-1
            )  # [B, ch]
            out = out.view(n_batch, self.output_ch)
        else:
            # Take all the channel
            out = (out[:, :-1, :, :] * SH_basis).sum(dim=-1)
            out = out.view(n_batch, n_rays, self.output_ch)
            if self.output_ch >= 7:
                out = torch.cat(
                    (
                        out[..., 0:1],
                        2.0 * math.pi * torch.sigmoid(out[..., 1:2]) - math.pi,
                        out[..., 2:3],
                        2.0 * math.pi * torch.sigmoid(out[..., 3:4]),
                        math.pi * torch.sigmoid(out[..., 4:5]),
                        2.0 * math.pi * torch.sigmoid(out[..., 5:6]),
                        math.pi * torch.sigmoid(out[..., 6:7]),
                        out[..., 7:],
                    ),
                    dim=-1,
                )
            elif self.output_ch == 6:
                out = torch.cat(
                    (
                        out[..., 0:1],
                        2.0 * math.pi * torch.sigmoid(out[..., 1:2]) - math.pi,
                        out[..., 2:3],
                        F.normalize(out[..., 3:6], dim=-1),
                    ),
                    dim=-1,
                )
            elif self.output_ch == 3:
                out = torch.cat(
                    (
                        out[..., 0:1],
                        torch.zeros_like(out[..., 1:2]),
                        out[..., 2:3],
                    ),
                    dim=-1,
                )
            else:
                raise RuntimeError("output ch for neural render is not correctly set")

        return out


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
        output_ch: int = 1,
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

        # TODO:
        self._LightField = (
            LightFieldNet(
                n_frequ_pos_encoding=4,
                n_feat_in=n_feat_in,
                n_pt_feat=self.n_pt_features,
                output_ch=output_ch,
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

    def forward(
        self,
        pts: torch.Tensor,
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
            pts                              (torch.Tensor):                [1, n_pts, dim=3]
            light_probe_pos                  (torch.Tensor):                [n_probes, dim=3]

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
        batch_size = rx_to_probe_and_tx_distance.shape[0]
        n_rays = rx_to_probe_and_tx_distance.shape[1] - 1
        K_closest = probe_to_pts_and_tx_distance.shape[2] - 1
        n_feat = self.n_pt_features
        n_pts = pts.shape[-2]

        # 1. Transform x so that bbox is a unit cube
        pts_x = pts[..., 0:3].transpose(2, 1)  # [B, 3, n_pts]
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
                directions=probe_to_pts_and_tx_dir[..., -1:, :],
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
            ray_dirs,
            rx_to_probe_and_tx_distance,
            rx_to_probe_and_tx_azimuth,
            rx_to_probe_and_tx_elevation,
        )

        return color
