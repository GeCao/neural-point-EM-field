import torch
import torch.nn as nn
import torch.nn.functional as F
from src.EM.renderer.pointcloud_encoding import PositionalEncoding
from src.EM.renderer.pointcloud_encoding import DenseLayer


class FeatureDistanceEncoder(nn.Module):
    def __init__(
        self,
        feat_dim_in: int = 128,
        W: int = 256,
        D: int = 1,
        n_frequ_pos_encoding: int = 4,
        key_len: int = 16,
        val_len: int = None,
        use_distance: bool = True,
        use_projected: bool = False,
        use_pitch: bool = True,
        use_azimuth: bool = True,
        use_tx_distance: bool = True,
        use_tx_azimuth: bool = True,
        use_tx_elevation: bool = True,
        azimuth_2d: bool = False,
        tx_azimuth_2d: bool = False,
        no_feat: bool = False,
    ):
        """
        feat_dim_in:
        W:
        D:
        n_frequ_pos_encoding:
        key_len:
        use_distance:
        use_projected:
        """
        super(FeatureDistanceEncoder, self).__init__()
        use_feat = not no_feat

        self.poseEncodingLen = PositionalEncoding(n_frequ_pos_encoding, 1)
        self.poseEncodingAng = PositionalEncoding(
            n_frequ_pos_encoding, 1, include_input=False
        )

        self.txEncodingLen = PositionalEncoding(n_frequ_pos_encoding, 1)
        self.txEncodingAng = PositionalEncoding(
            n_frequ_pos_encoding, 1, include_input=False
        )

        self.use_distance = use_distance
        self.use_projected = use_projected
        self.use_pitch = use_pitch
        self.use_azimuth = use_azimuth
        self.use_tx_distance = use_tx_distance
        self.use_tx_azimuth = use_tx_azimuth
        self.use_tx_elevation = use_tx_elevation
        self.az_2d = azimuth_2d
        self.tx_az_2d = tx_azimuth_2d
        self.use_feat = use_feat

        self.used_distances = {
            "distance": use_distance,
            "projected_distance": use_projected,
            "pitch": use_pitch,
            "azimuth": use_azimuth,
            "tx_distance": use_tx_distance,
            "tx_azimuth": use_tx_azimuth,
            "tx_elevation": use_tx_elevation,
        }

        self.feat_dim_in_key = feat_dim_in // 2
        self.feat_dim_in_val = feat_dim_in - self.feat_dim_in_key
        self.enc_dist_dim_key = (n_frequ_pos_encoding * 2) * (
            use_distance + use_projected + use_pitch + use_azimuth + azimuth_2d
        ) + 1 * (use_distance + use_projected + 2 * azimuth_2d)
        self.enc_dist_dim_val = (n_frequ_pos_encoding * 2) * (
            use_tx_distance + use_tx_azimuth + use_tx_elevation + tx_azimuth_2d
        ) + 1 * (use_tx_distance + 2 * tx_azimuth_2d)
        self.input_ch_key = self.feat_dim_in_key * use_feat + self.enc_dist_dim_key
        self.input_ch_val = self.feat_dim_in_val * use_feat + self.enc_dist_dim_val
        self.linear_key = nn.ModuleList(
            [DenseLayer(self.input_ch_key, W)]
            + [
                DenseLayer(
                    W,
                    W,
                )
                for i in range(D - 1)
            ]
        )
        self.linear_val = nn.ModuleList(
            [DenseLayer(self.input_ch_val, W)]
            + [
                DenseLayer(
                    W,
                    W,
                )
                for i in range(D - 1)
            ]
        )

        if val_len is None:
            self.val_out_ch = W - key_len
        else:
            self.val_out_ch = val_len

        self.value_linear = DenseLayer(W, self.val_out_ch)
        self.key_linear = DenseLayer(W, key_len)

    def forward(
        self,
        features: torch.Tensor,
        distance: torch.Tensor,
        projected_distance: torch.Tensor,
        pitch: torch.Tensor,
        azimuth: torch.Tensor,
        tx_distance: torch.Tensor = None,
        tx_azimuth: torch.Tensor = None,
        tx_elevation: torch.Tensor = None,
    ):
        """

        directions: [Batch_sz, N_rays, 3]
        features: [Batch_sz, N_rays, N_k_closest, N_feat_maps, dim_feat]
        distance: [Batch_sz, N_rays, N_k_closest]
        projected_distance: [Batch_sz, N_rays, N_k_closest]
        :return:
        :rtype:
        """
        n_batch, n_rays, k_closest, n_feat_maps, feat_len = features.shape

        x_key = torch.empty(n_batch, n_rays, k_closest, 0, device=distance.device)
        x_val = torch.empty(n_batch, n_rays, k_closest, 0, device=distance.device)

        # Normalize feature distances
        if self.used_distances["distance"]:
            enc_dist = self.poseEncodingLen(distance[..., None])
            x_key = torch.cat([x_key, enc_dist], dim=-1)

        if self.used_distances["projected_distance"]:
            x_proj_normalized = (
                projected_distance / projected_distance.max(dim=-1).values[..., None]
            )
            enc_proj = self.poseEncodingLen(x_proj_normalized[..., None])
            x_key = torch.cat([x_key, enc_proj], dim=-1)

        if self.used_distances["pitch"]:
            enc_pitch = self.poseEncodingAng(pitch[..., None])
            x_key = torch.cat([x_key, enc_pitch], dim=-1)

        if self.used_distances["azimuth"]:
            if not self.az_2d:
                enc_azimuth = self.poseEncodingAng(azimuth[..., None])
            else:
                enc_azimuth = torch.cat(
                    [
                        self.poseEncodingLen(torch.sin(azimuth)[..., None]),
                        self.poseEncodingLen(torch.cos(azimuth)[..., None]),
                    ],
                    dim=-1,
                )

            x_key = torch.cat([x_key, enc_azimuth], dim=-1)

        if self.used_distances["tx_distance"]:
            tx_distance = torch.where(
                tx_distance < 0.01,
                100 * torch.ones_like(tx_distance),
                1.0 / tx_distance,
            )
            enc_tx_dist = self.txEncodingLen(tx_distance[..., None])
            x_val = torch.cat([x_val, enc_tx_dist], dim=-1)

        if self.used_distances["tx_azimuth"]:
            if not self.az_2d:
                enc_tx_azimuth = self.txEncodingAng(tx_azimuth[..., None])
            else:
                enc_tx_azimuth = torch.cat(
                    [
                        self.txEncodingLen(torch.sin(tx_azimuth)[..., None]),
                        self.txEncodingLen(torch.cos(tx_azimuth)[..., None]),
                    ],
                    dim=-1,
                )

            x_val = torch.cat([x_val, enc_tx_azimuth], dim=-1)

        if self.used_distances["tx_elevation"]:
            enc_tx_elevation = self.txEncodingAng(tx_elevation[..., None])
            x_val = torch.cat([x_val, enc_tx_elevation], dim=-1)

        x_key = x_key[..., None, :].expand(
            n_batch, n_rays, k_closest, n_feat_maps, self.enc_dist_dim_key
        )
        x_val = x_val[..., None, :].expand(
            n_batch, n_rays, k_closest, n_feat_maps, self.enc_dist_dim_val
        )

        if self.use_feat:
            x_key = torch.cat(
                [
                    features[..., 0::2],
                    x_key,
                ],
                dim=-1,
            ).reshape(n_batch * n_rays, k_closest * n_feat_maps, self.input_ch_key)
            x_val = torch.cat(
                [
                    features[..., 1::2],
                    x_val,
                ],
                dim=-1,
            ).reshape(n_batch * n_rays, k_closest * n_feat_maps, self.input_ch_val)
        else:
            x_key = x_key.reshape(
                n_batch * n_rays, k_closest * n_feat_maps, self.input_ch_key
            )
            x_val = x_val.reshape(
                n_batch * n_rays, k_closest * n_feat_maps, self.input_ch_val
            )

        for i, layer in enumerate(self.linear_key):
            x_key = layer(x_key)
            x_key = F.relu(x_key)

        for i, layer in enumerate(self.linear_val):
            x_val = layer(x_val)
            x_val = F.relu(x_val)

        values = self.value_linear(x_val)
        keys = self.key_linear(x_key)

        return values, keys


# class FeatureDistanceEncoder(nn.Module):
#     def __init__(
#         self,
#         feat_dim_in=128,
#         W=256,
#         D=1,
#         n_frequ_pos_encoding=4,
#         key_len=16,
#         val_len=None,
#         use_distance=True,
#         use_projected=False,
#         use_pitch=True,
#         use_azimuth=True,
#         azimuth_2d=False,
#         no_feat=False,
#     ):
#         """
#         feat_dim_in:
#         W:
#         D:
#         n_frequ_pos_encoding:
#         key_len:
#         use_distance:
#         use_projected:
#         """
#         super(FeatureDistanceEncoder, self).__init__()
#         use_feat = not no_feat

#         self.poseEncodingLen = PositionalEncoding(n_frequ_pos_encoding, 1)
#         self.poseEncodingAng = PositionalEncoding(
#             n_frequ_pos_encoding, 1, include_input=False
#         )

#         self.use_distance = use_distance
#         self.use_projected = use_projected
#         self.use_pitch = use_pitch
#         self.use_azimuth = use_azimuth
#         self.az_2d = azimuth_2d
#         self.use_feat = use_feat

#         self.used_distances = {
#             "distance": use_distance,
#             "projected_distance": use_projected,
#             "pitch": use_pitch,
#             "azimuth": use_azimuth,
#         }

#         self.enc_dist_dim = (n_frequ_pos_encoding * 2) * (
#             use_distance + use_projected + use_pitch + use_azimuth + azimuth_2d
#         ) + 1 * (use_distance + use_projected + 2 * azimuth_2d)

#         self.input_ch = feat_dim_in * use_feat + self.enc_dist_dim

#         self.dim_feat = feat_dim_in * use_feat

#         self.linear = nn.ModuleList(
#             [DenseLayer(self.input_ch, W)]
#             + [
#                 DenseLayer(
#                     W,
#                     W,
#                 )
#                 for i in range(D - 1)
#             ]
#         )

#         if val_len is None:
#             self.val_out_ch = W - key_len
#         else:
#             self.val_out_ch = val_len

#         self.value_linear = DenseLayer(W, self.val_out_ch)
#         self.key_linear = DenseLayer(W, key_len)

#     def forward(
#         self,
#         features: torch.Tensor,
#         distance: torch.Tensor,
#         projected_distance: torch.Tensor,
#         pitch: torch.Tensor,
#         azimuth: torch.Tensor,
#         tx_distance: torch.Tensor = None,
#         tx_azimuth: torch.Tensor = None,
#         tx_elevation: torch.Tensor = None,
#     ):
#         """

#         directions: [Batch_sz, N_rays, 3]
#         features: [Batch_sz, N_rays, N_k_closest, N_feat_maps, dim_feat]
#         distance: [Batch_sz, N_rays, N_k_closest]
#         projected_distance: [Batch_sz, N_rays, N_k_closest]
#         :return:
#         :rtype:
#         """
#         n_batch, n_rays, k_closest, n_feat_maps, feat_len = features.shape

#         x = torch.empty(n_batch, n_rays, k_closest, 0, device=distance.device)

#         # Normalize feature distances
#         if self.used_distances["distance"]:
#             enc_dist = self.poseEncodingLen(distance[..., None])
#             x = torch.cat([x, enc_dist], dim=-1)

#         if self.used_distances["projected_distance"]:
#             x_proj_normalized = (
#                 projected_distance / projected_distance.max(dim=-1).values[..., None]
#             )
#             enc_proj = self.poseEncodingLen(x_proj_normalized[..., None])
#             x = torch.cat([x, enc_proj], dim=-1)

#         if self.used_distances["pitch"]:
#             enc_pitch = self.poseEncodingAng(pitch[..., None])
#             x = torch.cat([x, enc_pitch], dim=-1)

#         if self.used_distances["azimuth"]:
#             if not self.az_2d:
#                 enc_azimuth = self.poseEncodingAng(azimuth[..., None])
#             else:
#                 enc_azimuth = torch.cat(
#                     [
#                         self.poseEncodingLen(torch.sin(azimuth)[..., None]),
#                         self.poseEncodingLen(torch.cos(azimuth)[..., None]),
#                     ],
#                     dim=-1,
#                 )

#             x = torch.cat([x, enc_azimuth], dim=-1)

#         x = x[..., None, :].expand(
#             n_batch, n_rays, k_closest, n_feat_maps, self.enc_dist_dim
#         )

#         if self.use_feat:
#             x = torch.cat([features, x], dim=-1)

#         x = x.reshape(n_batch * n_rays, k_closest * n_feat_maps, self.input_ch)

#         for i, layer in enumerate(self.linear):
#             x = layer(x)
#             x = F.relu(x)

#         values = self.value_linear(x)
#         keys = self.key_linear(x)

#         return values, keys


class RayPointPoseEncoder(nn.Module):
    def __init__(
        self,
        W=64,
        D=1,
        n_frequ_pos_encoding=4,
        key_len=16,
        use_distance=True,
        use_projected=True,
        use_angle=True,
        feat_map_encoding=True,
    ):
        """
        feat_len:
        W:
        D:
        n_frequ_pos_encoding:
        key_len:
        use_distance:
        use_projected:
        """
        super(RayPointPoseEncoder, self).__init__()
        self.poseEncoding = PositionalEncoding(n_frequ_pos_encoding, 1)

        self.use_distance = use_distance
        self.use_projected = use_projected
        self.use_angle = use_angle
        self.use_feat_map_enc = feat_map_encoding

        self.input_ch = (n_frequ_pos_encoding * 2 + 1) * (
            use_distance + use_projected + use_angle
        ) + (n_frequ_pos_encoding * 2) * feat_map_encoding * 3

        self.linear = nn.ModuleList(
            [DenseLayer(self.input_ch, W)]
            + [
                DenseLayer(
                    W,
                    W,
                )
                for i in range(D - 1)
            ]
        )

        self.key_linear = DenseLayer(W, key_len)

        self.feat_map_enc = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.feat_map_enc = PositionalEncoding(
            n_frequ_pos_encoding,
            3,
            include_input=False,
        )(self.feat_map_enc)

    def forward(self, distance, projected_distance, angle, n_feat_maps=1):
        """
        distance: [Batch_sz, N_rays, N_k_closest]
        projected_distance: [Batch_sz, N_rays, N_k_closest]
        angle: [Batch_sz, N_rays, N_k_closest]
        :return:
        :rtype:
        """
        n_batch, n_rays, k_closest = distance.shape

        # Normalize feature distances
        x_proj_normalized = (
            projected_distance / projected_distance.max(dim=-1).values[..., None]
        )
        x_dist_normalized = distance / distance.max(dim=-1).values[..., None]

        # Positional encoding of ray distances and projected distance
        # TODO: Debug from here
        enc_proj = self.poseEncoding(x_proj_normalized[..., None])
        enc_dist = self.poseEncoding(x_dist_normalized[..., None])
        enc_angle = self.poseEncoding(angle[..., None])

        if self.use_feat_map_enc:
            if n_feat_maps == 6:
                feat_map_enc = self.feat_map_enc.to(enc_proj.device)

                x = torch.cat([enc_dist, enc_proj, enc_angle], dim=-1)
                x = x[..., None, :].repeat(1, 1, 1, n_feat_maps, 1)

                feat_map_enc = feat_map_enc[None, None, None].repeat(
                    n_batch, n_rays, k_closest, 1, 1
                )

                x = torch.cat([x, feat_map_enc], dim=-1)
                x = x.reshape(n_batch * n_rays, k_closest * n_feat_maps, self.input_ch)

            else:
                Warning("Not implemented yet.")
        else:
            x = torch.cat([enc_dist, enc_proj, enc_angle], dim=-1)
            x = x.reshape(n_batch * n_rays, k_closest, self.input_ch)

        # enc_proj = enc_proj.unsqueeze(3).repeat(1, 1, 1, n_feat_maps, 1)
        # enc_dist = enc_dist.unsqueeze(3).repeat(1, 1, 1, n_feat_maps, 1)
        # enc_angle = enc_angle.unsqueeze(3).repeat(1, 1, 1, n_feat_maps, 1)

        for i, layer in enumerate(self.linear):
            x = layer(x)
            x = F.relu(x)

        keys = self.key_linear(x)

        return keys


class RayEncoder(nn.Module):
    def __init__(
        self,
        W=64,
        D=1,
        q_len=16,
        n_frequ_pos_encoding=4,
    ):
        super(RayEncoder, self).__init__()
        self.poseEncoding = PositionalEncoding(n_frequ_pos_encoding, 3)

        self.input_ch = (n_frequ_pos_encoding * 2 + 1) * 3

        self.linear = nn.ModuleList(
            [DenseLayer(self.input_ch, W)]
            + [
                DenseLayer(
                    W,
                    W,
                )
                for i in range(D - 2)
            ]
        )

        self.query_linear = DenseLayer(
            W,
            q_len,
        )

    def forward(self, ray_direction):
        x = self.poseEncoding(ray_direction)

        for i, layer in enumerate(self.linear):
            x = layer(x)

        x = self.query_linear(x)

        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PointFeatureAttention(nn.Module):
    def __init__(
        self,
        feat_dim_in,
        feat_dim_out,
        embeded_dim=128,
        n_att_heads=8,
        kdim=128,
        vdim=128,
        Feat_D=2,
        Feat_W=256,
        Ray_D=2,
        Ray_W=64,
        new_encoding=False,
        no_feat=False,
    ):
        super(PointFeatureAttention, self).__init__()

        if new_encoding:
            n_frequ = 8
        else:
            n_frequ = 4

        self.dim_out = feat_dim_out
        self.FeatureEncoder = FeatureDistanceEncoder(
            feat_dim_in=feat_dim_in,
            W=Feat_W,
            D=Feat_D,
            n_frequ_pos_encoding=n_frequ,
            key_len=kdim,
            val_len=vdim,
            use_distance=True,
            use_projected=False,
            use_pitch=True,
            use_azimuth=True,
            azimuth_2d=new_encoding,
            no_feat=no_feat,
        )

        self.RayEncoder = RayEncoder(
            W=Ray_W,
            D=Ray_D,
            q_len=embeded_dim,
            n_frequ_pos_encoding=4,
        )

        self.n_att_heads = n_att_heads

        if self.n_att_heads > 0:
            self.attention = nn.MultiheadAttention(
                embed_dim=embeded_dim, num_heads=n_att_heads, kdim=kdim, vdim=vdim
            )
        else:
            self.attention = ScaledDotProductAttention(1.0)

        self.dim_reduction_out = nn.Linear(embeded_dim, feat_dim_out)

    def forward(
        self,
        directions: torch.Tensor,
        features: torch.Tensor,
        distance: torch.Tensor = None,
        projected_distance: torch.Tensor = None,
        pitch: torch.Tensor = None,
        azimuth: torch.Tensor = None,
        tx_distance: torch.Tensor = None,
        tx_azimuth: torch.Tensor = None,
        tx_elevation: torch.Tensor = None,
        **kwargs
    ):
        feat_dim = features.shape[-1]
        n_batch, n_rays, _ = directions.shape
        # Generate Key and values from projected point cloud features and positional encoded ray-point distance with an MLP
        val, key = self.FeatureEncoder(
            features,
            distance,
            projected_distance,
            pitch,
            azimuth,
            tx_distance,
            tx_azimuth,
            tx_elevation,
        )

        # Use encoded ray-dirs in MLP to get query vector
        query = self.RayEncoder(directions.reshape(-1, 3))
        query = query[:, None]

        # Attention
        # Query Vector + per point Key + per-point value vector in transformer-style attention + pooling to create conditioning vector
        # Transpose for torch attention module
        if self.n_att_heads > 0:
            query = query.transpose(1, 0)
            key = key.transpose(1, 0)
            val = val.transpose(1, 0)
            out, attn_weights = self.attention(query, key, val)
            out = self.dim_reduction_out(out)
            out = out.transpose(0, 1)
        else:
            out, attn_weights = self.attention(query, key, val)
            out = self.dim_reduction_out(out)

        out = out.reshape(n_batch, n_rays, self.dim_out)
        return out, attn_weights


class PointDistanceAttention(nn.Module):
    def __init__(self, v_len=128, kq_len=16):
        super(PointDistanceAttention, self).__init__()

        self.RayPointPoseEncoder = RayPointPoseEncoder(
            W=256,
            D=2,
            n_frequ_pos_encoding=4,
            key_len=kq_len,
            use_distance=True,
            use_projected=True,
            use_angle=True,
        )

        self.RayEncoder = RayEncoder(
            W=64,
            D=1,
            q_len=kq_len,
            n_frequ_pos_encoding=4,
        )

        self.kq_len = kq_len
        self.val_len = v_len

        # TODO: Integrate and Decide between multi head and single attention module
        self.attention = ScaledDotProductAttention(1.0)

    def forward(
        self, directions, features, distance, projected_distance, angle, **kwargs
    ):
        n_feat_maps, feat_dim = features.shape[-2:]
        n_batch, n_rays, _ = directions.shape

        # b) Take features from feature pints directly and encode with distance, projection and angles
        key = self.RayPointPoseEncoder(distance, projected_distance, angle, n_feat_maps)
        val = features.reshape(n_batch * n_rays, -1, feat_dim)

        # Use encoded ray-dirs in MLP to get query vector
        query = self.RayEncoder(directions.reshape(-1, 3))

        # Attention
        # Query Vector + per point Key + per-point value vector in transformer-style attention + pooling to create conditioning vector
        out, attn = self.attention(query[:, None], key, val)

        out = out.reshape(n_batch, n_rays, self.val_len)
        return out
