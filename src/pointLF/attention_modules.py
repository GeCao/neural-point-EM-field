import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pointLF.feature_mapping import PositionalEncoding
from src.pointLF.layer import DenseLayer


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
        azimuth_2d: bool = False,
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

        self.use_distance = use_distance
        self.use_projected = use_projected
        self.use_pitch = use_pitch
        self.use_azimuth = use_azimuth
        self.az_2d = azimuth_2d
        self.use_feat = use_feat

        self.used_distances = {
            "distance": use_distance,
            "projected_distance": use_projected,
            "pitch": use_pitch,
            "azimuth": use_azimuth,
        }

        self.enc_dist_dim = (n_frequ_pos_encoding * 2) * (
            use_distance + use_projected + use_pitch + use_azimuth + azimuth_2d
        ) + 1 * (use_distance + use_projected + 2 * azimuth_2d)

        self.input_ch = feat_dim_in * use_feat + self.enc_dist_dim

        self.dim_feat = feat_dim_in * use_feat

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
    ):
        """See Figure 4 from: https://arxiv.org/pdf/2112.01473.pdf
        This function is designed for point "value & key" part

        Args:
            features           (torch.Tensor): [B, n_rays, n_k_closest, n_feat_maps, dim_feat] | See $l_{k}$ in reference
            distance           (torch.Tensor): [B, n_rays, n_k_closest]                        | See $d_{k,j}$ in reference
            projected_distance (torch.Tensor): [B, n_rays, n_k_closest]                        | See $d_{k,j}$ in reference
            pitch              (torch.Tensor): [B, n_rays, n_k_closest]                        | See $\psi_{k,j}$ in reference
            azimuth            (torch.Tensor): [B, n_rays, n_k_closest]                        | See $\phi_{k,j}$ in reference

        Returns:
            torch.Tensor: keys   [B*n_rays, n_k_closest * n_feat_maps, key_len             ]
            torch.Tensor: values [B*n_rays, n_k_closest * n_feat_maps, val_len(W - key_len)]

        1. Stack all of the inputs (blue block):
            1.1 Blue_blocks = {
                EncodeLenth(distance),----------------> (2*n_freq + 1)
                EncodeLenth(projected_distance),------> (2*n_freq + 1)
                EncodeAngle(pitch),-------------------> (2*n_freq    )
                If azimuth_2d:
                    EncodeLength(sin(azimuth)),-------> (2*n_freq + 1)
                    EncodeLength(cos(azimuth)),-------> (2*n_freq + 1)
                else:
                    EncodeAngle(azimuth),-------------> (2*n_freq    )
            }  # [B, n_rays, n_k_closest, enc_dist_dim]

            1.2 expand To [B, n_rays, n_k_closest, n_feat_maps, enc_dist_dim]
        2. [If use_feat] Stack features ($l_{k}$) with blue blocks to get $v_{k,j}$ (green blocks):
            2.1 input = {
                features,
                Blue_blocks,
            }  # [B, n_rays, n_k_closest, n_feat_maps, enc_dist_dim + dim_feat]

            2.2 reshape as [B*n_rays, n_k_closest*n_feat_maps, enc_dist_dim + dim_feat]

            2.3 pass input into green blocks (MLP(W, W, W, .., W) the number of W is D) and get v_{k, j}.
            # [B*n_rays, n_k_closest*n_feat_maps, W]
        3. Get your keys (values) vai function $F_{\theta K}$ ($F_{\theta V$) (Gray Blocks):
            keys   = F_{\theta, K}(v_{k, j})  # [B*n_rays, n_k_closest*n_feat_maps, key_len             ]
            values = F_{\theta, V}(v_{k, j})  # [B*n_rays, n_k_closest*n_feat_maps, val_len(W - key_len)]
        """
        n_batch, n_rays, k_closest, n_feat_maps, feat_len = features.shape

        x = torch.empty(n_batch, n_rays, k_closest, 0, device=distance.device)

        # Normalize feature distances
        if self.used_distances["distance"]:
            enc_dist = self.poseEncodingLen(distance[..., None])
            x = torch.cat([x, enc_dist], dim=-1)

        if self.used_distances["projected_distance"]:
            x_proj_normalized = (
                projected_distance / projected_distance.max(dim=-1).values[..., None]
            )
            enc_proj = self.poseEncodingLen(x_proj_normalized[..., None])
            x = torch.cat([x, enc_proj], dim=-1)

        if self.used_distances["pitch"]:
            enc_pitch = self.poseEncodingAng(pitch[..., None])
            x = torch.cat([x, enc_pitch], dim=-1)

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

            x = torch.cat([x, enc_azimuth], dim=-1)

        x = x[..., None, :].expand(
            n_batch, n_rays, k_closest, n_feat_maps, self.enc_dist_dim
        )

        if self.use_feat:
            x = torch.cat([features, x], dim=-1)

        x = x.reshape(n_batch * n_rays, k_closest * n_feat_maps, self.input_ch)

        for i, layer in enumerate(self.linear):
            x = layer(x)
            x = F.relu(x)

        values = self.value_linear(x)
        keys = self.key_linear(x)

        return values, keys


class RayPointPoseEncoder(nn.Module):
    def __init__(
        self,
        W: int = 64,
        D: int = 1,
        n_frequ_pos_encoding: int = 4,
        key_len: int = 16,
        use_distance: bool = True,
        use_projected: bool = True,
        use_angle: bool = True,
        feat_map_encoding: bool = True,
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

    def forward(
        self,
        distance: torch.Tensor,
        projected_distance: torch.Tensor,
        angle: torch.Tensor,
        n_feat_maps: int = 1,
    ):
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
        directions,
        features,
        distance=None,
        projected_distance=None,
        pitch=None,
        azimuth=None,
        **kwargs
    ):
        feat_dim = features.shape[-1]
        n_batch, n_rays, _ = directions.shape
        # Generate Key and values from projected point cloud features and positional encoded ray-point distance with an MLP
        val, key = self.FeatureEncoder(
            features, distance, projected_distance, pitch, azimuth
        )  # [B*n_rays, n_k_closest*n_feat_map, key_dim(val_dim)]

        # Use encoded ray-dirs in MLP to get query vector
        query = self.RayEncoder(directions.reshape(-1, 3))
        query = query[:, None]  # [B*n_rays, q_len=embeded_dim, 1]

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
