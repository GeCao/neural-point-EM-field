import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st
from typing import Tuple, List


def ScaleToUintCube(points: torch.Tensor, scale: float = 1.0 / 1.4):
    assert points.dim() == 3

    batch_size = points.shape[0]
    dim = 3

    AABB_min, _ = points.view(batch_size, -1, dim).min(dim=1, keepdim=True)  # [B, 3]
    AABB_max, _ = points.view(batch_size, -1, dim).max(dim=1, keepdim=True)  # [B, 3]

    scaled_points = (points - AABB_min) * 2 * torch.reciprocal(
        AABB_max - AABB_min
    ) - 1.0
    scaled_points = torch.clamp(scaled_points, -1.0, 1.0)

    scaled_points = scaled_points * scale

    return scaled_points


def PostProcessFeatures(
    feat: torch.Tensor,
    scaled_pts: torch.Tensor,
    K_closest_mask: Tuple[List[int]],
    K_closest: int,
    feature_extractor,
    img_resolution=128,
    feature_resolution=16,
):
    """Post Procecss of encode features
    Args:
        feat (torch.Tensor): Encode Features with shape = [B, maps_per_Batch, n_features, feat_height, feat_width]
        scaled_pts (torch.Tensor): points coordinates with shape = [B, n_pts, 3], clamped to (-1, 1)
        K_closest_mask (Tuple of 2 list filled with int): scaled_pts[K_closest_mask] = [B*n_rays*K_closest, 3]
    Returns:
        torch.Tensor: pass
    """
    device = feat.device

    n_batch, maps_per_batch, n_features, feat_heigth, feat_width = feat.shape
    n_feat_maps = maps_per_batch * n_batch
    feat2img_f = img_resolution // feature_resolution
    feat = feat.reshape(-1, n_features, feat_heigth, feat_width)

    batch_size, n_pts, dim = scaled_pts.shape
    K_closest_scaled_pts = scaled_pts[K_closest_mask].reshape(batch_size, -1, 3)

    # TODO: Get coordinates in the feautre maps for each point
    coordinates, coord_x, coord_y, depth = feature_extractor._get_img_coord(
        K_closest_scaled_pts, resolution=img_resolution
    )
    # Adjust for downscaled feature maps
    coord_x = (
        torch.round(coord_x.view(n_feat_maps, -1, K_closest) / feat2img_f)
        .to(torch.long)
        .to(device)
    )
    coord_x = torch.minimum(
        coord_x.to(device), torch.tensor([feat_heigth - 1]).to(device)
    )
    coord_y = torch.round(
        coord_y.view(n_feat_maps, -1, K_closest).to(device) / feat2img_f
    ).to(torch.long)
    coord_y = torch.minimum(coord_y, torch.tensor([feat_width - 1]).to(device))
    feat = feat.permute(0, 2, 3, 1)
    pts_feat = torch.stack(
        [feat[i][tuple([coord_x[i], coord_y[i]])] for i in range(n_feat_maps)]
    )
    pts_feat = pts_feat.reshape(n_batch, maps_per_batch, -1, K_closest, n_features)
    pts_feat = pts_feat.permute(0, 2, 3, 1, 4)

    return pts_feat


def _export_obj(save_path: str, vertices: np.ndarray, faces: np.ndarray):
    np_faces = faces.reshape(-1, 3)
    np_vertices = vertices.reshape(-1, vertices.shape[-1])
    if np_faces.min() == 0:
        np_faces = np_faces + 1
    with open(save_path, "w") as f:
        f.write("# OBJ file\n")
        for i in range(np_vertices.shape[0]):
            if np_vertices.shape[-1] >= 6:
                f.write(
                    "v {} {} {} {} {} {}\n".format(
                        np_vertices[i, 0],
                        np_vertices[i, 1],
                        np_vertices[i, 2],
                        np_vertices[i, 3],
                        np_vertices[i, 4],
                        np_vertices[i, 5],
                    )
                )
            else:
                f.write(
                    "v {} {} {}\n".format(
                        np_vertices[i, 0], np_vertices[i, 1], np_vertices[i, 2]
                    )
                )
        for j in range(np_faces.shape[0]):
            f.write(
                "f {} {} {}\n".format(np_faces[j, 0], np_faces[j, 1], np_faces[j, 2])
            )
    f.close()


def export_asset(save_path: str, vertices: np.ndarray, faces: np.ndarray):
    if ".obj" in str(save_path):
        _export_obj(save_path=save_path, vertices=vertices, faces=faces)
    else:
        raise NotImplementedError("Currently no support for your file-format")


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def DrawHeatMap(tx: np.ndarray, ch: np.ndarray, radius: float = None, res_x: int = 256):
    tx_proj = tx[:, :, 0:2].reshape(-1, 2)  # [F*T, 2]
    tx_AABB_min = np.min(tx_proj, axis=0, keepdims=True)
    tx_AABB_max = np.max(tx_proj, axis=0, keepdims=True)
    tx_proj = (tx_proj - tx_AABB_min) / (tx_AABB_max - tx_AABB_min)  # [F*T, 2]

    gain = ch[:, :, 0, :, 0, :]  # [F, T, R, n_rays]
    gain = np.sum(gain.sum(axis=-1), axis=-1).reshape(-1, 1)  # [F*T, 1]
    if radius is None:
        radius = math.sqrt(1.0 / tx_proj.shape[0] / math.pi)

    AABB_length = tx_AABB_max[0] - tx_AABB_min[0]
    width = AABB_length[0]
    height = AABB_length[1]
    aspect = float(width / height)
    H, W = math.ceil(res_x / aspect), res_x
    r = int(radius * res_x)
    grid = np.zeros((1, H, W), dtype=np.float32)
    gs_kernel = gkern(kernlen=2 * r + 1)
    # splatt from particles to grid:
    for tx_idx in range(tx_proj.shape[0]):
        tx_pos = tx_proj[tx_idx]
        x = int(tx_pos[0] * W)
        y = int(tx_pos[1] * H)
        add_grid = gs_kernel * gain[tx_idx, 0]
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                index = [y + j, x + i]
                if index[0] < 0 or index[1] < 0 or index[0] >= H or index[1] >= W:
                    continue
                else:
                    grid[0, index[0], index[1]] += add_grid[j + r, i + r]

    grid_min, grid_max = np.min(grid), np.max(grid)
    grid = (grid - grid_min) / (grid_max - grid_min)
    color_b = (grid < 0.5) * (0.5 - grid) * 2.0
    color_r = (grid >= 0.5) * (grid - 0.5) * 2.0
    color = np.concatenate(
        (color_r, np.zeros_like(color_r), color_b), axis=0
    )  # [3, H, W]
    color = np.swapaxes(color * 255, axis1=0, axis2=2)  # [W, H, 3]

    return color
