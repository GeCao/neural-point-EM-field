import os
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


def gkern(
    kernlen=21, nsig=3, device: torch.device = torch.device("cpu"), dtype=torch.float32
):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    res = kern2d / kern2d.sum()
    return torch.from_numpy(res).to(device).to(dtype)


def DrawHeatMapTransmitters(
    tx: np.ndarray,
    ch: np.ndarray,
    radius: float = None,
    res_x: int = 256,
):
    env_idx = 0
    tx_idx = 0

    device = ch.device
    dtype = ch.dtype

    tx_proj = tx[env_idx, :, 0:2].reshape(-1, 2)  # [T, 2]
    tx_AABB_min = np.min(tx_proj, axis=0, keepdims=True)
    tx_AABB_max = np.max(tx_proj, axis=0, keepdims=True)
    tx_proj = tx_proj[0:1, :]
    tx_proj = (tx_proj - tx_AABB_min) / (tx_AABB_max - tx_AABB_min)  # [T, 2]

    gain = ch[:, :, 0, :, 0, :]  # [F, T, R, n_rays]
    gain = np.sum(gain.sum(axis=-1), axis=-1).reshape(-1, 1)  # [T, 1]
    if radius is None:
        radius = math.sqrt(1.0 / tx_proj.shape[0] / math.pi)

    AABB_length = tx_AABB_max[0] - tx_AABB_min[0]
    width = AABB_length[0]
    height = AABB_length[1]
    aspect = float(width / height)
    H, W = math.ceil(res_x / aspect), res_x
    r = int(radius * res_x)
    grid = np.zeros((1, H, W), dtype=np.float32)
    gs_kernel = gkern(kernlen=2 * r + 1, dtype=dtype, device=device)
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


def ApplyPositionBasedKernel(
    grid: torch.Tensor, kernel: torch.Tensor, grid_pos: torch.Tensor, gain: torch.Tensor
) -> torch.Tensor:
    assert gain.numel() * 2 == grid_pos.numel()
    gain = gain.flatten()
    assert grid_pos.shape[-1] == 2 and len(grid_pos.shape) == 2
    kernel_size_sqr = kernel.numel()
    kernel_size = int(math.sqrt(kernel_size_sqr))
    assert kernel_size % 2 == 1 and kernel_size * kernel_size == kernel_size_sqr
    r = (kernel_size - 1) // 2

    _, _, H, W = grid.shape
    device = grid.device
    dtype = grid.dtype

    kernel = kernel.reshape(1, 1, kernel_size, kernel_size)
    index = (
        grid_pos[:, 1].cpu().tolist(),
        grid_pos[:, 0].cpu().tolist(),
    )  # (y-idx, x-idx)
    grid = grid.reshape(1, 1, H, W)
    grid_pad = F.pad(grid, pad=(r, r, r, r), mode="constant", value=0)
    grid_pad = grid_pad.reshape(*grid_pad.shape[-2:])
    for i in range(kernel_size):
        for j in range(kernel_size):
            index_ij = ([ele + i for ele in index[0]], [ele + j for ele in index[1]])
            grid_pad[index_ij] += gain
    grid_pad = grid_pad.reshape(1, 1, *grid_pad.shape[-2:])
    output = torch.conv2d(input=grid_pad, weight=kernel)
    return output


def DrawHeatMapReceivers(
    rx: torch.Tensor,
    gain: torch.Tensor,
    radius: float = None,
    res_x: int = 1024,
    res_y: int = None,
) -> torch.Tensor:
    """
    Args:
        rx   (torch.Tensor): [R, 3]
        gain (torch.Tensor): [R, 1]
    """
    device = gain.device
    dtype = gain.dtype

    rx_proj = rx[:, 0:2]  # [R, 2]
    rx_AABB_min, _ = rx_proj.min(dim=0, keepdim=True)  # [1, 2]
    rx_AABB_max, _ = rx_proj.max(dim=0, keepdim=True)  # [1, 2]
    AABB_length = rx_AABB_max[0] - rx_AABB_min[0]
    rx_proj = (rx_proj - rx_AABB_min) / AABB_length  # [R, 2]

    if radius is None:
        radius = math.sqrt(1.0 / rx_proj.shape[0] / math.pi)

    if res_y is None:
        width = AABB_length[0]
        height = AABB_length[1]
        aspect = float(width / height)
        res_y = math.ceil(res_x / aspect)
    H, W = res_y, res_x
    r = 20
    grid = torch.zeros((1, 1, H, W)).to(dtype).to(device)
    gs_kernel = gkern(kernlen=2 * r + 1, device=device, dtype=dtype)
    gs_kernel = gs_kernel.reshape(1, 1, 2 * r + 1, 2 * r + 1)
    rx_proj[:, 0] *= W
    rx_proj[:, 1] *= H
    rx_proj = rx_proj.to(torch.int32)
    rx_proj[:, 0] = torch.clamp(rx_proj[:, 0], 0, W - 1)
    rx_proj[:, 1] = torch.clamp(rx_proj[:, 1], 0, H - 1)
    grid = ApplyPositionBasedKernel(
        grid=grid, kernel=gs_kernel, grid_pos=rx_proj, gain=gain
    )

    grid_min, grid_max = grid.min(), grid.max()
    grid = (grid - grid_min) / (grid_max - grid_min)
    grid = grid.reshape(1, *grid.shape[-2:])
    color_b = (grid < 0.5) * (0.5 - grid) * 2.0
    color_g = ((grid - 0.5).abs() < 0.25) * (0.25 - (grid - 0.5).abs()) * 4
    color_r = (grid >= 0.5) * (grid - 0.5) * 2.0
    color = torch.cat((color_r, color_g, color_b), axis=0)  # [3, H, W]
    color = torch.permute(color * 255, dims=(1, 2, 0))  # [W, H, 3]

    return color


def DeleteFloorOrCeil(
    verts: torch.Tensor, faces: torch.Tensor, axis: int = 2, mode="both"
) -> List[torch.Tensor]:
    verts_shape = verts.shape
    faces_shape = faces.shape
    verts = verts.reshape(-1, 3)
    faces = faces.reshape(-1, 3)

    eps = 1e-5
    z_min = verts[..., axis].min()
    z_max = verts[..., axis].max()
    face_verts = torch.index_select(
        verts, dim=0, index=faces.flatten()
    )  # [NumofFaces*3, 3]
    face_z = face_verts[..., axis].reshape(-1, 3)  # [NumofFaces, 3]
    if mode == "floor":
        remain_index = (face_z > z_min + eps).any(dim=-1)
    elif mode == "ceil":
        remain_index = (face_z < z_max - eps).any(dim=-1)
    elif mode == "both":
        remain_index = (face_z > z_min + eps).any(dim=-1) * (face_z < z_max - eps).any(
            dim=-1
        )
    else:
        raise RuntimeError("We only accept floor/ceil/both as the input of mode")

    faces = faces[remain_index]

    verts = verts.reshape(*verts_shape[:-2], *verts.shape)
    faces = faces.reshape(*faces_shape[:-2], *faces.shape)

    return [verts, faces]


def RenderRoom(
    pts: torch.Tensor, radius: float = None, res_x: int = 1024
) -> torch.Tensor:
    device = pts.device
    dtype = pts.dtype

    pts_proj = pts[..., 0:2].reshape(-1, 2)  # [n_pts, 2]
    rx_AABB_min, _ = pts_proj.min(dim=0, keepdim=True)  # [1, 2]
    rx_AABB_max, _ = pts_proj.max(dim=0, keepdim=True)  # [1, 2]
    AABB_length = rx_AABB_max[0] - rx_AABB_min[0]
    pts_proj = (pts_proj - rx_AABB_min) / AABB_length  # [n_pts, 2]

    gain = torch.ones((*pts_proj.shape[:-1], 1)).to(device).to(dtype)
    if radius is None:
        radius = math.sqrt(1.0 / pts_proj.shape[0] / math.pi)

    width = AABB_length[0]
    height = AABB_length[1]
    aspect = float(width / height)
    H, W = math.ceil(res_x / aspect), res_x
    r = 5
    grid = torch.zeros((1, 1, H, W)).to(dtype).to(device)
    gs_kernel = gkern(kernlen=2 * r + 1, device=device, dtype=dtype)
    gs_kernel = gs_kernel.reshape(1, 1, 2 * r + 1, 2 * r + 1)
    pts_proj[:, 0] *= W
    pts_proj[:, 1] *= H
    pts_proj = pts_proj.to(torch.int32)
    pts_proj[:, 0] = torch.clamp(pts_proj[:, 0], 0, W - 1)
    pts_proj[:, 1] = torch.clamp(pts_proj[:, 1], 0, H - 1)
    grid = ApplyPositionBasedKernel(
        grid=grid, kernel=gs_kernel, grid_pos=pts_proj, gain=gain
    )

    grid_min, grid_max = grid.min(), grid.max()
    grid = (grid - grid_min) / (grid_max - grid_min)
    grid = grid.reshape(1, *grid.shape[-2:])
    color = torch.permute(grid * 255, dims=(1, 2, 0))  # [W, H, 3]

    return color
