import sys, os
import argparse
import pywavefront
import math
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import cv2
import h5py
from typing import List
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.ops import sample_points_from_meshes

sys.path.append("../")

from src.EM.utils import (
    DumpCFGFile,
    LoadMeshes,
    LoadPointCloudFromMesh,
    LoadSingleMesh,
    create_3d_meshgrid_tensor,
    SplatFromParticlesToGrid,
)


def main(scene_name: str):
    add_rx = True
    add_tx = True
    with_light_probes = True
    n_samples_per_mesh = 20000 if "etoicenter" in scene_name else 300

    dtype = torch.float32
    device = torch.device("cpu")
    rx_pos = torch.Tensor([-150, 100, 1.5]).to(dtype)
    tx_pos = torch.Tensor([-180, -0, 1.5]).to(dtype)

    demo_path = os.path.abspath(os.curdir)
    root_path = os.path.abspath(os.path.join(demo_path, ".."))
    data_path = os.path.join(root_path, f"data/{scene_name}")
    asset_path = os.path.join(root_path, "assets")
    rx_path = os.path.join(asset_path, "stanford_bunny.obj")
    tx_path = os.path.join(asset_path, "tower.obj")

    obj_dir = os.path.join(data_path, "objs")
    data_files = os.listdir(obj_dir)
    data_files = sorted(data_files)

    # choice 1: Load ems per primitive
    vertices = []
    faces = []
    for filename in data_files:
        obj_path = os.path.join(obj_dir, filename)
        new_v, new_f = LoadSingleMesh(obj_path=obj_path, device=device, dtype=dtype)
        vertices.append(new_v)
        faces.append(new_f)
    if add_tx:
        scene = pywavefront.Wavefront(tx_path, collect_faces=True)
        new_v = torch.Tensor(scene.vertices).to(device).to(dtype)
        new_f = torch.Tensor(scene.meshes[None].faces).to(device).to(torch.int32)
        new_v = new_v.reshape(1, -1, 3)
        new_f = new_f.reshape(1, -1, 3)
        # y-up to z-up
        new_v = torch.cat((new_v[..., 0:1], new_v[..., 2:3], new_v[..., 1:2]), dim=-1)
        new_v = new_v * 10 + tx_pos
        vertices.append(new_v)
        faces.append(new_f)
        tx_center = (
            new_v.reshape(-1, 3).min(dim=0)[0] + new_v.reshape(-1, 3).max(dim=0)[0]
        ) / 2.0
        tx_center[..., 2] *= 2
    if add_rx:
        scene = pywavefront.Wavefront(rx_path, collect_faces=True)
        new_v = torch.Tensor(scene.vertices).to(device).to(dtype)
        new_f = torch.Tensor(scene.meshes[None].faces).to(device).to(torch.int32)
        new_v = new_v.reshape(1, -1, 3)
        new_f = new_f.reshape(1, -1, 3)
        # y-up to z-up
        new_v = torch.cat((new_v[..., 0:1], new_v[..., 2:3], new_v[..., 1:2]), dim=-1)
        new_v = new_v * 100 + rx_pos
        vertices.append(new_v)
        faces.append(new_f)
        rx_center = (
            new_v.reshape(-1, 3).min(dim=0)[0] + new_v.reshape(-1, 3).max(dim=0)[0]
        ) / 2.0

    vertices = [v.reshape(-1, 3) for v in vertices]
    faces = [f.reshape(-1, 3) for f in faces]
    n_textures = len(vertices)
    textures = [
        torch.Tensor([i / n_textures, 0.0, 0.0]).expand_as(vertices[i])
        for i in range(n_textures)
    ]

    pts, normals, textures = sample_points_from_meshes(
        Meshes(verts=vertices, faces=faces, textures=Textures(verts_rgb=textures)),
        num_samples=n_samples_per_mesh,
        return_normals=True,
        return_textures=True,
    )  # [F, NumOfSamples, 3]
    # pts = pts.reshape(1, -1, 3)
    # depths = pts[..., 2:3]
    # depths = depths + depths.min()
    # grid = SplatFromParticlesToGrid(
    #     particles=pts[..., 0:2], attributes=depths, res_x=30, res_y=32, support_radius=2
    # )
    # is_obs = grid > 0.1
    # grid[...] = 0
    # grid[is_obs] = 1
    # grid = grid[0].cpu().numpy()  # [1, H, W]
    # h5f = h5py.File(os.path.join(demo_path, f"{scene_name}_pmap.h5"), "w")
    # h5f.create_dataset("map", data=grid)  # [TR, H, W, P]
    # h5f.close()
    # print(pts.shape, pts.min(), pts.max())
    # print(grid.shape, grid.min(), grid.max())
    # cv2.imwrite(
    #     "/home/gecao2/homework/ACEM/neural-point-EM-field/demo/pmap.png",
    #     cv2.flip(grid[0] * 255, 0),
    # )
    # exit(0)

    # Choice 2: Load mesh in a integration
    # vertices, faces = LoadMeshes(
    #     data_path=data_path, device=torch.device("cpu"), dtype=dtype
    # )
    # pts, normals = sample_points_from_meshes(
    #     Meshes(verts=vertices, faces=faces),
    #     num_samples=4000,
    #     return_normals=True,
    #     return_textures=False,
    # )  # [F, NumOfSamples, 3]

    light_probe = None
    if add_rx and add_tx and with_light_probes:
        n_row = 8
        AABB_min = pts.reshape(-1, 3).min(dim=0)[0]
        AABB_max = pts.reshape(-1, 3).max(dim=0)[0]
        AABB_len = (AABB_max - AABB_min).abs()  # [dim,]
        max_len, long_dim = AABB_len.max(dim=0)
        aspect = AABB_len / max_len  # [dim,] -> expect to be 0 < aspect <= 1
        light_probe_shape = (aspect * n_row).to(torch.int32).cpu().tolist()
        light_probe_shape = [max(2, len_) for len_ in light_probe_shape]
        D, H, W = light_probe_shape[2], light_probe_shape[1], light_probe_shape[0]
        light_probe_shape = [1, 1, D, H, W]
        print("D H W = ", D, H, W)
        light_probe = create_3d_meshgrid_tensor(
            light_probe_shape, device=device, dtype=dtype
        )  # [1, 3, D, H, W]
        light_probe = light_probe + 0.5
        light_probe = light_probe.reshape(3, -1).transpose(0, 1)  # [DHW, 3]
        # Scale to AABB scene
        max_res = max(D, max(H, W))
        light_probe = light_probe / max_res  # range: 0 -> 1
        light_probe = light_probe * max_len + AABB_min
        n_probes = light_probe.shape[-2]

        n_rays = 8
        rx_probe_r = light_probe - rx_center
        rx_probe_dist = rx_probe_r.norm(dim=-1)
        rx_probe_nearest_dist, rx_probe_nearest_indices = torch.topk(
            input=rx_probe_dist, k=n_rays, largest=False, dim=-1
        )  # [n_rays,]
        rx_probe_nearest_r = rx_probe_r[
            rx_probe_nearest_indices.cpu().flatten().tolist()
        ]  # [n_rays, 3]

        K = 8
        tx_probe_r = light_probe - tx_center
        tx_probe_dist = tx_probe_r.norm(dim=-1)
        tx_probe_nearest_dist, tx_probe_nearest_indices = torch.topk(
            input=tx_probe_dist, k=n_rays, largest=False, dim=-1
        )  # [K,]
        tx_probe_nearest_r = tx_probe_r[
            tx_probe_nearest_indices.cpu().flatten().tolist()
        ]  # [K, 3]

        n_probes = n_probes
        light_probe_pts_r = pts[:-1].reshape(-1, 1, 3) - light_probe.reshape(
            1, -1, 3
        )  # [n_pts, DHW, 3]
        light_probe_pts_dist = light_probe_pts_r.norm(dim=-1)
        tx_probe_nearest_dist, tx_probe_nearest_indices = light_probe_pts_dist.min(
            dim=0
        )  # [DHW,]
        index = (
            tx_probe_nearest_indices.cpu().flatten().tolist(),
            torch.linspace(0, n_probes - 1, n_probes).tolist(),
        )
        light_probe_nearest_pts_r = light_probe_pts_r[index]  # [DHW, 3]

        n_samples_per_line = n_samples_per_mesh // (K + n_rays)
        t = (
            torch.linspace(0, n_samples_per_line - 1, n_samples_per_line).reshape(
                n_samples_per_line, 1, 1
            )
            / n_samples_per_line
        )
        rx_lines = t * rx_probe_nearest_r + rx_center
        tx_lines = t * tx_probe_nearest_r + tx_center

        lines = torch.cat((rx_lines, tx_lines), dim=0).reshape(1, -1, 3)
        print("rx/tx lines = ", lines.shape)
        pts = torch.cat((pts, lines), dim=0)

        n_samples_per_line = int(math.ceil(n_samples_per_mesh / n_probes))
        t = (
            torch.linspace(0, n_samples_per_line - 1, n_samples_per_line).reshape(
                n_samples_per_line, 1, 1
            )
            / n_samples_per_line
        )
        pts_lines = t * light_probe_nearest_pts_r + light_probe
        pts_lines = pts_lines.reshape(1, -1, 3)
        pts_lines = pts_lines[:, 0:n_samples_per_mesh]

        print("pts lines = ", pts_lines.shape)
        pts = torch.cat((pts, pts_lines), dim=0)

    print("pts = ", pts.shape)
    DumpCFGFile(
        save_path=data_path,
        save_name=scene_name,
        with_floor=False,
        point_clouds=pts,
        light_probe=light_probe,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default="sionna_etoicenter",
        choices=[
            "sionna_munich",
            "sionna_munich_shadowing_fastfading",
            "sionna_etoile",
            "sionna_etoicenter",
            "sionna_wiindoor",
            "wiindoor",
            "wi3rooms_0",
        ],
        help="scene_name",
    )
    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
