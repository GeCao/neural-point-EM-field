import sys, os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch
from typing import List

sys.path.append("../")

from src.EM.utils import DumpCFGFile


def SH_func(x_dir: np.ndarray, l: int, m: int, with_abs: bool = True) -> np.ndarray:
    result = None
    if l == 0:
        result = 0.5 * math.sqrt(0.5 / math.pi) * np.ones_like(x_dir[..., 0:1])
    elif l == 1:
        if m == -1:
            result = 0.5 * math.sqrt(1.5 / math.pi) * x_dir[..., 0:1]
        elif m == 0:
            result = 0.5 * math.sqrt(1.5 / math.pi) * x_dir[..., 2:3]
        elif m == 1:
            result = 0.5 * math.sqrt(1.5 / math.pi) * x_dir[..., 1:2]
    elif l == 2:
        if m == -2:
            result = (
                0.25
                * math.sqrt(7.5 / math.pi)
                * (
                    x_dir[..., 0:1] * x_dir[..., 0:1]
                    - x_dir[..., 1:2] * x_dir[..., 1:2]
                )
            )
        elif m == -1:
            result = (
                0.5 * math.sqrt(7.5 / math.pi) * (x_dir[..., 0:1] * x_dir[..., 2:3])
            )
        elif m == 0:
            result = (
                0.25
                * math.sqrt(2.5 / math.pi)
                * (3.0 * x_dir[..., 2:3] * x_dir[..., 2:3] - 1.0)
            )
        elif m == 1:
            result = (
                0.5 * math.sqrt(7.5 / math.pi) * (x_dir[..., 1:2] * x_dir[..., 2:3])
            )
        elif m == 2:
            result = (
                0.25
                * math.sqrt(7.5 / math.pi)
                * (x_dir[..., 1:2] * x_dir[..., 0:1] * 2)
            )
    else:
        raise NotImplementedError("Not implemented for high order SH coeffs")

    if with_abs:
        result = np.abs(result)
    return result


def SH_func_with_normals(x_dir: np.ndarray, l: int, m: int) -> np.ndarray:
    SH_primal = SH_func(x_dir, l=l, m=m)
    x_theta = np.cos(x_dir[..., 2:3])  # 0 -> PI
    x_cosphi = x_dir[..., 0:1] / np.sin(x_theta)
    x_sinphi = x_dir[..., 1:2] / np.sin(x_theta)
    x_phi = np.cos(x_cosphi)  # 0 -> PI
    x_phi[x_sinphi < 0] = (2 * math.pi - x_phi)[x_sinphi < 0]  # 0 -> 2PI
    eps = 0.001
    x1_dir = np.concatenate(
        (
            np.cos(x_phi - eps) * np.sin(x_theta),
            np.sin(x_phi - eps) * np.sin(x_theta),
            np.cos(x_theta),
        ),
        axis=-1,
    )
    x2_dir = np.concatenate(
        (
            np.cos(x_phi + eps) * np.sin(x_theta),
            np.sin(x_phi + eps) * np.sin(x_theta),
            np.cos(x_theta),
        ),
        axis=-1,
    )
    SH_left = SH_func(x1_dir, l=l, m=m)
    SH_right = SH_func(x2_dir, l=l, m=m)
    T_dir = SH_right * x2_dir - SH_left * x1_dir

    y1_dir = np.concatenate(
        (
            np.cos(x_phi) * np.sin(x_theta - eps),
            np.sin(x_phi) * np.sin(x_theta - eps),
            np.cos(x_theta - eps),
        ),
        axis=-1,
    )
    y2_dir = np.concatenate(
        (
            np.cos(x_phi) * np.sin(x_theta + eps),
            np.sin(x_phi) * np.sin(x_theta + eps),
            np.cos(x_theta + eps),
        ),
        axis=-1,
    )
    SH_down = SH_func(y1_dir, l=l, m=m)
    SH_up = SH_func(y2_dir, l=l, m=m)
    B_dir = SH_up * y2_dir - SH_down * y1_dir
    N_dir = np.cross(T_dir, B_dir, axis=-1)
    N_dir = N_dir / np.linalg.norm(N_dir)

    need_to_reflect = np.sum(N_dir * x_dir, axis=-1, keepdims=True) < 0
    need_to_reflect = np.tile(need_to_reflect, 3)  # [n_pts, 3]
    N_dir[need_to_reflect] = -N_dir[need_to_reflect]

    return [SH_primal, N_dir]


def squareToUniformCylinder(r1: float, r2: float) -> List[float]:
    fai = 2.0 * math.pi * r2
    return [math.cos(fai), math.sin(fai), 2 * r1 - 1]


def squareToUniformSphere(r1: float, r2: float) -> List[float]:
    Cylinder_res = squareToUniformCylinder(r1, r2)
    r = math.sqrt(1 - Cylinder_res[2] * Cylinder_res[2])
    return [r * Cylinder_res[0], r * Cylinder_res[1], Cylinder_res[2]]


def main():
    random.seed(40)
    n_rays = 25000
    ray_d = np.array(
        [squareToUniformSphere(random.random(), random.random()) for i in range(n_rays)]
    )

    demo_path = os.path.abspath(os.curdir)
    save_path = os.path.join(demo_path, f"SH.png")

    pts = None
    pts_color = None
    ls = [0, 1, 1, 1, 2, 2, 2, 2, 2]
    ms = [0, -1, 0, 1, -2, -1, 0, 1, 2]
    colors = [i for i in range(len(ls))]
    band_width = 1.0
    for i in range(len(ls)):
        l = ls[i]
        m = ms[i]
        color = colors[i]
        ray_t = SH_func(ray_d, l=l, m=m, with_abs=False)
        # get color vector
        ray_color = np.zeros((n_rays), dtype=np.int32)
        ray_color[(ray_t < 0).flatten()] = 1
        ray_t = np.abs(ray_t)
        ray = ray_d * ray_t
        # Translate
        ray[..., 1] += m * band_width
        ray[..., 2] -= l * band_width
        if pts is None:
            pts = ray
            pts_color = ray_color
        else:
            pts = np.concatenate((pts, ray), axis=0)
            pts_color = np.concatenate((pts_color, ray_color), axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pts[..., 0], pts[..., 1], pts[..., 2], marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.5, 0.5)
    # ax.set_zlim(-0.5, 0.5)
    plt.savefig(save_path)

    pts[..., 1] += 25
    DumpCFGFile(
        save_path=demo_path,
        save_name="SH",
        point_clouds=torch.from_numpy(pts).to(torch.float32),
        with_floor=False,
        colors=torch.from_numpy(pts_color).to(torch.float32),
    )

    # alpha = 0.01
    # pts = o3d.geometry.PointCloud()
    # pts.points = o3d.utility.Vector3dVector(pts)
    # pts.normals = o3d.utility.Vector3dVector(np.zeros((pts.shape[0], 3)))
    # pts.estimate_normals()
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #     pts, depth=5
    # )
    # filename = os.path.join(demo_path, "SH.obj")
    # o3d.io.write_triangle_mesh(filename, mesh)


if __name__ == "__main__":
    main()
