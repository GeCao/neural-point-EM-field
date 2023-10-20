import os, sys
import numpy as np
import math
from typing import List

sys.path.append("../")

from src.EM.managers import DataManager
from src.EM.utils import TrainType, export_asset, mkdir

demo_path = os.path.abspath(os.curdir)
root_path = os.path.abspath(os.path.join(demo_path, ".."))


def GetChannelsAndIntersections(
    env_idx: int = 0,
    tx_idx: int = 0,
    rx_idx: int = 0,
    data_set: str = "wiindoor",
    train_type: int = 0,
) -> List[np.ndarray]:
    data_path = os.path.join(root_path, "data", data_set)
    data_manager = DataManager(data_path=data_path)
    train_data, checkerboard_data, genz_data, gendiag_data = data_manager.LoadData(
        is_training=True, test_target='all'
    )

    # [F, T, 1, R, K, I, 4] for intersections
    # [F, T, 1, R, D=8, K] for channels
    if train_type == int(TrainType.TRAIN):
        ch, floor_idx, interactions, rx, tx = train_data
    elif train_type == int(TrainType.TEST):
        ch, floor_idx, interactions, rx, tx = checkerboard_data
    elif train_type == int(TrainType.VALIDATION):
        ch, floor_idx, interactions, rx, tx = gendiag_data

    n_nev, n_tx, _, n_rx, _, n_rays, _ = interactions.shape
    result_inter = interactions[
        env_idx, tx_idx, 0, rx_idx, :, :, :
    ]  # [n_rays, intersects, 4]
    result_ch = np.transpose(
        ch[env_idx, tx_idx, 0, rx_idx, :, :],
    )  # [n_rays, D=8]
    return result_ch, result_inter


def ConstructCylinderWithTwoPoints(x, y):
    eps = 1e-5
    x = np.array(x).reshape(3)
    y = np.array(y).reshape(3)
    dist = np.linalg.norm(y - x)
    normal = (y - x) * np.reciprocal(dist)
    radius = 0.05

    tangent = np.array([1.0, 0.0, 0.0])
    if tangent.dot(normal) < eps:
        tangent = np.array([0.0, 1.0, 0.0])

    bitangent = np.cross(normal, tangent)
    bitangent = bitangent * np.reciprocal(np.linalg.norm(bitangent))
    tangent = np.cross(bitangent, normal)
    tangent = tangent * np.reciprocal(np.linalg.norm(tangent))

    n_samples = 10
    verts = [[0.0, 0.0, 0.0] for i in range(2 * n_samples + 2)]
    faces = []
    r1 = tangent * radius
    r2 = bitangent * radius
    verts[0] = x
    verts[n_samples + 1] = y
    for i in range(n_samples):
        angle = (float(i) / n_samples) * (2.0 * math.pi)
        cosphi = math.cos(angle)
        sinphi = math.sin(angle)
        verts[i + 1] = x + r1 * cosphi + r2 * sinphi
        verts[i + n_samples + 2] = y + r1 * cosphi + r2 * sinphi

    for i in range(n_samples):
        faces.append([0, i + 1, ((i + 1) % n_samples) + 1])

    for i in range(n_samples):
        offset = n_samples + 1
        faces.append([offset, offset + i + 1, offset + ((i + 1) % n_samples) + 1])

    for i in range(n_samples):
        offset = n_samples + 1
        x_idx = [i + 1, ((i + 1) % n_samples) + 1]
        y_idx = [offset + i + 1, offset + ((i + 1) % n_samples) + 1]
        faces.append([x_idx[0], y_idx[0], y_idx[1]])
        faces.append([y_idx[1], x_idx[1], x_idx[0]])

    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    return verts, faces


if __name__ == "__main__":
    ch, interactions = GetChannelsAndIntersections()
    n_rays, intersects, _ = interactions.shape
    color = ch[:, 0]
    color = (color - np.min(color)) / (np.max(color) - np.min(color))
    color_b = (color < 0.5) * (0.5 - color) * 2.0
    color_r = (color >= 0.5) * (color - 0.5) * 2.0
    vertices = np.zeros((0, 6), dtype=np.float32)
    faces = np.zeros((0, 3), dtype=np.int32)
    for ray_idx in range(n_rays):
        vert_offset = len(vertices)
        for i in range(1, intersects):
            inter_type = int(interactions[ray_idx, i, 0])
            if inter_type >= 0 and inter_type < 6:
                x0 = interactions[ray_idx, i - 1, 1:4]
                x1 = interactions[ray_idx, i, 1:4]

                verts, face = ConstructCylinderWithTwoPoints(x=x0, y=x1)
                gain_color = np.array(
                    [[color_r[ray_idx], 0.0, color_b[ray_idx]]]
                ) * np.ones_like(verts)
                verts = np.concatenate((verts, gain_color), axis=-1)

                vert_offset = vertices.shape[0]
                vertices = np.concatenate((vertices, verts), axis=0)
                faces = np.concatenate((faces, face + vert_offset), axis=0)

    save_dir = os.path.join(demo_path, "Visualizations")
    mkdir(save_dir)
    save_path = os.path.join(save_dir, "save_rays.obj")
    export_asset(save_path=save_path, vertices=vertices, faces=faces)
