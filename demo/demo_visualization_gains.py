import os, sys
import numpy as np
import math
import torch
import torch.nn.functional as F
from typing import List
import cv2
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes

sys.path.append("../")

from src.EM.managers import DataManager
from src.EM.utils import (
    TrainType,
    mkdir,
    DrawHeatMapReceivers,
    FromGridToColor,
    ImageBlur,
    RenderRoom,
    LoadMeshes,
    DeleteFloorOrCeil,
    LoadPointCloudFromMesh,
    SplatFromParticlesToGrid,
    DumpGrayFigureToRGB,
)

data_set = "wiindoor"
demo_path = os.path.abspath(os.curdir)
root_path = os.path.abspath(os.path.join(demo_path, ".."))
data_path = os.path.join(root_path, "data", data_set)


def GetData(
    data_set: str = "wiindoor",
    train_type: int = 0,
    device: torch.device = torch.device("cpu"),
    dtype=torch.float32,
) -> List[torch.Tensor]:
    data_path = os.path.join(root_path, "data", data_set)
    data_manager = DataManager(data_path=data_path)
    data = data_manager.LoadData(is_training=True, device=device, dtype=dtype)

    # [F, T, 1, R, K, I, 4] for intersections
    # [F, T, 1, R, D=8, K] for channels
    if train_type == int(TrainType.TRAIN):
        ch, floor_idx, rx, tx = data["train"]
    elif train_type == int(TrainType.TEST):
        ch, floor_idx, rx, tx = data["genz"]
    elif train_type == int(TrainType.VALIDATION):
        ch, floor_idx, rx, tx = data["gendiag"]

    return ch, floor_idx, rx, tx


if __name__ == "__main__":
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_idx = 0
    tx_idx = 0

    ch, floor_idx, rx, tx = GetData(
        data_set=data_set, train_type=int(TrainType.TEST), device=device, dtype=dtype
    )
    meshes = LoadMeshes(data_path=data_path, device=device, dtype=dtype)

    verts = meshes.verts_list()[env_idx]
    faces = meshes.faces_list()[env_idx]
    verts, faces = DeleteFloorOrCeil(verts=verts, faces=faces, axis=2, mode="both")
    meshes = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0), textures=None)
    point_clouds = LoadPointCloudFromMesh(
        meshes=meshes, num_pts_samples=1000
    )  # [F, n_pts, 3]
    res_x = 256
    rendered_room = RenderRoom(point_clouds[env_idx], res_x=res_x)

    gt_color, tx_proj = DrawHeatMapReceivers(
        rx=rx[env_idx, tx_idx, 0, :, :],
        tx=tx[env_idx, tx_idx, :],
        gain=ch[env_idx, tx_idx, 0, :, 0:1, :].sum(dim=-1),
        res_x=rendered_room.shape[1],
        res_y=rendered_room.shape[0],
    )
    gt_color = SplatFromParticlesToGrid(
        particles=rx[env_idx, tx_idx, 0, :, 0:2],
        attributes=ch[env_idx, tx_idx, 0, :, 0:1, :].sum(dim=-1),
        res_x=rendered_room.shape[1],
        res_y=rendered_room.shape[0],
    )
    gt_color = gt_color.abs()
    rendered_room = SplatFromParticlesToGrid(
        particles=point_clouds[env_idx, :, 0:2],
        attributes=torch.ones_like(point_clouds[env_idx, :, 0:1]),
        res_x=rendered_room.shape[1],
        res_y=rendered_room.shape[0],
    )
    rendered_room = rendered_room.reshape(1, *rendered_room.shape[-2:])
    rendered_room = rendered_room / rendered_room.max()

    grid_min = gt_color.min()
    grid_max = gt_color.max()
    gt_color = (gt_color - grid_min) / (grid_max - grid_min)

    save_dir = os.path.join(demo_path, "Visualizations")
    mkdir(save_dir)
    save_path = os.path.join(save_dir, "heat_receivers.png")
    DumpGrayFigureToRGB(
        save_path=save_path,
        color=gt_color,
        mask=rendered_room > 0.05,
        extra_spot=tx_proj,
    )
