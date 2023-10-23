import os, sys
import numpy as np
import math
import torch
from typing import List
import cv2
from pytorch3d.structures import Meshes

sys.path.append("../")

from src.EM.managers import DataManager
from src.EM.utils import (
    TrainType,
    DrawHeatMapTransmitters,
    mkdir,
    DrawHeatMapReceivers,
    RenderRoom,
    LoadMeshes,
    DeleteFloorOrCeil,
    LoadPointCloudFromMesh,
)

data_set = "wi3rooms_0"
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
    train_data, checkerboard_data, genz_data, gendiag_data = data_manager.LoadData(
        is_training=True, test_target='all'
    )

    # [F, T, 1, R, K, I, 4] for intersections
    # [F, T, 1, R, D=8, K] for channels
    if train_type == int(TrainType.TRAIN):
        ch, floor_idx, interactions, rx, tx = train_data
    elif train_type == int(TrainType.TEST):
        ch, floor_idx, interactions, rx, tx = genz_data
    elif train_type == int(TrainType.VALIDATION):
        ch, floor_idx, interactions, rx, tx = gendiag_data

    ch = torch.from_numpy(ch).to(device).to(dtype)
    floor_idx = torch.from_numpy(floor_idx).to(device)
    interactions = torch.from_numpy(interactions).to(device).to(dtype)
    rx = torch.from_numpy(rx).to(device).to(dtype)
    tx = torch.from_numpy(tx).to(device).to(dtype)

    return ch, floor_idx, interactions, rx, tx


if __name__ == "__main__":
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_idx = 0

    ch, floor_idx, interactions, rx, tx = GetData(
        data_set=data_set, train_type=0, device=device, dtype=dtype
    )
    meshes = LoadMeshes(data_path=data_path, device=device, dtype=dtype)

    verts = meshes.verts_list()[env_idx]
    faces = meshes.faces_list()[env_idx]
    verts, faces = DeleteFloorOrCeil(verts=verts, faces=faces, axis=2, mode="both")
    meshes = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0), textures=None)
    point_clouds = LoadPointCloudFromMesh(
        meshes=meshes, num_pts_samples=1000
    )  # [F, n_pts, 3]
    rendered_room = RenderRoom(point_clouds).cpu().numpy()

    color_r = (
        DrawHeatMapReceivers(
            rx=rx, ch=ch, res_x=rendered_room.shape[1], res_y=rendered_room.shape[0]
        )
        .cpu()
        .numpy()
    )
    # color_t = DrawHeatMapTransmitters(tx=tx, ch=ch)

    print(rendered_room.shape, color_r.shape)
    color_r[..., 0:1][rendered_room > 10] = 0.0
    color_r[..., 1:2][rendered_room > 10] = 0.0
    color_r[..., 2:3][rendered_room > 10] = 0.0

    save_dir = os.path.join(demo_path, "Visualizations")
    mkdir(save_dir)

    save_path = os.path.join(save_dir, "heat_receivers.png")
    cv2.imwrite(save_path, color_r)

    # save_path = os.path.join(save_dir, "heat_transmitters.png")
    # cv2.imwrite(save_path, color_t)

    save_path = os.path.join(save_dir, "room.png")
    cv2.imwrite(save_path, rendered_room)
