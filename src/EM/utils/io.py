import os
import json
import numpy as np
import torch
import pywavefront
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.ops import sample_points_from_meshes
from pyevtk.hl import pointsToVTK


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def LoadMeshes(
    data_path: str, device: torch.device = torch.device("cpu"), dtype=torch.float32
) -> Meshes:
    """Load mesh data from disk
    We will typically load mesh from json file where,
        'verts'--------[NumOfVertices, 3]
        'faces'--------[NumOfFaces, 3]
        'edges'--------[NumOfEdges, 2, 3]
        'verts_rgba'---[1, NumofVertices, 4]
        'materials'----{NumOfFaces->str}
        'visible'------{NumOfFaces->bool}
    are given.
    """
    obj_path = os.path.join(data_path, "objs")
    if not os.path.exists(obj_path):
        raise RuntimeError(
            f"Mesh object path :{obj_path} not found, check your DataSet"
        )
    data_files = os.listdir(obj_path)
    data_files = sorted(data_files)

    meshes = []
    vertices = []
    faces = []
    edges = []
    verts_rgba = []
    materials = {}
    visible = {}
    json_found = False
    for filename in data_files:
        if filename[-5:] == ".json":
            json_found = True
            json_path = os.path.join(obj_path, filename)
            with open(json_path, "r") as load_f:
                json_data = json.load(load_f)
                new_v = torch.Tensor(json_data["verts"]).to(device).to(dtype)
                new_f = torch.Tensor(json_data["faces"]).to(device).to(torch.int32)
                new_e = torch.Tensor(json_data["edges"]).to(device).to(dtype)
                new_vcolor = torch.Tensor(json_data["verts_rgba"]).to(device).to(dtype)
                vertices.append(new_v)
                faces.append(new_f)
                edges.append(new_e)
                verts_rgba.append(new_vcolor.reshape(-1, 4))

            load_f.close()
            break

    if not json_found:
        for filename in data_files:
            if filename[-4:] == ".obj":
                json_path = os.path.join(obj_path, filename)
                scene = pywavefront.Wavefront(json_path, collect_faces=True)

                new_v = torch.Tensor(scene.vertices).to(device).to(dtype)
                new_f = (
                    torch.Tensor(scene.meshes[None].faces).to(device).to(torch.int32)
                )
                vertices.append(new_v)
                faces.append(new_f)
        tex = None
    else:
        tex = Textures(verts_rgb=verts_rgba)
    meshes = Meshes(verts=vertices, faces=faces, textures=tex)

    return meshes


def LoadPointCloudFromMesh(meshes: Meshes, num_pts_samples: int) -> torch.Tensor:
    point_clouds, normals = sample_points_from_meshes(
        meshes, num_samples=num_pts_samples, return_normals=True
    )  # [F, NumOfSamples, 3]
    return point_clouds


def ExportVTKFile(
    save_path: str, rx_pos: torch.Tensor, gain: torch.Tensor, point_clouds: torch.Tensor
):
    x = rx_pos[..., 0].flatten().cpu().numpy()
    y = rx_pos[..., 1].flatten().cpu().numpy()
    z = rx_pos[..., 2].flatten().cpu().numpy()
    gain = gain.cpu().flatten().numpy()

    x_obs = point_clouds[..., 0].flatten().cpu().numpy()
    y_obs = point_clouds[..., 1].flatten().cpu().numpy()
    z_obs = 0 * point_clouds[..., 2].flatten().cpu().numpy() + z.mean()
    obs_val = 0 * x_obs + gain.mean()

    x = np.concatenate((x, x_obs), axis=0)
    y = np.concatenate((y, y_obs), axis=0)
    z = np.concatenate((z, z_obs), axis=0)
    gain = np.concatenate((gain, obs_val), axis=0)

    pointsToVTK(save_path, x, y, z, data={"gain": gain})


def DumpGrayFigureToRGB(
    save_path: str,
    color: torch.Tensor,
    mask: torch.Tensor = None,
    extra_spot: torch.Tensor = None,
) -> None:
    if len(color.shape) == 4:
        _, _, H, W = color.shape
    elif len(color.shape) == 3 and color.shape[-1] == 1:
        H, W, _ = color.shape
    elif len(color.shape) == 3 and color.shape[0] == 1:
        _, H, W = color.shape
    elif len(color.shape) == 2:
        H, W = color.shape
    else:
        raise RuntimeError(
            f"Can not recognize input gray color with shape {color.shape}"
        )

    if extra_spot is not None:
        assert extra_spot.shape[-1] == 2 or extra_spot.shape[-1] == 3
        extra_spot = extra_spot[..., 0:2].reshape(-1, 2)

    color = color.reshape(H, W)
    mask = mask.reshape(H, W).to(torch.bool)
    color[mask] = 0.0

    aspect = int(W / H)
    fig, ax = plt.subplots(1, 1, figsize=(aspect * 6, 6))
    plt.pcolormesh(color.cpu().numpy())
    if extra_spot is not None:
        x, y = (
            extra_spot[:, 0].cpu().numpy(),
            extra_spot[:, 1].cpu().numpy(),
        )
        plt.scatter(x, y, c="red")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()
