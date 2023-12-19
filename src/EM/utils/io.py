import os
import json
import numpy as np
import torch
import pywavefront
import matplotlib.pyplot as plt
import open3d as o3d
from plyfile import PlyData, PlyElement
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
            elif filename[-4:] == ".ply":
                json_path = os.path.join(obj_path, filename)
                ply_data = o3d.io.read_triangle_mesh(json_path)
                new_v = np.asarray(ply_data.vertices)
                new_f = np.asarray(ply_data.triangles)

                new_v = new_v.reshape(1, -1, 3)
                new_f = new_f.reshape(1, -1, 3)

                new_v = torch.from_numpy(new_v).to(dtype).to(device)
                new_f = torch.from_numpy(new_f).to(torch.int32).to(device)

                if len(vertices) == 0:
                    vertices = new_v
                    faces = new_f
                else:
                    vert_offset = vertices.shape[0]
                    vertices = torch.cat((vertices, new_v), dim=-2)
                    faces = torch.cat((faces, new_f + vert_offset), dim=-2)
        tex = None
    else:
        tex = Textures(verts_rgb=verts_rgba)

    vertices = torch.from_numpy(np.array(vertices))
    faces = torch.from_numpy(np.array(faces, dtype=np.int32))
    vertices_min = vertices.reshape(-1, 3).min(dim=0)[0].reshape(3)
    vertices_max = vertices.reshape(-1, 3).max(dim=0)[0].reshape(3)
    vertices = 2.0 * (vertices - vertices_min) / (vertices_max - vertices_min) - 1.0
    meshes = Meshes(verts=vertices, faces=faces, textures=tex)

    return meshes, [vertices_min, vertices_max]


def LoadPointCloudFromMesh(meshes: Meshes, num_pts_samples: int) -> torch.Tensor:
    point_clouds, normals = sample_points_from_meshes(
        meshes, num_samples=num_pts_samples, return_normals=True
    )  # [F, NumOfSamples, 3]

    # DumpCFGFile(
    #     save_path="/home/gecao2/homework/ACEM/neural-point-EM-field/demo/demo.cfg",
    #     point_clouds=point_clouds,
    # )
    return point_clouds


def DumpCFGFile(save_path: str, point_clouds: torch.Tensor):
    point_clouds = point_clouds.reshape(-1, 3)
    bounding_box = [
        point_clouds.min(dim=0)[0].reshape(3),
        point_clouds.max(dim=0)[0].reshape(3),
    ]
    point_clouds = point_clouds + bounding_box[0].unsqueeze(0)
    point_clouds = point_clouds * 100
    bounding_box = [
        point_clouds.min(dim=0)[0].reshape(3),
        point_clouds.max(dim=0)[0].reshape(3),
    ]
    with open(save_path, "w+") as fp:
        fp.write(f"Number of particles = {point_clouds.shape[0]}\n")
        fp.write("A = 1 Angstrom (basic length-scale)\n")
        fp.write(f"H0(1,1) = {bounding_box[1][0] - bounding_box[0][0]} A\n")
        fp.write(f"H0(1,2) = {0} A\n")
        fp.write(f"H0(1,3) = {0} A\n")
        fp.write(f"H0(2,1) = {0} A\n")
        fp.write(f"H0(2,2) = {bounding_box[1][1] - bounding_box[0][1]} A\n")
        fp.write(f"H0(2,3) = {0} A\n")
        fp.write(f"H0(3,1) = {0} A\n")
        fp.write(f"H0(3,2) = {0} A\n")
        fp.write(f"H0(3,3) = {bounding_box[1][2] - bounding_box[0][2]} A\n")
        fp.write(".NO_VELOCITY.\n")
        fp.write(f"entry_count = {3}\n")

        mass = 1.0
        for i in range(point_clouds.shape[0]):
            fp.write(
                f"{mass}\n{'H'}\n{point_clouds[i, 0].item()} {point_clouds[i, 1].item()} {point_clouds[i, 2].item()}\n"
            )

    fp.close()


def DumpGrayFigureToRGB(
    save_path: str,
    color: torch.Tensor,
    gt_color: torch.Tensor = None,
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

    blank_wid = 5
    if extra_spot is not None:
        extra_spot = extra_spot[..., 0:2].reshape(-1, 2)
        extra_spot[..., 0] = 0.5 * (extra_spot[..., 0] + 1.0) * W
        extra_spot[..., 1] = 0.5 * (extra_spot[..., 1] + 1.0) * H

    color = color.reshape(H, W)
    if gt_color is not None:
        gt_color = gt_color.reshape(H, W)

    if mask is not None:
        mask = mask.reshape(H, W).to(torch.bool)
        color[mask] = 0.0
        if gt_color is not None:
            gt_color[mask] = 0.0

    if gt_color is not None:
        blank = torch.zeros((H, blank_wid)).to(color.device).to(color.dtype)
        color = torch.cat((color, blank, gt_color), dim=-1)

    color[color == 0.0] = np.inf

    aspect = int((W * 2 + 5) / H)
    fig, ax = plt.subplots(1, 1, figsize=(aspect * 6, 6))
    plt.pcolormesh(color.cpu().numpy())
    plt.title("Prediction - Ground truth")
    if extra_spot is not None:
        x, y = (
            extra_spot[:, 0].cpu().numpy(),
            extra_spot[:, 1].cpu().numpy(),
        )
        plt.scatter(x, y, c="red", marker="x")

        x, y = (
            (extra_spot[:, 0] + W + blank_wid).cpu().numpy(),
            extra_spot[:, 1].cpu().numpy(),
        )
        plt.scatter(x, y, c="red", marker="x")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()
