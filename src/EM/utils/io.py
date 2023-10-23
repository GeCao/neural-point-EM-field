import os
import json
import torch
import pywavefront
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.ops import sample_points_from_meshes


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
