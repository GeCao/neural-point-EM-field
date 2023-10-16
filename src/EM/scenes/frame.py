import os
import torch
import json
import pywavefront
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

from src.EM.scenes import AbstractScene


class Frame(object):
    def __init__(
        self,
        scene: AbstractScene,
        camera_index: int,
        frame_index: int,
        object2world_mat: torch.Tensor,
        data_path: str,
    ) -> None:
        self.scene = scene
        self.device = scene.device
        self.dtype = scene.dtype
        self.data_path = data_path
        self.num_pts_samples = 500

        self.edges = {}
        self.meshes = self.LoadMeshes(data_path=data_path)
        self.point_clouds = self.LoadPointCloudFromMesh()

        # Define paremts of frame: camera
        self.camera_index = camera_index
        self.frame_index = frame_index
        self.object2world_mat = object2world_mat

    def AddEdgesFromObject(
        self, obj_name: str, vertices: torch.Tensor, faces: torch.Tensor
    ):
        """
        Args:
            obj_name           (str): the name of object
            vertices           (torch.Tensor): [..., n_verts, dim=3]
            faces              (torch.Tensor): [..., n_faces, dim=3]

        Returns:
            None
        """
        if obj_name in self.edges.keys():
            self.scene.WarnLog(
                f"{obj_name}existed while trying to add it into frame{self.frame_index}, camera{self.camera_index}"
            )
            self.scene.WarnLog("We choose to replace this file in default")

        # Get Edegs from Vertices and Faces
        overlap_edges = torch.cat((faces, faces), dim=-1)
        edges = torch.unique(overlap_edges.reshape(-1, 2), dim=0).reshape(
            -1, 2
        )  # [:, 2]
        self.edges[obj_name] = edges

    def LoadPointCloudFromMesh(self):
        point_clouds, normals = sample_points_from_meshes(
            self.meshes, num_samples=self.num_pts_samples, return_normals=True
        )  # [1, NumOfSamples, 3]
        return point_clouds

    def GetPointCloud(self):
        return self.point_clouds

    def GetObjectToWorldTransformation(self):
        return self.object2world_mat

    def LoadMeshes(self, data_path):
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
            self.scene.ErrorLog(
                f"Mesh object path :{obj_path} not found, check your DataSet"
            )
        data_files = os.listdir(obj_path)
        device = self.device
        dtype = self.dtype

        vertices = torch.zeros((0, 3)).to(device).to(dtype)
        faces = torch.zeros((0, 3)).to(device).to(torch.int32)
        edges = torch.zeros((0, 2, 3)).to(device).to(dtype)
        verts_rgba = torch.zeros((1, 0, 4)).to(device).to(dtype)
        materials = {}
        visible = {}
        for filename in data_files:
            if filename[-5:] == ".json":
                json_path = os.path.join(obj_path, filename)
                with open(json_path, "r") as load_f:
                    json_data = json.load(load_f)
                    new_v = torch.Tensor(json_data["verts"]).to(device).to(dtype)
                    new_f = torch.Tensor(json_data["faces"]).to(device).to(torch.int32)
                    new_e = torch.Tensor(json_data["edges"]).to(device).to(dtype)
                    new_vcolor = (
                        torch.Tensor(json_data["verts_rgba"]).to(device).to(dtype)
                    )
                    vertices = torch.cat((vertices, new_v), dim=0)
                    faces = torch.cat((faces, new_f), dim=0)
                    edges = torch.cat((edges, new_e), dim=0)
                    verts_rgba = torch.cat((verts_rgba, new_vcolor), dim=1)

        tex = Textures(verts_rgb=verts_rgba)
        meshes = Meshes(
            verts=vertices.unsqueeze(0), faces=faces.unsqueeze(0), textures=tex
        )

        return meshes
