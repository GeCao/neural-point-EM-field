import torch

from src.EM.scenes import AbstractScene


class Frame(object):
    def __init__(
        self,
        scene: AbstractScene,
        transmitter_index: int,
        env_index: int,
        object2world_mat: torch.Tensor = None,
    ) -> None:
        self.scene = scene
        self.device = scene.device
        self.dtype = scene.dtype

        self.edges = {}

        # Define paremts of frame: transmitter_index
        self.transmitter_index = transmitter_index
        self.env_index = env_index
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
                f"{obj_name}existed while trying to add it into env{self.env_index}, transmitter{self.transmitter_index}"
            )
            self.scene.WarnLog("We choose to replace this file in default")

        # Get Edegs from Vertices and Faces
        overlap_edges = torch.cat((faces, faces), dim=-1)
        edges = torch.unique(overlap_edges.reshape(-1, 2), dim=0).reshape(
            -1, 2
        )  # [:, 2]
        self.edges[obj_name] = edges

    def GetPointCloud(self):
        return self.scene.GetPointCloud(env_index=self.env_index)

    def GetObjectToWorldTransformation(self):
        return self.object2world_mat
