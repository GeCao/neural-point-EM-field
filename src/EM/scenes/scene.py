import os
import torch
from typing import Dict, Optional, List, Union
from operator import itemgetter

from src.EM.managers import AbstractManager
from src.EM.scenes import AbstractScene, Frame, Camera, RaySampler
from src.EM.utils import TrainType, LoadMeshes, LoadPointCloudFromMesh


class NeuralScene(AbstractScene):
    def __init__(
        self,
        core_manager: AbstractManager,
        scene_opt: Dict = None,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
    ):
        """
        Data structure to store dynamic multi-object 3D scenes.

        Args:
            core_manager:
            scene_opt:
            device:
            dtype:
        """
        self.device = device
        self.dtype = dtype
        self.scene_opt = scene_opt
        self.log_manager = core_manager._log_manager

        self.K_closest = scene_opt["lightfield"].get("k_closest", 8)

        self.meshes = LoadMeshes(
            data_path=core_manager.GetDataPath(), device=device, dtype=dtype
        )
        self.point_clouds = LoadPointCloudFromMesh(
            meshes=self.meshes, num_pts_samples=1000
        )  # [F, n_pts, 3]
        core_manager.InfoLog(
            f"Point Clouds loading finished, with shape={self.point_clouds.shape}"
        )

        self.nodes = {}
        # Transmitters:
        self.transmitters = {
            int(TrainType.TRAIN): [],
            int(TrainType.TEST): [],
            int(TrainType.VALIDATION): [],
        }
        # Frames
        self.frames = {
            int(TrainType.TRAIN): [],
            int(TrainType.TEST): [],
            int(TrainType.VALIDATION): [],
        }  # traintype: List[Frame]

        data = core_manager.LoadData(core_manager.opt["is_training"])
        train_data = [
            torch.from_numpy(data["train"][i]).to(device).to(dtype)
            for i in range(len(data["train"]))
        ]
        checkerboard_data = [
            torch.from_numpy(data["checkerboard"][i]).to(device).to(dtype)
            for i in range(len(data["checkerboard"]))
        ]
        genz_data = [
            torch.from_numpy(data["genz"][i]).to(device).to(dtype)
            for i in range(len(data["genz"]))
        ]
        gendiag_data = [
            torch.from_numpy(data["gendiag"][i]).to(device).to(dtype)
            for i in range(len(data["gendiag"]))
        ]
        train_data = [
            train_data[0][:, 0:1, ...],
            train_data[1],
            train_data[2][:, 0:1, ...],
            train_data[3][:, 0:1, ...],
            train_data[4][:, 0:1, ...],
        ]
        checkerboard_data = [
            checkerboard_data[0][:, 0:1, ...],
            checkerboard_data[1],
            checkerboard_data[2][:, 0:1, ...],
            checkerboard_data[3][:, 0:1, ...],
            checkerboard_data[4][:, 0:1, ...],
        ]
        genz_data = [
            genz_data[0][:, 0:1, ...],
            genz_data[1],
            genz_data[2][:, 0:1, ...],
            genz_data[3][:, 0:1, ...],
            genz_data[4][:, 0:1, ...],
        ]
        gendiag_data = [
            gendiag_data[0][:, 0:1, ...],
            gendiag_data[1],
            gendiag_data[2][:, 0:1, ...],
            gendiag_data[3][:, 0:1, ...],
            gendiag_data[4][:, 0:1, ...],
        ]
        data["train"] = train_data
        data["checkerboard"] = checkerboard_data
        data["genz"] = genz_data
        data["gendiag"] = gendiag_data

        self.nodes["train"] = train_data
        self.nodes["test"] = data[core_manager.opt["test_target"]]
        self.nodes["validation"] = data[core_manager.opt["validation_target"]]

        # [F, T, 1, R, K, I, 4] for interactions
        _, _, train_interactions, _, _ = train_data
        _, _, test_interactions, _, _ = self.nodes["test"]
        _, _, validation_interactions, _, _ = self.nodes["validation"]

        self.n_train_env = train_interactions.shape[0]  # F
        self.n_train_transmitter = train_interactions.shape[1]  # T
        self.n_train_receivers = train_interactions.shape[3]  # R
        self.n_train_rays = train_interactions.shape[4]  # K
        self.n_train_interactions = train_interactions.shape[5]  # I

        self.n_test_env = test_interactions.shape[0]  # F
        self.n_test_transmitter = test_interactions.shape[1]  # T
        self.n_test_receivers = test_interactions.shape[3]  # R
        self.n_test_rays = test_interactions.shape[4]  # K
        self.n_test_interactions = test_interactions.shape[5]  # I

        self.n_validation_env = validation_interactions.shape[0]  # F
        self.n_validation_transmitter = validation_interactions.shape[1]  # T
        self.n_validation_receivers = validation_interactions.shape[3]  # R
        self.n_validation_rays = validation_interactions.shape[4]  # K
        self.n_validation_interactions = validation_interactions.shape[5]  # I

        self.ray_sampler = RaySampler(
            K_closest=self.K_closest,
            device=device,
            dtype=dtype,
        )

        # We need to transfer our data to point clouds firstly.
        self.Initialization()

    def RaySample(
        self, idx: int, train_type: int = 0, env_idx: int = None, tx_idx: int = None
    ) -> List[torch.Tensor]:
        return self.ray_sampler(idx, self, train_type, env_idx=env_idx, tx_idx=tx_idx)

    def Initialization(self):
        # Frames
        train_types = [
            int(TrainType.TRAIN),
            int(TrainType.TEST),
            int(TrainType.VALIDATION),
        ]
        for train_type in train_types:
            _, _, _, _, tx_data = self.GetData(train_type=train_type)
            n_transmitter = self.GetNumTransmitters(train_type=train_type)
            n_receivers = self.GetNumReceivers(train_type=train_type)

            self.frames[train_type] = [
                Frame(
                    self,
                    transmitter_index=tx_idx,
                    env_index=0,
                    object2world_mat=None,
                )
                for tx_idx in range(n_transmitter)
            ]

            self.transmitters[train_type] = [
                [
                    Camera(
                        eye=tx_data[0, tx_idx, :], device=self.device, dtype=self.dtype
                    )
                    for rx_idx in range(n_receivers)
                ]
                for tx_idx in range(n_transmitter)
            ]

        self.InfoLog("Neural Scene fully prepared.")

    def GetData(self, train_type: int) -> List[torch.Tensor]:
        if train_type == int(TrainType.TRAIN):
            return self.nodes["train"]
        elif train_type == int(TrainType.TEST):
            return self.nodes["test"]
        elif train_type == int(TrainType.VALIDATION):
            return self.nodes["validation"]
        else:
            self.ErrorLog(
                f"We only support Train: {int(TrainType.TRAIN)}, "
                f"Test: {int(TrainType.TEST)}, and Validation: {int(TrainType.VALIDATION)}, "
                f"while your intput is {train_type}, please take care of your input of scene dataset"
            )

    def GetFrames(self) -> List[Frame]:
        return self.frames

    def GetTransmitters(self) -> List[List[Camera]]:
        return self.transmitters

    def GetTransmitter(
        self, transmitter_idx: int, receiver_idx: int, train_type: int
    ) -> Camera:
        return self.transmitters[train_type][transmitter_idx][receiver_idx]

    def GetNumRays(self, train_type: int) -> int:
        if train_type == int(TrainType.TRAIN):
            return self.n_train_rays
        elif train_type == int(TrainType.TEST):
            return self.n_test_rays
        elif train_type == int(TrainType.VALIDATION):
            return self.n_validation_rays
        else:
            self.ErrorLog(
                "We only accept train type input as Train(0), Test(1) and Validation(2), "
                f"while your input is {train_type}"
            )

    def GetNumTransmitters(self, train_type: int) -> int:
        if train_type == int(TrainType.TRAIN):
            return self.n_train_transmitter
        elif train_type == int(TrainType.TEST):
            return self.n_test_transmitter
        elif train_type == int(TrainType.VALIDATION):
            return self.n_validation_transmitter
        else:
            self.ErrorLog(
                "We only accept train type input as Train(0), Test(1) and Validation(2), "
                f"while your input is {train_type}"
            )

    def GetNumReceivers(self, train_type: int) -> int:
        if train_type == int(TrainType.TRAIN):
            return self.n_train_receivers
        elif train_type == int(TrainType.TEST):
            return self.n_test_receivers
        elif train_type == int(TrainType.VALIDATION):
            return self.n_validation_receivers
        else:
            self.ErrorLog(
                "We only accept train type input as Train(0), Test(1) and Validation(2), "
                f"while your input is {train_type}"
            )

    def GetNumEnvs(self, train_type: int) -> int:
        if train_type == int(TrainType.TRAIN):
            return self.n_train_env
        elif train_type == int(TrainType.TEST):
            return self.n_test_env
        elif train_type == int(TrainType.VALIDATION):
            return self.n_validation_env
        else:
            self.ErrorLog(
                "We only accept train type input as Train(0), Test(1) and Validation(2), "
                f"while your input is {train_type}"
            )

    def GetPointCloud(self, env_index: int) -> torch.Tensor:
        return self.point_clouds[env_index]

    def InfoLog(self, *args, **kwargs):
        return self.log_manager.InfoLog(*args, **kwargs)

    def WarnLog(self, *args, **kwargs):
        return self.log_manager.WarnLog(*args, **kwargs)

    def ErrorLog(self, *args, **kwargs):
        return self.log_manager.ErrorLog(*args, **kwargs)
