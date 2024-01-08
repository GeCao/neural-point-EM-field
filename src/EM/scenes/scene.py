import os
import torch
from typing import Dict, Optional, List, Union
from operator import itemgetter

from src.EM.managers import AbstractManager
from src.EM.scenes import AbstractScene, Transmitter, Camera, RaySampler
from src.EM.utils import (
    TrainType,
    LoadMeshes,
    LoadPointCloudFromMesh,
    create_2d_meshgrid_tensor,
)


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
        self.opt = core_manager.opt
        self.log_manager = core_manager._log_manager

        self.K_closest = scene_opt["lightfield"].get("k_closest", 8)

        data_path = core_manager.GetDataPath()
        self.is_training = self.opt["is_training"]
        validation_target = self.opt["validation_target"]
        data_set = self.opt["data_set"]

        self.meshes, AABB_before_scale, self.AABB = LoadMeshes(
            data_path=data_path, device=torch.device("cpu"), dtype=dtype
        )
        self.InfoLog(f"Before scaling, scene AABB: {AABB_before_scale.tolist()}")
        self.InfoLog(f"After scaling, scene AABB: {self.AABB.tolist()}")
        self.AABB = self.AABB.to(device)
        AABB_before_scale = AABB_before_scale.to(device)
        self.point_clouds = LoadPointCloudFromMesh(
            meshes=self.meshes, num_pts_samples=4000
        )  # [F, n_pts, 3]
        self.point_clouds = self.point_clouds.to(device)
        self.log_manager.InfoLog(
            f"Point Clouds loading finished, with shape={self.point_clouds.shape}"
        )

        self.nodes = {"train": None, "test": None, "validation": None}
        # Transmitters:
        self.transmitters = {
            int(TrainType.TRAIN): [],
            int(TrainType.TEST): [],
            int(TrainType.VALIDATION): {},
        }
        # Frames
        self.receivers = {
            int(TrainType.TRAIN): [],
            int(TrainType.TEST): [],
            int(TrainType.VALIDATION): {},
        }  # traintype: List[Frame]

        data = core_manager.LoadData(
            self.is_training, validation_target, device=device, dtype=dtype
        )
        self.validation_target = data["target_list"]  # ["checkerboard"] or ["genz"] ...
        self.H = None
        self.W = None
        if "H" in data and "W" in data:
            self.H = data["H"]
            self.W = data["W"]
        # data structure:
        # "train"       -> [ch, rx, tx]
        # "validation"  -> "checkerboard" -> [ch, rx, tx]
        #                  "genz"         -> [ch, rx, tx]
        #                  "gendiag"         -> [ch, rx, tx]
        #                  ...
        # "target_list" -> ["checkerboard", "genz". "gendiag", ...]
        #
        # 1). Note target_list is controlled by opt["validation_target"]
        # 2). Note "train" key only exist when is_training.

        if self.is_training:
            data["train"] = [
                data["train"][0][0:1, ...],
                data["train"][1][0:1, ...],
                data["train"][2][0:1, ...],
            ]
            AABB_len, scale_dim = (AABB_before_scale[1] - AABB_before_scale[0]).max(
                dim=0
            )
            data["train"][1] = (
                2.0
                * (data["train"][1] - AABB_before_scale[0, scale_dim.item()])
                / AABB_len
                - 1.0
            )
            data["train"][2] = (
                2.0
                * (data["train"][2] - AABB_before_scale[0, scale_dim.item()])
                / AABB_len
                - 1.0
            )

            train_percent = 1.0
            tx_split = int(data["train"][0].shape[1] * train_percent)
            # TODO: rx_split

            self.nodes["train"] = [
                data["train"][0][:, 0:tx_split, ...],
                data["train"][1][:, 0:tx_split, ...],
                data["train"][2][:, 0:tx_split, ...],
            ]
            # [F, T, R, 1] for ch
            self.n_train_env = self.nodes["train"][0].shape[0]  # F
            self.n_train_transmitter = self.nodes["train"][0].shape[1]  # T
            self.n_train_receivers = self.nodes["train"][0].shape[2]  # R

            self.nodes["test"] = [
                data["train"][0][:, tx_split:, ...],
                data["train"][1][:, tx_split:, ...],
                data["train"][2][:, tx_split:, ...],
            ]
            # [F, T, R, 1] for ch
            self.n_test_env = self.nodes["test"][0].shape[0]  # F
            self.n_test_transmitter = self.nodes["test"][0].shape[1]  # T
            self.n_test_receivers = self.nodes["test"][0].shape[2]  # R

            self.log_manager.InfoLog(
                f"[Train]: env={self.n_train_env}, tx={self.n_train_transmitter}, rx={self.n_train_receivers}"
            )
            self.log_manager.InfoLog(
                f"[Test]: env={self.n_test_env}, tx={self.n_test_transmitter}, rx={self.n_test_receivers}"
            )
            print("Train tx: ", self.nodes["train"][2])

        self.n_validation_env = {}
        self.n_validation_transmitter = {}
        self.n_validation_receivers = {}
        for target_name in self.validation_target:
            data["validation"][target_name] = [
                data["validation"][target_name][0][0:1, ...],
                data["validation"][target_name][1][0:1, ...],
                data["validation"][target_name][2][0:1, ...],
            ]
            AABB_len, scale_dim = (AABB_before_scale[1] - AABB_before_scale[0]).max(
                dim=0
            )
            data["validation"][target_name][1] = (
                2.0
                * (
                    data["validation"][target_name][1]
                    - AABB_before_scale[0, scale_dim.item()]
                )
                / AABB_len
                - 1.0
            )
            data["validation"][target_name][2] = (
                2.0
                * (
                    data["validation"][target_name][2]
                    - AABB_before_scale[0, scale_dim.item()]
                )
                / AABB_len
                - 1.0
            )
            # [F, T, R, 1] for ch
            self.n_validation_env[target_name] = data["validation"][target_name][
                0
            ].shape[
                0
            ]  # F
            self.n_validation_transmitter[target_name] = data["validation"][
                target_name
            ][0].shape[
                1
            ]  # T
            self.n_validation_receivers[target_name] = data["validation"][target_name][
                0
            ].shape[
                2
            ]  # R

            self.log_manager.InfoLog(
                f"[Validation]: env={self.n_validation_env}, tx={self.n_validation_transmitter}, rx={self.n_validation_receivers}"
            )
            print("Validation tx: ", data["validation"][target_name][2])
        self.nodes["validation"] = data["validation"]

        self.ray_sampler = RaySampler(
            K_closest=self.K_closest,
            device=device,
            dtype=dtype,
        )

        # We need to transfer our data to point clouds firstly.
        self.Initialization()

    def RaySample(
        self,
        env_idx: int,
        tx_idx: int,
        rx_idx: int,
        validation_name: str = None,
        train_type: int = 0,
    ) -> List[torch.Tensor]:
        return self.ray_sampler(
            self,
            env_idx=env_idx,
            tx_idx=tx_idx,
            rx_idx=rx_idx,
            validation_name=validation_name,
            train_type=train_type,
        )

    def Initialization(self):
        # Frames
        train_types = (
            [
                int(TrainType.TRAIN),
                int(TrainType.TEST),
                int(TrainType.VALIDATION),
            ]
            if self.is_training
            else [int(TrainType.VALIDATION)]
        )
        env_idx = 0
        for train_type in train_types:
            _, rx_data, tx_data = self.GetData(
                train_type=train_type, validation_name=self.validation_target[0]
            )
            n_transmitter = self.GetNumTransmitters(train_type=train_type)
            n_receivers = self.GetNumReceivers(train_type=train_type)

            if train_type != int(TrainType.VALIDATION):
                self.transmitters[train_type] = [
                    Transmitter(
                        source_location=tx_data[env_idx, tx_idx],
                        device=self.device,
                        dtype=self.dtype,
                    )
                    for tx_idx in range(n_transmitter)
                ]

                self.receivers[train_type] = [
                    [
                        Camera(
                            eye=rx_data[env_idx, tx_idx, rx_idx],
                            device=self.device,
                            dtype=self.dtype,
                        )
                        for rx_idx in range(n_receivers)
                    ]
                    for tx_idx in range(n_transmitter)
                ]
            else:
                for validation_name in self.validation_target:
                    self.transmitters[train_type][validation_name] = [
                        Transmitter(
                            source_location=tx_data[env_idx, tx_idx],
                            device=self.device,
                            dtype=self.dtype,
                        )
                        for tx_idx in range(n_transmitter[validation_name])
                    ]

                    self.receivers[train_type][validation_name] = [
                        [
                            Camera(
                                eye=rx_data[env_idx, tx_idx, rx_idx],
                                device=self.device,
                                dtype=self.dtype,
                            )
                            for rx_idx in range(n_receivers[validation_name])
                        ]
                        for tx_idx in range(n_transmitter[validation_name])
                    ]

        # Light Probe, cover them on our map
        self.n_row = 8
        AABB = self.GetAABB()  # [2, dim] -> {min, max}
        AABB_len = (AABB[..., 1, :] - AABB[..., 0, :]).abs()  # [dim,]
        max_len, long_dim = AABB.max(dim=0)
        aspect = AABB_len / max_len  # [dim,] -> expect to be 0 < aspect <= 1
        light_probe_shape = (aspect * self.n_row).to(torch.int32).cpu().tolist()
        H, W = light_probe_shape[1], light_probe_shape[0]
        light_probe_shape = [1, 1, H, W]
        light_probe = create_2d_meshgrid_tensor(
            light_probe_shape, device=self.device, dtype=self.dtype
        )  # [1, 2, H, W]
        light_probe = light_probe + 0.5
        max_HW = max(H, W)
        light_probe = 2.0 * (light_probe / max_HW) - 1.0
        light_probe = light_probe.reshape(2, -1).transpose(0, 1)
        z_mean = self.GetPointCloud(env_index=env_idx)[..., 2:3].mean()
        light_probe = torch.cat(
            (light_probe, torch.ones_like(light_probe[:, 0:1]) * z_mean), dim=-1
        )  # [HW, 3]

        self.light_probe_pos = light_probe
        self.InfoLog(
            f"Light Probe position prepared, with shape = {self.light_probe_pos.shape}"
        )

        self.InfoLog("Neural Scene fully prepared.")

    def GetData(
        self, train_type: int, validation_name: str = None
    ) -> List[torch.Tensor]:
        if train_type == int(TrainType.TRAIN):
            return self.nodes["train"]
        elif train_type == int(TrainType.TEST):
            return self.nodes["test"]
        elif train_type == int(TrainType.VALIDATION):
            return self.nodes["validation"][validation_name]
        else:
            self.ErrorLog(
                f"We only support Train: {int(TrainType.TRAIN)}, "
                f"Test: {int(TrainType.TEST)}, and Validation: {int(TrainType.VALIDATION)}, "
                f"while your intput is {train_type}, please take care of your input of scene dataset"
            )

    def GetReceivers(self, train_type: int) -> List[List[Camera]]:
        return self.receivers[train_type]

    def GetReceiver(
        self,
        transmitter_idx: int,
        receiver_idx: int,
        train_type: int,
        validation_name: str = None,
    ) -> Camera:
        if train_type == int(TrainType.VALIDATION):
            return self.receivers[train_type][validation_name][transmitter_idx][
                receiver_idx
            ]
        else:
            return self.receivers[train_type][transmitter_idx][receiver_idx]

    def GetTransmitters(self, train_type: int) -> List[Transmitter]:
        return self.transmitters[train_type]

    def GetTransmitter(
        self, transmitter_idx: int, train_type: int, validation_name: str = None
    ) -> Transmitter:
        if train_type == int(TrainType.VALIDATION):
            return self.transmitters[train_type][validation_name][transmitter_idx]
        else:
            return self.transmitters[train_type][transmitter_idx]

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

    def GetAABB(self) -> torch.Tensor:
        return self.AABB

    def GetLightProbePosition(self) -> torch.Tensor:
        return self.light_probe_pos  # [n_probes, 3]

    def InfoLog(self, *args, **kwargs):
        return self.log_manager.InfoLog(*args, **kwargs)

    def WarnLog(self, *args, **kwargs):
        return self.log_manager.WarnLog(*args, **kwargs)

    def ErrorLog(self, *args, **kwargs):
        return self.log_manager.ErrorLog(*args, **kwargs)
