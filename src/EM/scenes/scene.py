import os
import torch
from typing import Dict, Optional, List, Union
from operator import itemgetter
from pytorch3d.structures import Meshes

from src.EM.managers import AbstractManager
from src.EM.scenes import AbstractScene, Transmitter, Camera, RaySampler
from src.EM.utils import (
    TrainType,
    LoadMeshes,
    LoadPointCloudFromMesh,
    create_2d_meshgrid_tensor,
    create_3d_meshgrid_tensor,
    DumpCFGFile,
    ModuleType,
    ScaleAABB,
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
        self.module_type = self.opt["module_type"]

        data_path = core_manager.GetDataPath()
        self.is_training = self.opt["is_training"]
        validation_target = self.opt["validation_target"]
        data_set = self.opt["data_set"]
        has_floor = os.path.exists(os.path.join(data_path, "floors"))

        verts, faces = LoadMeshes(
            data_path=data_path, device=torch.device("cpu"), dtype=dtype
        )
        num_pts_floor = 250 if has_floor else 0
        self.point_clouds = LoadPointCloudFromMesh(
            meshes=Meshes(verts=verts, faces=faces),
            num_pts_samples=1000 - num_pts_floor,
        )  # [F, n_pts, 3]
        if has_floor:
            verts_floor, faces_floor = LoadMeshes(
                data_path=data_path,
                obj_folder="floors",
                device=torch.device("cpu"),
                dtype=dtype,
            )
            self.point_clouds = torch.cat(
                (
                    self.point_clouds,
                    LoadPointCloudFromMesh(
                        meshes=Meshes(verts=verts_floor, faces=faces_floor),
                        num_pts_samples=num_pts_floor,
                    ),
                ),
                dim=-2,
            )
            vert_offset = verts.shape[-2]
            verts = torch.cat((verts, verts_floor), dim=-2)
            faces = torch.cat((faces, faces_floor + vert_offset), dim=-2)

        # Normalization: Compute previous AABB
        verts_min = verts.reshape(-1, 3).min(dim=0)[0].reshape(3)
        verts_max = verts.reshape(-1, 3).max(dim=0)[0].reshape(3)
        prev_AABB = torch.stack((verts_min, verts_max), dim=0)  # [2, dim]

        # Normalization: Scale your geometry to [-1, 1]
        verts = ScaleAABB(verts, AABB=prev_AABB, take_neg=True)
        self.point_clouds = ScaleAABB(self.point_clouds, AABB=prev_AABB, take_neg=True)
        self.meshes = Meshes(verts=verts, faces=faces, textures=None)

        # Normalization: Compute current AABB
        verts_min = verts.reshape(-1, 3).min(dim=0)[0].reshape(1, 3)
        verts_max = verts.reshape(-1, 3).max(dim=0)[0].reshape(1, 3)
        self.AABB = torch.cat((verts_min, verts_max), dim=0)  # [2, dim]

        # DumpCFGFile(
        #     save_path="/home/moritz/homework/neural-point-EM-field/demo/pointcloud.cfg",
        #     point_clouds=self.point_clouds,
        # )
        self.point_clouds = self.point_clouds.to(device)

        self.InfoLog(f"Before scaling, scene AABB: {prev_AABB.tolist()}")
        self.InfoLog(f"After scaling, scene AABB: {self.AABB.tolist()}")
        self.AABB = self.AABB.to(device)
        prev_AABB = prev_AABB.to(device)
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
                data["train"][i][0:1, ...] for i in range(len(data["train"]))
            ]
            data["train"][1] = ScaleAABB(
                data["train"][1], AABB=prev_AABB, take_neg=True
            )
            data["train"][2] = ScaleAABB(
                data["train"][2], AABB=prev_AABB, take_neg=True
            )

            train_percent = 1.0
            tx_split = int(data["train"][0].shape[1] * train_percent)
            # TODO: rx_split

            self.nodes["train"] = [
                data["train"][i][:, 0:tx_split, ...] for i in range(len(data["train"]))
            ]
            # [F, T, R, 1] for ch
            self.n_train_env = self.nodes["train"][0].shape[0]  # F
            self.n_train_transmitter = self.nodes["train"][0].shape[1]  # T
            self.n_train_receivers = self.nodes["train"][0].shape[2]  # R

            self.nodes["test"] = [
                data["train"][i][:, tx_split:, ...] for i in range(len(data["train"]))
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

        self.n_validation_env = {}
        self.n_validation_transmitter = {}
        self.n_validation_receivers = {}
        for target_name in self.validation_target:
            self.n_ch = data["validation"][target_name][0].shape[-1]
            data["validation"][target_name] = [
                data["validation"][target_name][i][0:1, ...]
                for i in range(len(data["validation"][target_name]))
            ]
            data["validation"][target_name][1] = ScaleAABB(
                data["validation"][target_name][1], AABB=prev_AABB, take_neg=True
            )
            data["validation"][target_name][2] = ScaleAABB(
                data["validation"][target_name][2], AABB=prev_AABB, take_neg=True
            )
            # [F, T, R, 1] for ch
            (
                self.n_validation_env[target_name],
                self.n_validation_transmitter[target_name],
                self.n_validation_receivers[target_name],
            ) = data["validation"][target_name][0].shape[
                0:3
            ]  # F, T, R
            self.gain_only = True if (self.n_ch == 4 or self.n_ch == 1) else False

            self.log_manager.InfoLog(
                f"[Validation]: env={self.n_validation_env}, tx={self.n_validation_transmitter}, rx={self.n_validation_receivers}"
            )
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
        env_idx = 0
        # Light Probe, cover them on our map
        if not self.is_ablation():
            self.n_row = 10
            AABB = self.GetAABB()  # [2, dim] -> {min, max}
            AABB_len = (AABB[..., 1, :] - AABB[..., 0, :]).abs()  # [dim,]
            max_len, long_dim = AABB_len.max(dim=0)
            aspect = AABB_len / max_len  # [dim,] -> expect to be 0 < aspect <= 1
            light_probe_shape = (aspect * self.n_row).to(torch.int32).cpu().tolist()
            D, H, W = light_probe_shape[2], light_probe_shape[1], light_probe_shape[0]
            light_probe_shape = [1, 1, D, H, W]
            print("D H W = ", D, H, W)
            light_probe = create_3d_meshgrid_tensor(
                light_probe_shape, device=self.device, dtype=self.dtype
            )  # [1, 3, D, H, W]
            light_probe = light_probe + 0.5
            max_res = max(D, max(H, W))
            light_probe = 2.0 * light_probe / max_res  # ->[0.0, 2.0]
            light_probe[:, 0, ...] += self.AABB[0, 0]
            light_probe[:, 1, ...] += self.AABB[0, 1]
            light_probe[:, 2, ...] += self.AABB[0, 2]
            light_probe = light_probe.reshape(3, -1).transpose(0, 1)  # [DHW, 3]

            # AABB = self.GetAABB()  # [2, dim] -> {min, max}
            # AABB_len = (AABB[..., 1, :] - AABB[..., 0, :]).abs()  # [dim,]
            # max_len, long_dim = AABB_len.max(dim=0)
            # aspect = AABB_len / max_len  # [dim,] -> expect to be 0 < aspect <= 1
            # light_probe_shape = (aspect * self.n_row).to(torch.int32).cpu().tolist()
            # H, W = light_probe_shape[1], light_probe_shape[0]
            # light_probe_shape = [1, 1, H, W]
            # print("H W = ", H, W)
            # light_probe = create_2d_meshgrid_tensor(
            #     light_probe_shape, device=self.device, dtype=self.dtype
            # )  # [1, 2, H, W]
            # light_probe = light_probe + 0.5
            # max_res = max(H, W)
            # light_probe = 2.0 * light_probe / max_res  # ->[0.0, 2.0]
            # light_probe[:, 0, ...] += self.AABB[0, 0]
            # light_probe[:, 1, ...] += self.AABB[0, 1]
            # light_probe = torch.cat(
            #     (light_probe, torch.zeros_like(light_probe[:, 0:1, ...])), dim=1
            # )
            # light_probe = light_probe.reshape(3, -1).transpose(0, 1)  # [DHW, 3]

            self.light_probe_pos = light_probe
            self.InfoLog(
                f"Light Probe position prepared, with shape = {light_probe.shape}, "
                f"AABB = min {self.AABB[0]} + max {self.AABB[1]}, "
                f"light probe range = min {light_probe.min(dim=0)[0]} + max {light_probe.max(dim=0)[0]}, "
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

    def GetInterections(
        self, train_type: int, validation_name: str = None
    ) -> torch.Tensor:
        if train_type == int(TrainType.TRAIN):
            if len(self.nodes["train"]) <= 3:
                self.WarnLog(
                    "You are requiring interactions information from dataset. However, it does not exist. Return None instead"
                )
                return None
            return self.nodes["train"][3]
        elif train_type == int(TrainType.TEST):
            if len(self.nodes["test"]) <= 3:
                self.WarnLog(
                    "You are requiring interactions information from dataset. However, it does not exist. Return None instead"
                )
                return None
            return self.nodes["test"][3]
        elif train_type == int(TrainType.VALIDATION):
            if len(self.nodes["validation"][validation_name]) <= 3:
                self.WarnLog(
                    "You are requiring interactions information from dataset. However, it does not exist. Return None instead"
                )
                return None
            return self.nodes["validation"][validation_name][3]
        else:
            self.ErrorLog(
                f"We only support Train: {int(TrainType.TRAIN)}, "
                f"Test: {int(TrainType.TEST)}, and Validation: {int(TrainType.VALIDATION)}, "
                f"while your intput is {train_type}, please take care of your input of scene dataset"
            )

    def GetTransmitterLocation(
        self, transmitter_idx: int, train_type: int, validation_name: str = None
    ) -> torch.Tensor:
        env_idx = 0
        if train_type == int(TrainType.TRAIN):
            return self.nodes["train"][2][env_idx, transmitter_idx]
        elif train_type == int(TrainType.TEST):
            return self.nodes["test"][2][env_idx, transmitter_idx]
        elif train_type == int(TrainType.VALIDATION):
            return self.nodes["validation"][validation_name][2][
                env_idx, transmitter_idx
            ]
        else:
            self.ErrorLog(
                f"We only support Train: {int(TrainType.TRAIN)}, "
                f"Test: {int(TrainType.TEST)}, and Validation: {int(TrainType.VALIDATION)}, "
                f"while your intput is {train_type}, please take care of your input of scene dataset"
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

    def is_ablation(self):
        return self.module_type == int(ModuleType.ABLATION)

    def is_MLP(self):
        return self.module_type == int(ModuleType.MLP)

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
