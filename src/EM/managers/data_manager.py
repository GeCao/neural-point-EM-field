import os
import numpy as np
import json
import h5py
from typing import Dict, Any, List, Union
import torch
from torch.utils.data import DataLoader, Dataset

from src.EM.managers import AbstractManager
from src.EM.scenes import NeuralScene
from src.EM.utils import TrainType


class SceneDataSet(Dataset):
    def __init__(self, scene: NeuralScene, train_type: int) -> None:
        super(SceneDataSet, self).__init__()
        self.scene = scene
        self.train_type = train_type  # See definition of Enum TrainType from utils

        self.num_envs = self.scene.GetNumEnvs(self.train_type)
        self.num_tx = self.scene.GetNumTransmitters(self.train_type)
        self.num_rx = self.scene.GetNumReceivers(self.train_type)

        self.validation_names = []
        if type(self.num_rx) == dict:
            self.lengths = []
            self.total_length = 0
            for i, key in enumerate(self.num_rx):
                self.num_tx[key] = 1  # TODO: tx_idx=3 only
                self.lengths.append(
                    self.num_envs[key] * self.num_tx[key] * self.num_rx[key]
                )
                self.total_length += self.lengths[i]
                self.validation_names.append(key)
        else:
            self.lengths = [self.num_envs * self.num_tx * self.num_rx]
            self.total_length = self.lengths[0]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        with torch.no_grad():
            index = index % self.__len__()
            # points, distance, proj_distance, pitch, azimuth
            for i, len_ in enumerate(self.lengths):
                if index < len_:
                    validation_name = (
                        None
                        if len(self.validation_names) == 0
                        else self.validation_names[i]
                    )

                    num_tx = (
                        self.num_tx
                        if type(self.num_tx) != dict
                        else self.num_tx[validation_name]
                    )
                    num_rx = (
                        self.num_rx
                        if type(self.num_rx) != dict
                        else self.num_rx[validation_name]
                    )

                    env_idx = index // (num_tx * num_rx)
                    tx_idx = (index % (num_tx * num_rx)) // num_rx
                    rx_idx = (index % (num_tx * num_rx)) % num_rx

                    if self.train_type == int(TrainType.VALIDATION):
                        tx_idx = 3  # TODO: tx_idx=3 only

                    return self.scene.RaySample(
                        env_idx=env_idx,
                        tx_idx=tx_idx,
                        rx_idx=rx_idx,
                        validation_name=validation_name,
                        train_type=self.train_type,
                    )
                else:
                    index = index - len_

            return None


class DataManager(object):
    def __init__(self, data_path: str):
        self.data_path = data_path

        self.log_path = None

        self._file_ptr = None

        self.initialized = False

    def Initialization(self):
        self.initialized = True

    def GetDataPath(self):
        return self.data_path

    def LoadData(
        self,
        is_training: bool = False,
        validation_target: str = "all",
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Load data from disk
        Note the data structure is given below:
        # F := #Environments
        # T := #tx locations
        # R := #rx locations
        # D := #8-dim attributes
            [
                complex_gain (dB),
                phase_of_complex_gain (radian),
                time_of_flights (ns),
                tx_eye_azimuth (radian),
                tx_elevation (radian),
                rx_azimuth (radian),
                rx_elevation (radian),
                validality (bool),
            ]
        # K := #paths
        # I := #Interactions
            [
                tx/rx location,
                diffraction (3d),
                reflection (3d),
                transmission (3d),
                reflection with ceiling,
                reflection with floor,
            ]
        """
        data_path = self.GetDataPath()

        if "coverage_map" in str(data_path):
            # Different data management
            data_files = os.listdir(data_path)
            data_files = [filename for filename in data_files if "obj" not in filename]
            data_files = [filename for filename in data_files if "geom" not in filename]
            # TODO: Sort directories
            data_files.sort()
            target_list = ["024"]
        elif "rt" in str(data_path):
            # Different data management
            data_files = os.listdir(data_path)
            target_list = ["022"]
        else:
            target_list = ["checkerboard", "genz", "gendiag"]
            if validation_target == "all":
                target_list = target_list
            elif validation_target in target_list:
                target_list = [validation_target]
            else:
                raise RuntimeError(
                    f"You claim your validation data named as {validation_target}, which is not match with our choices {target_list}"
                )
            data_files = os.listdir(data_path)

        result = {"train": None, "validation": {}, "target_list": target_list}
        for target_name in target_list:
            result["validation"][target_name] = None

        tx_idx = -1
        for filename in data_files:
            if filename[-3:] == ".h5":
                target_name = None
                if is_training and "train" in filename:
                    target_name = "train"
                else:
                    for name in target_list:
                        target_name = name if name in filename else target_name

                    if target_name is None:
                        continue

                # load train/test data:
                data = h5py.File(os.path.join(data_path, filename))
                if "channels" in data:
                    ch = np.array(
                        data["channels"][0:1, :, 0, :, 0, :]
                    )  # [F, T, 1, R, D=8, K] -> [F, T, R, K], float32
                    ch = ch.sum(axis=-1, keepdims=True)  # [F, T, R, 1], float32

                    rx = np.array(
                        data["rx"][0:1, :, 0, ...]
                    )  # [F, T, 1, R, dim=3] -> [F, T, R, dim=3]
                    tx = np.array(data["tx"][0:1, ...])  # [F, T, dim=3]
                elif "gain" in data:
                    ch = np.array(data["gain"])  # [T, R, 1]
                    rx = np.array(data["rx"])  # [T, R, dim=3]
                    tx = np.array(data["tx"])  # [T, dim=3]
                    ch = np.expand_dims(ch, axis=0)  # [F, T, R, 1]
                    rx = np.expand_dims(rx, axis=0)  # [F, T, R, dim=3]
                    tx = np.expand_dims(tx, axis=0)  # [F, T, dim=3]

                ch = torch.from_numpy(ch).to(dtype).to(device)
                rx = torch.from_numpy(rx).to(dtype).to(device)
                tx = torch.from_numpy(tx).to(dtype).to(device)

                if target_name is "train":
                    if result["train"] is None:
                        result["train"] = [ch, rx, tx]
                    else:
                        tensors = [
                            torch.cat((result[target_name][0], ch), dim=0),
                            torch.cat((result[target_name][1], rx), dim=0),
                            torch.cat((result[target_name][2], tx), dim=0),
                        ]
                        result["train"] = tensors
                else:
                    if result["validation"][target_name] is None:
                        result["validation"][target_name] = [ch, rx, tx]
                    else:
                        tensors = [
                            torch.cat((result[target_name][0], ch), dim=0),
                            torch.cat((result[target_name][1], rx), dim=0),
                            torch.cat((result[target_name][2], tx), dim=0),
                        ]
                        result["validation"][target_name] = tensors

        if "022" in result["validation"]:
            result["validation"]["022"] = [
                result["train"][0][:, 22:23, ...],
                result["train"][1][:, 22:23, ...],
                result["train"][2][:, 22:23, ...],
            ]

            result["train"] = [
                torch.cat(
                    (result["train"][0][:, 0:22, ...], result["train"][0][:, 23:, ...]),
                    dim=1,
                ),
                torch.cat(
                    (result["train"][1][:, 0:22, ...], result["train"][1][:, 23:, ...]),
                    dim=1,
                ),
                torch.cat(
                    (result["train"][2][:, 0:22, ...], result["train"][2][:, 23:, ...]),
                    dim=1,
                ),
            ]

        return result
