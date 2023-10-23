import os
import numpy as np
import h5py
from typing import Dict, Any, List
import torch
from torch.utils.data import DataLoader, Dataset

from src.EM.managers import AbstractManager
from src.EM.scenes import NeuralScene


class SceneDataSet(Dataset):
    def __init__(self, scene: NeuralScene, train_type: int) -> None:
        super(SceneDataSet, self).__init__()
        self.scene = scene
        self.train_type = train_type  # See definition of Enum TrainType from utils

        num_tx = self.scene.GetNumTransmitters(self.train_type)
        num_rx = self.scene.GetNumReceivers(self.train_type)
        self.length = num_tx * num_rx

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        with torch.no_grad():
            # points, distance, walk, pitch, azimuth
            return self.scene.RaySample(idx=index, train_type=self.train_type)


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

    def LoadData(self, is_training: bool = False, test_target: str = "all"):
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
        target_list = ["checkerboard", "genz", "gendiag"]
        data_files = os.listdir(data_path)

        result = {"train": None}
        for target_name in target_list:
            result[target_name] = None

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
                ch = np.array(
                    data["channels"][0:1, ...]
                )  # [F, T, 1, R, D=8, K], float32
                floor_idx = np.array(data["floor_idx"][0:1, ...])  # [3, ], int32
                interactions = np.array(
                    data["interactions"][0:1, ...]
                )  # [F, T, 1, R, K, I, 4]
                rx = np.array(data["rx"][0:1, ...])  # [F, T, 1, R, dim=3]
                tx = np.array(data["tx"][0:1, ...])  # [F, T, dim=3]

                print(f"{target_name}, intersection = {interactions.shape}")

                if result[target_name] is None:
                    result[target_name] = [ch, floor_idx, interactions, rx, tx]
                else:
                    tensors = [
                        np.concatenate((result[target_name][0], ch), axis=0),
                        np.concatenate((result[target_name][1], floor_idx), axis=0),
                        np.concatenate((result[target_name][2], interactions), axis=0),
                        np.concatenate((result[target_name][3], rx), axis=0),
                        np.concatenate((result[target_name][4], tx), axis=0),
                    ]
                    result[target_name] = tensors

        return [result["train"]] + [result[target_name] for target_name in target_list]
