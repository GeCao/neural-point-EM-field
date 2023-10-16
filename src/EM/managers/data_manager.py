import os
import numpy as np
import h5py
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader, Dataset

from src.EM.managers import AbstractManager
from src.EM.scenes import NeuralScene


class SceneDataSet(Dataset):
    def __init__(self, scene: NeuralScene, train_type: int) -> None:
        super(SceneDataSet, self).__init__()
        self.scene = scene
        self.train_type = train_type  # See definition of Enum TrainType from utils

    def __len__(self):
        return self.scene.cameras.shape[0] * self.scene.cameras.shape[1]

    def __getitem__(self, index: int) -> Any:
        with torch.no_grad():
            # points, distance, walk, pitch, azimuth
            return self.scene.RaySample(idx=index, train_type=self.train_type)


class DataManager(object):
    def __init__(self, core_manager: AbstractManager):
        self._core_manager = core_manager

        self.log_path = None

        self._file_ptr = None

        self.initialized = False

    def Initialization(self):
        self.initialized = True

    def LoadData(self, opt: Dict):
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
        is_training = opt.get("is_training")
        test_target = opt.get("test_target")
        data_path = self._core_manager.GetDataPath()
        target_list = ["checkerboard", "genz", "gendiag"]
        data_files = os.listdir(data_path)

        result = {"train": None}
        for target_name in target_list:
            result[target_name] = None

        for filename in data_files:
            if filename[-3:] == ".h5":
                if is_training and "train" in filename:
                    # load train data:
                    train_data = h5py.File(os.path.join(data_path, filename))
                    train_ch = np.array(
                        train_data["channels"]
                    )  # [F, T, 1, R, D=8, K], float32
                    train_floor_idx = np.array(train_data["floor_idx"])  # [3, ], int32
                    train_interactions = np.array(
                        train_data["interactions"]
                    )  # [F, T, 1, R, K, I, 4]
                    train_rx = np.array(train_data["rx"])  # [F, T, 1, R, dim=3]
                    train_tx = np.array(train_data["tx"])  # [F, T, dim=3]

                    result["train"] = [
                        train_ch,
                        train_floor_idx,
                        train_interactions,
                        train_rx,
                        train_tx,
                    ]
                else:
                    # load test data:
                    for target_name in target_list:
                        if target_name in filename and (
                            test_target == "all" or test_target == target_name
                        ):
                            test_data = h5py.File(os.path.join(data_path, filename))
                            test_ch = np.array(
                                test_data["channels"]
                            )  # [F, T, 1, R, D=8, K]
                            test_floor_idx = np.array(test_data["floor_idx"])  # [3, ]
                            test_interactions = np.array(
                                test_data["interactions"]
                            )  # [F, T, 1, R, K, I, 4]
                            test_rx = np.array(test_data["rx"])  # [F, T, 1, R, dim=3]
                            test_tx = np.array(test_data["tx"])  # [F, T, dim=3]

                            result[target_name] = [
                                test_ch,
                                test_floor_idx,
                                test_interactions,
                                test_rx,
                                test_tx,
                            ]
            elif filename == "objs":
                # TODO: Load obj files
                pass

        return [result["train"]] + [result[target_name] for target_name in target_list]
