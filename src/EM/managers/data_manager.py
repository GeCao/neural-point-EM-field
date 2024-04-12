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

        print("validation names = ", self.validation_names)

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
                        tx_idx = 0  # TODO: tx_idx=3 only

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

        if "sionna" in str(data_path):
            # Different data management
            data_files = os.listdir(data_path)
            if "etoicenter" in str(data_path):
                target_list = ["111"]
            elif "etoile" in str(data_path):
                target_list = ["054"]
            else:
                target_list = ["111"]
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

        for filename in data_files:
            if filename[-3:] == ".h5":
                target_name = None
                if "train" in filename:
                    target_name = "train"
                else:
                    for name in target_list:
                        target_name = name if name in filename else target_name

                    if target_name is None:
                        continue

                # load train/test data:
                data = h5py.File(os.path.join(data_path, filename))
                if "channels" in data:
                    # Note:
                    # departure: from tx
                    # arrival: to rx
                    ch = np.array(
                        data["channels"][0:1, :, 0, :, :, :]
                    )  # [F, T, 1, R, D=8, K] -> [F, T, R, D=8, n_rays], float32
                    # In original data, gain-phase-time-elevation-azimuth-elevation-azimuth-validality
                    # What we want, gain-phase-time-azimuth-elevation-azimuth-elevation-validality
                    ch = np.concatenate(
                        (
                            ch[..., 0:3, :],
                            ch[..., 4:5, :],
                            ch[..., 3:4, :],
                            ch[..., 6:7, :],
                            ch[..., 5:6, :],
                            ch[..., 7:, :],
                        ),
                        axis=-2,
                    )
                    # 1 - phase: [-PI, PI]
                    # 4/6 - elevation: [0, PI]
                    # 3/5 - azimuth: [0, 2*PI]

                    rx = np.array(
                        data["rx"][0:1, :, 0, ...]
                    )  # [F, T, 1, R, dim=3] -> [F, T, R, dim=3]
                    tx = np.array(data["tx"][0:1, ...])  # [F, T, dim=3]
                    interactions = np.array(
                        data["interactions"][0:1, :, 0, ...]
                    )  # [F, T, 1, R, K, I, dim=4] -> [F, T, R, n_rays, I, dim=4]
                    # intersects = np.array(
                    #     data["interactions"][0:1, :, 0, ...]
                    # )  # [F, T, 1, R, K, I, dim=4] -> [F, T, R, n_rays, I, dim=4]
                    # print("rx = ", rx[0, 0, 0])
                    # print("tx = ", tx[0, 0])
                    # print("intersects = ", intersects[0, 0, 0, 0])
                    # print("ch = ", ch[0, 2, 2546, :, :])
                    # azimuth_d = ch[0, 0, 0, 5, 0]
                    # elevation_d = ch[0, 0, 0, 6, 0]
                    # ch_dir_d = np.array(
                    #     [
                    #         np.cos(azimuth_d) * np.sin(elevation_d),
                    #         np.sin(azimuth_d) * np.sin(elevation_d),
                    #         np.cos(elevation_d),
                    #     ]
                    # )
                    # rxtx_dir_d = (rx[0, 0, 0] - tx[0, 0]) / np.linalg.norm(
                    #     (rx[0, 0, 0] - tx[0, 0])
                    # )
                    # print("ch_dir_d = ", ch_dir_d)
                    # print("rxtx_dir_d = ", rxtx_dir_d)
                    # exit(0)
                elif "gain" in data:
                    ch = np.array(data["gain"])  # [T, H, W, 4]
                    rx = np.array(data["rx"])  # [T, H, W, dim=3]
                    tx = np.array(data["tx"])  # [T, dim=3]
                    T, H, W, n_ch = ch.shape
                    if n_ch != 4:
                        raise RuntimeError("Regenerate your dataset!")
                    ch = ch.reshape(1, T, H * W, 4)  # [F, T, H*W, 1]
                    # ch = ch[..., 0:1]  # TODO:
                    rx = rx[0:T, ...].reshape(1, T, H * W, 3)  # [F, T, H*W, dim=3]
                    tx = np.expand_dims(tx[0:T, ...], axis=0)  # [F, T, dim=3]
                    result["H"] = H
                    result["W"] = W
                    print("Read channel: mean = ", ch.mean())
                    interactions = None

                ch = torch.from_numpy(ch).to(dtype).to(device)
                rx = torch.from_numpy(rx).to(dtype).to(device)
                tx = torch.from_numpy(tx).to(dtype).to(device)
                if interactions is not None:
                    interactions = torch.from_numpy(interactions).to(dtype).to(device)
                    interactions[
                        interactions[..., 0:1]
                        .to(torch.int32)
                        .sum(dim=-2, keepdim=True)
                        .repeat(1, 1, 1, 1, interactions.shape[-2], 4)
                        == 0
                    ] = -1

                if target_name is "train":
                    if result["train"] is None:
                        result["train"] = (
                            [ch, rx, tx]
                            if interactions is None
                            else [ch, rx, tx, interactions]
                        )
                    else:
                        tensors = [
                            torch.cat((result[target_name][0], ch), dim=0),
                            torch.cat((result[target_name][1], rx), dim=0),
                            torch.cat((result[target_name][2], tx), dim=0),
                        ]
                        if interactions is not None:
                            tensors = tensors + [
                                torch.cat((result[target_name][3], interactions), dim=0)
                            ]

                        result["train"] = tensors
                else:
                    if result["validation"][target_name] is None:
                        result["validation"][target_name] = (
                            [ch, rx, tx]
                            if interactions is None
                            else [ch, rx, tx, interactions]
                        )
                    else:
                        tensors = [
                            torch.cat((result[target_name][0], ch), dim=0),
                            torch.cat((result[target_name][1], rx), dim=0),
                            torch.cat((result[target_name][2], tx), dim=0),
                        ]
                        if interactions is not None:
                            tensors = tensors + [
                                torch.cat((result[target_name][3], interactions), dim=0)
                            ]

                        result["validation"][target_name] = tensors

        if "sionna" in str(data_path):
            vali_idx = []
            for validation_name in result["validation"]:
                idx = int(validation_name)
                if validation_name.isdigit():
                    vali_idx.append(idx)

                result["validation"][validation_name] = [
                    result["train"][0][:, idx : idx + 1, ...],
                    result["train"][1][:, idx : idx + 1, ...],
                    result["train"][2][:, idx : idx + 1, ...],
                ]

            if "etoicenter" in str(data_path):
                vali_idx = vali_idx + [30, 31, 32, 54, 55, 56, 112, 113]
            elif "etoile" in str(data_path):
                vali_idx = vali_idx + [33, 34, 35, 55, 56, 66, 67, 68]
            else:
                vali_idx = vali_idx + [30, 31, 32, 54, 55, 56, 112, 113]
            vali_idx = sorted(vali_idx, reverse=True)  # [large -> small]
            print(f"Load validation name {vali_idx} from train dataset")
            for i in range(len(vali_idx)):
                result["train"] = [
                    torch.cat(
                        (
                            result["train"][0][:, 0 : vali_idx[i], ...],
                            result["train"][0][:, vali_idx[i] + 1 :, ...],
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            result["train"][1][:, 0 : vali_idx[i], ...],
                            result["train"][1][:, vali_idx[i] + 1 :, ...],
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            result["train"][2][:, 0 : vali_idx[i], ...],
                            result["train"][2][:, vali_idx[i] + 1 :, ...],
                        ),
                        dim=1,
                    ),
                ]

        return result
