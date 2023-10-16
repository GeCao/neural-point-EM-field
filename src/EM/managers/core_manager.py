import os
import numpy as np
import torch
from typing import Dict
from tqdm import tqdm

from src.EM.managers import AbstractManager, LogManager, DataManager
from src.EM.utils import mkdir
from src.EM.models import PointLFEMModel
from src.EM.scenes import NeuralScene
from src.EM.global_variables import scene_opt


class CoreManager(AbstractManager):
    def __init__(self, opt: Dict) -> None:
        # 1. Resolve the path of project
        self._demo_path = os.path.abspath(os.curdir)
        self._root_path = os.path.abspath(os.path.join(self._demo_path, ".."))
        self._data_path = os.path.join(self._root_path, "data", opt["data_set"])
        self._save_path = os.path.join(self._root_path, "save", opt["data_set"])
        mkdir(self._save_path)

        print("The root path of our project: ", self._root_path)
        print("The data path of our project: ", self._data_path)

        self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = opt["dim"]

        self.opt = opt
        self._log_manager = LogManager(self, log_to_disk=opt["log_to_disk"])
        self._data_manager = DataManager(self)

        self.scene = NeuralScene(
            self, scene_opt=scene_opt, device=self.device, dtype=self.dtype
        )
        self.model = PointLFEMModel(
            self.scene, opt=opt, device=self.device, dtype=self.dtype
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt["lr"])
        self.loss = torch.nn.MSELoss()
        self.start_epoch = 0

        self.is_training = opt.get("is_training")
        self.use_check_point = opt.get("use_check_point")

    def Initialization(self):
        self._log_manager.Initialization()
        self._data_manager.Initialization()
        if self.opt["use_check_point"]:
            success = self.LoadCheckPoint()

    def run(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if self.is_training:
            self.InfoLog("Start Training with parameters: {}".format(self.opt))
            total_steps = self.opt["total_steps"]
            # TODO: Load data
            (
                train_data,
                checkerboard_data,
                genz_data,
                gendiag_data,
            ) = self._data_manager.LoadData(self.opt)
            if train_data is None:
                self.ErrorLog(
                    "Train DataSet not found, please ensure your data set has been placed carefully"
                )
            train_ch = gendiag_data[0]
            train_floor_idx = gendiag_data[1]
            train_interactions = gendiag_data[2]
            train_rx = gendiag_data[3]
            train_tx = gendiag_data[4]
            for epoch in tqdm(range(self.start_epoch, total_steps)):
                pass

    def SaveCheckPoint(self, epoch: int, loss: float):
        # Additional information
        save_filename = os.path.join(self.GetSavePath(), f"model_epoch{epoch}.pt")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            save_filename,
        )

    def LoadCheckPoint(self, save_filename: str = None, epoch: int = None) -> bool:
        if save_filename is not None and not os.path.exists(save_filename):
            save_filename = None
            self.WarnLog(
                "You have indicated a specific checkpoint which we cannot find."
            )
            if epoch is None:
                self.WarnLog(
                    "You did not indicate any sepcific epoch either, "
                    "we will find the most recent check point for you in default."
                )
            else:
                self.WarnLog(
                    f"However, you did indicate a specific epoch{epoch}, "
                    "we will find that check point for you."
                )

        if save_filename is None:
            pt_files = os.listdir(self.GetSavePath())

            max_saved_epoch = -1
            epoch_found = False

            for filename in pt_files:
                if ".pt" in filename:
                    left = filename.find("epoch") + 5
                    right = filename.find(".")
                    curr_epoch = int(filename[left:right])

                    max_saved_epoch = max(curr_epoch, max_saved_epoch)
                    if epoch is not None and curr_epoch == epoch:
                        epoch_found = True

            if epoch is not None and epoch_found:
                epoch = epoch
            else:
                if epoch is not None and not epoch_found:
                    self.WarnLog(
                        f"You indicated to read a checkpoint in specific epoch{epoch}, "
                        "which cannot be found by us. "
                        "We will find the most recent check point for you in default."
                    )

                if max_saved_epoch == -1:
                    self.WarnLog(
                        "No any saved check point found, we will re-train everything for you"
                    )
                    return False
                else:
                    epoch = max_saved_epoch

            save_filename = os.path.join(self.GetSavePath(), f"model_epoch{epoch}.pt")

        # load check point
        check_point = torch.load(save_filename)
        self.model.load_state_dict(check_point["model_state_dict"])
        self.optimizer.load_state_dict(check_point["optimizer_state_dict"])
        self.start_epoch = int(check_point["epoch"])
        loss = check_point["loss"]

        return True

    def GetRootPath(self) -> str:
        return self._root_path

    def GetSavePath(self) -> str:
        return self._save_path

    def GetDataPath(self) -> str:
        return self._data_path

    def GetDemoPath(self) -> str:
        return self._demo_path

    def GetDim(self) -> int:
        return self.dim

    def LoadData(self, *args, **kwargs):
        return self._data_manager.LoadData(*args, **kwargs)

    def InfoLog(self, *args, **kwargs):
        return self._log_manager.InfoLog(*args, **kwargs)

    def WarnLog(self, *args, **kwargs):
        return self._log_manager.WarnLog(*args, **kwargs)

    def ErrorLog(self, *args, **kwargs):
        return self._log_manager.ErrorLog(*args, **kwargs)
