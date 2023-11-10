import os
import numpy as np
import torch
import cv2
from typing import Dict
from tqdm import tqdm
from pytorch3d.structures import Meshes

from src.EM.managers import AbstractManager, LogManager, DataManager
from src.EM.utils import (
    mkdir,
    RenderRoom,
    DrawHeatMapReceivers,
    ExportVTKFile,
    FromGridToColor,
    DeleteFloorOrCeil,
    LoadPointCloudFromMesh,
)
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
        self._log_manager = LogManager(
            root_path=self._root_path, log_to_disk=opt["log_to_disk"]
        )
        self._log_manager.Initialization()
        self._data_manager = DataManager(self.GetDataPath())

        self.scene = NeuralScene(
            self, scene_opt=scene_opt, device=self.device, dtype=self.dtype
        )
        self.model = PointLFEMModel(
            self.scene, opt=opt, device=self.device, dtype=self.dtype
        )
        self.start_epoch = 0

        self.is_training = opt.get("is_training")
        self.use_check_point = opt.get("use_check_point")

    def Initialization(self):
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
            for epoch in tqdm(range(self.start_epoch, total_steps)):
                (
                    loss,
                    test_loss,
                    rx_pos,
                    predicted_gains,
                    gt_gains,
                ) = self.model.train_on_scene(epoch=epoch)

                self._log_manager.WriterAddScalar("train_loss", loss, epoch)
                self._log_manager.WriterAddScalar("test_loss", test_loss, epoch)
                self.InfoLog(
                    f"epoch = {epoch}, train_loss = {loss.item()}, test_loss = {test_loss.item()}"
                )

                if epoch % 10 == 0:
                    env_idx = self.model.test_dataloader.dataset.env_idx
                    meshes = self.scene.meshes
                    verts = meshes.verts_list()[env_idx]
                    faces = meshes.faces_list()[env_idx]
                    verts, faces = DeleteFloorOrCeil(
                        verts=verts, faces=faces, axis=2, mode="both"
                    )
                    meshes = Meshes(
                        verts=verts.unsqueeze(0),
                        faces=faces.unsqueeze(0),
                        textures=None,
                    )
                    point_clouds = LoadPointCloudFromMesh(
                        meshes=meshes, num_pts_samples=1000
                    )  # [F, n_pts, 3]
                    rendered_room = RenderRoom(point_clouds[env_idx], res_x=1024)
                    pred_color = DrawHeatMapReceivers(
                        rx=rx_pos,
                        gain=predicted_gains,
                        res_x=rendered_room.shape[1],
                        res_y=rendered_room.shape[0],
                    )
                    gt_color = DrawHeatMapReceivers(
                        rx=rx_pos,
                        gain=gt_gains,
                        res_x=rendered_room.shape[1],
                        res_y=rendered_room.shape[0],
                    )

                    grid_min, grid_max = gt_color.min(), gt_color.max()

                    pred_color = (pred_color - grid_min) / (grid_max - grid_min)
                    pred_color = FromGridToColor(grid=pred_color)  # [H, W, 3]

                    gt_color = (gt_color - grid_min) / (grid_max - grid_min)
                    gt_color = FromGridToColor(grid=gt_color)  # [H, W, 3]

                    pred_color[rendered_room.repeat(1, 1, 3) > 10] = 0.0
                    gt_color[rendered_room.repeat(1, 1, 3) > 10] = 0.0
                    gt_color = gt_color.cpu().numpy()
                    pred_color = pred_color.cpu().numpy()
                    rendered_room = rendered_room.cpu().numpy()

                    save_dir = os.path.join(self._save_path, "imgs")
                    mkdir(save_dir)

                    save_path = os.path.join(
                        save_dir, f"pred_env{env_idx}_epoch{epoch}.png"
                    )
                    cv2.imwrite(save_path, pred_color)

                    save_path = os.path.join(
                        save_dir, f"gt_env{env_idx}_epoch{epoch}.png"
                    )
                    cv2.imwrite(save_path, gt_color)

                    save_path = os.path.join(save_dir, f"pred_gain_{epoch}")
                    ExportVTKFile(
                        save_path=save_path,
                        rx_pos=rx_pos,
                        gain=predicted_gains,
                        point_clouds=point_clouds[env_idx],
                    )

                    save_path = os.path.join(save_dir, f"gt_gain_epoch{epoch}")
                    ExportVTKFile(
                        save_path=save_path,
                        rx_pos=rx_pos,
                        gain=gt_gains,
                        point_clouds=point_clouds[env_idx],
                    )

    def SaveCheckPoint(self, epoch: int, loss: float):
        # Additional information
        save_filename = os.path.join(self.GetSavePath(), f"model_epoch{epoch}.pt")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.GetRenderer().state_dict(),
                "optimizer_state_dict": self.model.GetOptimizer(),
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
        self.model.renderer.load_state_dict(check_point["model_state_dict"])
        self.model.optimizer.load_state_dict(check_point["optimizer_state_dict"])
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
