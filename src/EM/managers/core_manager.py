import os
import numpy as np
import torch
import time
from typing import Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes

from src.EM.managers import AbstractManager, LogManager, DataManager
from src.EM.utils import (
    mkdir,
    RenderRoom,
    SplatFromParticlesToGrid,
    DumpGrayFigureToRGB,
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
        ablation_prefix = "ablation" if scene_opt["lightfield"]["is_ablation"] else ""
        self._save_path = os.path.join(
            self._root_path, "save", f"{opt['data_set']}_{ablation_prefix}"
        )
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
        self.save_check_point = opt.get("save_check_point")

    def Initialization(self):
        self._data_manager.Initialization()

    def run(self):
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if self.opt["use_check_point"] or not self.is_training:
            success = self.LoadCheckPoint()

            for g in self.model.optimizer.param_groups:
                g["lr"] = self.opt["lr"]

        if self.is_training:
            self.InfoLog("Start Training with parameters: {}".format(self.opt))
            total_steps = self.opt["total_steps"]
            for epoch in tqdm(range(self.start_epoch, total_steps)):
                # Train:
                loss = self.model.train_on_scene(epoch=epoch)

                with torch.no_grad():
                    # Validation:
                    (
                        test_loss,
                        tx_pos,
                        rx_pos,
                        predicted_gains,
                        gt_gains,
                    ) = self.model.validation_on_scene()

                    self._log_manager.WriterAddScalar("train_loss", loss, epoch)
                    self._log_manager.WriterAddScalar("test_loss", test_loss, epoch)
                    self.InfoLog(
                        f"epoch = {epoch}, train_loss = {loss.item()}, test_loss = {test_loss.item()}"
                    )

                    if True:
                        self.validation_vis(
                            epoch=epoch,
                            tx_pos=tx_pos.detach().clone(),
                            rx_pos=rx_pos.detach().clone(),
                            predicted_gains=predicted_gains.detach().clone(),
                            gt_gains=gt_gains.detach().clone(),
                        )

                        if self.save_check_point:
                            self.SaveCheckPoint(epoch=epoch, loss=loss.item())
        else:
            # Validation:
            start_time = time.time()
            (
                test_loss,
                tx_pos,
                rx_pos,
                predicted_gains,
                gt_gains,
            ) = self.model.validation_on_scene()
            end_time = time.time()
            self.InfoLog(f"time for validation = {end_time - start_time}")

            self.validation_vis(
                epoch=None,
                tx_pos=tx_pos.detach().clone(),
                rx_pos=rx_pos.detach().clone(),
                predicted_gains=predicted_gains.detach().clone(),
                gt_gains=gt_gains.detach().clone(),
            )

    def validation_vis(
        self,
        epoch: int,
        tx_pos: torch.Tensor,
        rx_pos: torch.Tensor,
        predicted_gains: torch.Tensor,
        gt_gains: torch.Tensor,
    ):
        env_idx = 0
        save_dir = os.path.join(self._save_path, "imgs")
        mkdir(save_dir)

        if epoch is not None:
            save_path = os.path.join(save_dir, f"env{env_idx}_epoch{epoch}.png")
        else:
            save_path = os.path.join(save_dir, f"env{env_idx}_validation.png")

        if self.scene.H is not None and self.scene.W is not None:
            H, W = self.scene.H, self.scene.W
            aspect = int(W / H)
            fig, ax = plt.subplots(1, 1, figsize=(aspect * 14, 6))
            # plt.pcolormesh(new_gains)
            plt.title("Prediction - Ground truth")
            np_pred_gains = predicted_gains.cpu().numpy()
            np_gt_gains = gt_gains.cpu().numpy()
            np_pred_gains[np.abs(np_gt_gains) < 0.01] = np.inf
            np_gt_gains[np.abs(np_gt_gains) < 0.01] = np.inf
            np_pred_gains = np_pred_gains.reshape(H, W)
            np_gt_gains = np_gt_gains.reshape(H, W)
            blank_wid = 5
            blank_gain = np.zeros_like(np_gt_gains[..., 0:blank_wid])
            blank_gain[...] = np.inf
            np_pred_gt_gains = np.concatenate(
                (np_pred_gains, blank_gain, np_gt_gains),
                axis=-1,
            )
            if tx_pos is not None:
                tx_pos = tx_pos[..., 0:2].reshape(-1, 2)
                tx_pos[..., 0] = 0.5 * (tx_pos[..., 0] + 1.0) * W
                tx_pos[..., 1] = 0.5 * (tx_pos[..., 1] + 1.0) * H
            plt.imshow(np_pred_gt_gains, origin="lower")
            plt.colorbar()
            if tx_pos is not None:
                x, y = (
                    tx_pos[:, 0].cpu().numpy(),
                    tx_pos[:, 1].cpu().numpy(),
                )
                plt.scatter(x, y, c="red", marker="x")

                x, y = (
                    (tx_pos[:, 0] + W + blank_wid).cpu().numpy(),
                    tx_pos[:, 1].cpu().numpy(),
                )
                plt.scatter(x, y, c="red", marker="x")
            plt.savefig(save_path)
            plt.close()

            import h5py

            np_pred_gains[np_pred_gains == np.inf] = 0.0
            np_gt_gains[np_gt_gains == np.inf] = 0.0
            h5f = h5py.File(os.path.join(save_dir, "pred_and_gt.h5"), "w")
            h5f.create_dataset("pred", data=np_pred_gains.reshape(1, H, W))  # [H, W]
            h5f.create_dataset("gt", data=np_gt_gains.reshape(1, H, W))  # [H, W]
            h5f.close()

            return
        verts = self.scene.meshes.verts_list()[env_idx]
        faces = self.scene.meshes.faces_list()[env_idx]
        verts, faces = DeleteFloorOrCeil(verts=verts, faces=faces, axis=2, mode="both")
        meshes = Meshes(
            verts=verts.unsqueeze(0),
            faces=faces.unsqueeze(0),
            textures=None,
        )
        pts = LoadPointCloudFromMesh(
            meshes=meshes, num_pts_samples=5000
        )  # [F, n_pts, 3]
        rendered_room = RenderRoom(pts[env_idx], res_x=256)
        pred_color = SplatFromParticlesToGrid(
            particles=rx_pos[..., 0:2],
            attributes=predicted_gains,
            res_x=rendered_room.shape[1],
            res_y=rendered_room.shape[0],
        )
        gt_color = SplatFromParticlesToGrid(
            particles=rx_pos[..., 0:2],
            attributes=gt_gains,
            res_x=rendered_room.shape[1],
            res_y=rendered_room.shape[0],
        )
        rendered_room = SplatFromParticlesToGrid(
            particles=pts[env_idx, :, 0:2],
            attributes=torch.ones_like(pts[env_idx, :, 0:1]),
            res_x=rendered_room.shape[1],
            res_y=rendered_room.shape[0],
        )
        rendered_room = (rendered_room - rendered_room.min()) / (
            rendered_room.max() - rendered_room.min()
        )
        DumpGrayFigureToRGB(
            save_path,
            pred_color,
            gt_color,
            mask=rendered_room > 0.5,
            extra_spot=tx_pos,
        )

    def SaveCheckPoint(self, epoch: int, loss: float):
        # Additional information
        save_filename = os.path.join(
            self.GetSavePath(), f"model_{self.opt['data_set']}.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.GetRenderer().state_dict(),
                "optimizer_state_dict": self.model.GetOptimizer().state_dict(),
                "loss": loss,
            },
            save_filename,
        )
        self.InfoLog(f"model parameters have been restrored into: {save_filename}")

    def LoadCheckPoint(self, save_filename: str = None) -> bool:
        if save_filename is not None and not os.path.exists(save_filename):
            save_filename = None
            self.WarnLog(
                "You have indicated a specific checkpoint which we cannot find."
            )

        if save_filename is None:
            self.WarnLog(
                "You did not indicate any sepcific model to load, "
                "we will find the most recent check point for you in default."
            )
            pt_files = os.listdir(self.GetSavePath())

            model_file = None
            data_set = self.opt["data_set"]  # TODO:
            for filename in pt_files:
                if ".pt" in filename and data_set in filename:
                    model_file = filename

            if model_file is None:
                self.WarnLog(
                    "No any saved check point found, we will re-train everything for you"
                )
                return False
            else:
                self.InfoLog(
                    f"{model_file} found, we will load this checkpoint for you"
                )

            save_filename = os.path.join(self.GetSavePath(), f"model_{data_set}.pt")

        # load check point
        check_point = torch.load(save_filename)
        self.model.renderer.load_state_dict(check_point["model_state_dict"])
        self.model.optimizer.load_state_dict(check_point["optimizer_state_dict"])
        self.start_epoch = int(check_point["epoch"])
        loss = check_point["loss"]

        self.InfoLog(
            f"Sucessfully loaded a checkpoint start from epoch {self.start_epoch}"
        )

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
