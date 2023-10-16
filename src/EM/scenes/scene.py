import torch
from typing import Dict, Optional, List, Union
from operator import itemgetter

from src.EM.managers import AbstractManager
from src.EM.scenes import AbstractScene, Frame, Camera, RaySampler
from src.EM.utils import TrainType


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
        Create a scene graph for each frame from tracking data and respective camera paramters.
        To use a scene for training ground truth images need to be added to each frame.

        Args:
            dataset:
            models:
            global_transformation:
        """
        self.device = device
        self.dtype = dtype
        self.scene_opt = scene_opt
        self.core_manager = core_manager

        self.K_closest = scene_opt.get("k_closest", False)

        self.type2class = {}
        self.frames = {}
        self.nodes = {}
        # Cameras:
        self.cameras = []
        self.transmitter_attached_frames = {}  # int -> List[int]
        # Frames
        self.frames = []  # List[Frame]

        (
            train_data,
            checkerboard_data,
            genz_data,
            gendiag_data,
        ) = core_manager.LoadData(core_manager.opt)

        self.nodes["train"] = {core_manager.opt["data_set"]: train_data}
        self.nodes["test"] = {"checkerboard": checkerboard_data, "genz": genz_data}
        self.nodes["validation"] = {"gendiag": gendiag_data}

        # [F, T, 1, R, K, I, 4] for interactions
        (
            train_ch,
            train_floor_idx,
            train_interactions,
            train_rx,
            train_tx,
        ) = self.nodes["train"][core_manager.opt["data_set"]]

        self.n_env = train_interactions.shape[0]  # F
        self.n_transmitter = train_interactions.shape[1]  # T
        self.n_cameras = train_interactions.shape[3]  # R
        self.n_rays = train_interactions.shape[4]  # K
        self.n_interactions = train_interactions.shape[5]  # I

        self.ray_sampler = RaySampler(
            K_closest=self.K_closest,
            scene=self,
            device=device,
            dtype=dtype,
        )

        # We need to transfer our data to point clouds firstly.
        self.Initialization()

    def RaySample(self, idx: int, train_type: int = 0) -> List[torch.Tensor]:
        return self.ray_sampler(idx, self, train_type)

    def Initialization(self):
        data_path = self.core_manager.GetDataPath()  # wiindoor level

        # Frames
        self.frames = [
            Frame(self, camera_index=i, frame_index=0, data_path=data_path)
            for i in range(self.n_cameras)
        ]

        # Cameras
        #     The Movement of camera can also easily equaled as the movement of objects.
        self.transmitter_attached_frames = dict(
            zip(
                [i for i in range(self.n_transmitter)],
                [[j for j in range(self.n_cameras)] for i in range(self.n_transmitter)],
            )
        )
        #     TODO: Will we keep transmitters as the information of scene also?

        self.cameras = [
            [
                Camera(device=self.device, dtype=self.dtype)
                for j in range(self.n_cameras)
            ]
            for i in range(self.n_transmitter)
        ]

    def GetFrameIndexFromTransmitter(
        self,
        transmitter_index: Optional[Union[List[int], List[List[int]]]],
        unique=False,
    ):
        if type(transmitter_index) == int:
            frames_index = self.transmitter_attached_frames[transmitter_index]
        else:
            try:
                frames_index = list(
                    itemgetter(*transmitter_index)(self.transmitter_attached_frames)
                )
            except Exception as ex:
                self.ErrorLog(
                    f"Some wrong frame index {ex} is taken, hence terminate the programe"
                )

        if unique:
            frames_index_unique = torch.unique(
                torch.Tensor(
                    frames_index, device=torch.device("cpu"), dtype=self.dtype
                ).flatten(),
                sorted=True,
                return_inverse=False,
            ).tolist()
            return frames_index_unique

        return frames_index

    def GetFrameFromTransmitter(
        self, transmitter_index: Optional[Union[List[int], int]], unique: bool = False
    ) -> List[List[Frame]]:
        frames_index = self.GetFrameIndexFromTransmitter(
            transmitter_index=transmitter_index, unique=unique
        )  # List[int], or List[List[int]]
        if type(frames_index[0]) == int:
            selected_frames = list(itemgetter(*frames_index)(self.frames))
        else:
            selected_frames = [
                list(itemgetter(*(frames_index[i]))(self.frames))
                for i in range(len(frames_index))
            ]
        return selected_frames

    def GetData(self, train_type: int) -> Dict[str, List[torch.Tensor]]:
        if train_type == int(TrainType.Train):
            return self.nodes["train"]
        elif train_type == int(TrainType.TEST):
            return self.nodes["test"]
        elif train_type == int(TrainType.VALIDATION):
            return self.nodes["validation"]
        else:
            self.ErrorLog(
                f"We only support Train: {int(TrainType.Train)}, "
                f"Test: {int(TrainType.TEST)}, and Validation: {int(TrainType.VALIDATION)}, "
                f"while your intput is {train_type}, please take care of your input of scene dataset"
            )

    def GetFrames(self) -> List[Frame]:
        return self.frames

    def GetCameras(self) -> List[List[Camera]]:
        return self.cameras

    def GetCamera(self, transmitter_idx: int, camera_idx: int) -> Camera:
        return self.cameras[transmitter_idx][camera_idx]

    def GetNumRays(self) -> int:
        return self.n_rays

    def GetNumTransmitters(self) -> int:
        return self.n_transmitter

    def GetNumCameras(self) -> int:
        return self.n_cameras

    def GetNumEnvs(self) -> int:
        return self.n_env

    def InfoLog(self, *args, **kwargs):
        return self.core_manager.InfoLog(*args, **kwargs)

    def WarnLog(self, *args, **kwargs):
        return self.core_manager.WarnLog(*args, **kwargs)

    def ErrorLog(self, *args, **kwargs):
        return self.core_manager.ErrorLog(*args, **kwargs)
