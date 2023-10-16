from abc import ABC, abstractmethod
from typing import Dict, List
import torch

from src.EM.scenes import Camera


class AbstractScene(ABC):
    @abstractmethod
    def GetFrameIndexFromTransmitter(self, transmitter_index, unique):
        ...

    @abstractmethod
    def GetFrameFromTransmitter(self, transmitter_index, unique):
        ...

    @abstractmethod
    def GetFrames(self):
        ...

    @abstractmethod
    def GetCameras(self) -> List[Camera]:
        ...

    @abstractmethod
    def GetCamera(self, transmitter_idx: int, camera_idx: int) -> Camera:
        ...

    @abstractmethod
    def GetData(self, train_type: int) -> Dict[str, List[torch.Tensor]]:
        ...

    @abstractmethod
    def GetNumRays(self) -> int:
        ...

    @abstractmethod
    def GetNumTransmitters(self) -> int:
        ...

    @abstractmethod
    def GetNumCameras(self) -> int:
        ...

    @abstractmethod
    def GetNumEnvs(self) -> int:
        ...

    @abstractmethod
    def InfoLog(self):
        ...

    @abstractmethod
    def WarnLog(self):
        ...

    @abstractmethod
    def ErrorLog(self):
        ...
