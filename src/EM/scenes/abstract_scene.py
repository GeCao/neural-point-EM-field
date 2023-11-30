from abc import ABC, abstractmethod
from typing import Dict, List
import torch

from src.EM.scenes import Camera, Transmitter


class AbstractScene(ABC):
    @abstractmethod
    def GetReceivers(self, train_type: int) -> List[List[Camera]]:
        ...

    @abstractmethod
    def GetReceiver(
        self, transmitter_idx: int, receiver_idx: int, train_type: int
    ) -> Camera:
        ...

    @abstractmethod
    def GetTransmitters(self, train_type: int) -> List[Transmitter]:
        ...

    @abstractmethod
    def GetTransmitter(self, transmitter_idx: int, train_type: int) -> Transmitter:
        ...

    @abstractmethod
    def GetData(self, train_type: int) -> List[torch.Tensor]:
        ...

    @abstractmethod
    def GetNumRays(self, train_type: int) -> int:
        ...

    @abstractmethod
    def GetNumTransmitters(self, train_type: int) -> int:
        ...

    @abstractmethod
    def GetNumReceivers(self, train_type: int) -> int:
        ...

    @abstractmethod
    def GetNumEnvs(self, train_type: int) -> int:
        ...

    @abstractmethod
    def GetPointCloud(self, env_index: int) -> torch.Tensor:
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
