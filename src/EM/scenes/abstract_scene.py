from abc import ABC, abstractmethod
from typing import Dict, List
import torch

from src.EM.scenes import Camera, Transmitter


class AbstractScene(ABC):
    @abstractmethod
    def GetTransmitterLocation(
        self, transmitter_idx: int, train_type: int, validation_name: str
    ) -> Transmitter: ...

    @abstractmethod
    def GetData(self, train_type: int, validation_name: str) -> List[torch.Tensor]: ...

    @abstractmethod
    def GetInterections(
        self, train_type: int, validation_name: str
    ) -> torch.Tensor: ...

    @abstractmethod
    def GetNumTransmitters(self, train_type: int) -> int: ...

    @abstractmethod
    def GetNumReceivers(self, train_type: int) -> int: ...

    @abstractmethod
    def GetNumEnvs(self, train_type: int) -> int: ...

    @abstractmethod
    def GetPointCloud(self, env_index: int) -> torch.Tensor: ...

    @abstractmethod
    def GetAABB(self) -> torch.Tensor: ...

    @abstractmethod
    def GetLightProbePosition(self) -> torch.Tensor: ...

    @abstractmethod
    def InfoLog(self): ...

    @abstractmethod
    def WarnLog(self): ...

    @abstractmethod
    def ErrorLog(self): ...
