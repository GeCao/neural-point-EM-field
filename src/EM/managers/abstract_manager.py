from abc import ABC, abstractmethod
from typing import Dict
import torch


class AbstractManager(ABC):
    @abstractmethod
    def GetRootPath(self) -> str:
        ...

    @abstractmethod
    def GetDemoPath(self) -> str:
        ...

    @abstractmethod
    def GetDataPath(self) -> str:
        ...

    @abstractmethod
    def GetSavePath(self) -> str:
        ...

    @abstractmethod
    def GetDim(self) -> int:
        ...

    @abstractmethod
    def LoadData(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def WarnLog(self, *args, **kwargs):
        ...

    @abstractmethod
    def InfoLog(self, *args, **kwargs):
        ...

    @abstractmethod
    def ErrorLog(self, *args, **kwargs):
        ...
