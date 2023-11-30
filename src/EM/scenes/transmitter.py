import math
import torch
import torch.nn.functional as F
from typing import List


class Transmitter(object):
    def __init__(
        self,
        source_location: torch.Tensor,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
    ) -> None:
        self.device = device
        self.dtype = dtype

        self.source_location = source_location

    def GetSourceLocation(self) -> torch.Tensor:
        return self.source_location

    def Decompose(
        self,
        points: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Take ray direction and point information, return rendered wireless channel
        Args:
            points             (torch.Tensor):                [n_pts,             dim=3]

        Returns:
            torch.Tensor: [n_pts, 4] (distance, azimuth, elevation)
        """
        points = points.view(-1, 3)
        n_pts = points.shape[-2]

        eps = 1e-5
        r = self.GetSourceLocation().reshape(1, 3) - points  # [n_pts, 3]
        tx_dist = r.norm(dim=-1, keepdim=True)  # [n_pts, 1]
        tx_elevation = torch.acos(r[:, 2:3] / (tx_dist + eps))
        tx_azimuth = torch.acos(
            r[:, 0:1] / (tx_dist * torch.sin(tx_elevation) + eps)
        )  # [0, PI]
        tx_azimuth = torch.where(
            r[..., 1:2] < 0.0, 2.0 * math.pi - tx_azimuth, tx_azimuth
        )  # [0, 2PI]
        tx_info = torch.cat((tx_dist, tx_azimuth, tx_elevation), dim=-1)
        # TODO: What if add normal for material rendering?

        return tx_info
