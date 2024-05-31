import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


# Takes Points, weights  and rays and maps to color
class PointLightFieldMLP(nn.Module):
    def __init__(
        self,
        W: int = 64,
        n_feat: int = 32,
        input_ch: int = 7,
        output_ch: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        **kwargs,
    ):
        super(PointLightFieldMLP, self).__init__()
        self.device = device
        self.dtype = dtype

        self.W = W
        self.n_feat = n_feat
        self.input_ch = input_ch
        self.output_ch = output_ch

        self.spatial_MLP_fc1 = nn.Linear(self.input_ch, self.W).to(device).to(dtype)
        self.spatial_MLP_fc2 = nn.Linear(self.W, self.W).to(device).to(dtype)
        self.spatial_MLP_fc3 = nn.Linear(self.W, self.n_feat).to(device).to(dtype)

        self.directional_MLP_fc1 = nn.Linear(self.n_feat, self.W).to(device).to(dtype)
        self.directional_MLP_fc2 = nn.Linear(self.W, output_ch).to(device).to(dtype)

    def forward(
        self,
        pts: torch.Tensor,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        rx_to_tx_distance: torch.Tensor,
        rx_to_tx_azimuth: torch.Tensor,
        rx_to_tx_elevation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Take ray direction and point information, return rendered wireless channel
        Args:
            pts                      (torch.Tensor):                [1, n_pts, dim=3]

            ray_o                    (torch.Tensor):                [B, dim=3]

            rx_to_tx_distance        (torch.Tensor):                [B, 1]
            rx_to_tx_azimuth         (torch.Tensor):                [B, 1]
            rx_to_tx_elevation       (torch.Tensor):                [B, 1]

        Returns:
            torch.Tensor: rendered wireless channels with shape = [B, n_rays, 3] (TODO:)
        """
        x = torch.cat((ray_o, ray_d, rx_to_tx_distance), dim=-1)

        # Spatial MLP
        x = self.spatial_MLP_fc1(x)
        x = F.leaky_relu(x)
        x = self.spatial_MLP_fc2(x)
        x = F.leaky_relu(x)
        x = self.spatial_MLP_fc3(x)
        x = F.leaky_relu(x)  # [B, n_rays, 32]

        # Directional MLP
        x = self.directional_MLP_fc1(x)
        x = F.leaky_relu(x)
        x = self.directional_MLP_fc2(x)

        return x
