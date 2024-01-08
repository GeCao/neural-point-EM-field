import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CosineEmbeddingLoss


class ChannelLoss(nn.Module):
    def __init__(self):
        super(ChannelLoss, self).__init__()
        self.cosine_distance_loss = CosineEmbeddingLoss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.entroy = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Note inf (obs) points are un-relevant
        valid = y.abs() > 0.01

        # TODO: Try PSNR instead of a simple MSE
        loss = self.mse_loss(x[valid], y[valid])

        return loss
