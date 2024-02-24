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

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, valid_index: torch.Tensor = None
    ) -> torch.Tensor:
        # Note inf (obs) points are un-relevant
        scalar_index = [0, 2]
        angular_index = [1, 3, 4, 5, 6]
        bool_index = [8]  # TODO: Take this as punishment loss design

        if x.shape[-1] == 1 and y.shape[-1] == 1:
            valid = y.abs() > 0.01
            loss = self.mse_loss(x[valid], y[valid])

            return loss

        if x.shape[-1] == 4 and y.shape[-1] == 4:
            valid = y.abs() > 0.01
            loss = self.mse_loss(x[valid], y[valid])

            return loss

        if len(x.shape) == 3 and len(y.shape) == 3:
            assert x.shape[-1] == 6
            assert y.shape[-1] == 5
            x = x[valid_index.repeat(1, 1, x.shape[-1])].view(-1, x.shape[-1])
            y = y[valid_index.repeat(1, 1, y.shape[-1])].view(-1, y.shape[-1])

        loss = 0.0
        # loss += F.mse_loss(x[..., 0], torch.pow(10, y[..., 0] / 20.0))
        loss_gain = F.mse_loss(x[..., 0], y[..., 0])
        loss_time = F.mse_loss(x[..., 2], y[..., 2])
        # loss += (1.0 - torch.cos(x[..., 1] - y[..., 1])).mean()
        dir_x = x[..., -3:].view(-1, 3)
        dir_y = torch.stack(
            (
                torch.cos(y[..., 3]) * torch.sin(y[..., 4]),
                torch.sin(y[..., 3]) * torch.sin(y[..., 4]),
                torch.cos(y[..., 4]),
            ),
            dim=-1,
        ).view(-1, 3)
        dir_loss_func = nn.CosineSimilarity(dim=1)
        loss_angles = (1.0 - dir_loss_func(dir_x, dir_y)).mean()

        loss = loss_gain + loss_time + loss_angles
        # print("loss gain, time, angles = ", loss_gain, loss_time, loss_angles)

        return loss


class GainLoss(nn.Module):
    def __init__(self):
        super(GainLoss, self).__init__()
        self.cosine_distance_loss = CosineEmbeddingLoss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.entroy = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        valid = y.abs() > 0.01
        loss = self.mse_loss(x[valid], y[valid])

        return loss
