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
        scalar_index = [0, 2]
        angular_index = [1, 3, 4, 5, 6]
        bool_index = [7]  # TODO: Take this as punishment loss design

        x = x.view(1, -1, x.shape[-1])
        y = y.view(1, -1, y.shape[-1])

        x = x[..., :-1] * x[..., -1:]
        y = y[..., :-1] * y[..., -1:]

        # TODO: Try PSNR instead of a simple MSE
        loss = self.mse_loss(x[..., 0], y[..., 0])
        # loss += self.mse_loss(x[..., 2], y[..., 2])
        # loss += (1.0 - torch.cos(x[..., 1] - y[..., 1])).mean()  # phase

        if x.shape[-1] > 3:
            dir_x = torch.stack(
                (
                    torch.cos(x[..., 3]) * torch.sin(x[..., 4]),
                    torch.sin(x[..., 3]) * torch.sin(x[..., 4]),
                    torch.cos(x[..., 4]),
                ),
                dim=-1,
            ).view(-1, 3)
            dir_y = torch.stack(
                (
                    torch.cos(y[..., 3]) * torch.sin(y[..., 4]),
                    torch.sin(y[..., 3]) * torch.sin(y[..., 4]),
                    torch.cos(y[..., 4]),
                ),
                dim=-1,
            ).view(-1, 3)
            loss += self.cosine_distance_loss(
                dir_x,
                dir_y,
                torch.ones((dir_x.shape[0],), device=dir_x.device, dtype=dir_x.dtype),
            )
            dir_x = torch.stack(
                (
                    torch.cos(x[..., 5]) * torch.sin(x[..., 6]),
                    torch.sin(x[..., 5]) * torch.sin(x[..., 6]),
                    torch.cos(x[..., 6]),
                ),
                dim=-1,
            ).view(-1, 3)
            dir_y = torch.stack(
                (
                    torch.cos(y[..., 5]) * torch.sin(y[..., 6]),
                    torch.sin(y[..., 5]) * torch.sin(y[..., 6]),
                    torch.cos(y[..., 6]),
                ),
                dim=-1,
            ).view(-1, 3)
            loss += self.cosine_distance_loss(
                dir_x,
                dir_y,
                torch.ones((dir_x.shape[0],), device=dir_x.device, dtype=dir_x.dtype),
            )

            # TODO: Try entroy
            # loss += self.mse_loss(x[..., 7:8], y[..., 7:8])

        return loss
