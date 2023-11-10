from typing import Union, Optional, Tuple
from src.EM.utils import AbstractKernel
import torch
import torch.utils.checkpoint
from torch import nn
import math
import numpy as np


def get_grid_shape(cell_size, bbox):
    bbox_diff = bbox[..., 1] - bbox[..., 0]
    grid_shape = (bbox_diff / cell_size).round().int().squeeze()
    return grid_shape


def flatten_index(index, grid_shape):
    """Converts a nd index to a flat (linear) index that can index a flatten view of a tensor"""

    grid_shape = torch.as_tensor(grid_shape)
    ndim = index.shape[-1]

    index_flat = index[..., -1].clone().to(torch.int64)
    for d in range(ndim - 1):
        index_flat += index[..., d] * torch.prod(grid_shape[d + 1 :])
    return index_flat


# TODO fix kernel inconsistencies (e.g. CubicSPH compute the norm, dyadic abs, Linear batch_size??)
# TODO optimize points grouping and chunk size
class Splatter(nn.Module):
    _valid_grad_ckpt = ["off", "kernel", "scatter"]

    def __init__(
        self,
        kernel: AbstractKernel,
        cell_size: Union[torch.Tensor, float, np.ndarray],
        normalization: bool = True,
        ndim: int = 3,
        grad_ckpt: str = "off",
        chunk_size: Optional[int] = 2**14,
    ):
        """Converts points to grid

        Args:
            kernel: kernel instance with radius set in grid coord
            cell_size:size of cells/voxels in the generate grid
            normalization: Flag if the data should be normalized
            ndim: number of dimensions, 2D / 3D
            grad_ckpt: Whether gradient checkpointing should be used to save memory (increases computation time).
                Options are ["off", "kernel", "scatter"], scatter enables checkpoint for both, kernels computation and
                 scattering onto the grid.
            chunk_size: How many points are processed at once.
        """
        super().__init__()

        if grad_ckpt not in self._valid_grad_ckpt:
            raise ValueError(
                f"Invalid 'grad_ckpt', expected one of {self._valid_grad_ckpt}, got: {grad_ckpt}"
            )

        self.kernel = kernel
        self.cell_size = cell_size
        self.normalization = normalization
        self.ndim = ndim
        self.grad_ckpt = grad_ckpt
        self.chunk_size: torch.Tensor = chunk_size
        self._init_offset()

    def _init_offset(self):
        # points are at most 0.5 away from a grid node
        self.kernel_radius = math.ceil(self.kernel.support_radius - 0.5)

        self.offsets = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(
                        -self.kernel_radius,
                        self.kernel_radius + 1,
                        dtype=torch.int,
                    )
                    for _ in range(self.ndim)
                ]
            ),
            dim=-1,
        )

    def _compute_kernel_weigths(self, scatter_index, points_grid):
        batch_size = scatter_index.shape[0]

        if self.kernel.dyadic:
            mid = scatter_index.shape[1] // 2
            mid = slice(mid, mid + 1)

            if self.ndim == 3:
                d1 = scatter_index[:, mid, mid, :, :, 2:] - points_grid[..., 2:]
                w1 = self.kernel(d1)
                d2 = scatter_index[:, mid, :, mid, :, 1:2] - points_grid[..., 1:2]
                w2 = self.kernel(d2)
                d3 = scatter_index[:, :, mid, mid, :, 0:1] - points_grid[..., 0:1]
                w3 = self.kernel(d3)
                weights = w3 * w2 * w1
            elif self.ndim == 2:
                d1 = scatter_index[:, mid, :, :, 1:] - points_grid[..., 1:]
                w1 = self.kernel(d1)
                d2 = scatter_index[:, :, mid, :, 0:1] - points_grid[..., 0:1]
                w2 = self.kernel(d2)
                weights = w2 * w1
            else:
                raise NotImplementedError(
                    f"dyadic kernel splatting in {self.ndim}D not implemented."
                )
        else:
            # distance = torch.norm(scatter_index - points_grid, dim=-1, keepdim=True)
            # weights = self.kernel(distance)
            weights = self.kernel(scatter_index - points_grid)

        return weights.view(batch_size, -1, 1)

    def _compute_flat_scatter_values(self, points, densities, grid_shape, bbox):
        # TODO consider partially moving to a new method:
        #   AbstractKernel.weights(points, cell_size, extent=None)
        # that computes offsets and return weights around each points
        # or AbstractKernel.weights(points, offsets)
        # + special override in CubicDyadic to leverage linear decomposition property

        self.offsets = self.offsets.to(points.device)

        ndim = points.shape[-1]
        batch_size = points.shape[0]
        # convert points to grid coords
        points_grid = (
            (points - bbox[:1][None] - self.cell_size / 2)
            / (bbox[1:][None] - bbox[:1][None])
            * grid_shape[None]
        )

        points_grid_int = points_grid.round().to(torch.int32)

        # use broadcast semantic to compute position of kernel vals around each point
        # [batch, kernel_shape[0], kernel_shape[1], ..., n_points, ndim]
        points_grid_int = points_grid_int.view(
            points_grid_int.shape[0:1] + (1,) * ndim + points_grid_int.shape[1:]
        )
        points_grid = points_grid.view(
            points_grid.shape[0:1] + (1,) * ndim + points_grid.shape[1:]
        )
        densities = densities.view(
            densities.shape[0:1] + (1,) * ndim + densities.shape[1:]
        )

        scatter_index = points_grid_int + self.offsets[None, ..., None, :]
        densities = densities.expand(scatter_index.shape[:-1] + (1,)).reshape(
            batch_size, -1
        )
        weights = self._compute_kernel_weigths(scatter_index, points_grid)
        scatter_index = scatter_index.view(batch_size, -1, ndim)
        scatter_values = densities * weights[..., 0]
        scatter_index_flat = flatten_index(scatter_index, grid_shape)

        return scatter_values, weights[..., 0], scatter_index_flat

    def _compute_scattered_update(self, points, densities, grid_shape, bbox):
        scatter_values, weights, scatter_index_flat = self._compute_flat_scatter_values(
            points,
            densities,
            grid_shape,
            bbox,
        )
        batch_size = points.shape[0]
        grid = torch.zeros(
            [batch_size, torch.prod(grid_shape)],
            dtype=points.dtype,
            device=points.device,
        )
        grid_accumulated_weights = torch.zeros(
            [batch_size, torch.prod(grid_shape)],
            dtype=points.dtype,
            device=points.device,
        )

        grid.scatter_add_(
            dim=1,
            index=scatter_index_flat,
            src=scatter_values,
        )
        grid_accumulated_weights.scatter_add_(
            dim=1,
            index=scatter_index_flat,
            src=weights,
        )

        return grid, grid_accumulated_weights

    def _pad(self, bbox):
        bbox = bbox.clone()
        bbox[:, :, 0] -= (self.kernel_radius + 1) * self.cell_size
        bbox[:, :, 1] += (self.kernel_radius + 1) * self.cell_size

        return bbox

    def _trim(self, grid):
        if self.kernel_radius > 0:
            mid = slice(self.kernel_radius + 1, -(self.kernel_radius + 1))
            grid = grid[(slice(None), slice(None)) + (mid,) * self.ndim]

        return grid

    def forward(
        self,
        points: torch.Tensor,
        densities: torch.Tensor,
        bbox: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bbox = self._pad(bbox)
        grid_shape = get_grid_shape(self.cell_size, bbox)
        grid_shape = grid_shape.flip(dims=(0,))

        # TODO build the bbox in the right order in the first place (incompatible with current code, doc suggest otherwise)
        bbox = bbox[0].permute((1, 0)).flip(dims=(1,))

        chunk_size = points.shape[1] if self.chunk_size is None else self.chunk_size

        points = points.flip(dims=(-1,))
        batch_size = points.shape[0]

        grid = torch.zeros(
            [batch_size, torch.prod(grid_shape)],
            dtype=points.dtype,
            device=points.device,
        )
        grid_accumulated_weights = torch.zeros(
            [batch_size, torch.prod(grid_shape)],
            dtype=points.dtype,
            device=points.device,
        )

        for points_chunk, densities_chunk in zip(
            points.split(chunk_size, dim=1),
            densities.split(chunk_size, dim=1),
        ):
            if self.training and self.grad_ckpt == "scatter":
                (
                    grid_update,
                    grid_accumulated_weights_update,
                ) = torch.utils.checkpoint.checkpoint(
                    self._compute_scattered_update,
                    points_chunk,
                    densities_chunk,
                    grid_shape,
                    bbox,
                    preserve_rng_state=False,
                )
                grid += grid_update
                grid_accumulated_weights += grid_accumulated_weights_update
            else:
                if not self.training or self.grad_ckpt == "off":
                    (
                        scatter_values,
                        weights,
                        scatter_index_flat,
                    ) = self._compute_flat_scatter_values(
                        points_chunk,
                        densities_chunk,
                        grid_shape,
                        bbox,
                    )
                else:
                    (
                        scatter_values,
                        weights,
                        scatter_index_flat,
                    ) = torch.utils.checkpoint.checkpoint(
                        self._compute_flat_scatter_values,
                        points_chunk,
                        densities_chunk,
                        grid_shape,
                        bbox,
                        preserve_rng_state=False,
                    )

                grid.scatter_add_(dim=1, index=scatter_index_flat, src=scatter_values)
                grid_accumulated_weights.scatter_add_(
                    dim=1, index=scatter_index_flat, src=weights
                )

        if self.normalization:
            grid = torch.where(
                grid_accumulated_weights > 0,
                grid / (grid_accumulated_weights + 1e-9),
                grid,
            )

        grid = grid.view([batch_size, 1, *grid_shape])
        grid_accumulated_weights = grid_accumulated_weights.view(
            [batch_size, 1, *grid_shape]
        )

        grid = self._trim(grid)
        grid_accumulated_weights = self._trim(grid_accumulated_weights)

        return grid, grid_accumulated_weights
