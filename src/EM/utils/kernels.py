from abc import ABC, abstractmethod
import torch


class AbstractKernel(ABC):
    @property
    def dyadic(self):
        raise NotImplementedError

    @abstractmethod
    def weight(self, normalized_distance: torch.Tensor):
        ...


class Linear(AbstractKernel):
    dyadic = True

    def __init__(self, support_radius: torch.tensor):
        # divided by two to match the function in p2gg2p common process
        self.support_radius = support_radius

        # normalization factor
        h = self.support_radius
        self.normalization_factor = 1 / h

    def weight(self, distance: torch.Tensor) -> torch.Tensor:
        """Compute the weight of a distance vector
            or it can be done by each channel as described here:
                https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
                Page: 32

        Args:
            distance (torch.Tensor): B x nPoints x 1/2/3

        Returns:
            torch.Tensor: The weight of the weight per channel
        """
        normalized_distance = distance / self.support_radius
        x_abs = torch.abs(normalized_distance)
        return torch.where(
            x_abs >= 1.0,
            torch.zeros_like(normalized_distance),
            self.normalization_factor * (1.0 - x_abs),
        )

    __call__ = weight


class Quadratic(AbstractKernel):
    dyadic = True

    def __init__(
        self,
        support_radius: torch.tensor,
    ):
        # always use 2 cell to splat the values to
        self.max_distance = 1.5

        # divided by two to match the function in p2gg2p common process
        self.support_radius = support_radius

        # normalization factor
        h = self.support_radius / self.max_distance
        self.normalization_factor = 1 / h

    def weight(self, distance: torch.Tensor) -> torch.Tensor:
        """Compute the weight of a distance. This can either be a distance vector
            or it can be done by each channel as described here:
                https://www.seas.upenn.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
                Page: 32

        Args:
            distance (torch.Tensor): B x nPoints x 1/2/3

        Returns:
            torch.Tensor: The weight of the distance per channel
        """
        normalized_distance = distance / self.support_radius * self.max_distance
        # run per channel version
        x_abs = torch.abs(normalized_distance)
        return torch.where(
            x_abs >= self.max_distance,
            torch.zeros_like(normalized_distance),
            torch.where(
                x_abs >= self.max_distance / 3,
                self.normalization_factor * (1.0 / 2.0 * (3.0 / 2.0 - x_abs) ** 2),
                self.normalization_factor * (3.0 / 4.0 - x_abs**2),
            ),
        )

    __call__ = weight
