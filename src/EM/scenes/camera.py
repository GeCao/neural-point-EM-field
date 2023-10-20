import math
import torch
import torch.nn.functional as F
from typing import List


class Camera(object):
    def __init__(
        self,
        eye: torch.Tensor,
        up: torch.Tensor = None,
        lookat: torch.Tensor = None,
        aspect: float = 1.0,
        fov: float = 60,
        near: float = 0.01,
        far: float = 1000.0,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
    ) -> None:
        self.device = device
        self.dtype = dtype

        up = (
            up if up is not None else torch.Tensor([0.0, 1.0, 0.0]).to(dtype).to(device)
        )
        lookat = (
            lookat
            if lookat is not None
            else torch.Tensor([0.0, 0.0, -1.0]).to(dtype).to(device)
        )

        self.eye = eye
        self.up = up
        self.lookat = lookat
        self.aspect = aspect  # width / height
        self.fov = math.radians(fov)  # raidan, fov on y-axis
        self.near = near
        self.far = far

        self.P_mat = self.ConstructPerspectiveMatrix()
        self.V_mat = self.ConstructViewMatrix()

    def GetViewMatrix(self) -> torch.Tensor:
        if self.V_mat is None:
            self.V_mat = self.ConstructViewMatrix()

        return self.V_mat

    def GetPerspectiveMatrix(self) -> torch.Tensor:
        if self.P_mat is None:
            self.P_mat = self.ConstructPerspectiveMatrix()

        return self.P_mat

    def ConstructPerspectiveMatrix(self) -> torch.Tensor:
        """https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/projection-matrix-introduction.html
        We assume our vector is column based.
        """
        assert self.eye is not None
        assert self.up is not None
        assert self.lookat is not None

        fov_cot = 1.0 / math.tan(self.fov / 2.0)
        P_mat = (
            torch.Tensor(
                [
                    [fov_cot / self.aspect, 0, 0, 0],
                    [0, fov_cot, 0, 0],
                    [
                        0,
                        0,
                        -(self.far + self.near) / (self.far - self.near),
                        -2.0 * self.far * self.near / (self.far - self.near),
                    ],
                    [0, 0, -1.0, 0],
                ]
            )
            .to(self.device)
            .to(self.dtype)
        )

        return P_mat

    def ConstructViewMatrix(self) -> torch.Tensor:
        """https://learnopengl.com/Getting-started/Camera
        We assume our vector is column based.
        """
        assert self.eye is not None
        assert self.up is not None
        assert self.lookat is not None

        z_axis = F.normalize(self.eye - self.lookat, dim=-1)
        y_axis = F.normalize(self.up, dim=-1)
        x_axis = F.normalize(torch.cross(y_axis, z_axis, dim=-1), dim=-1)
        V_mat = (
            torch.Tensor(
                [
                    [x_axis[0], x_axis[1], x_axis[2], -x_axis.dot(self.eye)],
                    [y_axis[0], y_axis[1], y_axis[2], -y_axis.dot(self.eye)],
                    [z_axis[0], z_axis[1], z_axis[2], -z_axis.dot(self.eye)],
                    [0, 0, 0.0, 1.0],
                ]
            )
            .to(self.device)
            .to(self.dtype)
        )

        return V_mat

    def FrustumCulling(self, points: torch.Tensor) -> torch.Tensor:
        points = points.view(-1, 3)

        negz_axis = F.normalize(self.lookat - self.eye, dim=-1)
        y_axis = self.up  # Cannot always ensure indicated z, y ortho
        x_axis = F.normalize(torch.cross(y_axis, -negz_axis, dim=-1), dim=-1)
        y_axis = F.normalize(torch.cross(x_axis, negz_axis, dim=-1), dim=-1)

        # 1. Along with the -z axis, get all of the distance between eye and points
        negz_walk = ((points - self.eye) * negz_axis).sum(
            dim=-1, keepdim=False
        )  # [-1,]
        negz_proj_points = self.eye + negz_walk.unsqueeze(-1) * negz_axis  # [-1, 3]
        negz_distance_x = (
            ((points - negz_proj_points) * x_axis).sum(dim=-1, keepdim=False).abs()
        )  # [-1,]
        negz_distance_y = (
            ((points - negz_proj_points) * y_axis).sum(dim=-1, keepdim=False).abs()
        )  # [-1,]

        # 2. every z_distance affines a view plane
        half_height = negz_walk * math.tan(self.fov / 2.0)
        half_width = self.aspect * half_height

        mark_as_include = (
            (negz_distance_x <= half_width)
            * (negz_distance_y <= half_height)
            * (negz_walk > self.near)
            * (negz_walk < self.far)
        ).to(torch.bool)
        (include_index,) = torch.where(mark_as_include > 0)

        if include_index.shape[0] == 0:
            return None

        return include_index

    def FindKClosest(
        self,
        ray_o: torch.Tensor,
        ray_d: torch.Tensor,
        points: torch.Tensor,
        K_closest: int,
    ) -> List[torch.Tensor]:
        """See reference from fig 3. of https://arxiv.org/pdf/2112.01473.pdf"""
        points = points.view(1, -1, 3)
        ray_o = ray_o.view(-1, 1, 3)
        ray_d = ray_d.view(-1, 1, 3)

        n_pts = points.shape[-2]
        n_rays = ray_d.shape[0]

        negz_axis = F.normalize(self.lookat - self.eye, dim=-1)
        y_axis = self.up  # Cannot always ensure indicated z, y ortho
        x_axis = F.normalize(torch.cross(y_axis, -negz_axis, dim=-1), dim=-1)
        y_axis = F.normalize(torch.cross(x_axis, negz_axis, dim=-1), dim=-1)

        ray_cospitch = (F.normalize(points - ray_o, dim=-1) * ray_d).sum(
            dim=-1, keepdim=False
        )  # [nrays, npts,]
        ray_sinpitch = torch.sqrt(1.0 - ray_cospitch * ray_cospitch)  # [nrays, npts,]
        ray_distance = ((points - ray_o) * (points - ray_o)).sum(
            dim=-1, keepdim=False
        ) * ray_sinpitch  # [nrays, npts,]

        # Find topK
        index = None
        if K_closest < n_pts:
            _, topK_indices = torch.topk(
                input=ray_distance, k=K_closest, largest=False, dim=-1
            )  # [n_rays, n_K]
            index = (
                torch.linspace(
                    0, n_rays - 1, n_rays, device=ray_d.device, dtype=torch.int32
                )
                .unsqueeze(-1)
                .repeat(1, K_closest)
                .flatten()
                .cpu()
                .tolist(),
                topK_indices.flatten().cpu().tolist(),
            )
        else:
            raise RuntimeError(
                f"In your input, K-closest = {K_closest}, meanwhile number of points only {n_pts}"
            )

        topK_points = points.repeat(n_rays, 1, 1)[index].reshape(
            n_rays, K_closest, 3
        )  # [nrays, n_K, 3]
        topK_ray_walk = ((topK_points - ray_o) * ray_d).sum(
            dim=-1, keepdim=True
        )  # [nrays, n_K, 1]
        topK_ray_cospitch = (F.normalize(topK_points - ray_o, dim=-1) * ray_d).sum(
            dim=-1, keepdim=True
        )  # [nrays, n_K, 1]
        topK_ray_pitch = torch.acos(topK_ray_cospitch)  # [nrays, n_K, 1]
        topK_ray_sinpitch = torch.sqrt(1.0 - topK_ray_cospitch * topK_ray_cospitch)
        topK_ray_distance = ((topK_points - ray_o) * (topK_points - ray_o)).sum(
            dim=-1, keepdim=True
        ) * topK_ray_sinpitch

        # TODO: this is not as same as the original code
        topK_ray_up = y_axis - torch.sum(y_axis * ray_d, axis=-1, keepdim=True) * ray_d
        topK_ray_up = F.normalize(topK_ray_up, dim=-1)
        topK_ray_to_points_vector = topK_points - (ray_o + topK_ray_walk * ray_d)
        topK_ray_to_points_vector_projected = F.normalize(
            topK_ray_to_points_vector
            - torch.sum(topK_ray_to_points_vector * ray_d, axis=-1, keepdim=True)
            * ray_d,
            dim=-1,
        )
        topK_ray_azimuth = torch.sum(
            topK_ray_to_points_vector_projected * topK_ray_up, dim=-1, keepdim=True
        )

        return [
            topK_indices.unsqueeze(-1),
            topK_points,
            topK_ray_distance,
            topK_ray_walk,
            topK_ray_azimuth,
            topK_ray_pitch,
        ]
