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

        self.P_mat = None
        self.V_mat = None

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

    def FrustumCulling(
        self, points: torch.Tensor, ray_o: torch.Tensor, ray_d: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            points   (torch.Tensor): [n_pts, 3]
            ray_o    (torch.Tensor): [n_rays, 3]
            ray_d    (torch.Tensor): [n_rays, 3]

        Returns:
            torch.Tensor: TODO
        """
        device = points.device
        dtype = points.dtype
        points = points.view(1, -1, 3)
        ray_o = ray_o.view(-1, 1, 3)
        ray_d = ray_d.view(-1, 1, 3)
        fov = math.radians(10)
        near = 0.1
        far = 1000
        aspect = 1.0

        negz_axis = ray_d
        y_axis = torch.Tensor([[[0.0, 1.0, 0.0]]]).to(device).to(dtype)
        x_axis = F.normalize(torch.cross(y_axis, -negz_axis, dim=-1), dim=-1)
        y_axis = F.normalize(torch.cross(x_axis, negz_axis, dim=-1), dim=-1)

        # 1. Along with the -z axis, get all of the distance between eye and points
        negz_walk = ((points - ray_o) * negz_axis).sum(
            dim=-1, keepdim=True
        )  # [n_rays, n_pts, 1]
        negz_proj_points = ray_o + negz_walk * negz_axis  # [n_rays, n_pts, 3]
        negz_distance_x = (
            ((points - negz_proj_points) * x_axis).sum(dim=-1, keepdim=True).abs()
        )  # [n_rays, n_pts, 1]
        negz_distance_y = (
            ((points - negz_proj_points) * y_axis).sum(dim=-1, keepdim=True).abs()
        )  # [n_rays, n_pts, 1]

        # 2. every z_distance affines a view plane
        half_height = negz_walk * math.tan(fov / 2.0)
        half_width = aspect * half_height

        mark_as_include = (
            (
                (negz_distance_x <= half_width)
                * (negz_distance_y <= half_height)
                * (negz_walk > near)
                * (negz_walk < far)
            )
            .squeeze(-1)
            .to(torch.bool)
        )

        return mark_as_include

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

        negz_axis = ray_d
        y_axis = self.up.reshape(1, 1, 3)  # Cannot always ensure indicated z, y ortho
        x_axis = F.normalize(torch.cross(y_axis, -negz_axis, dim=-1), dim=-1)
        y_axis = F.normalize(torch.cross(x_axis, negz_axis, dim=-1), dim=-1)

        ray_cosphi = (F.normalize(points - ray_o, dim=-1) * ray_d).sum(
            dim=-1, keepdim=False
        )  # [nrays, npts,]
        ray_sinphi = torch.sqrt(1.0 - ray_cosphi * ray_cosphi)  # [nrays, npts,]
        ray_distance = ((points - ray_o) * (points - ray_o)).sum(
            dim=-1, keepdim=False
        ).sqrt() * ray_sinphi  # [nrays, npts,]

        # Exert Frustum Culling
        in_frustum_mask = self.FrustumCulling(points=points, ray_o=ray_o, ray_d=ray_d)
        assert len(in_frustum_mask.shape) == 2

        sky_mask = (in_frustum_mask.sum(dim=1) < 1).to(torch.bool)
        ray_distance = torch.cross(points - ray_o, ray_d, dim=-1).norm(
            dim=-1
        )  # [nrays, npts,]

        # Find topK
        index = None
        if K_closest < n_pts:
            topK_ray_distance, topK_indices = torch.topk(
                input=ray_distance, k=K_closest, largest=False, dim=-1
            )  # [n_rays, n_K]
            topK_indices = topK_indices.to(torch.device("cpu"))
            index = (
                torch.linspace(
                    0, n_rays - 1, n_rays, device=torch.device("cpu"), dtype=torch.int32
                )
                .unsqueeze(-1)
                .repeat(1, K_closest)
                .flatten()
                .tolist(),
                topK_indices.flatten().tolist(),
            )
        else:
            raise RuntimeError(
                f"In your input, K-closest = {K_closest}, meanwhile number of points only {n_pts}"
            )

        topK_points = points.repeat(n_rays, 1, 1)[index].reshape(
            n_rays, K_closest, 3
        )  # [nrays, n_K, 3]
        topK_ray_cosphi = (F.normalize(topK_points - ray_o, dim=-1) * ray_d).sum(
            dim=-1, keepdim=True
        )  # [nrays, n_K, 1]
        topK_ray_cosphi = torch.clamp(topK_ray_cosphi, -1.0, 1.0)
        topK_ray_sinphi = torch.sqrt(1.0 - topK_ray_cosphi * topK_ray_cosphi)
        topK_ray_distance = ((topK_points - ray_o) * (topK_points - ray_o)).sum(
            dim=-1, keepdim=True
        ) * topK_ray_sinphi

        # Take the formula from reference
        topK_y = y_axis - (y_axis * ray_d).sum(dim=-1, keepdim=True) * ray_d
        topK_y = F.normalize(topK_y, dim=-1)  # [n_rays, 1, 3]
        topK_proj_coord_x = (topK_y * topK_points).sum(dim=-1, keepdim=True)
        topK_proj_coord_y = (ray_d.cross(topK_y, dim=-1) * topK_points).sum(
            dim=-1, keepdim=True
        )
        # topK_ray_proj_distance = torch.sqrt(
        #     topK_proj_coord_x * topK_proj_coord_x
        #     + topK_proj_coord_y * topK_proj_coord_y
        # )
        topK_ray_proj_distance = ((topK_points - ray_o) * (topK_points - ray_o)).sum(
            dim=-1, keepdim=True
        ) * topK_ray_cosphi
        topK_ray_azimuth = torch.arctan(topK_proj_coord_x / topK_proj_coord_y)
        topK_ray_pitch = torch.arccos(
            (ray_d * F.normalize(topK_points, dim=-1)).sum(dim=-1, keepdim=True)
        )

        return [
            topK_indices.unsqueeze(-1),
            topK_ray_distance,
            topK_ray_proj_distance,
            topK_ray_azimuth,
            topK_ray_pitch,
            sky_mask,
        ]
