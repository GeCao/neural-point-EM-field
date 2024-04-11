import os
import json
import numpy as np
import torch
import pywavefront
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from PIL import Image
from typing import List
from plyfile import PlyData, PlyElement
from pytorch3d.renderer import Textures
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def LoadMeshes(
    data_path: str, obj_folder: str = "objs", device: torch.device = torch.device("cpu"), dtype=torch.float32
) -> List[torch.Tensor]:
    """Load mesh data from disk
    We will typically load mesh from json file where,
        'verts'--------[NumOfVertices, 3]
        'faces'--------[NumOfFaces, 3]
        'edges'--------[NumOfEdges, 2, 3]
        'verts_rgba'---[1, NumofVertices, 4]
        'materials'----{NumOfFaces->str}
        'visible'------{NumOfFaces->bool}
    are given.
    """
    is_sionna = "sionna" in data_path
    obj_path = os.path.join(data_path, obj_folder)
    if not os.path.exists(obj_path):
        raise RuntimeError(
            f"Mesh object path :{obj_path} not found, check your DataSet"
        )
    data_files = os.listdir(obj_path)
    data_files = sorted(data_files)

    vertices = []
    faces = []
    edges = []
    verts_rgba = []
    if not is_sionna:
        for filename in data_files:
            if filename[-5:] == ".json" and "1" in filename:
                json_path = os.path.join(obj_path, filename)
                with open(json_path, "r") as load_f:
                    json_data = json.load(load_f)
                    new_v = torch.Tensor(json_data["verts"]).to(device).to(dtype)
                    new_f = torch.Tensor(json_data["faces"]).to(device).to(torch.int32)
                    new_e = torch.Tensor(json_data["edges"]).to(device).to(dtype)
                    new_vcolor = (
                        torch.Tensor(json_data["verts_rgba"]).to(device).to(dtype)
                    )
                    vertices.append(new_v)
                    faces.append(new_f)
                    edges.append(new_e)
                    verts_rgba.append(new_vcolor.reshape(-1, 4))

                load_f.close()
                break
            elif filename[-4:] == ".obj" and "1" in filename:
                json_path = os.path.join(obj_path, filename)
                scene = pywavefront.Wavefront(json_path, collect_faces=True)

                new_v = torch.Tensor(scene.vertices).to(device).to(dtype)
                new_f = (
                    torch.Tensor(scene.meshes[None].faces).to(device).to(torch.int32)
                )
                vertices.append(new_v)
                faces.append(new_f)
                break
    else:
        for filename in data_files:
            if filename[-4:] == ".obj":
                json_path = os.path.join(obj_path, filename)
                scene = pywavefront.Wavefront(json_path, collect_faces=True)

                new_v = torch.Tensor(scene.vertices).to(device).to(dtype)
                new_f = (
                    torch.Tensor(scene.meshes[None].faces).to(device).to(torch.int32)
                )

                new_v = new_v.reshape(1, -1, 3)
                new_f = new_f.reshape(1, -1, 3)

                if len(vertices) == 0:
                    vertices = new_v
                    faces = new_f
                else:
                    vert_offset = vertices.shape[-2]
                    vertices = torch.cat((vertices, new_v), dim=-2)
                    faces = torch.cat((faces, new_f + vert_offset), dim=-2)
            elif filename[-4:] == ".ply":
                json_path = os.path.join(obj_path, filename)
                ply_data = o3d.io.read_triangle_mesh(json_path)
                new_v = np.asarray(ply_data.vertices)
                new_f = np.asarray(ply_data.triangles)

                new_v = new_v.reshape(1, -1, 3)
                new_f = new_f.reshape(1, -1, 3)

                new_v = torch.from_numpy(new_v).to(dtype).to(device)
                new_f = torch.from_numpy(new_f).to(torch.int32).to(device)

                if len(vertices) == 0:
                    vertices = new_v
                    faces = new_f
                else:
                    vert_offset = vertices.shape[-2]
                    vertices = torch.cat((vertices, new_v), dim=-2)
                    faces = torch.cat((faces, new_f + vert_offset), dim=-2)

    if type(vertices) == type([]):
        vertices = torch.cat(vertices, dim=-2).reshape(1, -1, 3)
        faces = torch.cat(faces, dim=-2).to(torch.int64).reshape(1, -1, 3)

    return [vertices, faces]


def LoadPointCloudFromMesh(meshes: Meshes, num_pts_samples: int) -> torch.Tensor:
    point_clouds, normals = sample_points_from_meshes(
        meshes, num_samples=num_pts_samples, return_normals=True
    )  # [F, NumOfSamples, 3]
    return point_clouds


def export_asset(save_path: str, vertices: torch.Tensor, faces: torch.Tensor):
    np_faces = faces.reshape(-1, 3).to(torch.int).cpu().numpy()
    np_vertices = vertices.reshape(-1, 3).cpu().numpy()
    if np_faces.min() == 0:
        np_faces = np_faces + 1
    with open(save_path, "w") as f:
        f.write("# OBJ file\n")
        for i in range(np_vertices.shape[0]):
            f.write(
                "v {} {} {}\n".format(
                    np_vertices[i, 0], np_vertices[i, 1], np_vertices[i, 2]
                )
            )
        for j in range(np_faces.shape[0]):
            f.write(
                "f {} {} {}\n".format(np_faces[j, 0], np_faces[j, 1], np_faces[j, 2])
            )
    f.close()


def DumpCFGFile(save_path: str, point_clouds: torch.Tensor):
    point_clouds = point_clouds.reshape(-1, 3)
    bounding_box = [
        point_clouds.min(dim=0)[0].reshape(3),
        point_clouds.max(dim=0)[0].reshape(3),
    ]
    point_clouds = point_clouds + bounding_box[0].unsqueeze(0)
    point_clouds = point_clouds * 100
    bounding_box = [
        point_clouds.min(dim=0)[0].reshape(3),
        point_clouds.max(dim=0)[0].reshape(3),
    ]
    with open(save_path, "w+") as fp:
        fp.write(f"Number of particles = {point_clouds.shape[0]}\n")
        fp.write("A = 1 Angstrom (basic length-scale)\n")
        fp.write(f"H0(1,1) = {bounding_box[1][0] - bounding_box[0][0]} A\n")
        fp.write(f"H0(1,2) = {0} A\n")
        fp.write(f"H0(1,3) = {0} A\n")
        fp.write(f"H0(2,1) = {0} A\n")
        fp.write(f"H0(2,2) = {bounding_box[1][1] - bounding_box[0][1]} A\n")
        fp.write(f"H0(2,3) = {0} A\n")
        fp.write(f"H0(3,1) = {0} A\n")
        fp.write(f"H0(3,2) = {0} A\n")
        fp.write(f"H0(3,3) = {bounding_box[1][2] - bounding_box[0][2]} A\n")
        fp.write(".NO_VELOCITY.\n")
        fp.write(f"entry_count = {3}\n")

        mass = 1.0
        floor_height = point_clouds[:, 2].min()
        for i in range(point_clouds.shape[0]):
            particle_type = "H"
            if point_clouds[i, 2] <= floor_height + 0.01:
                particle_type = "F"
            fp.write(
                f"{mass}\n{particle_type}\n{point_clouds[i, 0].item()} {point_clouds[i, 1].item()} {point_clouds[i, 2].item()}\n"
            )

    fp.close()


def DumpGrayFigureToRGB(
    save_path: str,
    pred_color: torch.Tensor,
    gt_color: torch.Tensor = None,
    mask: torch.Tensor = None,
    extra_spot: torch.Tensor = None,
) -> None:
    eps = 1e-6
    if len(pred_color.shape) == 4:
        _, _, H, W = pred_color.shape
    elif len(pred_color.shape) == 3 and pred_color.shape[-1] == 1:
        H, W, _ = color.shape
    elif len(pred_color.shape) == 3 and pred_color.shape[0] == 1:
        _, H, W = pred_color.shape
    elif len(pred_color.shape) == 2:
        H, W = pred_color.shape
    else:
        raise RuntimeError(
            f"Can not recognize input gray color with shape {pred_color.shape}"
        )

    blank_wid = 5
    if extra_spot is not None:
        extra_spot = extra_spot.reshape(-1, 3)[:, 0:2] * max(H, W)

    pred_color = pred_color.reshape(H, W)
    if gt_color is not None:
        gt_color = gt_color.reshape(H, W)

    if mask is not None:
        mask = mask.reshape(H, W).to(torch.bool)
        pred_color[mask] = 0.0
        if gt_color is not None:
            gt_color[mask] = 0.0

    if gt_color is not None:
        blank = torch.zeros((H, blank_wid)).to(pred_color.device).to(pred_color.dtype)
        color = torch.cat((pred_color, blank, gt_color), dim=-1)

    color[color == 0.0] = np.inf

    aspect = int((W * 2 + 5) / H)
    fig, ax = plt.subplots(1, 1, figsize=(aspect * 6, 6))
    plt.pcolormesh(color.cpu().numpy())
    plt.title("Prediction - Ground truth")
    if extra_spot is not None:
        x, y = (
            extra_spot[:, 0].cpu().numpy(),
            extra_spot[:, 1].cpu().numpy(),
        )
        plt.scatter(x, y, c="red", marker="x")

        x, y = (
            (extra_spot[:, 0] + W + blank_wid).cpu().numpy(),
            extra_spot[:, 1].cpu().numpy(),
        )
        plt.scatter(x, y, c="red", marker="x")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

    print("pred = ", pred_color.min(), pred_color.max(), pred_color.mean())
    print("gt = ", gt_color.min(), gt_color.max(), gt_color.mean())

    save_dir = os.path.dirname(save_path)
    zero_mask = (gt_color.abs() < 1e-6).reshape(H, W)
    pred_image = SaveGrayTensorToColor(
        save_path=os.path.join(save_dir, "pred_time.png"),
        x=pred_color,
        colormap="Greens",
        inf_masks=[mask, zero_mask],
        y_flip=True,
        ignore_zeros=True,
    )
    gt_image = SaveGrayTensorToColor(
        save_path=os.path.join(save_dir, "gt_time.png"),
        x=gt_color,
        colormap="Greens",
        inf_masks=[mask, zero_mask],
        y_flip=True,
        ignore_zeros=True,
    )

    pred_image[pred_image == np.inf] = 0
    gt_image[gt_image == np.inf] = 0
    pred_mse = ((pred_image - gt_image) ** 2).mean()
    f_max = gt_image.max()
    pred_psnr = 20 * np.log10(f_max / np.sqrt(pred_mse))
    print(f"report MSE = {pred_mse}, psnr = {pred_psnr}")


def SaveGrayTensorToColor(
    save_path: str,
    x: torch.Tensor,
    colormap: str = "rainbow",
    inf_masks: List[torch.Tensor] = [],
    y_flip=True,
    ignore_zeros: bool = True,
) -> torch.Tensor:
    cm = plt.get_cmap(colormap)

    x = NormalizeTensor(x, ignore_zeros=ignore_zeros, keep_vacancy=False)
    x = x.squeeze()  # [(D), H, W]
    if len(x.shape) > 3 or len(x.shape) < 2:
        raise RuntimeError(
            f"Mapping gray image to color, the maximum valid shape is [(D), H, W]"
            f"However, your input dimension is [{x.shape}]"
        )
    elif len(x.shape) == 3:
        x = x.mean(dim=0)  # [H, W]

    np_x = x.cpu().numpy()  # [H, W]
    if y_flip:
        np_x = np.flip(np_x, axis=0)  # [H, W]

    np_x = cm(np_x) * 255.0  # [H, W, 4]
    result_x = np_x / 255.0
    H, W, n_ch = np_x.shape
    for inf_mask in inf_masks:
        if isinstance(inf_mask, torch.Tensor):
            inf_mask = inf_mask.squeeze()
            if len(inf_mask.shape) == 3:
                inf_mask = inf_mask.mean(dim=0)  # [H, W]

            assert inf_mask.shape[0] == H
            assert inf_mask.shape[1] == W
            assert len(inf_mask.shape) == 2
            inf_mask = inf_mask.unsqueeze(-1).repeat(1, 1, n_ch)
            np_inf_mask = inf_mask.cpu().numpy()

            if y_flip:
                np_inf_mask = np.flip(np_inf_mask, axis=0)
        elif inf_mask is None:
            continue
        else:
            raise RuntimeError("We only accept format: torch.Tensor as input mask")

        np_x[np_inf_mask] = np.inf
        result_x[np_inf_mask] = 0

    Image.fromarray((np_x).astype(np.uint8)).save(save_path)

    return result_x


def NormalizeTensor(
    x: torch.Tensor,
    min_val: float = 0,
    max_val: float = 1,
    ignore_zeros: bool = True,
    keep_vacancy: bool = False,
):
    vacancy_mask = ~torch.isfinite(x)
    x[vacancy_mask] = x.mean()

    zero_mask = None
    if ignore_zeros:
        zero_mask = x.abs() < 1e-8
        x[zero_mask] = x.mean()

    x = (x - x.min()) / (x.max() - x.min())  # -> min 0, max 1
    x = x * (max_val - min_val) + min_val

    if zero_mask is not None:
        x[zero_mask] = 0

    if keep_vacancy:
        x[vacancy_mask] = torch.inf
    else:
        x[vacancy_mask] = 0

    return x


def create_2d_meshgrid_tensor(
    size: List[int],
    device: torch.device = torch.device("cpu"),
    dtype=torch.float32,
) -> torch.Tensor:
    [batch, _, height, width] = size
    y_pos, x_pos = torch.meshgrid(
        [
            torch.arange(0, height, device=device, dtype=dtype),
            torch.arange(0, width, device=device, dtype=dtype),
        ]
    )
    mgrid = torch.stack([x_pos, y_pos], dim=0)  # [C, H, W]
    mgrid = mgrid.unsqueeze(0)  # [B, C, H, W]
    mgrid = mgrid.repeat(batch, 1, 1, 1)
    return mgrid


def create_3d_meshgrid_tensor(
    size: List[int],
    device: torch.device = torch.device("cpu"),
    dtype=torch.float32,
) -> torch.Tensor:
    [batch, _, depth, height, width] = size
    z_pos, y_pos, x_pos = torch.meshgrid(
        [
            torch.arange(0, depth, device=device, dtype=dtype),
            torch.arange(0, height, device=device, dtype=dtype),
            torch.arange(0, width, device=device, dtype=dtype),
        ]
    )

    mgrid = torch.stack([x_pos, y_pos, z_pos], dim=0)  # [C, D, H, W]
    mgrid = mgrid.unsqueeze(0)  # [B, C, D, H, W]
    mgrid = mgrid.repeat(batch, 1, 1, 1, 1)
    return mgrid
