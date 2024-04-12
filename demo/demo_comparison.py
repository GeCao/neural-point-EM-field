import sys
import os
import h5py
import numpy as np
import torch
import argparse
from typing import Dict

sys.path.append("../")

from src.EM.managers import CoreManager


def get_rmse_psnr(scene_name: str, module_type: str):
    demo_path = os.path.abspath(os.curdir)
    root_path = os.path.abspath(os.path.join(demo_path, ".."))
    save_path = os.path.join(root_path, f"save/{scene_name}{module_type}/imgs")
    filename = os.path.join(save_path, "pred_and_gt.h5")
    h5f = h5py.File(filename)
    pred = np.asarray(h5f["pred"])
    gt = np.asarray(h5f["gt"])
    h5f.close()

    pred = pred / np.abs(pred).max()
    gt = gt / np.abs(gt).max()

    mse = ((pred - gt) * (pred - gt)).mean()
    rmse = np.sqrt(mse)

    f_max = np.abs(gt).max()
    psnr = 20 * np.log10(f_max / rmse)

    return [rmse, psnr]


def main():
    scene_name = "sionna_etoile"
    rmse_MLP, psnr_MLP = get_rmse_psnr(scene_name=scene_name, module_type="_MLP")
    rmse_ablation, psnr_ablation = get_rmse_psnr(
        scene_name=scene_name, module_type="_ablation"
    )
    rmse_default, psnr_default = get_rmse_psnr(scene_name=scene_name, module_type="")

    print("rmse_MLP = ", rmse_MLP)
    print("rmse_ablation = ", rmse_ablation)
    print("rmse_default = ", rmse_default)

    print("psnr_MLP = ", psnr_MLP)
    print("psnr_ablation = ", psnr_ablation)
    print("psnr_default = ", psnr_default)


if __name__ == "__main__":
    main()
