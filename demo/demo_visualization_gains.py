import os, sys
import numpy as np
from typing import List
import cv2

sys.path.append("../")

from src.EM.managers import DataManager
from src.EM.utils import TrainType, DrawHeatMap, mkdir

demo_path = os.path.abspath(os.curdir)
root_path = os.path.abspath(os.path.join(demo_path, ".."))


def GetData(
    data_set: str = "wiindoor",
    train_type: int = 0,
) -> List[np.ndarray]:
    data_path = os.path.join(root_path, "data", data_set)
    data_manager = DataManager(data_path=data_path)
    train_data, checkerboard_data, genz_data, gendiag_data = data_manager.LoadData(
        is_training=True, test_target='all'
    )

    # [F, T, 1, R, K, I, 4] for intersections
    # [F, T, 1, R, D=8, K] for channels
    if train_type == int(TrainType.TRAIN):
        ch, floor_idx, interactions, rx, tx = train_data
    elif train_type == int(TrainType.TEST):
        ch, floor_idx, interactions, rx, tx = checkerboard_data
    elif train_type == int(TrainType.VALIDATION):
        ch, floor_idx, interactions, rx, tx = gendiag_data

    return ch, floor_idx, interactions, rx, tx


if __name__ == "__main__":
    ch, floor_idx, interactions, rx, tx = GetData()
    color = DrawHeatMap(tx=tx[0:1, ...], ch=ch[0:1, ...])
    save_dir = os.path.join(demo_path, "Visualizations")
    mkdir(save_dir)
    save_path = os.path.join(save_dir, "heat.png")
    cv2.imwrite(save_path, color)
