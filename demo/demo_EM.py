import sys
import torch
import argparse
from typing import Dict

sys.path.append("../")

from src.EM.managers import CoreManager


def main(opt: Dict):
    core_manager = CoreManager(opt)
    core_manager.Initialization()
    core_manager.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument("--is_training", type=bool, default=True, help="Train?")
    parser.add_argument(
        "--test_target",
        type=str,
        default="genz",
        choices=["checkerboard", "genz", "gendiag"],
        help="Test?",
    )
    parser.add_argument(
        "--validation_target",
        type=str,
        default="gendiag",
        choices=["checkerboard", "genz", "gendiag"],
        help="Test?",
    )
    parser.add_argument(
        "--data_set",
        type=str,
        default="wi3rooms_0",
        choices=["wiindoor, wi3rooms_0, wi3rooms_1"],
        help="Train?",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Size of Batch")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="When sample data from dataset, indicate a number of work threads",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--use_check_point", type=bool, default=True, help="Use Check Point?"
    )
    parser.add_argument("--total_steps", type=int, default=500, help="Total Steps")

    parser.add_argument(
        "--dim", type=int, default=3, choices=[3], help="dimension: 2D or 3D"
    )
    parser.add_argument("--log_to_disk", type=bool, default=False, help="Log to disk?")

    opt = vars(parser.parse_args())
    print(opt)
    main(opt)
