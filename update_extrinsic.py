import os
import json
import argparse

import numpy as np
from tqdm.contrib import tenumerate, tzip

def parse_args():
    parser = argparse.ArgumentParser(description="Update extrinsic parameters")
    parser.add_argument(
        "--aruco_poses",
        type=str,
        default="/home/testbed/Projects/instant-ngp/data/Benchmark_scene_c/transforms.json",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--refined_poses",
        type=str,
        default="/home/testbed/Projects/instant-ngp/data/Benchmark_scene_c/base_extrinsics.json",
        help="Path to the dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """Loads transforms.json saves a copy as transforms_aruco.json and updates the
    transform_matrix of each frame with the refined pose from base_extrinsics.json
    (refined poses from instant-ngp)
    """
    args = parse_args()
    aruco_poses = args.aruco_poses
    with open(aruco_poses) as f:
        aruco_dict = json.load(f)

    # save aruco_dict as a copy
    aruco_copy = aruco_poses.replace(".json", "_aruco.json")
    with open(aruco_copy, "w") as f:
        json.dump(aruco_dict, f, indent=2)

    # load refined poses
    refined_poses = args.refined_poses
    with open(refined_poses) as f:
        refined_poses = json.load(f)

    aruco_frames = aruco_dict["frames"]

    for a_frame, r_frame in zip(aruco_frames, refined_poses):
        T_mtx_refined = np.array(r_frame["transform_matrix"])
        T_mtx_refined = np.vstack((T_mtx_refined, np.array([0, 0, 0, 1])))

        a_frame["transform_matrix"] = T_mtx_refined.tolist()

    # aruco_dict["orientation_override"] = "none"
    # aruco_dict["applied_scale"] = 1.0
    # aruco_dict["applied_transform"] = np.eye(4)[:3, :].tolist()

    # save aruco_dict
    with open(
        aruco_poses,
        "w",
    ) as f:
        json.dump(aruco_dict, f, indent=2)
