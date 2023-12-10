import numpy as np
import os
import cv2
from CameraPose.campose import CameraPose
from CameraPose.board_file import BoardDefinition
from CameraPose.ngpdatacreator import ngp_data_creator

from CameraPose.DataParser import RealsenseDataParser

from tqdm import tqdm

if __name__ == '__main__':

    # data_set_path = "/home/testbed/Projects/instant-ngp/data/Benchmark_scene_c"
    # transform_file_location = "/home/testbed/Projects/instant-ngp/data/Benchmark_scene_c"
    data_set_path = "captures/Benchmark_scene_test"
    transform_file_location = "captures/Benchmark_scene_test"

    data_parser = RealsenseDataParser(data_set_path)
    frame_path = data_parser.get_rgb_path()
    K = data_parser.get_K()
    dist = data_parser.get_dist_coeffs()
    img_size = data_parser.get_resolution()
    Bd = BoardDefinition()
    Cp = CameraPose(K, dist, Bd)
    Cp.set_base_marker(0)
    # print(Bd.board_center())
    ngp_data = ngp_data_creator(K, dist, img_size, Bd.board_center())

    for i in tqdm(range(len(os.listdir(frame_path)))):
        image_name =  str(i).zfill(6) + '.png'
        image_name = os.path.join(frame_path, image_name)
        # print(image_name)
        image = cv2.imread(image_name)
        image, pose = Cp.find_camera_pose(image)
        if (pose == np.eye(4)).all():
            continue
        ngp_data.add_image(image, image_name, pose)

    # save json file
    ngp_data.save_file(transform_file_location)