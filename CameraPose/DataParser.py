import os
import json

import numpy as np


class RealsenseDataParser:
    def __init__(self, path):
        self.path = path
        self.intrinsic_data = self.read_intrinsics()

    def get_rgb_path(self):
        rgb_path = os.path.join(self.path, 'color')
        return rgb_path

    def read_intrinsics(self):
        intrinsics_path = os.path.join(self.path, 'intrinsics.json')
        with open(intrinsics_path) as f:
            data = json.load(f)
        return data

    def get_K(self):
        K = np.eye(3)
        K[0, 0] = self.intrinsic_data['fx']
        K[1, 1] = self.intrinsic_data['fy']
        K[0, 2] = self.intrinsic_data['ppx']
        K[1, 2] = self.intrinsic_data['ppy']
        return K

    def get_dist_coeffs(self):
        dist_coeffs = np.array(self.intrinsic_data['coeffs']).reshape(1, 5)
        return dist_coeffs

    def get_resolution(self):
        return (self.intrinsic_data['width'], self.intrinsic_data['height'])




