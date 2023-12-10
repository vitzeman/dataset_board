import os.path

import cv2
import math
import json

import numpy as np


class ngp_data_creator:
    def __init__(self, intrinsics, dist_coeffs, resolution, center_translation):
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
        self.resolution = resolution
        self.transform = {}
        self.create_header()
        self.flip_mat = np.array([
			[1, 0, 0, 0],
			[0, -1, 0, 0],
			[0, 0, -1, 0],
			[0, 0, 0, 1]
		])
        self.center_translation = np.array([-center_translation[0], center_translation[1], center_translation[2]]) / 1000


    def variance_of_laplacian(self,image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def sharpness(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = self.variance_of_laplacian(gray)
        return fm

    def create_header(self):
        # get instrinsics
        fl_x = self.intrinsics[0][0]
        fl_y = self.intrinsics[1][1]
        c_x = self.intrinsics[0][2]
        c_y = self.intrinsics[1][2]
        w = self.resolution[0]
        h = self.resolution[1]

        # get distortion coefficients
        k1 = self.dist_coeffs[0][0]
        k2 = self.dist_coeffs[0][1]
        p1 = self.dist_coeffs[0][2]
        p2 = self.dist_coeffs[0][3]
        # k3 = self.dist_coeffs[0][4]

        camera_angle_x = math.atan(w / (fl_x * 2)) * 2
        camera_angle_y = math.atan(h / (fl_y * 2)) * 2
        # fovx = camera_angle_x * 180 / math.pi
        # fovy = camera_angle_y * 180 / math.pi

        # create header
        self.transform['fl_x'] = fl_x
        self.transform['fl_y'] = fl_y
        self.transform['cx'] = c_x
        self.transform['cy'] = c_y
        self.transform['w'] = w
        self.transform['h'] = h
        self.transform['k1'] = k1
        self.transform['k2'] = k2
        self.transform['p1'] = p1
        self.transform['p2'] = p2
        # self.transform['k3'] = k3
        self.transform['camera_angle_x'] = camera_angle_x
        self.transform['camera_angle_y'] = camera_angle_y
        self.transform['aabb_scale'] = 2
        self.transform['scale'] = 1
        self.transform['orientation_override'] = 'none'
        self.transform['applied_scale'] = 1.0
        self.transform['applied_transform'] = np.eye(4)[:3, :].tolist()
        self.transform["offset"]: [0.0, 0.0, 0.0]
        self.transform['frames'] = []


    def add_image(self,image, img_name, pose):
        # add image name
        sharpness = self.sharpness(image)
        if sharpness < 200:
            return
        img_name = os.path.split(img_name)[1]
        img_name = 'color/' + img_name
        # pose = np.linalg.inv(pose)
        pose  = np.matmul(pose, self.flip_mat)
        pose[0:3, 3] += self.center_translation
        self.transform['frames'].append({'file_path': img_name, 'transform_matrix': pose.tolist(), 'sharpness': sharpness})

    def normalize_camera(self, pose):
        nframes = len(self.transform["frames"])
        avglen = 0.
        for f in self.transform["frames"]:
            f["transform_matrix"] = np.array(f["transform_matrix"])
            avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
        avglen /= nframes
        self.transform["scale_pose"] = avglen
        print("avg camera distance from origin", avglen)
        for f in self.transform["frames"]:
            f["transform_matrix"][0:3, 3] *= 1.0 / avglen  # scale to "nerf sized"
            f["transform_matrix"] = f["transform_matrix"].tolist()

    def save_file(self, location):
        locations = location.split('/')
        
        location = os.path.join(*locations)

        location = os.path.join(location, 'transforms.json')
        self.normalize_camera(self.transform)
        with open(location, 'w') as outfile:
            json.dump(self.transform, outfile, indent=4)

