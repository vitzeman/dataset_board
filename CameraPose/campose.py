# import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# from board_file import BoardDefinition
# # from scipy.spatial.transform import Rotation as R
# # from check_pose import get_camera_frustum, frustums2lineset
# # import open3d as o3d
# from ngpdatacreator import ngp_data_creator

class CameraPose:
    def __init__(self, camera_matrix, dist_coeffs, BoardDefinition):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.Bd = BoardDefinition
        self.scale = 1000 # in meters change to 1 for mm
        self.marker_size = self.Bd.marker_size_mm / self.scale # convert to meters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.base_marker_id = None
        self.new_coordinate_system = None

    def set_base_marker(self, base_marker_id):
        self.new_coordinate_system = {}
        self.base_marker_id = base_marker_id
        x,y,z = self.Bd.marker_loaction_in_mm[base_marker_id]

        for id in self.Bd.marker_loaction_in_mm:
            self.new_coordinate_system[id] = (self.Bd.marker_loaction_in_mm[id] - np.array([x,y,z])) / self.scale

    def pose_esitmation(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedCandidates = self.detector.detectMarkers(gray)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        marker_points = np.array([[-self.marker_size / 2, self.marker_size / 2, 0],
                                  [self.marker_size / 2, self.marker_size / 2, 0],
                                  [self.marker_size / 2, -self.marker_size / 2, 0],
                                  [-self.marker_size / 2, -self.marker_size / 2, 0]], dtype=np.float32)

        marker_pose = {}
        i = 0
        try:
            for id in ids:
                marker_pose[id[0]] = {}
                corner = corners[i]
                nada, R, t = cv2.solvePnP(marker_points, corner, self.camera_matrix, self.dist_coeffs)
                cv2.putText(frame, str(id[0]), (int(corner[0][0][0]), int(corner[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                marker_pose[id[0]]['R'] = R
                marker_pose[id[0]]['t'] = t
                i += 1

                # cv2.drawFrameAxes(image=frame, cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coeffs, rvec=R, tvec=t, length=0.07)

            return marker_pose, frame
        except:
            return None, frame


    def find_camera_pose(self, frame):
        marker_pose, frame = self.pose_esitmation(frame)

        # frustums = []
        # things_to_draw = []
        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2, resolution=10)
        # sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
        # sphere.paint_uniform_color((1, 0, 0))
        #
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
        # img_size = (frame.shape[0], frame.shape[1])

        pose = np.eye(4)
        if marker_pose is not None:
            W2C_average = np.zeros((4, 4, len(marker_pose)))
            # For transformed pose`
            i = 0
            for id in marker_pose:

                Rot = marker_pose[id]['R']
                tra = marker_pose[id]['t']
                RoTmat = cv2.Rodrigues(Rot)[0]

                # Transformation matrix of marker
                Transformation_marker = np.eye(4)
                Transformation_marker[0:3, 0:3] = RoTmat
                Transformation_marker[0:3, 3] = tra.reshape(3)

                # transformation matrix of the new coordinate system
                Transformation_new = np.eye(4)
                Transformation_new[0:3, 0:3] = np.eye(3)
                Transformation_new[0:3, 3] = self.new_coordinate_system[id].reshape(3)

                Transformation = np.matmul(Transformation_marker, Transformation_new)

                # convert to camera coordinate system
                W2C = np.linalg.inv(Transformation)
                W2C_average[:, :, i] = W2C
                i += 1


            # for real marker pose
            for id in marker_pose:
                # real poses of the markers
                Rot = marker_pose[id]['R']
                tra = marker_pose[id]['t']

                RoTmat = cv2.Rodrigues(Rot)[0]

                Transformation = np.eye(4)
                Transformation[0:3, 0:3] = RoTmat
                Transformation[0:3, 3] = tra.reshape(3)

                # convert to camera coordinate system
                W2C = np.linalg.inv(Transformation)

                # frustums.append(get_camera_frustum(img_size, self.camera_matrix, W2C, frustum_length=0.05, color=[0, 0, 1]))
            # maybe add pose refinment here cv2.

            # inlier = self.remove_outliers(W2C_average)
            # inlier_average_pose = np.median(inlier, axis=2)

            pose = self.average_pose(W2C_average)
            # frustums.append(get_camera_frustum(img_size, self.camera_matrix, inlier_average_pose, frustum_length=0.05, color=[0, 1, 0]))

            # # average of all the poses
            # median = np.median(W2C_average, axis=2)

            # frustums.append(get_camera_frustum(img_size, self.camera_matrix, median, frustum_length=0.05,    color=[1, 0, 0]))

            # pose = inlier_average_pose


            # for i in range(W2C_average.shape[2]):
            #     frustums.append(get_camera_frustum(img_size, self.camera_matrix, W2C_average[:, :, i], frustum_length=0.05, color=[0, 1, 0]))

        # cameras = frustums2lineset(frustums)
        # things_to_draw.append(cameras)
        # things_to_draw.append(coord_frame)
        # things_to_draw.append(sphere)
        #
        # o3d.visualization.draw_geometries(things_to_draw,
        #                                   zoom=0.04,
        #                                   front=[0, 0, -1],
        #                                   lookat=[0, 0, 0],
        #                                   up=[0, -1, 0])


        return frame, pose

    def average_pose(self, W2C) -> np.ndarray:
        """ Computes the average pose of the camera

        Args:
            W2C (np.ndarray): Transformation matrices of the camera poses(4x4xN)      

        Returns:
            np.ndarray: Average pose of the camera(4x4)
        """         

        # move last axis to first
        W2C = np.moveaxis(W2C, -1, 0)
        Rtxs = W2C[:, 0:3, 0:3] # rotation matrices
        trls = W2C[:, 0:3, 3] # translation vectors


        # convert rotation matrices to angles 
        r = R.from_matrix(Rtxs)
        quats = r.as_quat()

        # average quaternions
        quat = np.median(quats, axis=0)
        
        delta_quats = np.linalg.norm(quats - quat, axis=1)
        std_quats = np.std(delta_quats)
        inliers_quat = delta_quats < 3*std_quats
        # print(np.sum(inliers_quat), "/", inliers_quat.shape[0], "quaternion inliers")

        # average translation vectors
        trl = np.median(trls[inliers_quat], axis=0)
        delta_trls = np.linalg.norm(trls - trl, axis=1) 
        std_trls = np.std(delta_trls)
        inliers_trl = delta_trls < 3*std_trls

        inliers = np.logical_and(inliers_quat, inliers_trl)
        # print(np.sum(inliers), "/", inliers.shape[0], "inliers")

        # compute the average again with the inliers
        quat = np.median(quats[inliers], axis=0)
        # Check if nan
        if not np.isnan(quat).any():
            # print(quat)
            r = R.from_quat(quat)
            Rtx = r.as_matrix()
            trl = np.median(trls[inliers], axis=0)

            # create transformation matrix
            pose = np.eye(4)
            pose[0:3, 0:3] = Rtx
            pose[0:3, 3] = trl.reshape(3)

        else :
            pose = np.eye(4)


        # angles = r.as_euler('xyz', degrees=True)
        # # average angles 
        # angle = np.median(angles, axis=0)
        # # convert back to rotation matrices
        # r = R.from_euler('xyz', angles, degrees=True)
        # Rtx = r.as_matrix()
        # # average translation vectors
        # trl = np.median(trls, axis=0)

        # # compute standard deviation of the  delta of the angles and translation vectors
        # delta_trls = np.linalg.norm(trls - trl, axis=1)
        # delta_angles = np.linalg.norm(angles - angle, axis=1)
        # std_trls = np.std(delta_trls)
        # std_angles = np.std(delta_angles)
        # inliers_trl = delta_trls < std_trls
        # inliers_angl = delta_angles < std_angles 

        # inlier_mask = np.logical_and(inliers_trl, inliers_angl)
        
        # print(np.sum(inlier_mask), "/", inlier_mask.shape[0], "trl inliers", np.sum(inliers_trl), "rtx inliers", np.sum(inliers_angl))
        # # compute the average again with the inliers
        # angle = np.median(angles[inlier_mask], axis=0)
        # r = R.from_euler('xyz', angle, degrees=True)
        # Rtx = r.as_matrix()
        # trl = np.median(trls[inlier_mask], axis=0)

        # # create transformation matrix
        # pose = np.eye(4)
        # pose[0:3, 0:3] = Rtx
        # pose[0:3, 3] = trl.reshape(3)

        # print(np.linalg.det(Rtx))

        return pose

    def remove_outliers(self, W2C):
        median = np.median(W2C, axis=2)
        inlier = []
        for i in range(W2C.shape[2]):
            if np.linalg.norm(W2C[:, :, i] - median) < 0.02:
                inlier.append(W2C[:, :, i])

        if len(inlier) == 0:
            return W2C
        inlier = np.array(inlier)
        inlier = np.swapaxes(inlier, 0, 2)
        inlier = np.transpose(inlier, (1, 0, 2))

        return inlier


# Example run
# if __name__ == '__main__':
#
#     board_file = '/home/varun/PycharmProjects/dataset_board/config/board.json'
#     frame = '/home/varun/PycharmProjects/dataset_board/test_capture/color/'
#     Bd = BoardDefinition()
#
#     camera_matrix = np.array( [[921.77, 0, 641.26],
#                                [0, 921.060, 358.95],
#                                [0, 0, 1]])
#     dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0]).reshape(1, 4)
#     Cp = CameraPose(camera_matrix, dist_coeffs, Bd)
#     Cp.set_base_marker(0)
#     img_size = (1280, 720)
#     ngp_data = ngp_data_creator(camera_matrix, dist_coeffs, img_size)
#
#     for i in range(len(os.listdir(frame))):
#         image_name =  str(i).zfill(6) + '.png'
#         image_name = os.path.join(frame, image_name)
#         print(image_name)
#         image = cv2.imread(image_name)
#         image, pose = Cp.find_camera_pose(image)
#         if (pose == np.eye(4)).all():
#             continue
#         ngp_data.add_image(image, image_name, pose)
#
#     location = "/home/varun/PycharmProjects/dataset_board/test_capture/"
#     # save json file
#     # ngp_data.save_file(location)
#
#
#
#         # cv2.imshow('frame', image)
#         # cv2.waitKey(0)






