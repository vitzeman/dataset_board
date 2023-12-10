import argparse 
import os
import json 
import copy


import png

import cv2
import numpy as np
import pyrealsense2 as rs
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str, help="Path 2 save the recording", default="data")
    args = parser.parse_args()
    return args

class IntelConfig:
    """
    Class for loading json config file into depth camera.
    """

    def __init__(self):
        """
        IntelConfig object constructor.

        Args:
            config_path (str): Path to the config file.
        """

        self.DS5_product_ids = [
            "0AD1",
            "0AD2",
            "0AD3",
            "0AD4",
            "0AD5",
            "0AF6",
            "0AFE",
            "0AFF",
            "0B00",
            "0B01",
            "0B03",
            "0B07",
            "0B3A",
            "0B5C",
        ]

    def find_device_that_supports_advanced_mode(self) -> rs.device:
        """
        Searches devices connected to the PC for one compatible with advanced mode.

        Returns:
            rs.device: RealSense device which supports advanced mode.
        """

        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            if (
                dev.supports(rs.camera_info.product_id)
                and dev.supports(rs.camera_info.name)
                and str(dev.get_info(rs.camera_info.product_id)) in self.DS5_product_ids
            ):
                print(
                    "[INFO] Found device that supports advanced mode:",
                    dev.get_info(rs.camera_info.name),
                )
                return dev

        raise Exception(
            "[ERROR] No RealSense camera that supports advanced mode was found"
        )

    def load_config(self, config_path: str):
        """
        Loads json config file into the camera.

        Args:
            config_path (str): Path to the config file.
        """

        # Open camera in advanced mode
        dev = self.find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)

        # Read configuration JSON file as string and print it to console
        # serialized_string = advnc_mode.serialize_json()
        # print(serialized_string)

        # Write configuration file to camera
        with open(config_path) as file:
            data = json.load(file)
        json_string = str(data).replace("'", '"')
        advnc_mode.load_json(json_string)
        print("[INFO] Loaded RealSense camera config from file:", config_path)


class RealSenseCamera:
    """Class for realsesnse camera"""

    def __init__(self) -> None:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise Exception("No device connected, please connect a RealSense device")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        config_name = "D435_camera_config_defaults.json"
        self.ic = IntelConfig()
        self.ic.load_config(config_name)
        

        # dev = self.find
        # Enable streams with the same resolution
        # self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        # self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        # # Align depth frame to color frame
        # self.align = rs.align(rs.stream.color)

        # # Start streaming
        # self.profile = self.pipeline.start(self.config)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

        # Create object for aligning depth frame to RGB frame, so that they have equal resolution
        self.align = rs.align(rs.stream.color)

        # Create object for filling missing depth pixels where the sensor was not able to detect depth
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 2)

        # Create object for colorizing depth frames
        self.colorizer = rs.colorizer()
        self.colorizer.set_option(rs.option.color_scheme, 0)

        # Start video stream
        self.profile = self.pipeline.start(self.config)

        
        # Used intrinsics 
        self.color_stream = self.profile.get_stream(rs.stream.color)
        self.color_intrinsics = self.color_stream.as_video_stream_profile().get_intrinsics()
        print("Color intrinsics: ", self.color_intrinsics)
        self.start_rec = False
        self.log_json = {
            "pause_frmes": [],
        }


    def end_stream(self):
        self.pipeline.stop()

    def get_intrinsics(self):
        return self.color_intrinsics

    def get_aligned_frames(self):
        framset = self.pipeline.wait_for_frames()
        color_frame = framset.get_color_frame()
        depth_frame = framset.get_depth_frame()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(framset)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        return color_frame, aligned_depth_frame
    
    def save_depth_image(self, depth, name:str):
        """Saves depth image into png format with 16 bit

        Args:
            depth (np.ndarray): Uint16 numpy array containing depth values
            name (str): Name of the file to save
        """            
        depth = depth.astype(np.uint16)
        w_depth = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16, greyscale=True)
        with open(name, "wb") as f:
            w_depth.write(f, depth)


    def read_depth_image(self, name:str) -> np.ndarray:
        """Reads the depth image from png file

        Args:
            name (str): Name of the file to read

        Returns:
            np.ndarray: Depth image as numpy array
        """        
        r = png.Reader(filename=name)
        w,h, depth, meta = r.read()
        depth = np.array(list(depth)).reshape((h,w))
        return depth

    def variance_of_laplacian(self,image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def sharpness(self, image):
        """Calculates the sharpness of the image

        Args:
            img (np.ndarray): Image as numpy array

        Returns:
            float: Sharpness value
        """        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = self.variance_of_laplacian(gray)
        return fm
    
    def save_on_key(self, save_path:str):
        save_folder = save_path
        save_color_path = os.path.join(save_folder, "color")
        save_depth_path = os.path.join(save_folder, "depth")
        os.makedirs(save_color_path, exist_ok=True)
        os.makedirs(save_depth_path, exist_ok=True)
        cv2.namedWindow("color", cv2.WINDOW_AUTOSIZE)
        i = 0
        while True:
            color_frame, aligned_depth_frame = self.get_aligned_frames()
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                
                cv2.imwrite(os.path.join(save_color_path, str(i).zfill(6) + ".png"), color_image)
                self.save_depth_image(depth_image, os.path.join(save_depth_path, str(i).zfill(6) + ".png"))
                i += 1
                print("Saved frame: ", i)

            
            cv2.imshow("color", color_image)

    
    def make_recoring(self, save_path:str):
        save_folder = save_path
        save_color_path = os.path.join(save_folder, "color")
        save_depth_path = os.path.join(save_folder, "depth")
        os.makedirs(save_color_path, exist_ok=True)
        os.makedirs(save_depth_path, exist_ok=True)
        intrinsics = self.get_intrinsics()
        print("Intrinsics: ", type(intrinsics))

        intrinsics_d = {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "height": intrinsics.height,
            "width": intrinsics.width,
            "model": str(intrinsics.model),
            "coeffs": intrinsics.coeffs,
        }
        with open(os.path.join(save_folder, "intrinsics.json"), "w") as f:
            json.dump(intrinsics_d, f, indent=2)
        i = 0
        # colors = np.zeros((1500, intrinsics.height, intrinsics.width, 3), dtype=np.uint8)
        # depths = np.zeros((1500, intrinsics.height, intrinsics.width), dtype=np.uint16)
        while True:
            name = str(i).zfill(6)
            color_frame, aligned_depth_frame = self.get_aligned_frames()
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            sharpness = self.sharpness(color_image)

            img2show = copy.deepcopy(color_image)
            img2show = cv2.putText(img2show, "Frame: " + str(i), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            img2show = cv2.putText(img2show, "Sharpness: " + str(sharpness), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("color", img2show)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.start_rec = True
            elif key == ord('p'): 
                self.start_rec = False
                self.log_json["pause_frmes"].append(i)
    
            # colors[i, :,:,:] = color_image
            # depths[i, :,:] = depth_image
            if self.start_rec:
                cv2.imwrite(os.path.join(save_color_path, name + ".png"), color_image)
                self.save_depth_image(depth_image, os.path.join(save_depth_path, name + ".png"))
                self.log_json[i] = {
                    "sharpness": sharpness,
                }
                i += 1
            
                if i ==1000000:
                    break
            # if i == 1000:
            #     break

        with open(os.path.join(save_folder, "log.json"), "w") as f:
            json.dump(self.log_json, f, indent=2)

        # for e in tqdm(range(i)):
        #     name = str(e).zfill(6)
        #     cv2.imwrite(os.path.join(save_color_path, name + ".png"), colors[e, :,:,:])
        #     self.save_depth_image(depths[e, :,:], os.path.join(save_depth_path, name + ".png"))
        
if __name__ == "__main__":
    args = parse_args()
    save_path = args.save_path
    cam = RealSenseCamera()
    # cam.make_recoring(save_path)
    cam.save_on_key(save_path)


    # while True:
    #     color_frame, aligned_depth_frame = cam.get_aligned_frames()
    #     color_image = np.asanyarray(color_frame.get_data())
    #     depth_image = np.asanyarray(aligned_depth_frame.get_data())
    #     cv2.imshow("color", color_image)
    #     cv2.imshow("depth", depth_image)
    #     # print("Color image shape: ", color_image.shape, "Depth image shape: ", depth_image.shape)

    #     # print(color_image.dtype, depth_image.dtype)
    #     key = cv2.waitKey(1)
    #     if key == ord('q'):
    #         break
    #     elif key == ord('s'):

    #         cv2.imwrite("color.png", color_image)
    #         cv2.imwrite("depth.png", depth_image)
    #         # SAVE the depth as uint16 with 1 channel
    #         depth_image = depth_image.astype(np.uint16)
    #         w_depth = png.Writer(width=depth_image.shape[1], height=depth_image.shape[0], bitdepth=16, greyscale=True)
    #         with open("depth.png", "wb") as f:
    #             w_depth.write(f, depth_image)

    #         r_depth = png.Reader(filename="depth.png")
    #         depth_loaded = r_depth.read()
    #         print(depth_loaded)
    #         w,h, depth, meta = depth_loaded
    #         depth_loaded = np.array(list(depth)).reshape((h,w))
    #         print("Depth loaded shape: ", depth_loaded.shape)
    #         print(np.unique(depth_image, return_counts=True))
    #         print(np.unique(depth_loaded, return_counts=True))
    #         # print(np.unique(depth_loaded[:,:,0], return_counts=True))
    #         print((depth_image == depth_loaded).all())

    # cam.end_stream()