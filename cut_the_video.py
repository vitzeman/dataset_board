import cv2
import numpy as np
import os

def cut_video_into_images(path2video):
    video = cv2.VideoCapture(path2video)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("frame_count: ", frame_count)
    print("fps: ", fps)
    print("width: ", width)
    print("height: ", height)
    os.makedirs("captures/Mobile_capture", exist_ok=True)
    # save frames
    curr_frame = 0
    while True:
        ret, frame = video.read()
        if ret:
            cv2.imwrite(os.path.join("captures", "Mobile_capture", str(curr_frame).zfill(6) + ".png"), frame)
            cv2.putText(frame, str(curr_frame), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            curr_frame += 1
        else:
            break

if __name__ == '__main__':
    path2video = "captures/Mobile_capture.mp4"
    cut_video_into_images(path2video)
    