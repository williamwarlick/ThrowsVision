import cv2
import numpy as np

#Camera indices are 0, 1, 2, 3, etc. On Macbook, the built-in camera is usually 0 and a connected iPhone is 2.
# A useful command you can also run in terminal is $ ffmpeg -f avfoundation -list_devices true -i 
def list_cameras(max_tests=3):
    available_cameras = []
    for i in range(max_tests):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"No camera found at index {i}")
    return available_cameras

# Test the first 3 indices
available_cameras = list_cameras(3)
print("Available camera indices:", available_cameras)
