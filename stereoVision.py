import argparse
import cv2
import numpy as np
import triangulation as tri
import StereoVisionDepthEstimation.calibration as calibration

parser = argparse.ArgumentParser(description='Stereo Vision')
parser.add_argument('baseline', type=float, help='Baseline between cameras (in meters)')
parser.add_argument('focal_length', type=float, help='Focal length of the cameras (in pixels)')
parser.add_argument('horizontal_fov', type=float, help='Horizontal field of view (in degrees)')

args = parser.parse_args()

# Access the parsed arguments
baseline = args.baseline
focal_length = args.focal_length
horizontal_fov = args.horizontal_fov
image_right_path = '../../April9Testing/right.png'
image_left_path = '../../April9Testing/left.png'

# Stereo vision setup parameters
B = baseline         #Distance between the cameras [cm]
f = focal_length       #Camera lense's focal length [mm]
alpha = horizontal_fov   #Camera field of view in the horisontal plane [degrees]

# Read the right and left images
frame_right = cv2.imread(image_right_path)
frame_left = cv2.imread(image_left_path)

# Calibrate the frames
frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

# Mouse click event handler
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth = tri.find_depth((x, y), (x, y), frame_right, frame_left, B, alpha)
        print("Clicked depth: ", str(round(depth, 1)))

# Create a named window and set the mouse callback
cv2.namedWindow("frame right")
cv2.setMouseCallback("frame right", mouse_click)

while True:
    # Display the frames
    cv2.imshow("frame right", frame_right)
    cv2.imshow("frame left", frame_left)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()