import sys
import cv2
import numpy as np
import time

# Function for stereo vision and depth estimation
import triangulation as tri
import StereoVisionDepthEstimation.calibration as calibration

# Initialize points
points_right = []
points_left = []

def click_event_right(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_right.append((x, y))

def click_event_left(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_left.append((x, y))

# Open both cameras
cap_right = cv2.VideoCapture(1)                    
cap_left = cv2.VideoCapture(0)
# 2,3 is mirr
# cap_cannon = cv2.VideoCapture(3)
# while cap_cannon.isOpened():
#     success_cannon, frame_cannon = cap_cannon.read()
#     if not success_cannon:
#         print("Failed to capture images.")                
#         break
#     cv2.imshow("Frame Cannon", frame_cannon)
#     cv2.setMouseCallback("Frame Cannon", click_event_right)
#     if len(points_right) > 0:
#         print(points_right[-1])
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap_cannon.release()
#         cv2.destroyAllWindows()
        


# Stereo vision setup parameters
parser.add_argument('baseline', type=float, help='Baseline between cameras (in meters)')
parser.add_argument('focal_length', type=float, help='Focal length of the cameras (in pixels)')
parser.add_argument('horizontal_fov', type=float, help='Horizontal field of view (in degrees)')
args = parser.parse_args()


while cap_right.isOpened() and cap_left.isOpened():
    success_right, frame_right = cap_right.read()
    success_left, frame_left = cap_left.read()

    # Calibration (if needed)
    frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

    if not success_right or not success_left:
        print("Failed to capture images.")                
        break

    cv2.imshow("Frame Right", frame_right)
    cv2.imshow("Frame Left", frame_left)
    cv2.setMouseCallback("Frame Right", click_event_right)
    cv2.setMouseCallback("Frame Left", click_event_left)

    if len(points_right) > 0 and len(points_left) > 0:
        # Assuming the last clicked points are the ones to be used
        startTime = time.time()
        depth = tri.find_depth(points_right[-1], points_left[-1], frame_right, frame_left, B, alpha)
        
        print("Time to cal depth: ", time.time() - startTime)
        print("Depth: ", depth)
        points_right.clear()  # Clear the points to allow for new measurements
        points_left.clear()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
