import cv2
import numpy as np
import os

# Function to get the last frame of a video
def get_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, frame = cap.read()
    cap.release()
    return frame

# Load the stereo calibration parameters from the XML file
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
projMatrixL = cv_file.getNode('projMatrixL').mat()
projMatrixR = cv_file.getNode('projMatrixR').mat()
cv_file.release()

cameraMatrixL = np.loadtxt('cameraMatrixL.txt')
distCoeffsL = np.loadtxt('distCoeffsL.txt')
cameraMatrixR = np.loadtxt('cameraMatrixR.txt')
distCoeffsR = np.loadtxt('distCoeffsR.txt')
trans = np.loadtxt('trans.txt')


# Define the video path
left_video = '../../April9Testing/left/left-may6.MP4'
right_video = '../../April9Testing/right/may6_right.MP4'

# Get the last frames from the left and right videos
frame_left = get_last_frame(left_video)
frame_right = get_last_frame(right_video)

# Get the image dimensions
height, width, _ = frame_left.shape

# Create new matrices for rectification
left_rectified = np.zeros((height, width, 3), dtype=np.uint8)
right_rectified = np.zeros((height, width, 3), dtype=np.uint8)

# Rectify the frames
left_rectified = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
right_rectified = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

# Convert the rectified frames to grayscale
left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

# Create a stereo matcher
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=256,
    blockSize=5,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    P1=8*3*5**2,
    P2=32*3*5**2
)

# Compute the disparity map
disparity = stereo.compute(left_gray, right_gray)

# Scale the disparity values for subpixel accuracy
# disparity_scaled = disparity / 16.0

cv2.namedWindow("Left Frame")
cv2.namedWindow("Right Frame")

point_left = None
point_right = None

def mouse_callback_left(event, x, y, flags, param):
    global point_left
    if event == cv2.EVENT_LBUTTONDOWN:
        point_left = (x, y)
        cv2.circle(frame_left, point_left, 5, (0, 255, 0), -1)
        cv2.imshow("Left Frame", frame_left)

def mouse_callback_right(event, x, y, flags, param):
    global point_right
    if event == cv2.EVENT_LBUTTONDOWN:
        point_right = (x, y)
        cv2.circle(frame_right, point_right, 5, (0, 255, 0), -1)
        cv2.imshow("Right Frame", frame_right)

cv2.setMouseCallback("Left Frame", mouse_callback_left)
cv2.setMouseCallback("Right Frame", mouse_callback_right)

cv2.imshow("Left Frame", frame_left)
cv2.imshow("Right Frame", frame_right)

# Wait for the user to select points in both frames
while point_left is None or point_right is None:
    cv2.waitKey(1)

if point_left is not None and point_right is not None:
    points_left = np.array([[point_left[0], point_left[1]]], dtype=np.float32)
    points_right = np.array([[point_right[0], point_right[1]]], dtype=np.float32)
    # Undistort the points using the camera intrinsic parameters
    points_left_undistorted = cv2.undistortPoints(points_left, cameraMatrixL, distCoeffsL)
    points_right_undistorted = cv2.undistortPoints(points_right, cameraMatrixR, distCoeffsR)

    points_4d = cv2.triangulatePoints(projMatrixL, projMatrixR, points_left_undistorted, points_right_undistorted)
    points_3d = points_4d[:3] / points_4d[3]
    depth = points_3d[2, 0]
    print(f"Depth is: {depth}")

cv2.destroyAllWindows()