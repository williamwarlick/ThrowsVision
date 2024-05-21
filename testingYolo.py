import cv2
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Ball Detection Yolo')
parser.add_argument('left_image1', type=str, help='left.png')
parser.add_argument('right_image1', type=str, help='right.png')
args = parser.parse_args()

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the confidence threshold and NMS threshold
model.conf = 0.3  # Adjust the confidence threshold as needed

# Images
left_img_path = args.left_image1
right_img_path = args.right_image1

# Read the images
left_img = cv2.imread(left_img_path)
right_img = cv2.imread(right_img_path)

# Read the stereo calibration parameters from the XML file
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

# Undistort the images using the stereo rectification maps
left_img_undistorted = cv2.remap(left_img, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
right_img_undistorted = cv2.remap(right_img, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

# Detect balls in the undistorted left image
left_results = model(left_img_undistorted)
left_filtered_results = left_results.xyxy[0][left_results.xyxy[0][:, 5] == 32]

# Detect balls in the undistorted right image
right_results = model(right_img_undistorted)
right_filtered_results = right_results.xyxy[0][right_results.xyxy[0][:, 5] == 32]

# Get the coordinates of the first detected ball in the left image
if len(left_filtered_results) > 0:
    x1, y1, x2, y2, conf, cls = left_filtered_results[0]
    cx_left = (x1 + x2) / 2
    cy_left = (y1 + y2) / 2
    left_ball_coordinate = (cx_left, cy_left)

    # Draw bounding box on the undistorted left image
    cv2.rectangle(left_img_undistorted, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.putText(left_img_undistorted, 'Ball', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
else:
    print("No ball detected in the left image.")

# Get the coordinates of the first detected ball in the right image
if len(right_filtered_results) > 0:
    x1, y1, x2, y2, conf, cls = right_filtered_results[0]
    cx_right = (x1 + x2) / 2
    cy_right = (y1 + y2) / 2
    right_ball_coordinate = (cx_right, cy_right)

    # Draw bounding box on the undistorted right image
    cv2.rectangle(right_img_undistorted, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    cv2.putText(right_img_undistorted, 'Ball', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
else:
    print("No ball detected in the right image.")

# Save the matched ball coordinates to a file
if len(left_filtered_results) > 0 and len(right_filtered_results) > 0:
    with open('ball_coordinates.txt', 'w') as f:
        f.write(f'{left_ball_coordinate[0]} {left_ball_coordinate[1]} {right_ball_coordinate[0]} {right_ball_coordinate[1]}\n')
else:
    print("Ball not detected in both images. Coordinates not saved.")

# Display the undistorted images with bounding boxes
cv2.imshow('Left Image', left_img_undistorted)
cv2.imshow('Right Image', right_img_undistorted)

# Save the undistorted images with bounding boxes
experiment_number = 1
cv2.imwrite(f'./run/exp{experiment_number}/ball_detection_left.png', left_img_undistorted)
cv2.imwrite(f'./run/exp{experiment_number}/ball_detection_right.png', right_img_undistorted)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()