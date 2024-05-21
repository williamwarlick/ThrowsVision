import argparse
import cv2
import numpy as np

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

# Read the stereo calibration parameters from the XML file
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

# Read the right and left images
frame_right = cv2.imread(image_right_path)
frame_left = cv2.imread(image_left_path)

# Rectify the stereo images
frame_right_rectified = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
frame_left_rectified = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

ball_coordinates = []
with open('ball_coordinates.txt', 'r') as f:
    for line in f:
        left_x, left_y, right_x, right_y = map(float, line.strip().split())
        ball_coordinates.append(((left_x, left_y), (right_x, right_y)))

# Estimate the distance for each ball
for i, (left_coord, right_coord) in enumerate(ball_coordinates):
    # Convert coordinates to integers
    left_coord = tuple(map(int, left_coord))
    right_coord = tuple(map(int, right_coord))

    # Triangulate the 3D point using OpenCV's triangulatePoints function
    left_point = np.array([[left_coord[0]], [left_coord[1]]], dtype=np.float32)
    right_point = np.array([[right_coord[0]], [right_coord[1]]], dtype=np.float32)

    # Assuming you have the projection matrices (P1 and P2) from stereo calibration
    points4D = cv2.triangulatePoints(P1, P2, left_point, right_point)
    points3D = points4D[:3] / points4D[3]

    # Extract the depth (Z-coordinate) from the triangulated point
    depth = points3D[2][0]

    if depth > 0:
        cv2.putText(frame_right_rectified, f'Ball {i+1}: {round(depth, 1)} cm', (right_coord[0], right_coord[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(frame_right_rectified, f'Ball {i+1}: Invalid depth', (right_coord[0], right_coord[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

experiment_number = 1
cv2.imwrite(f'./run/exp{experiment_number}/distances_left.png', frame_left_rectified)
cv2.imwrite(f'./run/exp{experiment_number}/distances_right.png', frame_right_rectified)

# Display the frames with ball labels and distances
cv2.imshow("Frame Left", frame_left_rectified)
cv2.imshow("Frame Right", frame_right_rectified)

# Save the frames with ball labels and distances
cv2.waitKey(0)
cv2.destroyAllWindows()