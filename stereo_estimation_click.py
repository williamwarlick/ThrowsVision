import cv2
import numpy as np

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

# Load the translation vector (baseline)
trans = np.loadtxt('trans.txt')

# Define the image paths
left_image_path = './images/StereoLeft/imageL8.png'
right_image_path = './images/StereoRight/imageR8.png'

# Get the last frames from the left and right images
frame_left = cv2.imread(left_image_path)
frame_right = cv2.imread(right_image_path)

# Get the image dimensions
height, width, _ = frame_left.shape

# Create new matrices for rectification
left_rectified = np.zeros((height, width, 3), dtype=np.uint8)
right_rectified = np.zeros((height, width, 3), dtype=np.uint8)

# Rectify the frames
left_rectified = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
right_rectified = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

# Create a window and set the mouse callback
cv2.namedWindow("Stereo Depth")

point_left = None
point_right = None
click_counter = 0

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global point_left, point_right, click_counter
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_counter == 0:
            point_left = (x, y)
            cv2.circle(combined_frame, point_left, 5, (0, 255, 0), -1)
            click_counter += 1
        elif click_counter == 1:
            point_right = (x, y)
            cv2.circle(combined_frame, point_right, 5, (0, 255, 0), -1)
            click_counter += 1

cv2.setMouseCallback("Stereo Depth", mouse_callback)

# Display the frames side by side
combined_frame = cv2.hconcat([frame_left, frame_right])
cv2.imshow("Stereo Depth", combined_frame)

# Wait for the user to select two points
while click_counter < 2:
    cv2.waitKey(1)

# Define the find_depth function
def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):
    # Convert focal length from mm to pixels
    height_right, width_right, _ = frame_right.shape
    height_left, width_left, _ = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)
    else:
        print('Left and right camera frames do not have the same pixel width')
        return None

    x_right = right_point[0]
    x_left = left_point[0]

    # Ensure disparity is positive
    disparity =  x_right - x_left

    if disparity == 0:
        print("Disparity is zero. Cannot calculate depth.")
        return None

    # Calculate depth
    zDepth = (baseline * f_pixel) / disparity  # Depth in cm
    return zDepth

# Extract baseline from translation vector
baseline = np.linalg.norm(trans) / 10
import numpy as np

# Assuming you have already loaded the camera matrix
cameraMatrixL = np.loadtxt('cameraMatrixL.txt')

# Calculate focal length in pixels from camera matrix
focal_length = cameraMatrixL[0, 0] 

# Assuming the sensor width is 18 mm
sensor_width = 18  # mm

# Calculate alpha (field of view)
alpha = 60

print(f"Focal Length: {focal_length} pixels")
print(f"Field of View (alpha): {alpha} degrees")

# Calculate and print the depth
if point_left is not None and point_right is not None:
    depth = find_depth(point_right, point_left, frame_right, frame_left, baseline, focal_length, alpha)
    if depth is not None:
        print(f"Depth is: {depth} cm")

cv2.destroyAllWindows()
