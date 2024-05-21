import cv2
import numpy as np


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

# Normalize the disparity map for visualization
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display the disparity map
# cv2.imshow("Disparity Map", disparity_normalized)
# cv2.waitKey(0)

# Convert disparity to depth
focal_length = cameraMatrixL[0, 0]  # Focal length of the camera
baseline = np.linalg.norm(trans)  # Baseline distance between the cameras (using the translation vector)
disparity_scaled = disparity / 16.0  # Adjust the scaling factor as needed

# Handle zero disparity values
disparity_scaled[disparity_scaled == 0] = 0.1  # Replace zero disparity with a small value

depth = (focal_length * baseline) / disparity_scaled

# Normalize the depth map for visualization
min_depth = np.min(depth)
max_depth = np.max(depth)
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Create a color map for the depth visualization
color_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

# Add a color scale key to the depth map
scale_percent = 0.3  # Adjust the scale percentage as needed
height, width, _ = color_map.shape
scale_width = int(width * scale_percent)
scale_height = 20

# Create a blank scale image with the same width as the color map
scale_image = np.zeros((scale_height, width, 3), dtype=np.uint8)

# Fill the scale image with color gradient
for i in range(scale_width):
    color = color_map[0, int((i / scale_width) * width)]
    scale_image[:, int((i / scale_width) * width)] = color

# Add depth values to the scale
cv2.putText(scale_image, f"{min_depth:.2f}", (0, scale_height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(scale_image, f"{max_depth:.2f}", (width - 100, scale_height - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Combine the depth map and scale image vertically
combined_image = np.vstack((color_map, scale_image))

# Display the depth map with the color scale key
cv2.imshow("Depth Map", combined_image)
cv2.waitKey(0)

cv2.destroyAllWindows()