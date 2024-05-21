import cv2
import numpy as np

# Load the images
left_img = cv2.imread('./images/stereoLeft/left_last_frame.png')
right_img = cv2.imread('./images/stereoRight/right_last_frame.png')

# Define a function to get pixel coordinates from mouse click
def get_pixel_coords(event, x, y, flags, param):
    global left_coords, right_coords, left_img_copy, right_img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        if param == 'left':
            left_coords = (x, y)
            cv2.circle(left_img_copy, (x, y), 5, (0, 0, 255), -1)
        elif param == 'right':
            right_coords = (x, y)
            cv2.circle(right_img_copy, (x, y), 5, (0, 0, 255), -1)

        if left_coords and right_coords:
            print(f"Left Image Coordinates: {left_coords}")
            print(f"Right Image Coordinates: {right_coords}")
            cv2.imshow('Left Image', left_img_copy)
            cv2.imshow('Right Image', right_img_copy)

# Create copies of the images for drawing
left_img_copy = left_img.copy()
right_img_copy = right_img.copy()

# Initialize coordinate variables
left_coords = None
right_coords = None

# Create windows and set mouse callback functions
cv2.namedWindow('Left Image')
cv2.setMouseCallback('Left Image', get_pixel_coords, param='left')
cv2.namedWindow('Right Image')
cv2.setMouseCallback('Right Image', get_pixel_coords, param='right')

# Display the images
cv2.imshow('Left Image', left_img_copy)
cv2.imshow('Right Image', right_img_copy)

# Wait for user to click on both images
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()