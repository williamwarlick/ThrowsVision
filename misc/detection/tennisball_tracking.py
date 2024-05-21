"""
This script detects a green ball in video streams from two sources and draws lines representing the ball's trajectory. 
Key computer vision techniques applied include color-based segmentation using the HSV color space, contour detection, 
and object tracking to isolate the green ball.
 """


from collections import deque
import numpy as np
import argparse
import cv2
import imutils
import time

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help="path to the (optional) video file")
parser.add_argument("-b", "--buffer", type=int, default=100, help="max buffer size")
args = vars(parser.parse_args())

# Define the lower and upper boundaries of the "green" ball in the HSV color space
#(this is dependent on specific object and lighting conditions -- I used a tennis ball in top down indoor lighting conditions) 
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)


pts = deque(maxlen=args["buffer"])

# Initialize the video streams
vs1 = cv2.VideoCapture(0) if not args.get("video", False) else cv2.VideoCapture(args["video"])
vs2 = cv2.VideoCapture(2) if not args.get("video", False) else cv2.VideoCapture(args["video"])

# Allow the camera or video file to warm up
time.sleep(0.5)

while True:
    # Grab the current frame from both video streams
    ret1, frame1 = vs1.read()
    ret2, frame2 = vs2.read()

    frame1 = frame1 if not args.get("video", False) else frame1
    frame2 = frame2 if not args.get("video", False) else frame2

    if frame1 is None or frame2 is None:
        break

    # Resize the frames, blur them, and convert them to the HSV color space
    frame1 = imutils.resize(frame1, width=600)
    frame2 = imutils.resize(frame2, width=600)
    blurred1 = cv2.GaussianBlur(frame1, (11, 11), 0)
    blurred2 = cv2.GaussianBlur(frame2, (11, 11), 0)
    hsv1 = cv2.cvtColor(blurred1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(blurred2, cv2.COLOR_BGR2HSV)

    #  mask 
    mask1 = cv2.inRange(hsv1, greenLower, greenUpper)
    mask2 = cv2.inRange(hsv2, greenLower, greenUpper)
    mask1 = cv2.erode(mask1, None, iterations=2)
    mask2 = cv2.erode(mask2, None, iterations=2)
    mask1 = cv2.dilate(mask1, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)

    cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)
    cnts2 = imutils.grab_contours(cnts2)
    center1 = None
    center2 = None

    # Only proceed if at least one contour was found
    if len(cnts1) > 0:
        # Find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts1, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center1 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Only proceed if the radius meets a minimum size
        if radius > 10:
            # Draw the circle and centroid on the frame, then update the list of tracked points
            cv2.circle(frame1, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame1, center1, 5, (0, 0, 255), -1)

    if len(cnts2) > 0:
        # Find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
        c = max(cnts2, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Only proceed if the radius meets a minimum size
        if radius > 10:
            # Draw the circle and centroid on the frame, then update the list of tracked points
            cv2.circle(frame2, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame2, center2, 5, (0, 0, 255), -1)

    # Update the points queue
    pts.appendleft(center1)
    pts.appendleft(center2)

    # Loop over the set of tracked points
    for i in range(1, len(pts), 2):
        # If either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame1, pts[i - 1], pts[i], (0, 0, 255), thickness)

    for i in range(2, len(pts), 2):
        # If either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame2, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Show the frames to our screen
    cv2.imshow("Frame 1", frame1)
    cv2.imshow("Frame 2", frame2)
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# Release the video capture objects and close all windows
vs1.release()
vs2.release()
cv2.destroyAllWindows()
