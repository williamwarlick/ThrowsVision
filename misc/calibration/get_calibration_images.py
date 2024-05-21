import os
import numpy as np
import cv2

print('Starting the Calibration. Press "q" to exit the script\n')
print('Push (s) to save the image you want')

id_image = 0
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Call the two cameras
base_folder = "../../April9Testing/"
left_folder = os.path.join(base_folder, "left")
right_folder = os.path.join(base_folder, "right")
left_video = os.path.join(left_folder, f"left_1_calibration.mp4")
right_video = os.path.join(right_folder, f"right_1_calibration.mp4")
CamR = cv2.VideoCapture(right_video)  # 0 -> Right Camera
CamL = cv2.VideoCapture(left_video)   # 2 -> Left Camera

while True:
    retR, frameR = CamR.read()
    retL, frameL = CamL.read()

    if not retR or not retL:
        break

    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retR, cornersR = cv2.findChessboardCorners(grayR, (6, 4), None)  # Define the number of chess corners (here 9 by 6) we are looking for with the right Camera
    retL, cornersL = cv2.findChessboardCorners(grayL, (6, 4), None)  # Same with the left camera

    cv2.imshow('imgR', frameR)
    cv2.imshow('imgL', frameL)

    # If found, add object points, image points (after refining them)
    if (retR == True) & (retL == True):
        corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)  # Refining the Position
        corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)

        # Draw and display the corners
        cv2.drawChessboardCorners(grayR, (6, 4), corners2R, retR)
        cv2.drawChessboardCorners(grayL, (6, 4), corners2L, retL)
        cv2.imshow('VideoR', grayR)
        cv2.imshow('VideoL', grayL)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        str_id_image = str(id_image)
        print(f'Images {str_id_image} saved for right and left cameras')
        cv2.imwrite(f'chessboard-R{str_id_image}.png', frameR)  # Save the image in the file where this Programm is located
        cv2.imwrite(f'chessboard-L{str_id_image}.png', frameL)
        id_image += 1
    elif key == ord('q'):
        print('Exiting the script')
        break

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()