import numpy as np
import cv2 as cv
import glob

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9, 6)
frameSize = (1920, 1080)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2) * 30

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpointsL = []  # 2d points in image plane.
imgpointsR = []  # 2d points in image plane.


num_images = 9  # Number of images to use for calibration

for i in range(num_images):
    imgL = cv.imread(f'images/stereoLeft/imageL{i}.png')
    imgR = cv.imread(f'images/stereoRight/imageR{i}.png')
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR:
        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners (optional)
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('Left Chessboard', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('Right Chessboard', imgR)
        cv.waitKey(500)
    else:
        print(f"Chessboard corners not found in images {i}. Skipping...")

cv.destroyAllWindows()

############## CALIBRATION #######################################################

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))


# Save the camera matrices and distortion coefficients to files
np.savetxt('cameraMatrixL.txt', cameraMatrixL)
np.savetxt('distCoeffsL.txt', distL)
np.savetxt('cameraMatrixR.txt', cameraMatrixR)
np.savetxt('distCoeffsR.txt', distR)


# Reprojection error
print("Reprojection error - Left camera: ", retL)
print("Reprojection error - Right camera: ", retR)

# Check if the intrinsic parameters are similar for both cameras
print("Left Camera Intrinsic Parameters:")
print(cameraMatrixL)
print("Right Camera Intrinsic Parameters:")
print(cameraMatrixR)

# Check focal lengths
fxLeft = cameraMatrixL[0, 0]
fyLeft = cameraMatrixL[1, 1]
fxRight = cameraMatrixR[0, 0]
fyRight = cameraMatrixR[1, 1]


# Check if the intrinsic parameters are similar for both cameras
print("Left Camera Focal:")
print(fxLeft)
print("Right Camera Focal:")
print(fxRight)

# Check principal points
cxLeft = cameraMatrixL[0, 2]
cyLeft = cameraMatrixL[1, 2]
cxRight = cameraMatrixR[0, 2]
cyRight = cameraMatrixR[1, 2]



########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC

# Here we fix the intrinsic camera matrices so that only Rot, Trns, Emat and Fmat are calculated.

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to find the transformation between the two cameras and calculate Essential and Fundamental matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1],
    criteria_stereo, flags)

np.savetxt('trans.txt', trans)


# Stereo reprojection error
print("Stereo reprojection error prior to rectification: ", retStereo)

# Check the quality of the stereo calibration
print("Essential Matrix:")
print(essentialMatrix)
print("Fundamental Matrix:")
print(fundamentalMatrix)

########## Stereo Rectification #################################################

rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR,
                                                                           distR, grayL.shape[::-1], rot, trans,
                                                                           rectifyScale, (0, 0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)
cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])
cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])
cv_file.write('projMatrixL', projMatrixL)
cv_file.write('projMatrixR', projMatrixR)
cv_file.release()
