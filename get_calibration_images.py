import os
import cv2
import argparse

def put_text_on_frame(frame, text, position, font_scale=0.7, color=(0, 255, 0), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

parser = argparse.ArgumentParser(description='Calibration Images')
parser.add_argument('left_video', type=str, help='left.mp4')
parser.add_argument('right_video', type=str, help='right.mp4')
args = parser.parse_args()

print('Starting the Calibration. Press "q" to exit the script\n')
print('Push (s) to save the image you want')

capL = cv2.VideoCapture(args.left_video)
capR = cv2.VideoCapture(args.right_video)

# To use live feed use commented out code below
# capL = cv2.VideoCapture(0)
# capR = cv2.VideoCapture(2)

if not capL.isOpened() or not capR.isOpened():
    print("Error: One of the video files could not be found.")
    exit()

id_image = 0
chessboard_size = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
saved_images = []
approve_images = True  # Set this to True if you want to approve images manually

resolution = f'{int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))}'
fps = f'FPS: {int(capL.get(cv2.CAP_PROP_FPS))}'

instructions = "Press 'q' to exit. Press 's' to save the image."
calibration_note = "Gather 18 calibration images."

while capL.isOpened() and capR.isOpened():
    succes1, frameL = capL.read()
    succes2, frameR = capR.read()

    if not succes1 or not succes2:
        print("Failed to capture images from one or both cameras.")
        break

    put_text_on_frame(frameL, resolution + " " + fps, (10, 30))
    put_text_on_frame(frameL, instructions, (10, 60))
    put_text_on_frame(frameL, calibration_note, (10, 90))

    put_text_on_frame(frameR, resolution + " " + fps, (10, 30))
    put_text_on_frame(frameR, instructions, (10, 60))
    put_text_on_frame(frameR, calibration_note, (10, 90))

    cv2.imshow('VideoR', frameR)
    cv2.imshow('VideoL', frameL)

    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):  # wait for 'q' key to exit
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

        if retR and retL:
            corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            saved_images.append((frameR, frameL))
            print(f'Images {id_image} saved for right and left cameras')
            id_image += 1

        if id_image >= 18:
            break

capL.release()
capR.release()
cv2.destroyAllWindows()

if approve_images:
    print('Image capture completed. Press "a" to approve or "d" to disapprove each image.')
    for i, (imgR, imgL) in enumerate(saved_images):
        cv2.imshow(f'Image {i} - Right', imgR)
        cv2.imshow(f'Image {i} - Left', imgL)
        while True:
            r = cv2.waitKey(0) & 0xFF
            if r == ord('a'):  # Approve the image
                cv2.imwrite(f'images/stereoRight/imageR{i}.png', imgR)
                cv2.imwrite(f'images/stereoLeft/imageL{i}.png', imgL)
                print(f'Images {i} approved and saved')
                break
            elif r == ord('d'):  # Disapprove the image
                print(f'Images {i} disapproved and deleted')
                break
else:
    for i, (imgR, imgL) in enumerate(saved_images):
        cv2.imwrite(f'images/stereoRight/imageR{i}.png', imgR)
        cv2.imwrite(f'images/stereoLeft/imageL{i}.png', imgL)
        print(f'Images {i} saved')

cv2.destroyAllWindows()
print('Assessment completed.')
