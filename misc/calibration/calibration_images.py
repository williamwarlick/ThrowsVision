import cv2
import argparse

parser = argparse.ArgumentParser(description='Calibration Images')
parser.add_argument('left_video',  help='left.mpt')
parser.add_argument('right_video', help='right.mp4')

args = parser.parse_args()

# Access the parsed arguments


# cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
# cap2 = cv2.VideoCapture(2, cv2.CAP_AVFOUNDATION)

cap = cv2.VideoCapture(args.left_video)
cap2 = cv2.VideoCapture(args.right_video)

if not cap.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

num = 0

while cap.isOpened() and cap2.isOpened():
    succes1, img = cap.read()
    succes2, img2 = cap2.read()

    if not succes1 or not succes2:
        print("Failed to capture images from one or both cameras.")
        break

    k = cv2.waitKey(5)

    if k == 27:  # wait for 'ESC' key to exit
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        if succes1 and succes2:  # Ensure both images were successfully captured
            cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img)
            cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', img2)
            print("Images saved!")
            num += 1
        else:
            print("Could not save images, one or both captures failed.")

    if succes1:
        cv2.imshow('Img 1', img)
    if succes2:
        cv2.imshow('Img 2', img2)

    if num >= 10:
        # Release and destroy all windows before termination
        cap.release()
        cap2.release()
        cv2.destroyAllWindows()


