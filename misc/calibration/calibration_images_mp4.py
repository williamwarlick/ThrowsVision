import cv2

cap = cv2.VideoCapture('../../April9Testing/apr9-1-left.MP4')
cap2 = cv2.VideoCapture('../../April9Testing/apr9-1-right.MP4')

if not cap.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both videos.")
    exit()

# Set the starting frame for each video
left_start_frame = 0  # Replace with the actual starting frame number for the left video
right_start_frame = 0  # Replace with the actual starting frame number for the right video

cap.set(cv2.CAP_PROP_POS_FRAMES, left_start_frame)
cap2.set(cv2.CAP_PROP_POS_FRAMES, right_start_frame)

num = 0
while cap.isOpened() and cap2.isOpened():
    succes1, img = cap.read()
    succes2, img2 = cap2.read()

    if not succes1 or not succes2:
        print("Failed to capture images from one or both videos.")
        break

    k = cv2.waitKey(5)
    if k == 27:  # wait for 'ESC' key to exit
        break
    elif k == ord('s'):  # wait for 's' key to save and exit
        if succes1 and succes2:
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
        break

# Release and destroy all windows before termination
cap.release()
cap2.release()
cv2.destroyAllWindows()