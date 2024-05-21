import cv2
import os
import sys
import subprocess


def run_calibration_images(left_video, right_video):
    command = f"python cali_img_2.py {left_video} {right_video}"
    subprocess.run(command, shell=True)


def run_stereo_calibration():
    command = "python stereo_calibration.py"
    subprocess.run(command, shell=True)


def run_testing_yolo(left_image, right_image):
    command = f"python testingYolo.py {left_image} {right_image}"
    subprocess.run(command, shell=True)


def run_stereo_vision_balls():
    command = f"python stereoVisionBalls.py"
    subprocess.run(command, shell=True)


def get_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, frame = cap.read()
    cap.release()
    return frame


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <experiment_number> <start_process>")
        print("start_process: 'calibration', 'yolo', or 'stereo'")
        sys.exit(1)

    # experiment_number = int(sys.argv[1])
    start_process = sys.argv[1]

    base_folder = "./test_footage"
    
    left_video = os.path.join(base_folder, f"calibration_check_L.mp4")
    right_video = os.path.join(base_folder, f"calibration_check_R.mp4")

    os.makedirs("./images/stereoLeft", exist_ok=True)
    os.makedirs("./images/stereoRight", exist_ok=True)

    if start_process == 'calibration':
        run_calibration_images(left_video, right_video)
        run_stereo_calibration()

    left_image = "./images/stereoLeft/left_last_frame.png"
    right_image = "./images/stereoRight/right_last_frame.png"

    if start_process in ['calibration', 'yolo']:
        left_last_frame = get_last_frame(left_video)
        right_last_frame = get_last_frame(right_video)

        cv2.imwrite(left_image, left_last_frame)
        cv2.imwrite(right_image, right_last_frame)

        run_testing_yolo(left_image, right_image)

    if start_process in ['calibration', 'yolo', 'stereo']:
        with open("proj_matrices.txt", "r") as f:
            proj_matrices = f.read().split()
            p1 = proj_matrices[0]
            p2 = proj_matrices[1]

        run_stereo_vision_balls()


if __name__ == "__main__":
    main()
