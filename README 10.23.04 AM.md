Stereo Vision Depth Estimation
=============================
A library for live and (working on) past video recording stero camera calibration and depth estimation.

How to run it 
-
```
pip install -r requirements. txt
install yolov5 model
calibration_images.py (add flags for live or past)
stereo_calibration.py (flags for chessboard size and picture resolution)
testingYolo.py 
stereo_visionBalls.py (w/ args from calibration_values.text)
```
Here is what is happening in each file:

testingYolo:
Used an yolomodel(todo: fine tuned for this task w/ roboflow) to idenity
all the cordinations of sports ball in the given image.




Before you Begin 
Throught my research, I ran into a variety of roadblocks unknowleing. If you don't get these thigns right from the beggining, 
a
you can run into a lot of problems later and remains unsure of the solution.
1. Ensure use of a quality calibration frame.
   2. Calibration is a very particular matter that I overlooked for the majority of my project.
   3. I initally used a piece of 6X4 chessboard on printer paper. I belived that it was scaled properly and made measurmetns to confirm this. 