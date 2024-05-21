import cv2
import numpy as np
import torch
import os


class YoloDetector():
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 1.1  # Further adjust to better detect smaller objects
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, height, width, confidence_shot=0.3):
        labels, cord = results
        detections = []
        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]
            label = self.class_to_label(labels[i])
            confidence = row[4]

            if label == 'sports ball' and confidence >= confidence_shot:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], confidence, 'Shot'))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Shot: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame, detections


image_path = './images/basic.jpg' 
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The image path {image_path} does not exist. Please check the path.")

img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Failed to read the image from {image_path}. Please check the file integrity.")

detector = YoloDetector(model_name=None)  # can be replaced with any fine tuned model of your liking

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

results = detector.score_frame(img)
img, detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence_shot=0.3)

cv2.imshow('img', img)
cv2.imwrite('./images/shotcheck.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
