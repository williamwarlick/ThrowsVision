import cv2
import numpy as np
import torch
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

# YOLO Detector class
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
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, height, width, confidence=0.5):
        labels, cord = results
        detections = []
        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]
            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                if self.class_to_label(labels[i]) == 'sports ball':
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'sports ball'))

        return frame, detections

# Initialize video capture
cap = cv2.VideoCapture('../First Video Trails/MVI_7993.MP4') 

# Initialize YOLO detector
detector = YoloDetector(model_name=None)


os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize DeepSort object tracker
object_tracker = DeepSort(max_age=10,
                          n_init=1,
                          nms_max_overlap=1.0,
                          max_cosine_distance=0.2,
                          nn_budget=None,
                          override_track_class=None,  
                          embedder="mobilenet",
                          half=True,
                          bgr=True,
                          embedder_gpu=True,
                          embedder_model_name=None,
                          embedder_wts=None,
                          polygon=False,
                          today=None)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    results = detector.score_frame(img)
    img, detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5)

    tracks = object_tracker.update_tracks(detections, frame=img)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = ltrb
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()
