import cv2 as cv
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Detector
options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path="face_detection_short_range.tflite"),
    running_mode=vision.RunningMode.live_stream,
    min_detection_confidence=0.5
)

detector = vision.FaceDetector.create_from_options(options)

cap = cv.VideoCapture(0)

#Results
for detection in detection_result.detections:
    bbox = detection.bounding_box                               # .origin_x, .origin_y, .width, .height
    keypoints = detection.keypoints                             # list of 6 NormalizedKeypoint (eyes, nose, mouth, ears)
    print("Bounding box:", bbox)
    print("Keypoints:", keypoints)

    confidence = detection.categories[0].score    
    print("Confidence:", confidence)
