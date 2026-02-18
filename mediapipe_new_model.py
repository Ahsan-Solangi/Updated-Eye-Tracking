import mediapipe as mp
import cv2 as cv
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "D:\\PoC and Paper Implementation\\face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=on_result
)
landmarker = FaceLandmarker.create_from_options(options)

cap = cv.VideoCapture(0)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        break

    # Converting frame for mediapipe
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Send frame to landmarker
    frame_count += 1
    landmarker.detect_async(mp_image, frame_count)

    # Draw landmarks if available
    if latest_result and latest_result.face_landmarks:
        for face_landmarks in latest_result.face_landmarks:
            for landmark in face_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv.imshow("Face Landmarks", frame)
    if cv.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
landmarker.close()
