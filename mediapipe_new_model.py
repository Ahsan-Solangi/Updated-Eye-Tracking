import mediapipe as mp
import cv2 as cv
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gaze_estimation import GazeEstimator
from calibration import run_calibration
from visualization import draw_eye_landmarks, draw_gaze_pointer

model_path = "D:\\PoC and Paper Implementation\\face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# --- Gaze estimator ---
gaze = GazeEstimator(smoothing_window=5)

# --- Calibration ---
print("Starting calibration...")
print("Look at each green dot and press SPACE. Press ESC to cancel.")
coeffs = run_calibration(None, gaze)
if coeffs is None:
    print("Calibration cancelled. Exiting.")
    exit()
gaze.set_calibration(coeffs)
print("Calibration done! Starting gaze tracking...")

# --- Landmarker (with iris refinement) ---
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_faces=1,
    refine_landmarks=True,
    result_callback=on_result
)
landmarker = FaceLandmarker.create_from_options(options)

# --- Gaze display window ---
gaze_window = np.zeros((coeffs['screen_h'], coeffs['screen_w'], 3), dtype=np.uint8)
cv.namedWindow("Gaze", cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty("Gaze", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# --- Main loop ---
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

    # Process landmarks if available
    if latest_result and latest_result.face_landmarks:
        face_lm = latest_result.face_landmarks[0]

        # Draw eye landmarks on webcam feed
        draw_eye_landmarks(frame, face_lm)

        # Calculate gaze and map to screen
        h_ratio, v_ratio = gaze.get_gaze_ratios(face_lm)
        screen_x, screen_y = gaze.gaze_to_screen(h_ratio, v_ratio)

        if screen_x is not None:
            draw_gaze_pointer(gaze_window, screen_x, screen_y)
            cv.imshow("Gaze", gaze_window)

    cv.imshow("Face Landmarks", frame)
    if cv.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
landmarker.close()
