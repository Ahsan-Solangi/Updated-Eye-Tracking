"""
Face Landmark Detection — Webcam (New MediaPipe Tasks API)

Install:
    pip install mediapipe opencv-python

Download the model bundle once:
    wget -O face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Usage:
    python face_landmarks_webcam.py

Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import threading

# ── MediaPipe Tasks imports ──────────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# ── Drawing imports (still valid for visualization) ──────────────────────────
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh  # only used for connection map constants

# ── Model path ───────────────────────────────────────────────────────────────
MODEL_PATH = "D:\\PoC and Paper Implementation\\face_landmarker.task"

# ── Shared state for async results ──────────────────────────────────────────
latest_result: FaceLandmarkerResult | None = None
result_lock = threading.Lock()


def on_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """Callback invoked by the landmarker on every processed frame."""
    global latest_result
    with result_lock:
        latest_result = result


def draw_landmarks_on_image(frame: np.ndarray, result: FaceLandmarkerResult) -> np.ndarray:
    """Draw face mesh tesselation, contours, and irises on the frame."""
    if not result or not result.face_landmarks:
        return frame

    for face_landmarks in result.face_landmarks:
        # Convert Tasks API NormalizedLandmark list → proto NormalizedLandmarkList
        # so we can reuse the built-in drawing utilities.
        face_landmarks_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            mp.framework.formats.landmark_pb2.NormalizedLandmark(
                x=lm.x, y=lm.y, z=lm.z
            )
            for lm in face_landmarks
        ])

        # Tesselation (full mesh)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks_proto,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        # Contours (eyes, brows, lips, face oval)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks_proto,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        # Irises
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks_proto,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return frame


def main():
    global latest_result

    # ── Configure the FaceLandmarker in LIVE_STREAM mode ─────────────────────
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    print("Webcam opened — press 'q' to quit.")

    with FaceLandmarker.create_from_options(options) as landmarker:
        timestamp_ms = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the frame for a selfie-view
            frame = cv2.flip(frame, 1)

            # Convert BGR → RGB and wrap in mp.Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Send frame for async detection (non-blocking)
            landmarker.detect_async(mp_image, timestamp_ms)
            timestamp_ms += 33  # ~30 fps step; must be monotonically increasing

            # Draw the most recent result on the frame
            with result_lock:
                current_result = latest_result

            frame = draw_landmarks_on_image(frame, current_result)

            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Face Landmark Detection", frame)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()