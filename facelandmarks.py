"""
Face Landmark Detection — Webcam (New MediaPipe Tasks API, No Legacy mp.solutions)

Install:
    pip install mediapipe opencv-python numpy

Download the model bundle once (run in your terminal):
    On Windows (PowerShell):
        Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker.task"
    On Linux/macOS:
        wget -O face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Usage:
    python face_landmarks_webcam.py

Press 'q' to quit.
"""

import os
import sys
import ctypes
import cv2
import numpy as np
import threading

# ═══════════════════════════════════════════════════════════════════════════════
# Windows fix: MediaPipe looks for free() inside its own DLL, but on Windows
# free() lives in ucrtbase.dll. We patch it before importing mediapipe.
# ═══════════════════════════════════════════════════════════════════════════════
if os.name == "nt":
    try:
        ucrt = ctypes.CDLL("ucrtbase.dll")
        # Pre-load so MediaPipe's ctypes lookup finds free()
    except OSError:
        pass  # Very old Windows without ucrtbase — unlikely but safe to skip

import mediapipe as mp

# If on Windows, patch the mediapipe native library so it can find free()
if os.name == "nt":
    try:
        import mediapipe.tasks.python.core as _mp_core  # trigger DLL load
        # Find the loaded mediapipe DLL and attach free from ucrt
        for name, mod in sorted(sys.modules.items()):
            if hasattr(mod, "__file__") and mod.__file__ and "mediapipe" in str(mod.__file__):
                pass  # just ensuring modules are loaded
        ucrt = ctypes.CDLL("ucrtbase.dll")
        ucrt.free.argtypes = [ctypes.c_void_p]
        ucrt.free.restype = None
    except Exception:
        pass

# ── MediaPipe Tasks API imports ──────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# ── Model path ───────────────────────────────────────────────────────────────
MODEL_PATH = "D:\\PoC and Paper Implementation\\face_landmarker.task"

# ═══════════════════════════════════════════════════════════════════════════════
# Face mesh connection definitions (from MediaPipe source, avoids mp.solutions)
# ═══════════════════════════════════════════════════════════════════════════════

FACEMESH_LIPS = frozenset([
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405),
    (405, 321), (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291), (78, 95),
    (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318),
    (318, 324), (324, 308), (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
])

FACEMESH_LEFT_EYE = frozenset([
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
])

FACEMESH_LEFT_EYEBROW = frozenset([
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334),
    (334, 296), (296, 336),
])

FACEMESH_RIGHT_EYE = frozenset([
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
])

FACEMESH_RIGHT_EYEBROW = frozenset([
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66),
    (66, 107),
])

FACEMESH_FACE_OVAL = frozenset([
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
])

FACEMESH_LEFT_IRIS = frozenset([
    (474, 475), (475, 476), (476, 477), (477, 474),
])

FACEMESH_RIGHT_IRIS = frozenset([
    (469, 470), (470, 471), (471, 472), (472, 469),
])

FACEMESH_CONTOURS = frozenset().union(
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW,
    FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL,
)

FACEMESH_IRISES = frozenset().union(FACEMESH_LEFT_IRIS, FACEMESH_RIGHT_IRIS)


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing utilities (pure OpenCV, no mp.solutions needed)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_connections(
    frame: np.ndarray,
    landmarks: list,
    connections: frozenset,
    color: tuple = (0, 255, 0),
    thickness: int = 1,
):
    """Draw connection lines between landmark pairs on the frame."""
    h, w = frame.shape[:2]
    for idx_a, idx_b in connections:
        if idx_a >= len(landmarks) or idx_b >= len(landmarks):
            continue
        pt_a = (int(landmarks[idx_a].x * w), int(landmarks[idx_a].y * h))
        pt_b = (int(landmarks[idx_b].x * w), int(landmarks[idx_b].y * h))
        cv2.line(frame, pt_a, pt_b, color, thickness, cv2.LINE_AA)


def draw_landmarks_on_image(
    frame: np.ndarray,
    result: FaceLandmarkerResult,
    draw_mesh_points: bool = True,
) -> np.ndarray:
    """Draw face contours, irises, and optionally all 478 points."""
    if not result or not result.face_landmarks:
        return frame

    for face_landmarks in result.face_landmarks:
        # Optionally draw every landmark as a tiny dot (gives a mesh feel)
        if draw_mesh_points:
            h, w = frame.shape[:2]
            for lm in face_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (128, 128, 128), -1, cv2.LINE_AA)

        # Contours: face oval, eyes, eyebrows, lips
        draw_connections(frame, face_landmarks, FACEMESH_FACE_OVAL, (180, 180, 180), 1)
        draw_connections(frame, face_landmarks, FACEMESH_LEFT_EYE, (0, 255, 0), 1)
        draw_connections(frame, face_landmarks, FACEMESH_RIGHT_EYE, (0, 255, 0), 1)
        draw_connections(frame, face_landmarks, FACEMESH_LEFT_EYEBROW, (0, 200, 200), 1)
        draw_connections(frame, face_landmarks, FACEMESH_RIGHT_EYEBROW, (0, 200, 200), 1)
        draw_connections(frame, face_landmarks, FACEMESH_LIPS, (0, 128, 255), 1)

        # Irises
        draw_connections(frame, face_landmarks, FACEMESH_IRISES, (0, 255, 255), 2)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Async result handling
# ═══════════════════════════════════════════════════════════════════════════════

latest_result: FaceLandmarkerResult | None = None
result_lock = threading.Lock()


def on_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with result_lock:
        latest_result = result


# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global latest_result

    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found.")
        print("Download it with:")
        print(f'  wget -O {MODEL_PATH} https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task')
        sys.exit(1)

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

            # Mirror for selfie-view
            frame = cv2.flip(frame, 1)

            # Convert BGR → RGB and wrap in mp.Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Send frame (non-blocking)
            landmarker.detect_async(mp_image, timestamp_ms)
            timestamp_ms += 33  # must be monotonically increasing

            # Draw latest result
            with result_lock:
                current_result = latest_result

            frame = draw_landmarks_on_image(frame, current_result)

            cv2.putText(
                frame, "Press 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("Face Landmark Detection", frame)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()