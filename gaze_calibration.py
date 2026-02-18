"""
Gaze Calibration -- Visual Axis Estimation via Kappa Angle Correction

This script builds on facelandmarks.py to estimate where a user is truly
looking (visual axis) by correcting the optical axis derived from iris
landmark positions. The default kappa angle (~5 deg) varies per person, so
we run a user-specific calibration to measure the offset.

Calibration Process:
  1. Show user 9 pairs of dots (blue = fixation target, green = estimation dot).
  2. User focuses on the BLUE dot; system estimates gaze toward the GREEN dot.
  3. The offset between estimated gaze and true gaze (blue dot) is recorded.
  4. After all 9 points, an affine correction model is fitted and saved as JSON.
  5. In live mode, the correction is applied so gaze estimates reflect the
     visual axis, not just the optical axis.

Install:
    pip install mediapipe opencv-python numpy

Download model:
    wget -O face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Usage:
    python gaze_calibration.py                   # Run calibration then live demo
    python gaze_calibration.py --calibrate-only   # Only calibrate, save profile
    python gaze_calibration.py --live-only         # Skip calibration, load profile
    python gaze_calibration.py --user alice        # Name the calibration profile

Press 'q' to quit at any time.
"""

import argparse
import ctypes
import json
import os
import sys
import time
import threading
from pathlib import Path

# -- Windows fix: MediaPipe looks for free() in the wrong DLL ----------------
# On Windows, free() lives in ucrtbase.dll, not inside libmediapipe.dll.
# We patch it before importing mediapipe so the DLL load doesn't crash.
if os.name == "nt":
    try:
        _ucrt = ctypes.CDLL("ucrtbase.dll")
        # Pre-load so mediapipe's ctypes call finds free() on the search path
        ctypes.CDLL("ucrtbase.dll", mode=ctypes.DEFAULT_MODE)
    except OSError:
        pass  # not on Windows or ucrtbase unavailable -- proceed normally

import cv2
import mediapipe as mp
import numpy as np

# -- MediaPipe Tasks imports --------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# -- No mp.solutions needed -- we draw landmarks manually with OpenCV ----------

# -- Configuration ------------------------------------------------------------
MODEL_PATH = "face_landmarker.task"           # adjust to your local path
CALIBRATION_DIR = "calibration_profiles"
DEFAULT_KAPPA_DEG = 5.0                        # typical human kappa angle

# MediaPipe iris landmark indices (within the 478-landmark face mesh)
# Left eye iris: 468-472 (468 = center), Right eye iris: 473-477 (473 = center)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
# Eye corner landmarks for reference frame
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
# Nose tip for head pose reference
NOSE_TIP = 1

# -- Shared state for async results -------------------------------------------
latest_result: FaceLandmarkerResult | None = None
result_lock = threading.Lock()


def on_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with result_lock:
        latest_result = result


# =============================================================================
#  GAZE ESTIMATION -- Optical Axis from Iris Landmarks
# =============================================================================

def get_iris_positions(face_landmarks, frame_w: int, frame_h: int):
    """
    Extract iris center positions in pixel coordinates for both eyes.
    Returns dict with 'left' and 'right' iris (x, y) or None if not found.
    """
    if face_landmarks is None or len(face_landmarks) < 478:
        return None  # need full 478 landmarks (incl. iris)

    left_iris = face_landmarks[LEFT_IRIS_CENTER]
    right_iris = face_landmarks[RIGHT_IRIS_CENTER]

    return {
        "left":  (left_iris.x * frame_w,  left_iris.y * frame_h),
        "right": (right_iris.x * frame_w, right_iris.y * frame_h),
    }


def get_eye_corners(face_landmarks, frame_w: int, frame_h: int):
    """Get inner/outer corners for both eyes in pixel coords."""
    if face_landmarks is None or len(face_landmarks) < 478:
        return None

    return {
        "left_inner":  (face_landmarks[LEFT_EYE_INNER].x * frame_w,
                        face_landmarks[LEFT_EYE_INNER].y * frame_h),
        "left_outer":  (face_landmarks[LEFT_EYE_OUTER].x * frame_w,
                        face_landmarks[LEFT_EYE_OUTER].y * frame_h),
        "right_inner": (face_landmarks[RIGHT_EYE_INNER].x * frame_w,
                        face_landmarks[RIGHT_EYE_INNER].y * frame_h),
        "right_outer": (face_landmarks[RIGHT_EYE_OUTER].x * frame_w,
                        face_landmarks[RIGHT_EYE_OUTER].y * frame_h),
    }


def estimate_optical_gaze(face_landmarks, frame_w: int, frame_h: int):
    """
    Estimate the raw optical-axis gaze point on screen from iris position
    relative to eye corners.

    Returns (gaze_x, gaze_y) in screen coordinates, or None.

    The idea: the iris position within the eye opening gives a normalized
    gaze direction. We map this to screen coordinates assuming the user
    faces roughly toward the camera/screen center.
    """
    iris = get_iris_positions(face_landmarks, frame_w, frame_h)
    corners = get_eye_corners(face_landmarks, frame_w, frame_h)
    if iris is None or corners is None:
        return None

    gaze_ratios = []
    for side in ["left", "right"]:
        ix, iy = iris[side]
        inner = corners[f"{side}_inner"]
        outer = corners[f"{side}_outer"]

        # Horizontal ratio: where is the iris between outer and inner corner?
        eye_width = np.linalg.norm(np.array(inner) - np.array(outer))
        if eye_width < 1e-3:
            continue

        # Distance from outer corner to iris, normalized by eye width
        ratio_x = (ix - outer[0]) / (inner[0] - outer[0] + 1e-6)

        # Vertical: iris y relative to the midline of the eye
        eye_mid_y = (inner[1] + outer[1]) / 2.0
        eye_height = eye_width * 0.4  # approximate eye aspect ratio
        ratio_y = (iy - eye_mid_y) / (eye_height + 1e-6)

        gaze_ratios.append((ratio_x, ratio_y))

    if not gaze_ratios:
        return None

    # Average both eyes
    avg_rx = np.mean([r[0] for r in gaze_ratios])
    avg_ry = np.mean([r[1] for r in gaze_ratios])

    # Map to screen coordinates
    # ratio_x ~ 0 means looking right (from camera's perspective = screen left for mirrored)
    # ratio_x ~ 1 means looking left
    # We invert so that screen-left corresponds to low x
    gaze_x = avg_rx * frame_w
    gaze_y = (0.5 + avg_ry) * frame_h  # center vertically + offset

    return (gaze_x, gaze_y)


# =============================================================================
#  CALIBRATION PROFILE -- Save / Load
# =============================================================================

def profile_path(user: str) -> str:
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    return os.path.join(CALIBRATION_DIR, f"{user}_calibration.json")


def save_calibration(user: str, calib_data: dict):
    """Save calibration correction matrix and metadata to JSON."""
    path = profile_path(user)
    # Convert numpy arrays to lists for JSON
    serializable = {}
    for k, v in calib_data.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Calibration saved -> {path}")


def load_calibration(user: str) -> dict | None:
    """Load a previously saved calibration profile."""
    path = profile_path(user)
    if not os.path.exists(path):
        print(f"No calibration found for user '{user}' at {path}")
        return None
    with open(path) as f:
        data = json.load(f)
    # Restore numpy arrays
    for key in ["affine_matrix"]:
        if key in data and data[key] is not None:
            data[key] = np.array(data[key])
    print(f"Calibration loaded <- {path}")
    return data


# =============================================================================
#  CORRECTION MODEL -- Affine Transform from Optical -> Visual Axis
# =============================================================================

def fit_affine_correction(raw_gaze_points: list, true_screen_points: list) -> np.ndarray:
    """
    Fit a 2D affine transformation that maps raw (optical-axis) gaze
    estimates to corrected (visual-axis) screen coordinates.

    raw_gaze_points:   list of (x, y) -- where the system thinks you looked
    true_screen_points: list of (x, y) -- where you actually looked (blue dots)

    Returns a 2x3 affine matrix A such that:
        [corrected_x, corrected_y]^T = A @ [raw_x, raw_y, 1]^T
    """
    assert len(raw_gaze_points) >= 3, "Need at least 3 calibration points"

    src = np.array(raw_gaze_points, dtype=np.float64)
    dst = np.array(true_screen_points, dtype=np.float64)

    # Build system: for each point  dst_i = A @ [src_i, 1]
    # We solve via least-squares for the 2x3 matrix A
    N = len(src)
    # Augment source with ones column
    ones = np.ones((N, 1), dtype=np.float64)
    src_aug = np.hstack([src, ones])  # Nx3

    # Solve for each target dimension separately
    # dst_x = src_aug @ a_x   and   dst_y = src_aug @ a_y
    a_x, _, _, _ = np.linalg.lstsq(src_aug, dst[:, 0], rcond=None)
    a_y, _, _, _ = np.linalg.lstsq(src_aug, dst[:, 1], rcond=None)

    affine = np.vstack([a_x, a_y])  # 2x3
    return affine


def apply_correction(raw_gaze: tuple, affine: np.ndarray) -> tuple:
    """Apply the affine correction to a raw gaze point."""
    pt = np.array([raw_gaze[0], raw_gaze[1], 1.0])
    corrected = affine @ pt
    return (float(corrected[0]), float(corrected[1]))


# =============================================================================
#  CALIBRATION PROCEDURE -- 9 Dual-Dot Pairs
# =============================================================================

def generate_calibration_points(screen_w: int, screen_h: int):
    """
    Generate 9 calibration positions in a 3x3 grid.
    Each position has:
      - blue_dot: where the user should look (fixation target)
      - green_dot: offset position for gaze estimation reference

    The green dot is placed at a fixed angular offset from the blue dot,
    simulating the expected kappa-angle discrepancy.
    """
    margin_x = int(screen_w * 0.15)
    margin_y = int(screen_h * 0.15)

    cols = np.linspace(margin_x, screen_w - margin_x, 3).astype(int)
    rows = np.linspace(margin_y, screen_h - margin_y, 3).astype(int)

    # The green dot offset simulates what an uncalibrated system would
    # estimate (~5 deg kappa -> roughly 50-80 pixel offset at typical distance)
    kappa_offset_px = int(min(screen_w, screen_h) * 0.04)

    points = []
    for r in rows:
        for c in cols:
            blue = (int(c), int(r))
            # Green dot offset in a consistent direction (nasal/up shift)
            green = (int(c) + kappa_offset_px, int(r) - kappa_offset_px // 2)
            points.append({"blue": blue, "green": green})

    return points


def run_calibration(landmarker, cap, screen_w: int, screen_h: int, user: str):
    """
    Run the interactive 9-point calibration.

    For each point:
      1. Display blue dot (fixation) and green dot (estimation reference).
      2. Wait for user to fixate on blue dot, then press SPACE to record.
      3. Collect multiple gaze samples while user holds fixation.
      4. Store the median raw gaze estimate paired with the blue dot location.

    Returns the calibration data dict (including the affine correction matrix).
    """
    global latest_result

    cal_points = generate_calibration_points(screen_w, screen_h)
    raw_gaze_collected = []    # where system estimated the gaze
    true_targets = []          # where the user was actually looking (blue dots)

    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    timestamp_ms = 0
    print("\n+--------------------------------------------------------------+")
    print("|              GAZE CALIBRATION - 9 Point Procedure          |")
    print("|                                                            |")
    print("|  * Look at the BLUE dot                                    |")
    print("|  * Press SPACE when you are fixating steadily              |")
    print("|  * Hold your gaze for ~2 seconds while samples collect     |")
    print("|  * Press 'q' to abort calibration                         |")
    print("+--------------------------------------------------------------+\n")

    for idx, point in enumerate(cal_points):
        blue = point["blue"]
        green = point["green"]
        samples = []
        collecting = False
        collect_start = 0
        COLLECT_DURATION = 2.0  # seconds to collect samples

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            # Feed to landmarker
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp_ms)
            timestamp_ms += 33

            # Build calibration display
            display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            display[:] = (30, 30, 30)  # dark background

            # Draw green dot (estimation reference)
            cv2.circle(display, green, 18, (0, 200, 0), -1)
            cv2.circle(display, green, 20, (0, 255, 0), 2)

            # Draw blue dot (fixation target) -- on top
            cv2.circle(display, blue, 18, (200, 100, 0), -1)
            cv2.circle(display, blue, 20, (255, 150, 0), 2)
            # Small white center for precise fixation
            cv2.circle(display, blue, 4, (255, 255, 255), -1)

            # Status text
            status = f"Point {idx + 1}/9  --  Look at the BLUE dot"
            cv2.putText(display, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            if not collecting:
                cv2.putText(display, "Press SPACE when fixating on blue dot",
                            (20, screen_h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            else:
                elapsed = time.time() - collect_start
                remaining = max(0, COLLECT_DURATION - elapsed)
                bar_w = int((elapsed / COLLECT_DURATION) * 300)
                cv2.rectangle(display, (20, screen_h - 60), (20 + bar_w, screen_h - 40),
                              (0, 255, 100), -1)
                cv2.rectangle(display, (20, screen_h - 60), (320, screen_h - 40),
                              (100, 100, 100), 2)
                cv2.putText(display, f"Collecting... {remaining:.1f}s",
                            (340, screen_h - 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)

                # Collect gaze samples
                with result_lock:
                    current = latest_result
                if current and current.face_landmarks:
                    gaze = estimate_optical_gaze(current.face_landmarks[0],
                                                 frame_w, frame_h)
                    if gaze is not None:
                        # Scale from webcam frame coords to screen coords
                        sx = gaze[0] / frame_w * screen_w
                        sy = gaze[1] / frame_h * screen_h
                        samples.append((sx, sy))

                        # Draw estimated gaze on display (red dot)
                        cv2.circle(display, (int(sx), int(sy)), 6, (0, 0, 255), -1)

                if elapsed >= COLLECT_DURATION:
                    break

            # Overlay small webcam feed in corner
            thumb_h, thumb_w = 120, 160
            thumb = cv2.resize(frame, (thumb_w, thumb_h))
            display[10:10+thumb_h, screen_w-thumb_w-10:screen_w-10] = thumb

            cv2.imshow("Calibration", display)

            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                print("Calibration aborted.")
                cv2.destroyWindow("Calibration")
                return None
            elif key == ord(" ") and not collecting:
                collecting = True
                collect_start = time.time()
                samples = []

        # Process samples for this point
        if len(samples) >= 5:
            median_gaze = (float(np.median([s[0] for s in samples])),
                           float(np.median([s[1] for s in samples])))
            raw_gaze_collected.append(median_gaze)
            true_targets.append((float(blue[0]), float(blue[1])))
            print(f"  Point {idx+1}: target={blue}, "
                  f"estimated=({median_gaze[0]:.0f}, {median_gaze[1]:.0f}), "
                  f"samples={len(samples)}")
        else:
            print(f"  Point {idx+1}: insufficient samples ({len(samples)}), skipping.")

    cv2.destroyWindow("Calibration")

    # -- Fit correction model ---------------------------------------------
    if len(raw_gaze_collected) < 4:
        print("ERROR: Not enough valid calibration points (need >= 4).")
        return None

    affine = fit_affine_correction(raw_gaze_collected, true_targets)

    # Compute calibration accuracy
    errors = []
    for raw, true in zip(raw_gaze_collected, true_targets):
        corrected = apply_correction(raw, affine)
        err = np.linalg.norm(np.array(corrected) - np.array(true))
        errors.append(err)

    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))

    # Estimate effective kappa angle (rough approximation)
    avg_offset = np.mean(np.array(raw_gaze_collected) - np.array(true_targets), axis=0)
    offset_magnitude = np.linalg.norm(avg_offset)
    # Assume ~57 px per degree at typical viewing distance
    estimated_kappa = offset_magnitude / 57.0

    calib_data = {
        "user": user,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_points": len(raw_gaze_collected),
        "screen_w": screen_w,
        "screen_h": screen_h,
        "affine_matrix": affine,
        "mean_error_px": mean_err,
        "max_error_px": max_err,
        "estimated_kappa_deg": float(estimated_kappa),
        "avg_offset_xy": avg_offset.tolist(),
        "raw_gaze_points": raw_gaze_collected,
        "true_target_points": true_targets,
    }

    print(f"\n-- Calibration Results ------------------------------------")
    print(f"  Points used:        {len(raw_gaze_collected)}/9")
    print(f"  Mean residual error: {mean_err:.1f} px")
    print(f"  Max residual error:  {max_err:.1f} px")
    print(f"  Est. kappa angle:    {estimated_kappa:.2f} deg")
    print(f"  Avg offset (x, y):   ({avg_offset[0]:.1f}, {avg_offset[1]:.1f}) px")
    print(f"-------------------------------------------------------------\n")

    save_calibration(user, calib_data)
    return calib_data


# =============================================================================
#  DRAWING UTILITIES
# =============================================================================

def draw_landmarks_on_image(frame, result):
    """
    Draw face mesh landmarks on frame using pure OpenCV.
    No mp.solutions required -- just draws iris circles, eye contours,
    and face oval from the raw landmark coordinates.
    """
    if not result or not result.face_landmarks:
        return frame

    h, w = frame.shape[:2]

    for face_landmarks in result.face_landmarks:
        lms = face_landmarks  # list of NormalizedLandmark

        # -- Draw all 468 face mesh points as tiny dots -------------------
        for i, lm in enumerate(lms[:468]):
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(frame, (px, py), 1, (100, 100, 100), -1)

        # -- Draw iris landmarks (468-477) with larger green circles ------
        if len(lms) >= 478:
            # Left iris: 468 (center), 469-472 (ring)
            for i in range(468, 473):
                px = int(lms[i].x * w)
                py = int(lms[i].y * h)
                color = (0, 255, 0) if i == 468 else (0, 200, 0)
                radius = 3 if i == 468 else 2
                cv2.circle(frame, (px, py), radius, color, -1)

            # Right iris: 473 (center), 474-477 (ring)
            for i in range(473, 478):
                px = int(lms[i].x * w)
                py = int(lms[i].y * h)
                color = (0, 255, 0) if i == 473 else (0, 200, 0)
                radius = 3 if i == 473 else 2
                cv2.circle(frame, (px, py), radius, color, -1)

            # Draw iris circles (fit circle through the 4 ring points)
            for center_idx, ring_start in [(468, 469), (473, 474)]:
                cx = int(lms[center_idx].x * w)
                cy = int(lms[center_idx].y * h)
                # Average distance from center to ring points = radius
                dists = []
                for ri in range(ring_start, ring_start + 4):
                    rx = int(lms[ri].x * w)
                    ry = int(lms[ri].y * h)
                    dists.append(np.sqrt((rx - cx)**2 + (ry - cy)**2))
                r = int(np.mean(dists))
                cv2.circle(frame, (cx, cy), r, (0, 255, 0), 1)

        # -- Draw eye contours --------------------------------------------
        # Left eye contour indices
        LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                    173, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                     466, 388, 387, 386, 385, 384, 398]

        for eye_indices in [LEFT_EYE, RIGHT_EYE]:
            pts = []
            for idx in eye_indices:
                if idx < len(lms):
                    pts.append((int(lms[idx].x * w), int(lms[idx].y * h)))
            if len(pts) > 2:
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.polylines(frame, [pts_arr], True, (0, 200, 200), 1)

        # -- Draw face oval -----------------------------------------------
        FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                     288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                     150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54,
                     103, 67, 109]
        oval_pts = []
        for idx in FACE_OVAL:
            if idx < len(lms):
                oval_pts.append((int(lms[idx].x * w), int(lms[idx].y * h)))
        if len(oval_pts) > 2:
            cv2.polylines(frame, [np.array(oval_pts, np.int32)],
                          True, (80, 80, 80), 1)

    return frame


# =============================================================================
#  LIVE GAZE DEMO -- With Correction Applied
# =============================================================================

def run_live_gaze(landmarker, cap, screen_w: int, screen_h: int,
                  calib_data: dict | None):
    """
    Live gaze visualization with optional calibration correction.
    Shows both raw (optical axis) and corrected (visual axis) gaze.
    """
    global latest_result

    affine = None
    if calib_data and "affine_matrix" in calib_data:
        affine = calib_data["affine_matrix"]
        if isinstance(affine, list):
            affine = np.array(affine)

    cv2.namedWindow("Live Gaze", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Live Gaze", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    timestamp_ms = 0
    # Smoothing buffers
    raw_history = []
    corrected_history = []
    SMOOTH_N = 5

    print("Live gaze tracking -- press 'q' to quit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, timestamp_ms)
        timestamp_ms += 33

        with result_lock:
            current = latest_result

        # Build display
        display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        display[:] = (25, 25, 25)

        # Draw a subtle grid for reference
        for i in range(1, 3):
            x = int(screen_w * i / 3)
            y = int(screen_h * i / 3)
            cv2.line(display, (x, 0), (x, screen_h), (40, 40, 40), 1)
            cv2.line(display, (0, y), (screen_w, y), (40, 40, 40), 1)

        raw_gaze = None
        corrected_gaze = None

        if current and current.face_landmarks:
            face_lm = current.face_landmarks[0]
            gaze = estimate_optical_gaze(face_lm, frame_w, frame_h)

            if gaze is not None:
                # Scale to screen
                raw_gaze = (gaze[0] / frame_w * screen_w,
                            gaze[1] / frame_h * screen_h)

                # Smooth raw gaze
                raw_history.append(raw_gaze)
                if len(raw_history) > SMOOTH_N:
                    raw_history.pop(0)
                smooth_raw = (np.mean([p[0] for p in raw_history]),
                              np.mean([p[1] for p in raw_history]))

                # Draw raw gaze (red -- optical axis)
                rx, ry = int(smooth_raw[0]), int(smooth_raw[1])
                cv2.circle(display, (rx, ry), 14, (0, 0, 180), -1)
                cv2.circle(display, (rx, ry), 16, (0, 0, 255), 2)
                cv2.putText(display, "Optical", (rx + 20, ry - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Apply correction if available
                if affine is not None:
                    corrected_gaze = apply_correction(raw_gaze, affine)

                    corrected_history.append(corrected_gaze)
                    if len(corrected_history) > SMOOTH_N:
                        corrected_history.pop(0)
                    smooth_corr = (np.mean([p[0] for p in corrected_history]),
                                   np.mean([p[1] for p in corrected_history]))

                    # Draw corrected gaze (green -- visual axis)
                    cx, cy = int(smooth_corr[0]), int(smooth_corr[1])
                    cv2.circle(display, (cx, cy), 14, (0, 180, 0), -1)
                    cv2.circle(display, (cx, cy), 16, (0, 255, 0), 2)
                    cv2.putText(display, "Visual", (cx + 20, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Draw connecting line
                    cv2.line(display, (rx, ry), (cx, cy), (100, 100, 0), 1)

        # HUD info
        y_pos = 30
        cv2.putText(display, "Live Gaze Tracking", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_pos += 30
        cv2.circle(display, (20, y_pos - 5), 6, (0, 0, 255), -1)
        cv2.putText(display, "Optical axis (raw)", (35, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_pos += 25
        if affine is not None:
            cv2.circle(display, (20, y_pos - 5), 6, (0, 255, 0), -1)
            cv2.putText(display, "Visual axis (corrected)", (35, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y_pos += 25
            cv2.putText(display,
                        f"Kappa ~{calib_data.get('estimated_kappa_deg', '?'):.1f} deg  |  "
                        f"Calib err: {calib_data.get('mean_error_px', '?'):.0f} px",
                        (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        else:
            cv2.putText(display, "No calibration loaded -- showing raw only",
                        (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)

        # Webcam thumbnail with mesh
        frame_annotated = draw_landmarks_on_image(frame.copy(), current)
        thumb_h, thumb_w = 150, 200
        thumb = cv2.resize(frame_annotated, (thumb_w, thumb_h))
        display[screen_h-thumb_h-10:screen_h-10,
                screen_w-thumb_w-10:screen_w-10] = thumb

        cv2.putText(display, "Press 'q' to quit", (20, screen_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        cv2.imshow("Live Gaze", display)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cv2.destroyWindow("Live Gaze")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gaze calibration: estimate visual axis from optical axis")
    parser.add_argument("--user", default="default",
                        help="Calibration profile name (default: 'default')")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="Only run calibration, don't start live demo")
    parser.add_argument("--live-only", action="store_true",
                        help="Skip calibration, load existing profile")
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"Path to face_landmarker.task (default: {MODEL_PATH})")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    # -- Set up face landmarker -------------------------------------------
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("Download it with:")
        print("  wget -O face_landmarker.task "
              "https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        sys.exit(1)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    # Get screen size (approximate -- OpenCV fullscreen will use actual screen)
    # Try to detect screen resolution; fall back to 1920x1080
    try:
        import screeninfo
        monitor = screeninfo.get_monitors()[0]
        screen_w, screen_h = monitor.width, monitor.height
    except ImportError:
        screen_w, screen_h = 1920, 1080
        print(f"(screeninfo not installed -- assuming {screen_w}x{screen_h}; "
              f"pip install screeninfo for accuracy)")

    print(f"Screen: {screen_w}x{screen_h}  |  User: '{args.user}'")

    calib_data = None

    with FaceLandmarker.create_from_options(options) as landmarker:
        # -- Calibration phase --------------------------------------------
        if not args.live_only:
            calib_data = run_calibration(landmarker, cap,
                                         screen_w, screen_h, args.user)
            if calib_data is None and not args.calibrate_only:
                print("Calibration failed/aborted. Continuing without correction.")

        # -- Load existing calibration if live-only -----------------------
        if args.live_only or (calib_data is None and not args.calibrate_only):
            calib_data = load_calibration(args.user)

        # -- Live demo ----------------------------------------------------
        if not args.calibrate_only:
            run_live_gaze(landmarker, cap, screen_w, screen_h, calib_data)

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
