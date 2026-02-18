import cv2 as cv
import numpy as np
import mediapipe as mp
from gaze_estimation import GazeEstimator


def run_calibration(landmarker, gaze_estimator):
    """Run a 9-point calibration routine.

    Shows a fullscreen window with dots at 9 positions. The user looks at each
    dot and presses any key to record the gaze ratio at that point. After all
    9 points, computes a polynomial fit mapping gaze ratios to screen coords.

    Returns calibration coefficients dict, or None if cancelled.
    """
    # Window size for calibration
    win_w = 800
    win_h = 600
    cv.namedWindow("calibration", cv.WINDOW_NORMAL)
    cv.resizeWindow("calibration", win_w, win_h)

    # Get actual screen resolution for gaze mapping
    screen_w = 1920
    screen_h = 1080
    try:
        from ctypes import windll
        screen_w = windll.user32.GetSystemMetrics(0)
        screen_h = windll.user32.GetSystemMetrics(1)
    except Exception:
        pass

    # 3x3 grid of calibration points within the window
    margin_x = int(win_w * 0.1)
    margin_y = int(win_h * 0.1)
    cols = [margin_x, win_w // 2, win_w - margin_x]
    rows = [margin_y, win_h // 2, win_h - margin_y]

    calibration_points = []
    for y in rows:
        for x in cols:
            calibration_points.append((x, y))

    # Open webcam for calibration
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera for calibration.")
        return None

    collected_ratios = []  # (h_ratio, v_ratio) for each point
    frame_count = 0
    latest_result = [None]

    def on_result(result, output_image, timestamp_ms):
        latest_result[0] = result

    # We reuse the passed-in landmarker, but need to feed it frames
    # The caller's callback will update latest_result in the main module,
    # but here we need our own detection loop.
    # Instead, we'll create a temporary landmarker for calibration.
    from mediapipe.tasks.python import vision

    cal_options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path="D:\\PoC and Paper Implementation\\face_landmarker.task"
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    cal_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(cal_options)

    for i, (px, py) in enumerate(calibration_points):
        collecting = True
        while collecting:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            # Detect face landmarks
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = cal_landmarker.detect_for_video(mp_image, frame_count)

            # Draw calibration screen
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

            # Draw the target dot
            cv.circle(canvas, (px, py), 20, (0, 255, 0), -1)
            cv.circle(canvas, (px, py), 22, (255, 255, 255), 2)

            # Instructions
            text = f"Look at the green dot and press SPACE ({i + 1}/9)"
            text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (win_w - text_size[0]) // 2
            cv.putText(canvas, text, (text_x, win_h - 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv.imshow("calibration", canvas)
            key = cv.waitKey(5) & 0xFF

            if key == 27:  # Escape to cancel
                cap.release()
                cal_landmarker.close()
                cv.destroyWindow("calibration")
                return None

            if key == 32 and result.face_landmarks:  # Space to record
                face_lm = result.face_landmarks[0]
                h_ratio, v_ratio = gaze_estimator.get_gaze_ratios(face_lm)
                collected_ratios.append((h_ratio, v_ratio))
                collecting = False

    cap.release()
    cal_landmarker.close()
    cv.destroyWindow("calibration")

    # Compute mapping: gaze ratios → screen coordinates
    # Scale calibration points from window coords to screen coords
    h_ratios = np.array([r[0] for r in collected_ratios])
    v_ratios = np.array([r[1] for r in collected_ratios])
    screen_xs = np.array([p[0] * screen_w / win_w for p in calibration_points])
    screen_ys = np.array([p[1] * screen_h / win_h for p in calibration_points])

    # Fit a 2nd-degree polynomial for each axis
    x_coeffs = np.polyfit(h_ratios, screen_xs, 2)
    y_coeffs = np.polyfit(v_ratios, screen_ys, 2)

    coeffs = {
        'x_coeffs': x_coeffs,
        'y_coeffs': y_coeffs,
        'screen_w': screen_w,
        'screen_h': screen_h,
    }

    print("Calibration complete!")
    return coeffs
