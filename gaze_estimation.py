import numpy as np
from collections import deque

# MediaPipe Face Mesh landmark indices
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Eye corners (inner = near nose, outer = near temple)
LEFT_EYE_INNER = 263
LEFT_EYE_OUTER = 362
RIGHT_EYE_INNER = 33
RIGHT_EYE_OUTER = 133

# Eye top/bottom
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145


class GazeEstimator:
    def __init__(self, smoothing_window=5):
        self.calibration_coeffs = None
        self.history_x = deque(maxlen=smoothing_window)
        self.history_y = deque(maxlen=smoothing_window)

    def get_gaze_ratios(self, face_landmarks):
        """Calculate where the iris sits within the eye (0-1 range).

        Returns (horizontal_ratio, vertical_ratio) averaged across both eyes.
        0,0 = looking top-left, 1,1 = looking bottom-right.
        """
        lm = face_landmarks

        # Left eye horizontal ratio
        left_iris_x = lm[LEFT_IRIS_CENTER].x
        left_inner_x = lm[LEFT_EYE_INNER].x
        left_outer_x = lm[LEFT_EYE_OUTER].x
        left_h_range = left_outer_x - left_inner_x
        left_h_ratio = (left_iris_x - left_inner_x) / left_h_range if left_h_range != 0 else 0.5

        # Right eye horizontal ratio
        right_iris_x = lm[RIGHT_IRIS_CENTER].x
        right_inner_x = lm[RIGHT_EYE_INNER].x
        right_outer_x = lm[RIGHT_EYE_OUTER].x
        right_h_range = right_outer_x - right_inner_x
        right_h_ratio = (right_iris_x - right_inner_x) / right_h_range if right_h_range != 0 else 0.5

        # Left eye vertical ratio
        left_iris_y = lm[LEFT_IRIS_CENTER].y
        left_top_y = lm[LEFT_EYE_TOP].y
        left_bottom_y = lm[LEFT_EYE_BOTTOM].y
        left_v_range = left_bottom_y - left_top_y
        left_v_ratio = (left_iris_y - left_top_y) / left_v_range if left_v_range != 0 else 0.5

        # Right eye vertical ratio
        right_iris_y = lm[RIGHT_IRIS_CENTER].y
        right_top_y = lm[RIGHT_EYE_TOP].y
        right_bottom_y = lm[RIGHT_EYE_BOTTOM].y
        right_v_range = right_bottom_y - right_top_y
        right_v_ratio = (right_iris_y - right_top_y) / right_v_range if right_v_range != 0 else 0.5

        # Average both eyes
        h_ratio = (left_h_ratio + right_h_ratio) / 2
        v_ratio = (left_v_ratio + right_v_ratio) / 2

        return h_ratio, v_ratio

    def set_calibration(self, coeffs):
        """Set calibration coefficients from calibration routine.

        coeffs: dict with keys 'x_coeffs' and 'y_coeffs', each a tuple of
        (slope, intercept) from linear regression.
        """
        self.calibration_coeffs = coeffs

    def gaze_to_screen(self, h_ratio, v_ratio):
        """Map gaze ratios to screen coordinates using calibration data."""
        if self.calibration_coeffs is None:
            return None, None

        x_coeffs = self.calibration_coeffs['x_coeffs']
        y_coeffs = self.calibration_coeffs['y_coeffs']

        # Apply polynomial: coeffs are [degree-n, ..., degree-1, degree-0]
        screen_x = np.polyval(x_coeffs, h_ratio)
        screen_y = np.polyval(y_coeffs, v_ratio)

        # Clamp to screen bounds
        sw = self.calibration_coeffs['screen_w']
        sh = self.calibration_coeffs['screen_h']
        screen_x = max(0, min(sw, screen_x))
        screen_y = max(0, min(sh, screen_y))

        # Smooth with moving average
        self.history_x.append(screen_x)
        self.history_y.append(screen_y)
        smooth_x = int(np.mean(self.history_x))
        smooth_y = int(np.mean(self.history_y))

        return smooth_x, smooth_y
