import cv2 as cv

# Landmark indices for drawing
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382,
                    362, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155,
                     133, 246, 161, 160, 159, 158, 157, 173]


def draw_eye_landmarks(frame, face_landmarks):
    """Draw iris circles and eye contours on the webcam frame."""
    h, w = frame.shape[:2]

    # Draw eye contours (green lines)
    for contour in [LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR]:
        points = []
        for idx in contour:
            lm = face_landmarks[idx]
            points.append((int(lm.x * w), int(lm.y * h)))
        for i in range(len(points)):
            cv.line(frame, points[i], points[(i + 1) % len(points)],
                    (0, 255, 0), 1)

    # Draw iris points (cyan dots)
    for iris in [LEFT_IRIS, RIGHT_IRIS]:
        for idx in iris:
            lm = face_landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv.circle(frame, (x, y), 2, (255, 255, 0), -1)

        # Draw iris center with a larger circle
        center = face_landmarks[iris[0]]
        cx, cy = int(center.x * w), int(center.y * h)
        cv.circle(frame, (cx, cy), 4, (0, 255, 255), 2)


def draw_gaze_pointer(gaze_window, screen_x, screen_y):
    """Draw a gaze dot on the fullscreen gaze canvas.

    Clears the canvas and draws a circle at the predicted gaze position.
    """
    gaze_window[:] = 0  # Clear to black

    # Gaze dot with glow effect
    cv.circle(gaze_window, (screen_x, screen_y), 20, (0, 80, 0), -1)
    cv.circle(gaze_window, (screen_x, screen_y), 12, (0, 180, 0), -1)
    cv.circle(gaze_window, (screen_x, screen_y), 6, (0, 255, 0), -1)

    # Crosshair
    cv.line(gaze_window, (screen_x - 30, screen_y), (screen_x + 30, screen_y),
            (0, 150, 0), 1)
    cv.line(gaze_window, (screen_x, screen_y - 30), (screen_x, screen_y + 30),
            (0, 150, 0), 1)

    # Coordinates text
    cv.putText(gaze_window, f"({screen_x}, {screen_y})",
               (screen_x + 25, screen_y - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
