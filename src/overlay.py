import cv2
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec, draw_landmarks
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

_POSE_CONNECTIONS = PoseLandmarksConnections.POSE_LANDMARKS

# Drawing specs
DEFAULT_LANDMARK_STYLE = DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
DEFAULT_CONNECTION_STYLE = DrawingSpec(color=(0, 255, 0), thickness=2)
ALERT_LANDMARK_STYLE = DrawingSpec(color=(0, 100, 255), thickness=2, circle_radius=2)
ALERT_CONNECTION_STYLE = DrawingSpec(color=(0, 100, 255), thickness=2)


def draw_skeleton(frame, result, has_alerts=False):
    """Draw pose skeleton on the frame.

    Args:
        frame: BGR image (numpy array).
        result: PoseLandmarkerResult from PoseEstimator, or None.
        has_alerts: If True, draw skeleton in orange/red instead of green.
    """
    if result is None:
        return
    landmark_style = ALERT_LANDMARK_STYLE if has_alerts else DEFAULT_LANDMARK_STYLE
    connection_style = ALERT_CONNECTION_STYLE if has_alerts else DEFAULT_CONNECTION_STYLE
    # result.pose_landmarks is list[list[NormalizedLandmark]]; take first person
    landmarks = result.pose_landmarks[0]
    draw_landmarks(
        frame, landmarks, _POSE_CONNECTIONS,
        landmark_drawing_spec=landmark_style,
        connection_drawing_spec=connection_style,
    )


def draw_feedback(frame, messages, rep_count=0):
    """Draw feedback messages and rep count on the frame."""
    h, w = frame.shape[:2]

    # Rep count top-right
    cv2.putText(frame, f"Reps: {rep_count}", (w - 200, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    # Feedback messages top-left
    y = 45
    for msg in messages:
        # "+ N more" indicator in lighter color
        if msg.startswith("+"):
            color = (180, 180, 180)
            scale = 0.7
            thickness = 2
        else:
            color = (0, 100, 255)
            scale = 0.85
            thickness = 2
        cv2.putText(frame, msg, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        y += 35


def draw_status(frame, calibrated, view_mode):
    """Draw calibration status and view mode at the bottom."""
    h, w = frame.shape[:2]

    cal_text = "Calibrated" if calibrated else "Calibrating..."
    cal_color = (0, 255, 0) if calibrated else (0, 255, 255)
    cv2.putText(frame, cal_text, (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, cal_color, 3)

    if view_mode == "unclear":
        mode_text = "Please face camera or turn fully sideways"
        mode_color = (0, 255, 255)
    else:
        mode_text = f"Mode: {view_mode.capitalize()}"
        mode_color = (255, 255, 255)
    cv2.putText(frame, mode_text, (10, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 3)
