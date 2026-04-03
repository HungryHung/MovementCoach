import cv2
from pose_estimator import PoseEstimator
from body_calibration import BodyCalibration
from view_detector import ViewDetector
from overlay import draw_skeleton, draw_feedback, draw_status


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    estimator = PoseEstimator()
    calibration = BodyCalibration()
    view_detector = ViewDetector()

    print("MovementCoach started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame for more natural interaction
        frame = cv2.flip(frame, 1)

        # Pose estimation
        result = estimator.estimate(frame)

        if result is not None:
            # Update body calibration
            calibration.update(result)

            # Detect view (front / side / unclear)
            view_mode = view_detector.detect_view(result)
        else:
            view_mode = "front"

        # Draw skeleton (green for now, no alerts yet)
        draw_skeleton(frame, result, has_alerts=False)

        # Status bar
        draw_status(frame, calibrated=calibration.is_calibrated(), view_mode=view_mode)

        cv2.imshow("MovementCoach", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    estimator.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
