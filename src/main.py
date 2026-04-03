import cv2
from pose_estimator import PoseEstimator
from body_calibration import BodyCalibration
from view_detector import ViewDetector
from feedback_engine import FeedbackEngine
from overlay import draw_skeleton, draw_feedback, draw_status


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    estimator = PoseEstimator()
    calibration = BodyCalibration()
    view_detector = ViewDetector()
    feedback = FeedbackEngine()

    # Make window resizable so video scales when maximized
    cv2.namedWindow("MovementCoach", cv2.WINDOW_NORMAL)

    print("MovementCoach started. Press 'q' to quit, 'r' to reset reps.")

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

            # Run feedback rules
            messages, rep_count, has_alerts = feedback.evaluate(
                result, view_mode,
                ratios=calibration.get_ratios(),
                near_side=view_detector.near_side,
            )
        else:
            view_mode = "front"
            messages = []
            rep_count = 0
            has_alerts = False

        # Draw skeleton (color changes based on alerts)
        draw_skeleton(frame, result, has_alerts=has_alerts)

        # Draw feedback messages and rep count
        draw_feedback(frame, messages, rep_count=rep_count)

        # Status bar
        draw_status(frame, calibrated=calibration.is_calibrated(),
                    view_mode=view_mode)

        cv2.imshow("MovementCoach", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            feedback.reset_reps()
            print("Reps reset.")

    estimator.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
