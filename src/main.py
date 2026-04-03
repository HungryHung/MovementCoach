import cv2
from pose_estimator import PoseEstimator
from overlay import draw_skeleton, draw_feedback, draw_status


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    estimator = PoseEstimator()

    print("MovementCoach started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame for more natural interaction
        frame = cv2.flip(frame, 1)

        # Pose estimation
        result = estimator.estimate(frame)

        # Draw skeleton (green for now, no alerts yet)
        draw_skeleton(frame, result, has_alerts=False)

        # Placeholder status (will be replaced by real calibration/view detection)
        draw_status(frame, calibrated=False, view_mode="front")

        cv2.imshow("MovementCoach", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    estimator.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
