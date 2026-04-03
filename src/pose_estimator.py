import os

import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_DEFAULT_MODEL = os.path.join(_MODEL_DIR, "pose_landmarker_lite.task")


class PoseEstimator:
    def __init__(self, model_path=None, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        model_path = model_path or _DEFAULT_MODEL
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def estimate(self, frame):
        """Run pose estimation on a BGR frame.

        Returns a PoseLandmarkerResult, or None if no person detected.
        The result has:
          .pose_landmarks  — list[list[NormalizedLandmark]] (x, y, z normalized 0-1)
          .pose_world_landmarks — list[list[Landmark]] (real-world 3D in meters)
        """
        rgb = frame[:, :, ::-1]  # BGR to RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_timestamp_ms += 33  # ~30 fps
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
        if not result.pose_landmarks:
            return None
        return result

    def close(self):
        self._landmarker.close()
