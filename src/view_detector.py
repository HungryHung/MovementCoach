import math

# Landmark indices
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24

# Thresholds for shoulder_width / torso_height ratio
_FRONT_THRESHOLD = 0.55   # above this → front-facing
_SIDE_THRESHOLD = 0.30    # below this → side-facing
# Between these two → "unclear" (diagonal)

_HYSTERESIS_FRAMES = 8  # require N consecutive frames before switching


def _pixel_distance(lm_a, lm_b):
    """2D pixel-space distance using normalized coordinates."""
    return math.sqrt((lm_a.x - lm_b.x) ** 2 + (lm_a.y - lm_b.y) ** 2)


class ViewDetector:
    def __init__(self):
        self._current_view = "front"
        self._pending_view = "front"
        self._pending_count = 0
        self._near_side = "left"

    def detect_view(self, result):
        """Detect whether the person is facing front, side, or at an unclear angle.

        Args:
            result: PoseLandmarkerResult from PoseEstimator.

        Returns:
            "front", "side", or "unclear"
        """
        landmarks = result.pose_landmarks[0]

        left_shoulder = landmarks[_LEFT_SHOULDER]
        right_shoulder = landmarks[_RIGHT_SHOULDER]
        left_hip = landmarks[_LEFT_HIP]
        right_hip = landmarks[_RIGHT_HIP]

        shoulder_width = _pixel_distance(left_shoulder, right_shoulder)

        # Reference: average torso height (shoulder to hip, both sides)
        left_torso = _pixel_distance(left_shoulder, landmarks[_LEFT_HIP])
        right_torso = _pixel_distance(right_shoulder, landmarks[_RIGHT_HIP])
        torso_height = (left_torso + right_torso) / 2

        if torso_height < 0.01:  # avoid division by zero
            raw_view = self._current_view
        else:
            ratio = shoulder_width / torso_height
            if ratio > _FRONT_THRESHOLD:
                raw_view = "front"
            elif ratio < _SIDE_THRESHOLD:
                raw_view = "side"
            else:
                raw_view = "unclear"

        # Hysteresis: require N consecutive frames of the same new view
        if raw_view != self._current_view:
            if raw_view == self._pending_view:
                self._pending_count += 1
            else:
                self._pending_view = raw_view
                self._pending_count = 1

            if self._pending_count >= _HYSTERESIS_FRAMES:
                self._current_view = raw_view
                self._pending_count = 0
        else:
            self._pending_count = 0

        # Determine near side when in side mode
        if self._current_view == "side":
            left_vis = (left_shoulder.visibility + landmarks[_LEFT_HIP].visibility) / 2
            right_vis = (right_shoulder.visibility + landmarks[_RIGHT_HIP].visibility) / 2
            self._near_side = "left" if left_vis >= right_vis else "right"

        return self._current_view

    @property
    def near_side(self):
        """Which side of the body is closer to the camera in side mode."""
        return self._near_side
