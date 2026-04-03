import math
from collections import deque
from statistics import median

# Landmark indices for body segments
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_KNEE = 25
_RIGHT_KNEE = 26
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14

# Segments to measure: (name, landmark_a, landmark_b)
_SEGMENTS = [
    ("left_femur", _LEFT_HIP, _LEFT_KNEE),
    ("right_femur", _RIGHT_HIP, _RIGHT_KNEE),
    ("left_tibia", _LEFT_KNEE, _LEFT_ANKLE),
    ("right_tibia", _RIGHT_KNEE, _RIGHT_ANKLE),
    ("left_torso", _LEFT_SHOULDER, _LEFT_HIP),
    ("right_torso", _RIGHT_SHOULDER, _RIGHT_HIP),
    ("left_upper_arm", _LEFT_SHOULDER, _LEFT_ELBOW),
    ("right_upper_arm", _RIGHT_SHOULDER, _RIGHT_ELBOW),
]

_BUFFER_SIZE = 100
_VISIBILITY_THRESHOLD = 0.5
_IQR_THRESHOLD = 0.02  # relative IQR (iqr / median) below which a segment is "stable"
_DRIFT_THRESHOLD = 0.15  # 15% deviation triggers recalibration
_CALIBRATED_SAMPLE_INTERVAL = 30  # measure every Nth frame once calibrated


def _distance_3d(lm_a, lm_b):
    """Euclidean distance between two world landmarks."""
    return math.sqrt(
        (lm_a.x - lm_b.x) ** 2
        + (lm_a.y - lm_b.y) ** 2
        + (lm_a.z - lm_b.z) ** 2
    )


def _iqr(values):
    """Compute interquartile range of a list of values."""
    s = sorted(values)
    n = len(s)
    if n < 4:
        return float("inf")
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    return q3 - q1


class BodyCalibration:
    def __init__(self):
        self._buffers = {name: deque(maxlen=_BUFFER_SIZE) for name, _, _ in _SEGMENTS}
        self._medians = {}
        self._stable = {name: False for name, _, _ in _SEGMENTS}
        self._calibrated = False
        self._frame_count = 0

    def update(self, result):
        """Update calibration with new pose data.

        Args:
            result: PoseLandmarkerResult from PoseEstimator. Must not be None.
        """
        self._frame_count += 1

        # Once calibrated, only measure every Nth frame
        if self._calibrated and (self._frame_count % _CALIBRATED_SAMPLE_INTERVAL != 0):
            return

        world_landmarks = result.pose_world_landmarks[0]

        for name, idx_a, idx_b in _SEGMENTS:
            lm_a = world_landmarks[idx_a]
            lm_b = world_landmarks[idx_b]

            # Skip if either landmark has low visibility
            if lm_a.visibility < _VISIBILITY_THRESHOLD or lm_b.visibility < _VISIBILITY_THRESHOLD:
                continue

            dist = _distance_3d(lm_a, lm_b)
            buf = self._buffers[name]

            # Auto-reset: check for drift if we have a stable median
            if self._stable[name] and self._medians.get(name):
                if abs(dist - self._medians[name]) / self._medians[name] > _DRIFT_THRESHOLD:
                    self._reset()
                    return

            buf.append(dist)

            # Update median and stability
            if len(buf) >= 10:
                med = median(buf)
                self._medians[name] = med
                relative_iqr = _iqr(list(buf)) / med if med > 0 else float("inf")
                self._stable[name] = relative_iqr < _IQR_THRESHOLD

        # Key segments that must be stable for calibration
        key_segments = ["left_femur", "right_femur", "left_tibia", "right_tibia",
                        "left_torso", "right_torso"]
        self._calibrated = all(self._stable.get(s, False) for s in key_segments)

    def _reset(self):
        """Clear all buffers and restart calibration."""
        for buf in self._buffers.values():
            buf.clear()
        self._medians.clear()
        self._stable = {name: False for name, _, _ in _SEGMENTS}
        self._calibrated = False

    def is_calibrated(self):
        return self._calibrated

    def get_ratios(self):
        """Get body proportion ratios. Returns defaults if not calibrated."""
        femur = self._avg("left_femur", "right_femur")
        tibia = self._avg("left_tibia", "right_tibia")
        torso = self._avg("left_torso", "right_torso")

        if femur and tibia and torso:
            leg = femur + tibia
            return {
                "femur_tibia_ratio": femur / tibia if tibia > 0 else 1.0,
                "torso_leg_ratio": torso / leg if leg > 0 else 0.5,
                "femur_length": femur,
                "tibia_length": tibia,
                "torso_length": torso,
            }
        # Population-average fallback
        return {
            "femur_tibia_ratio": 1.0,
            "torso_leg_ratio": 0.5,
            "femur_length": None,
            "tibia_length": None,
            "torso_length": None,
        }

    def _avg(self, left_name, right_name):
        """Average the medians of left and right segments."""
        left = self._medians.get(left_name)
        right = self._medians.get(right_name)
        if left and right:
            return (left + right) / 2
        return left or right
