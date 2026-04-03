import math
import numpy as np

# --- Landmark indices ---
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_KNEE = 25
_RIGHT_KNEE = 26
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28
_LEFT_HEEL = 29
_RIGHT_HEEL = 30
_LEFT_FOOT_INDEX = 31
_RIGHT_FOOT_INDEX = 32
_NOSE = 0
_LEFT_EAR = 7
_RIGHT_EAR = 8
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16

# --- Priority levels (lower = more important) ---
_PRIORITY_HIGH = 1     # safety: valgus, back alignment
_PRIORITY_MEDIUM = 2   # form: depth, knee-over-toe, balance
_PRIORITY_LOW = 3      # refinement: stance width, weight distribution

# --- Rep counter thresholds (degrees) ---
_SQUAT_ENTER_ANGLE = 130   # knee angle below this → in squat
_SQUAT_EXIT_ANGLE = 160    # knee angle above this → standing

# --- Feedback thresholds ---
_DEPTH_WARNING_ANGLE = 120        # knee angle above this during squat → too shallow
_VALGUS_RATIO_THRESHOLD = 0.75   # knee_dist / ankle_dist below this → valgus
_BALANCE_DEVIATION = 0.12         # CoG x deviation from midline (normalized coords)
_FORWARD_LEAN_ANGLE = 145         # shoulder-hip-knee angle below this → leaning
_MAX_DISPLAY_MESSAGES = 2

# --- Winter's segment mass model (% of total body mass) ---
_SEGMENT_MASS = {
    "head": 0.081,
    "torso": 0.497,
    "upper_arm": 0.028,   # per arm
    "forearm": 0.016,     # per arm
    "thigh": 0.100,       # per leg
    "shank": 0.047,       # per leg
    "foot": 0.014,        # per foot
}


def calc_angle(a, b, c):
    """Compute angle at point b (in degrees) given three landmarks with x, y, z."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def calc_2d_cog(landmarks):
    """Estimate 2D center of gravity using Winter's segment mass model.

    Args:
        landmarks: list of NormalizedLandmark (x, y in 0-1 range).

    Returns:
        (cog_x, cog_y) in normalized coordinates.
    """
    def _midpoint(i, j):
        return ((landmarks[i].x + landmarks[j].x) / 2,
                (landmarks[i].y + landmarks[j].y) / 2)

    def _segment_cog(i, j, fraction=0.5):
        """CoG along the segment from i to j, at the given fraction from i."""
        x = landmarks[i].x + fraction * (landmarks[j].x - landmarks[i].x)
        y = landmarks[i].y + fraction * (landmarks[j].y - landmarks[i].y)
        return (x, y)

    segments = [
        # (cog_position, mass_fraction)
        # Head: midpoint of ears, full head mass
        (_midpoint(_LEFT_EAR, _RIGHT_EAR), _SEGMENT_MASS["head"]),
        # Torso: 50% along shoulders-midpoint to hips-midpoint
        (_midpoint(_LEFT_SHOULDER, _RIGHT_SHOULDER)[0] * 0.5 + _midpoint(_LEFT_HIP, _RIGHT_HIP)[0] * 0.5,
         _midpoint(_LEFT_SHOULDER, _RIGHT_SHOULDER)[1] * 0.5 + _midpoint(_LEFT_HIP, _RIGHT_HIP)[1] * 0.5,
         _SEGMENT_MASS["torso"]),
        # Left upper arm: 43% from shoulder
        (_segment_cog(_LEFT_SHOULDER, _LEFT_ELBOW, 0.43), _SEGMENT_MASS["upper_arm"]),
        # Right upper arm
        (_segment_cog(_RIGHT_SHOULDER, _RIGHT_ELBOW, 0.43), _SEGMENT_MASS["upper_arm"]),
        # Left forearm: 43% from elbow
        (_segment_cog(_LEFT_ELBOW, _LEFT_WRIST, 0.43), _SEGMENT_MASS["forearm"]),
        # Right forearm
        (_segment_cog(_RIGHT_ELBOW, _RIGHT_WRIST, 0.43), _SEGMENT_MASS["forearm"]),
        # Left thigh: 43% from hip
        (_segment_cog(_LEFT_HIP, _LEFT_KNEE, 0.43), _SEGMENT_MASS["thigh"]),
        # Right thigh
        (_segment_cog(_RIGHT_HIP, _RIGHT_KNEE, 0.43), _SEGMENT_MASS["thigh"]),
        # Left shank: 43% from knee
        (_segment_cog(_LEFT_KNEE, _LEFT_ANKLE, 0.43), _SEGMENT_MASS["shank"]),
        # Right shank
        (_segment_cog(_RIGHT_KNEE, _RIGHT_ANKLE, 0.43), _SEGMENT_MASS["shank"]),
        # Left foot: 50% from heel to toe
        (_segment_cog(_LEFT_HEEL, _LEFT_FOOT_INDEX, 0.5), _SEGMENT_MASS["foot"]),
        # Right foot
        (_segment_cog(_RIGHT_HEEL, _RIGHT_FOOT_INDEX, 0.5), _SEGMENT_MASS["foot"]),
    ]

    total_mass = 0.0
    cog_x = 0.0
    cog_y = 0.0

    for item in segments:
        if len(item) == 3:
            # Torso special case: (x, y, mass) directly
            sx, sy, mass = item
        else:
            (sx, sy), mass = item
        cog_x += sx * mass
        cog_y += sy * mass
        total_mass += mass

    if total_mass > 0:
        cog_x /= total_mass
        cog_y /= total_mass

    return (cog_x, cog_y)


class FeedbackEngine:
    def __init__(self):
        self._rep_count = 0
        self._in_squat = False
        self._min_knee_angle = 180  # track deepest point during current rep

    def evaluate(self, result, view_mode, ratios, near_side="left"):
        """Run feedback rules and update rep counter.

        Args:
            result: PoseLandmarkerResult (must not be None).
            view_mode: "front", "side", or "unclear".
            ratios: dict from BodyCalibration.get_ratios().
            near_side: "left" or "right" (used in side mode).

        Returns:
            (messages, rep_count, has_alerts) where messages is a list of strings
            (max 2 + optional "... + N more" indicator).
        """
        norm_lm = result.pose_landmarks[0]
        world_lm = result.pose_world_landmarks[0]

        # Update rep counter (works in all modes)
        self._update_rep_counter(world_lm)

        if view_mode == "unclear":
            return [], self._rep_count, False

        # Collect feedback
        if view_mode == "front":
            alerts = self._evaluate_front(norm_lm, world_lm, ratios)
        else:
            alerts = self._evaluate_side(norm_lm, world_lm, ratios, near_side)

        # Sort by priority, take top 2, add indicator if more
        alerts.sort(key=lambda x: x[0])
        messages = [msg for _, msg in alerts[:_MAX_DISPLAY_MESSAGES]]
        remaining = len(alerts) - _MAX_DISPLAY_MESSAGES
        if remaining > 0:
            messages.append(f"+ {remaining} more issue{'s' if remaining > 1 else ''}")

        has_alerts = len(alerts) > 0
        return messages, self._rep_count, has_alerts

    def reset_reps(self):
        """Reset rep counter."""
        self._rep_count = 0
        self._in_squat = False
        self._min_knee_angle = 180

    def _update_rep_counter(self, world_lm):
        """Track squat reps via knee angle state transitions."""
        left_angle = calc_angle(world_lm[_LEFT_HIP], world_lm[_LEFT_KNEE], world_lm[_LEFT_ANKLE])
        right_angle = calc_angle(world_lm[_RIGHT_HIP], world_lm[_RIGHT_KNEE], world_lm[_RIGHT_ANKLE])
        knee_angle = (left_angle + right_angle) / 2

        if self._in_squat:
            self._min_knee_angle = min(self._min_knee_angle, knee_angle)
            if knee_angle > _SQUAT_EXIT_ANGLE:
                self._rep_count += 1
                self._in_squat = False
                self._min_knee_angle = 180
        else:
            if knee_angle < _SQUAT_ENTER_ANGLE:
                self._in_squat = True
                self._min_knee_angle = knee_angle

    def _evaluate_front(self, norm_lm, world_lm, ratios):
        """Front-facing squat rules. Returns list of (priority, message)."""
        alerts = []

        # Average knee angle
        left_knee_angle = calc_angle(world_lm[_LEFT_HIP], world_lm[_LEFT_KNEE], world_lm[_LEFT_ANKLE])
        right_knee_angle = calc_angle(world_lm[_RIGHT_HIP], world_lm[_RIGHT_KNEE], world_lm[_RIGHT_ANKLE])
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        # 1. Squat depth — only flag when in squat phase and not deep enough
        if self._in_squat and avg_knee_angle > _DEPTH_WARNING_ANGLE:
            alerts.append((_PRIORITY_MEDIUM,
                           f"Depth: {avg_knee_angle:.0f}\u00b0 — aim below {_DEPTH_WARNING_ANGLE}\u00b0"))

        # 2. Knee valgus — knees caving inward
        knee_dist = abs(norm_lm[_LEFT_KNEE].x - norm_lm[_RIGHT_KNEE].x)
        ankle_dist = abs(norm_lm[_LEFT_ANKLE].x - norm_lm[_RIGHT_ANKLE].x)
        if ankle_dist > 0.01:  # avoid div by zero
            valgus_ratio = knee_dist / ankle_dist
            if valgus_ratio < _VALGUS_RATIO_THRESHOLD and self._in_squat:
                alerts.append((_PRIORITY_HIGH,
                               f"Knee valgus: knees inside ankles ({valgus_ratio:.2f}) — push knees outward"))

        # 3. Left-right balance
        cog_x, _ = calc_2d_cog(norm_lm)
        ankle_midpoint_x = (norm_lm[_LEFT_ANKLE].x + norm_lm[_RIGHT_ANKLE].x) / 2
        if ankle_dist > 0.01:
            deviation = (cog_x - ankle_midpoint_x) / ankle_dist
            if abs(deviation) > _BALANCE_DEVIATION:
                direction = "left" if deviation < 0 else "right"
                alerts.append((_PRIORITY_MEDIUM,
                               f"Balance: weight shifted {direction} ({abs(deviation):.0%}) — center over feet"))

        return alerts

    def _evaluate_side(self, norm_lm, world_lm, ratios, near_side):
        """Side-facing squat rules. Returns list of (priority, message)."""
        alerts = []

        # Pick near-side landmarks
        if near_side == "left":
            shoulder, hip, knee, ankle = _LEFT_SHOULDER, _LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE
            heel, toe = _LEFT_HEEL, _LEFT_FOOT_INDEX
        else:
            shoulder, hip, knee, ankle = _RIGHT_SHOULDER, _RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE
            heel, toe = _RIGHT_HEEL, _RIGHT_FOOT_INDEX

        knee_angle = calc_angle(world_lm[hip], world_lm[knee], world_lm[ankle])

        # 1. Squat depth
        if self._in_squat and knee_angle > _DEPTH_WARNING_ANGLE:
            alerts.append((_PRIORITY_MEDIUM,
                           f"Depth: {knee_angle:.0f}\u00b0 — aim below {_DEPTH_WARNING_ANGLE}\u00b0"))

        # 2. Knee-over-toe
        knee_x = norm_lm[knee].x
        toe_x = norm_lm[toe].x
        # Adjust threshold based on femur:tibia ratio — longer femurs need more forward travel
        ft_ratio = ratios.get("femur_tibia_ratio", 1.0)
        # Allow more forward travel for higher ratios; base tolerance ~0
        tolerance = (ft_ratio - 1.0) * 0.05  # normalized coordinate units
        if self._in_squat:
            # In side view, "forward" depends on which direction the person faces
            # Use heel-to-toe direction to determine forward
            forward_sign = 1 if toe_x > norm_lm[heel].x else -1
            knee_past_toe = (knee_x - toe_x) * forward_sign
            if knee_past_toe > tolerance:
                alerts.append((_PRIORITY_MEDIUM,
                               f"Knees past toes by {abs(knee_past_toe):.2f} — shift weight to heels"))

        # 3. Back alignment (forward lean)
        back_angle = calc_angle(world_lm[shoulder], world_lm[hip], world_lm[knee])
        if self._in_squat and back_angle < _FORWARD_LEAN_ANGLE:
            alerts.append((_PRIORITY_HIGH,
                           f"Forward lean: {back_angle:.0f}\u00b0 — keep chest up"))

        # 4. Heel-to-toe weight distribution
        cog_x, _ = calc_2d_cog(norm_lm)
        heel_x = norm_lm[heel].x
        toe_x_val = norm_lm[toe].x
        foot_length = abs(toe_x_val - heel_x)
        if foot_length > 0.005 and self._in_squat:
            # Position along foot: 0 = heel, 1 = toe
            foot_pos = (cog_x - heel_x) / (toe_x_val - heel_x) if (toe_x_val - heel_x) != 0 else 0.5
            if foot_pos < 0.2:
                alerts.append((_PRIORITY_LOW,
                               f"Weight on heels ({foot_pos:.0%} heel-to-toe) — shift slightly forward"))
            elif foot_pos > 0.8:
                alerts.append((_PRIORITY_LOW,
                               f"Weight on toes ({foot_pos:.0%} heel-to-toe) — shift to midfoot"))

        return alerts
