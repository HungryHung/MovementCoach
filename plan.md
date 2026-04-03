# Plan: MovementCoach Squat POC

Build a working Python desktop POC that captures webcam video, runs pose estimation (MediaPipe), detects squat form via joint-angle rules, and overlays real-time coaching feedback on the video feed. Supports front-facing and side-facing squat analysis with auto-detection. Body proportions calibrate on-the-fly (no forced pose). Architecture keeps feedback logic decoupled for future AI/LLM extension.

## Phase 1 — Project Bootstrap (do first)

1. **Verify/install Python 3.10+** — check `python --version`. Install from python.org if missing.
2. **Create virtual environment & dependencies**
   - `python -m venv .venv` in the repo root.
   - Create `requirements.txt` with: `mediapipe`, `opencv-python`, `numpy`.
   - `pip install -r requirements.txt`.
3. **Rename `gitignore_Version2.txt` → `.gitignore`** so it actually takes effect.
4. **Create initial project structure:**
   ```
   MovementCoach/
   ├── .gitignore
   ├── requirements.txt
   ├── README.md
   ├── MovementCoach_ProjectSpec.md
   └── src/
       ├── main.py              # entry point: webcam loop + orchestration
       ├── pose_estimator.py    # wraps MediaPipe pose detection
       ├── body_calibration.py  # on-the-fly limb ratio measurement
       ├── view_detector.py     # auto-detect front vs side facing
       ├── feedback_engine.py   # rule-based feedback logic (angles → messages)
       └── overlay.py           # draws skeleton + feedback text on frames
   ```

   **Why this structure:** separating pose estimation, calibration, view detection, feedback rules, and rendering keeps each concern independent. When you later want LLM feedback, you add a new feedback module without touching other code.

## Phase 2 — Webcam Capture & Pose Estimation

5. **`main.py` — webcam capture loop** — use `cv2.VideoCapture(0)` to open the camera, read frames in a loop, display with `cv2.imshow()`, and quit on `q` key press.
6. **`pose_estimator.py` — MediaPipe Pose wrapper** — initialize `mp.solutions.pose.Pose()`, expose a function `estimate(frame) → landmarks` that takes a BGR frame, converts to RGB, runs pose, and returns the 33 landmark positions (or `None` if no person detected).
7. **`overlay.py` — draw skeleton** — use `mp.solutions.drawing_utils.draw_landmarks()` to render the pose skeleton on the frame. Verify visually that landmarks track correctly.

*Checkpoint: running `python src/main.py` shows the webcam feed with a skeleton overlaid on your body.*

## Phase 3 — On-the-fly Body Calibration

8. **`body_calibration.py` — continuous limb measurement:**
   - Each frame (if landmark visibility > threshold), compute 3D segment lengths: hip→knee (femur), knee→ankle (tibia), shoulder→hip (torso), shoulder→elbow, elbow→wrist.
   - Store measurements in rolling buffers (e.g., collections.deque, maxlen=100 per segment).
   - Use **running median** for the calibrated value (robust to outliers from noisy z-coordinates).
   - Track **interquartile range (IQR)** per segment — when IQR drops below a threshold, mark that segment as "calibrated."
   - Expose `is_calibrated() → bool` (True when all key segments are stable) and `get_ratios() → dict` (e.g., `{"femur_tibia_ratio": 1.12, "torso_leg_ratio": 0.48, ...}`).
   - Once calibrated, reduce measurement frequency to every ~30th frame.
   - **Auto-reset:** if new measurements deviate significantly (>15%) from calibrated median (e.g., different person steps in), clear buffers and recalibrate.
9. **Show calibration status on overlay** — small indicator (e.g., "Calibrating..." → "Calibrated ✓") so user knows when personalized feedback kicks in.

*Checkpoint: stand in front of camera, move naturally for 1-2 seconds; overlay shows "Calibrated ✓" and ratios are stable.*

## Phase 4 — View Auto-Detection (Front vs Side)

10. **`view_detector.py` — detect facing direction:**
    - Compute pixel distance between left_shoulder and right_shoulder landmarks.
    - Compare to a reference length (e.g., shoulder→hip distance, which is relatively stable across views).
    - When shoulder_width / reference_length > threshold → **front-facing mode**.
    - When ratio drops below threshold → **side-facing mode**.
    - Expose `detect_view(landmarks) → "front" | "side" | "unclear"`.
    - Apply hysteresis (require N consecutive frames before switching) to avoid flickering between modes.
    - **Determine near side** in side mode: compare visibility scores of left vs right landmarks; higher-visibility side is the near side. Use near-side landmarks for feedback.
    - **"Unclear" state**: when shoulder-width ratio falls in the ambiguous middle zone (~45° diagonal), return `"unclear"`. Feedback rules are paused and overlay shows a repositioning prompt instead of coaching feedback.
11. **Show current mode on overlay** — "Mode: Front", "Mode: Side", or **"Please face the camera or turn fully sideways"** when unclear.

*Checkpoint: face camera → "Mode: Front"; turn sideways → "Mode: Side"; stand at ~45° → repositioning prompt. All transitions stable.*

## Phase 5 — Squat Feedback Rules (Dual Mode)

12. **Define utilities** in `feedback_engine.py`:
    - `calc_angle(a, b, c) → float` — angle at point `b` using `numpy.arctan2`.
    - `calc_2d_cog(landmarks) → (x, y)` — weighted center of gravity using Winter's segment mass model (standard population percentages: head 8.1%, torso 50.1%, thigh 10.0% each, shank 4.65% each, etc.). Use 2D (x, y) only.
13. **Front-facing squat rules** (`evaluate_squat_front(landmarks, ratios) → list[str]`):
    - **Knee angle** (hip→knee→ankle): detect squat depth. Threshold adjusted by femur:tibia ratio from calibration.
    - **Knee valgus** (knees caving inward): compare knee x-positions to ankle x-positions. Flag if knees collapse inward.
    - **Left-right balance**: compute 2D CoG x-position relative to midpoint between ankles. Flag if shifted >15% toward one side.
    - **Stance width**: distance between ankles relative to hip width. Flag if too narrow or too wide.
14. **Side-facing squat rules** (`evaluate_squat_side(landmarks, ratios, near_side) → list[str]`):
    - **Knee-over-toe**: compare near-side knee x-position to toe x-position. Threshold adjusted by femur:tibia ratio.
    - **Back alignment**: shoulder→hip→knee angle — flag excessive forward lean.
    - **Heel-to-toe weight estimate**: compute 2D CoG x-position relative to near-side foot (heel landmark to toe landmark). Classify as "weight back / centered / weight forward." Flag if too far forward or back.
    - **Squat depth**: same knee angle check as front mode, using near-side landmarks.
15. **Rep counter** — track state transitions (standing → squat → standing) based on knee angle crossing thresholds. Works in both modes. Display rep count on overlay.
16. **Pre-calibration fallback** — before calibration completes, use population-average thresholds. Switch to personalized thresholds once `is_calibrated()` returns True. Show this transition on overlay.

*Checkpoint: front-facing squats show balance + valgus feedback; side-facing squats show knee-over-toe + lean + weight distribution.*

## Phase 6 — Rep Summary

17. **Per-rep data capture** in `feedback_engine.py`:
    - At each rep completion (squat→standing transition), snapshot key metrics into a list of dicts: min knee angle (depth), max forward lean angle, max CoG deviation, knee valgus amount, which rules triggered, view mode during rep.
    - Store in a `RepHistory` class or simple list, reset when user presses `r`.
18. **Summary computation** — function `compute_summary(rep_history) → dict`:
    - Average/min/max squat depth across reps
    - Consistency score (std deviation of key metrics — lower = more consistent)
    - Most common form issue (which rule triggered most often)
    - Trend detection: compare first half vs second half of reps — flag if form degrades (fatigue indicator)
    - Best rep / worst rep identification
19. **Summary overlay** — press `s` to toggle summary view:
    - Pauses live feedback, shows a summary screen overlaid on the (still-running) video
    - Displays: rep count, average depth, consistency rating, trend, top form issues
    - Press `s` again to return to live coaching
    - Summary data can also be printed to terminal for easy copy/review

*Checkpoint: do 5+ squats, press `s` → summary overlay shows per-set statistics. Press `s` again → back to live coaching.*

## Phase 7 — Overlay & UX Polish

20. **`overlay.py` — render feedback text** — draw feedback strings using `cv2.putText()`. Position at top-left, readable font/color, rep count at top-right, calibration status + view mode at bottom.
21. **Visual cues** — skeleton green (good form) / red-orange (rule triggered). Use `mp.solutions.drawing_utils.DrawingSpec`.
22. **Smooth feedback** — hold messages for ~1 second or average angles over ~5 frames to reduce flicker.
23. **CoG visualization** (optional) — draw a small circle on the frame at the estimated CoG position. In side mode, draw a line from CoG down to the foot segment to visualize weight distribution.

*Checkpoint: feedback is readable, stable, and mode-appropriate.*

## Phase 8 — Testing & Iteration

24. **Self-test front mode** — squats with intentionally bad form: knees caving in, shallow depth, leaning to one side.
25. **Self-test side mode** — squats with knees too far forward, excessive forward lean, weight too far on toes.
26. **Test mode switching** — turn from front to side during exercise; verify smooth transition and correct rule set activation. Test diagonal → repositioning prompt.
27. **Test calibration** — verify calibration completes within ~2 seconds; verify auto-reset works when a different person appears.
28. **Test rep summary** — do 10 squats with varying form, press `s`, verify summary is accurate and readable.
29. **Tune thresholds** — adjust angle thresholds based on testing. Document chosen values.
30. **Edge cases** — no person in frame (no crash), partial visibility, person at unusual distance.

## Relevant Files

- `src/main.py` — webcam loop; orchestrates: pose → calibration → view detection → feedback → overlay
- `src/pose_estimator.py` — wraps MediaPipe Pose, returns landmarks
- `src/body_calibration.py` — on-the-fly limb ratio measurement with rolling median + IQR convergence + auto-reset
- `src/view_detector.py` — auto-detect front vs side from shoulder width ratio + hysteresis
- `src/feedback_engine.py` — `calc_angle()`, `calc_2d_cog()`, `evaluate_squat_front()`, `evaluate_squat_side()`, rep counter; **swap/extend point for future LLM feedback**
- `src/overlay.py` — `draw_skeleton()`, `draw_feedback()`, `draw_cog()`, mode/calibration indicators
- `requirements.txt` — `mediapipe`, `opencv-python`, `numpy`
- `.gitignore` — renamed from `gitignore_Version2.txt`

## Hotkeys

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Toggle rep summary overlay |
| `r` | Reset rep history |

## Verification

1. After Phase 1: `python -c "import cv2; import mediapipe; print('OK')"` prints OK.
2. After Phase 2: `python src/main.py` shows webcam with skeleton overlay. Close with `q`.
3. After Phase 3: stand naturally for ~2s → overlay shows "Calibrated ✓".
4. After Phase 4: face camera → "Mode: Front"; turn sideways → "Mode: Side"; stand at ~45° → repositioning prompt. All stable.
5. After Phase 5 (front): squats facing camera → balance + knee valgus feedback appears for bad form.
6. After Phase 5 (side): squats sideways → knee-over-toe + lean + weight distribution feedback appears.
7. After Phase 6: do 5+ squats, press `s` → summary overlay with stats. Press `s` → back to live.
8. After Phase 7: feedback is readable and stable (no flickering).
9. After Phase 8: all intentional bad-form cases trigger correct warnings; mode switching and diagonal prompt work; calibration converges; summary is accurate; empty frame is safe.

## Decisions

- **Single exercise (squat)** for POC scope. Additional exercises follow the same pattern.
- **MediaPipe Pose** — easiest install, no GPU required.
- **On-the-fly calibration** with rolling median + IQR convergence — no forced T-pose needed. Auto-resets for new users.
- **Auto-detect front vs side** from shoulder-width ratio with hysteresis — no manual toggle. Diagonal/unclear angle shows repositioning prompt.
- **2D CoG only** — 3D CoG deferred due to MediaPipe z-coordinate noise.
- **Heel-to-toe: side mode only** — 2D projection makes it ~70-75% accurate from the side vs. ~40-50% from the front.
- **Population-average fallback** — feedback works immediately with default thresholds; personalizes once calibration converges.
- **Rep summary** — manual trigger (`s` key) shows per-set statistics. Reset with `r` key.
- **No AI/LLM integration in this POC** — `feedback_engine.py` is isolated for future `llm_feedback.py`.
- **No persistent storage / recording** in this POC.
- **.gitignore rename** needed.

## Future Considerations

- **AI/LLM feedback**: create `src/llm_feedback.py` that accepts landmarks + user prompt and calls an LLM API. Toggle in `main.py` to switch between rule-based and LLM. Overlay unchanged.
- **3D CoG**: requires camera calibration and better depth filtering. Defer to next phase with better tooling or depth camera.
- **Pressure mat integration**: for true heel-to-toe weight measurement. Would validate/replace the 2D CoG estimation.
- **Multiple exercises**: each exercise gets its own rule set in `feedback_engine.py` following the same pattern (front rules + side rules + calibration-adjusted thresholds).
