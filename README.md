# MovementCoach

MovementCoach is an experimental Python desktop application that uses a webcam to analyze human movement in real-time (pose estimation) and provide actionable coaching feedback on exercise/movement form.

For the full vision and feature plan, see [MovementCoach_ProjectSpec.md](MovementCoach_ProjectSpec.md).
For the detailed implementation plan, see [plan.md](plan.md).

## Current Status

**Phases 1–5 of 8 complete.** The app captures webcam video, estimates body pose, auto-calibrates to the user's body proportions, detects front/side facing, and provides real-time squat form feedback with rep counting. Currently in testing and threshold tuning.

### What works

| Feature | Status | Details |
|---------|--------|---------|
| Webcam capture + display | Done | Mirrored feed, resizable window (`cv2.WINDOW_NORMAL`) |
| Pose estimation | Done | MediaPipe PoseLandmarker (tasks API v0.10.33), 33 body landmarks |
| Skeleton overlay | Done | Green when form is good, orange when alerts are active |
| On-the-fly body calibration | Done | Measures limb segment lengths (3D world landmarks) using rolling median + IQR convergence. Auto-resets for new users. No forced T-pose needed. |
| View auto-detection | Done | Front / Side / Unclear based on shoulder-width:torso-height ratio with 8-frame hysteresis. Unclear state shows repositioning prompt. |
| Front-facing squat feedback | Done | Squat depth, knee valgus, left-right balance, stance width |
| Side-facing squat feedback | Done | Knee-over-toe (calibration-adjusted), forward lean, heel-to-toe weight distribution (2D CoG via Winter's model) |
| Rep counter | Done | Tracks standing↔squat transitions via knee angle thresholds |
| Feedback display | Done | Top 2 most important alerts shown (priority-ranked), "+ N more" indicator for additional issues. Clinical/neutral style with degree values. |

### What's next (not yet implemented)

- **Phase 6** — Rep summary (press `s` after a set to see per-set statistics)
- **Phase 7** — Overlay polish (smoothed feedback, CoG visualization)
- **Phase 8** — Testing and threshold tuning

## Setup

```bash
# Clone the repo
git clone https://github.com/HungryHung/MovementCoach.git
cd MovementCoach

# Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Download the pose model (~5.5 MB)
mkdir models
# Download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
# Save to: models/pose_landmarker_lite.task
```

## Running

```bash
.venv\Scripts\activate
cd src
python main.py
```

Stand far enough from the camera for your full body to be visible.

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset rep counter |

## Project Structure

```
MovementCoach/
├── .gitignore
├── requirements.txt          # mediapipe, opencv-python, numpy
├── README.md                 # this file
├── MovementCoach_ProjectSpec.md  # original vision / brainstorm
├── plan.md                   # detailed 8-phase implementation plan
├── models/                   # (not in git) pose_landmarker_lite.task
└── src/
    ├── main.py               # entry point: webcam loop + orchestration
    ├── pose_estimator.py     # MediaPipe PoseLandmarker wrapper
    ├── body_calibration.py   # on-the-fly limb ratio measurement
    ├── view_detector.py      # auto-detect front vs side facing
    ├── overlay.py            # skeleton + feedback text + status rendering
    ├── feedback_engine.py    # squat feedback rules (front + side), rep counter, CoG
    └── ...
```