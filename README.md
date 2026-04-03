# MovementCoach

MovementCoach is an experimental Python desktop application that uses a webcam to analyze human movement in real-time (pose estimation) and provide actionable coaching feedback on exercise/movement form.

For the full vision and feature plan, see [MovementCoach_ProjectSpec.md](MovementCoach_ProjectSpec.md).
For the detailed implementation plan, see [plan.md](plan.md).

## Current Status

**Phases 1–4 of 8 complete.** The app captures webcam video, estimates body pose, auto-calibrates to the user's body proportions, and detects whether the user is facing front or sideways. No squat feedback rules yet — that's Phase 5.

### What works

| Feature | Status | Details |
|---------|--------|---------|
| Webcam capture + display | Done | Mirrored feed, resizable window (`cv2.WINDOW_NORMAL`) |
| Pose estimation | Done | MediaPipe PoseLandmarker (tasks API v0.10.33), 33 body landmarks |
| Skeleton overlay | Done | Green skeleton drawn on the video feed |
| On-the-fly body calibration | Done | Measures limb segment lengths (3D world landmarks) using rolling median + IQR convergence. Auto-resets for new users. No forced T-pose needed. |
| View auto-detection | Done | Front / Side / Unclear based on shoulder-width:torso-height ratio with 8-frame hysteresis. Unclear state shows repositioning prompt. |

### What's next (not yet implemented)

- **Phase 5** — Squat feedback rules (dual mode: front + side)
- **Phase 6** — Rep summary (press `s` after a set to see stats)
- **Phase 7** — Overlay polish (skeleton color cues, smoothed feedback, CoG visualization)
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

Press `q` to quit. Stand far enough from the camera for your full body to be visible.

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
    ├── overlay.py            # skeleton + status text rendering
    ├── feedback_engine.py    # (not yet created) squat feedback rules
    └── ...
```