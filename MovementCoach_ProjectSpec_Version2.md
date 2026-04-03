# MovementCoach – Project Specification

## Vision
Build an experimental prototype ("proof of concept") desktop application in Python that uses a webcam to capture the user's movement, estimates body pose in real-time, and provides actionable feedback/coaching on exercise form and movement quality.

## Key Project Insights (so far)
- Goal is to explore and experiment with analysis/feedback, not perfect UI or mobile deployment.
- Start with a minimal PC-based solution using familiar Python libraries (e.g., MediaPipe, OpenCV).
- Deliver simple, rule-based feedback first (e.g., "Your knee is too bent during a squat"); advanced AI feedback (e.g., natural language prompts, vision LLMs) is a possible stretch goal.
- User should ideally be able to request/check specific movement aspects and receive relevant feedback.
- No prior ML or pose estimation experience required, but open to using tutorials/examples/found code.
- Internet access is available; may experiment with cloud-based vision/AI APIs for richer feedback.

## Minimal Viable Product (MVP)
- Capture video stream from the user's webcam.
- Run pose estimation on each video frame (e.g., using MediaPipe).
- Simple rule-based feedback (for example: squats, arm raises — detect angles/joint positions and respond if movement doesn't meet basic criteria).
- Feedback can be provided as simple text (“Bend your knee more!”), in terminal, or overlaid on the output video feed.

## Possible Stretch Goals
- Support multiple types of movements/exercises.
- Let user "prompt" coach for what to focus on (“Watch my landing”, “Check my arm position”).
- Integrate with cloud/AI vision APIs (e.g., OpenAI Vision, Gemini Pro, or similar) to enable more advanced, context-aware feedback.
- Record/save video and/or feedback history.
- Improve UI/UX for more interactive experience.

## Implementation Plan / Checklist

1. **Research & Set Up**
    - Install Python, OpenCV, and MediaPipe (and Jupyter if desired).
    - Test webcam capture and visualize initial frames.

2. **Pose Estimation Baseline**
    - Integrate MediaPipe (or similar) to extract pose data.
    - Overlay pose skeleton on video output.

3. **Rule-Based Feedback System**
    - Select one or two example movements (e.g., squat, arm raise).
    - Write feedback rules (e.g., check joint angles) and display feedback (terminal or overlay).
    - Allow user to select which feedback/“coach” to activate.

4. **Experiment & Iterate**
    - Run on yourself, test feedback, adjust rules.
    - Get feedback from possible users (family/friends if available).

5. **Stretch (optional)**
    - Try uploading still frames or video to a vision LLM API (like OpenAI Vision or Claude 3) via Python to generate richer feedback.
    - Refactor to accept arbitrary user text prompts for analysis and route to LLM if needed.
    - Add more exercises and rules if time allows.

## Further Brainstorming
- What feedback is actually useful to users? (Realistic suggestions welcomed.)
- Which movements are easiest/most impactful to analyze as a first MVP?
- What’s the simplest possible metric to start with (e.g., single joint angle, range of motion)?

---

*Summary: Focus on building a working prototype for movement feedback using pose estimation in Python, prioritize quick experiments and insights over polish or mobile deployment. Use AI and advanced tools for feedback augmentation if time and energy allow.*
