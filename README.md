# Tabby Zero

**AI Composition Pipeline Prototype**

Tabby Zero is the software-only prototype of [Tabby One](#) — an autonomous AI photo-capturing robot. This version strips away all the robotics and runs entirely on your laptop using your webcam. The goal is to prove that the AI composition pipeline works before bolting it onto a moving robot.

Think of Tabby Zero as the brain of Tabby One, running in a test lab. No wheels, no motors — just the AI logic that decides whether a photo is well-framed and what needs to change.

---

## What It Does

- Captures live frames from your webcam
- Runs object and person detection using YOLOv8 (draws bounding boxes in real time)
- Sends the frame to a Vision-Language Model (VLM) via cloud API
- Gets back structured JSON advice: who is in the frame, how good the composition is, and what to adjust
- Simulates camera movement by shifting/cropping the frame — no robot required
- Runs a full composition loop: evaluate, adjust, re-evaluate, until the shot is good enough

---

## What It Is Not

Tabby Zero does not control any hardware. There are no motors, no ESP32, no Orange Pi NPU. This is intentional. Tabby Zero exists to validate the AI pipeline in isolation before integration with the robot in Phase 3 of the Tabby One roadmap.

---

## Project Structure

```
tabby-zero/
  README.md
  requirements.txt
  config/
    settings.yaml          # Model paths, API keys, thresholds — all config lives here
  camera/
    webcam.py              # Opens webcam, captures frames
  detection/
    yolo_detector.py       # YOLO wrapper class — swappable model backend
  composition/
    vlm_client.py          # VLM API wrapper — returns structured JSON
    composition_loop.py    # Main decision loop: detect → evaluate → adjust
    frame_simulator.py     # Simulates camera movement by cropping/shifting frames
  utils/
    logger.py              # Logging setup
  main.py                  # Entry point
```

---

## How It Works

The pipeline runs in a loop. Here is what happens from the moment you give a command to the moment a composition decision is made:

```
User types: "Take a portrait of me with this trophy"
        |
        v
camera/webcam.py captures a live frame
        |
        v
detection/yolo_detector.py runs YOLO — returns bounding boxes
        |
        v
composition/vlm_client.py sends frame + detections to VLM API
        |
        v
VLM returns structured JSON:
  - subjects present / missing
  - framing quality score (0.0 to 1.0)
  - is_good_enough: true / false
  - action: move direction, distance, tilt
  - reasoning: plain text explanation
        |
        v
If not good enough:
  frame_simulator.py shifts the frame to simulate movement
  Loop repeats
        |
        v
If good enough (or max attempts reached):
  Final frame is saved
```

---

## VLM Output Format

The VLM is always forced to return structured JSON. No prose, no extra text. Example output:

```json
{
  "subjects_present": ["person", "trophy"],
  "subjects_missing": [],
  "framing_quality": 0.72,
  "is_good_enough": false,
  "action": {
    "move": "backward",
    "distance_cm": 30,
    "tilt_camera": "up",
    "tilt_degrees": 5
  },
  "reasoning": "Trophy is partially cropped at the top of the frame"
}
```

Why JSON and not plain text? Because plain text is unpredictable. JSON gives the pipeline deterministic, parseable output every single time.

---

## Setup

### Requirements

- Python 3.10 or higher
- A webcam connected to your laptop
- A Gemini Flash API key (free tier available) or a Claude API key

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure

Copy the example config and fill in your API key:

```bash
cp config/settings.yaml.example config/settings.yaml
```

Edit `config/settings.yaml`:

```yaml
yolo:
  model_path: "yolov8n.pt"
  confidence_threshold: 0.5
  target_classes: ["person", "cup", "bottle", "cell phone"]

vlm:
  provider: "gemini"          # "gemini" or "claude"
  api_key: "YOUR_API_KEY"
  max_retries: 2

composition:
  max_attempts: 5
  good_enough_threshold: 0.80

camera:
  device_index: 0             # 0 is usually the built-in webcam
```

### Run

```bash
python main.py
```

Then type a command when prompted:

```
> Take a photo of me with this cup
```

---

## Modules

### `camera/webcam.py`

Opens the webcam using OpenCV and returns frames on demand. Keeping this as its own module means swapping in a different camera source (IP stream, file, Orange Pi CSI) later is a one-line change.

### `detection/yolo_detector.py`

Wraps YOLOv8 into a clean class. Call `detector.detect(frame)` and get back a list of detections with label, confidence score, and bounding box coordinates. No display logic lives here — this module only deals with data.

### `composition/vlm_client.py`

Handles all communication with the VLM API. Sends a frame (as base64) along with the user's intent and YOLO detections. Parses the JSON response. If the API returns invalid JSON, it retries once and then returns a safe default response so the loop does not crash.

### `composition/frame_simulator.py`

Since there is no robot, this module fakes movement. If the VLM says "move left 30cm", the simulator shifts the frame left by a proportional number of pixels. This lets the composition loop run multiple iterations and actually improve the framing — all on a still webcam frame.

### `composition/composition_loop.py`

The main loop. Ties all the modules together. Runs until the VLM says `is_good_enough: true` or the max attempt count is reached. Every iteration is logged with what YOLO found, what the VLM advised, and the framing quality score.

---

## Exit Criteria (Phase 1 Complete)

Tabby Zero is considered complete when all three of these are true:

| Check | Target |
|-------|--------|
| End-to-end response time | Webcam frame to structured JSON advice in under 3 seconds |
| Composition loop improvement | Framing quality score improves across at least 3 iterations on a staged scene |
| Screen recording | A 2-minute recording showing the full pipeline working exists and is committed to the repo |

---

## Relationship to Tabby One

Tabby Zero covers Phase 1 of the Tabby One roadmap (Months 3-5). Once these exit criteria are met, the pipeline moves to Phase 2 where the robot chassis is built separately, and then Phase 3 where the two are integrated.

| Phase | Project | What it covers |
|-------|---------|----------------|
| Phase 1 | Tabby Zero | AI pipeline on laptop, no hardware |
| Phase 2 | Tabby One (hardware) | Robot chassis, motors, sensors — no AI yet |
| Phase 3 | Tabby One (integrated) | AI pipeline running on the robot |

---

## Development Notes

- All configuration lives in `config/settings.yaml` — no hardcoded values anywhere in the code
- Use Python's `logging` module, not `print` statements — log levels matter when debugging
- Keep each module's interface narrow and clean — the goal is that swapping any one module does not break the others
- Commit often with descriptive messages — your future self debugging at 2am will thank you

---

## License

MIT License. See `LICENSE` for details.
