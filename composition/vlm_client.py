from openai import OpenAI
from dataclasses import dataclass
from typing import Optional
import cv2
import base64
import json
import math
import time
import logging

from detection.yolo_detector import Detection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CompositionAdvice:
    """
    Structured output from the VLM.
    Every field maps directly to the JSON the VLM returns.
    Using a dataclass means the rest of the pipeline works
    with clean typed objects, not raw dictionaries.
    """
    subjects_present: list[str]
    subjects_missing: list[str]
    framing_quality: float      # 0.0 = terrible, 1.0 = perfect
    is_good_enough: bool
    move_direction: str         # "left" | "right" | "forward" | "backward" | "none"
    move_distance_cm: int
    tilt_direction: str         # "up" | "down" | "none"
    tilt_degrees: int
    reasoning: str              # plain English explanation from the VLM


# Safe default returned when the API fails completely.
# Tells the loop to stop moving and not crash.
SAFE_DEFAULT = CompositionAdvice(
    subjects_present=[],
    subjects_missing=[],
    framing_quality=0.0,
    is_good_enough=False,
    move_direction="none",
    move_distance_cm=0,
    tilt_direction="none",
    tilt_degrees=0,
    reasoning="VLM call failed, returning safe default."
)


# ---------------------------------------------------------------------------
# Scene change detector
# ---------------------------------------------------------------------------

class SceneChangeDetector:
    """
    Decides whether the scene has changed enough to warrant a new VLM call.

    Two conditions must both be true to trigger a call:
    1. The scene changed meaningfully (new object appeared, or significant movement)
    2. Enough time has passed since the last VLM call (cooldown)

    This keeps API costs near zero by only calling the VLM when something
    actually changed, not on every frame.

    Think of it like a motion sensor light — it only turns on when
    something moves, not every second of the day.
    """

    def __init__(
        self,
        cooldown_seconds: float = 3.0,
        movement_threshold_px: int = 40,
    ) -> None:
        # Minimum seconds between VLM calls regardless of scene change
        self.cooldown_seconds = cooldown_seconds

        # How many pixels a bounding box center must move
        # before we consider it a meaningful scene change
        self.movement_threshold_px = movement_threshold_px

        self._last_call_time: float = 0.0
        self._last_detections: list[Detection] = []

    def should_call_vlm(self, current_detections: list[Detection]) -> bool:
        """
        Returns True if a VLM call is warranted.
        Returns False if scene is the same or cooldown has not passed.
        """
        now = time.time()
        cooldown_passed = (now - self._last_call_time) >= self.cooldown_seconds

        if not cooldown_passed:
            return False

        return self._scene_changed(current_detections)

    def mark_called(self, detections: list[Detection]) -> None:
        """
        Call this immediately after making a VLM call.
        Records the timestamp and detection snapshot for next comparison.
        """
        self._last_call_time = time.time()
        self._last_detections = detections

    def _scene_changed(self, current: list[Detection]) -> bool:
        """
        Compare current detections to the last snapshot.

        Scene is considered changed if:
        - Number of detected objects changed
        - Any object class changed
        - Any object's center point moved more than the threshold
        """
        previous = self._last_detections

        # First call ever — no previous snapshot, always trigger
        if not previous:
            return True

        # Different number of objects
        if len(current) != len(previous):
            return True

        # No detections in either — nothing to compare
        if not current and not previous:
            return False

        # Different classes appeared
        current_labels = sorted(d.label for d in current)
        previous_labels = sorted(d.label for d in previous)
        if current_labels != previous_labels:
            return True

        # Check if any object moved significantly
        for curr, prev in zip(
            sorted(current, key=lambda d: d.label),
            sorted(previous, key=lambda d: d.label)
        ):
            curr_cx = (curr.x1 + curr.x2) / 2
            curr_cy = (curr.y1 + curr.y2) / 2
            prev_cx = (prev.x1 + prev.x2) / 2
            prev_cy = (prev.y1 + prev.y2) / 2

            distance = math.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2)

            if distance > self.movement_threshold_px:
                return True

        return False


# ---------------------------------------------------------------------------
# VLM client
# ---------------------------------------------------------------------------

class VlmClient:
    """
    Handles all communication with the Qwen VL API.

    Uses the OpenAI-compatible SDK pointed at Alibaba Cloud's DashScope.
    The rest of the pipeline calls evaluate() and gets back a
    CompositionAdvice — swapping the backend later is a one-file change.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-vl-plus",
        max_retries: int = 2
    ) -> None:
        self.model_name = model
        self.max_retries = max_retries

        # Same OpenAI library, different base_url — that is the only difference
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        logger.info(f"Qwen VLM client ready. Model: {model}")

    def evaluate(
        self,
        frame: cv2.typing.MatLike,
        intent: str,
        detections: list[Detection]
    ) -> CompositionAdvice:
        """
        Send a frame to Qwen VL and get composition advice back.

        intent: what the user wants, e.g. "take a portrait of me with this trophy"
        detections: list of Detection objects from YOLO
        Returns: CompositionAdvice, or SAFE_DEFAULT if all attempts fail
        """
        frame_b64 = self._encode_frame(frame)

        detection_summary = ", ".join(
            f"{d.label} ({d.confidence:.2f})" for d in detections
        ) or "nothing detected"

        prompt = self._build_prompt(intent, detection_summary)

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Qwen VL (attempt {attempt + 1}/{self.max_retries})")

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{frame_b64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                )

                raw_text = response.choices[0].message.content
                advice = self._parse_response(raw_text)
                logger.info(
                    f"VLM response: quality={advice.framing_quality:.2f}, "
                    f"good_enough={advice.is_good_enough}"
                )
                return advice

            except Exception as e:
                logger.warning(f"Qwen attempt {attempt + 1} failed: {e}")

        logger.error("All Qwen attempts failed. Returning safe default.")
        return SAFE_DEFAULT

    def _build_prompt(self, intent: str, detection_summary: str) -> str:
        """
        Build the system prompt for the VLM.

        Key principle: give the VLM a concrete example of the exact
        JSON format you expect. VLMs follow examples much better
        than abstract instructions alone.
        """
        return f"""
You are a professional photography composition assistant.
The user wants to: {intent}

YOLO has detected the following in the current frame: {detection_summary}

Evaluate the composition of the image and return ONLY a JSON object.
No explanation before or after. No markdown code blocks. Raw JSON only.

Return exactly this structure:
{{
  "subjects_present": ["list", "of", "detected", "subjects", "relevant", "to", "intent"],
  "subjects_missing": ["list", "of", "subjects", "needed", "but", "not", "visible"],
  "framing_quality": 0.0,
  "is_good_enough": false,
  "action": {{
    "move": "left | right | forward | backward | none",
    "distance_cm": 0,
    "tilt_camera": "up | down | none",
    "tilt_degrees": 0
  }},
  "reasoning": "One sentence explaining what is wrong and why you suggested this action."
}}

Rules:
- framing_quality is a float between 0.0 (terrible) and 1.0 (perfect)
- is_good_enough is true only if framing_quality is above 0.75
- distance_cm should be between 0 and 100
- tilt_degrees should be between 0 and 30
- If framing is already good, set move to "none" and distance_cm to 0
"""

    def _encode_frame(self, frame: cv2.typing.MatLike) -> str:
        """
        Convert an OpenCV frame to a base64 JPEG string.
        The API cannot accept raw OpenCV arrays directly.
        JPEG encoding keeps the payload size small.
        """
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Failed to encode frame as JPEG.")
        return base64.b64encode(buffer).decode("utf-8")

    def _parse_response(self, raw_text: str) -> CompositionAdvice:
        """
        Parse the VLM's raw text response into a CompositionAdvice.

        VLMs sometimes wrap JSON in markdown code fences like:
            ```json
            { ... }
            ```
        This method strips that wrapper before parsing.
        Falls back to SAFE_DEFAULT if parsing fails.
        """
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])

        try:
            data = json.loads(cleaned)
            action = data.get("action", {})

            return CompositionAdvice(
                subjects_present=data.get("subjects_present", []),
                subjects_missing=data.get("subjects_missing", []),
                framing_quality=float(data.get("framing_quality", 0.0)),
                is_good_enough=bool(data.get("is_good_enough", False)),
                move_direction=action.get("move", "none"),
                move_distance_cm=int(action.get("distance_cm", 0)),
                tilt_direction=action.get("tilt_camera", "none"),
                tilt_degrees=int(action.get("tilt_degrees", 0)),
                reasoning=data.get("reasoning", "No reasoning provided.")
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse VLM response: {e}")
            logger.warning(f"Raw response was: {raw_text}")
            return SAFE_DEFAULT