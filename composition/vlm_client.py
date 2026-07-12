from openai import OpenAI
from dataclasses import dataclass
from typing import Optional
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
# Static system prompt
#
# Kept as a constant (not rebuilt per call) so the API sees an identical
# prefix on every request — DashScope caches repeated prefixes, which cuts
# both latency and token cost on every call after the first.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a professional photography composition assistant.
You do NOT see the actual image. Instead, you receive structured data from
computer vision models (YOLO object detection and NIMA aesthetic scoring).
Your job is to IMAGINE what the photo looks like based on this data,
then evaluate the composition and suggest specific improvements.

YOUR TASK:
1. Based on the object positions, sizes, and relationships provided, IMAGINE
   what this photograph looks like. Consider where subjects are placed
   relative to the rule of thirds grid, how much of the frame they fill,
   and how they relate spatially to each other.

2. Evaluate the composition quality considering:
   - Rule of thirds alignment
   - Subject placement and balance
   - Frame coverage (too tight, too loose, or well-framed)
   - Spatial relationships between objects
   - Whether the detected objects match the user's intent
   - The NIMA aesthetic score as an objective quality baseline

3. Suggest a specific camera movement to improve the composition.
   If a move history is provided, learn from it: do NOT repeat a move
   that lowered the quality score, and do NOT oscillate between opposite
   directions. Converge on the best composition in as few moves as possible.

Return ONLY a JSON object. No explanation before or after.
No markdown code blocks. Raw JSON only.

Return exactly this structure:
{
  "subjects_present": ["list", "of", "detected", "subjects", "relevant", "to", "intent"],
  "subjects_missing": ["list", "of", "subjects", "needed", "but", "not", "visible"],
  "framing_quality": 0.0,
  "is_good_enough": false,
  "action": {
    "move": "left | right | forward | backward | none",
    "distance_cm": 0,
    "tilt_camera": "up | down | none",
    "tilt_degrees": 0
  },
  "reasoning": "Describe what you imagine the scene looks like, what composition issues you identified, and why you suggested this specific movement. Reference the NIMA score and specific composition principles."
}

Rules:
- framing_quality is a float between 0.0 (terrible) and 1.0 (perfect)
- is_good_enough is true only if framing_quality is above 0.75 AND NIMA score is above 5.0
- distance_cm should be between 0 and 100
- tilt_degrees should be between 0 and 30
- If framing is already good, set move to "none" and distance_cm to 0
"""


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
    Handles all communication with the Qwen LLM API.

    Instead of sending raw images, this client builds a rich text
    description of the scene from YOLO detections and NIMA scores.
    The LLM imagines the composition from the data and suggests
    improvements — no vision model required.

    Uses the OpenAI-compatible SDK pointed at Alibaba Cloud's DashScope.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus",
        max_retries: int = 2
    ) -> None:
        self.model_name = model
        self.max_retries = max_retries

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        logger.info(f"Qwen LLM client ready (text-only). Model: {model}")

    def evaluate(
        self,
        intent: str,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
        nima_score: float = 0.0,
        nima_std: float = 0.0,
        nima_rating: str = "unknown",
        history: Optional[list[str]] = None,
    ) -> CompositionAdvice:
        """
        Build a scene description from detection data and NIMA scores,
        send it to the LLM as text only, and get composition advice back.

        The LLM never sees the actual image — it imagines the scene
        from the structured data and reasons about composition.

        history: short per-iteration summaries of previous moves and their
        resulting scores. Giving the model memory of what it already tried
        stops it oscillating (left, right, left...) and cuts iterations.
        """
        scene_description = self._build_scene_description(
            detections, frame_width, frame_height
        )
        user_message = self._build_user_message(
            intent, scene_description, frame_width, frame_height,
            nima_score, nima_std, nima_rating, history
        )

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Calling Qwen LLM (attempt {attempt + 1}/{self.max_retries})")

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    # Static system prompt + dynamic user message:
                    # the identical system prefix is cacheable server-side
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    # Force valid JSON output — removes markdown-fence noise
                    # and makes parsing reliable
                    response_format={"type": "json_object"},
                    # Cap generation length: the JSON never needs more than
                    # this, and fewer tokens = faster responses
                    max_tokens=800,
                )

                raw_text = response.choices[0].message.content
                # _parse_response now RAISES on bad JSON instead of silently
                # returning SAFE_DEFAULT, so a garbage response triggers a
                # retry here instead of wasting a whole loop iteration
                advice = self._parse_response(raw_text)
                logger.info(
                    f"LLM response: quality={advice.framing_quality:.2f}, "
                    f"good_enough={advice.is_good_enough}"
                )
                return advice

            except Exception as e:
                logger.warning(f"Qwen attempt {attempt + 1} failed: {e}")
                # Short backoff before retrying — immediate retries tend to
                # hit the same transient error (rate limit, network blip)
                if attempt + 1 < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))

        logger.error("All Qwen attempts failed. Returning safe default.")
        return SAFE_DEFAULT

    # ------------------------------------------------------------------
    # Scene description builder
    # ------------------------------------------------------------------

    def _build_scene_description(
        self,
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
    ) -> str:
        """
        Convert YOLO detections into a rich text description of the scene.

        For each object, includes:
        - Label and confidence
        - Bounding box coordinates
        - Position in frame (left/center/right, top/middle/bottom)
        - Size relative to the frame (% of frame area)
        - Center point

        Also computes spatial relationships between objects.
        """
        if not detections:
            return "No objects detected in the frame. The scene appears empty."

        frame_area = frame_w * frame_h
        lines = []

        for i, d in enumerate(detections, 1):
            # Bounding box dimensions
            box_w = d.x2 - d.x1
            box_h = d.y2 - d.y1
            area_pct = (box_w * box_h) / frame_area * 100

            # Center of the object
            cx = (d.x1 + d.x2) / 2
            cy = (d.y1 + d.y2) / 2

            # Horizontal position
            x_ratio = cx / frame_w
            if x_ratio < 0.33:
                h_pos = "left third"
            elif x_ratio < 0.66:
                h_pos = "center"
            else:
                h_pos = "right third"

            # Vertical position
            y_ratio = cy / frame_h
            if y_ratio < 0.33:
                v_pos = "upper"
            elif y_ratio < 0.66:
                v_pos = "middle"
            else:
                v_pos = "lower"

            # Rule of thirds proximity
            thirds_x = [frame_w / 3, 2 * frame_w / 3]
            thirds_y = [frame_h / 3, 2 * frame_h / 3]
            near_vertical_third = any(abs(cx - tx) < frame_w * 0.05 for tx in thirds_x)
            near_horizontal_third = any(abs(cy - ty) < frame_h * 0.05 for ty in thirds_y)
            on_thirds = near_vertical_third or near_horizontal_third

            lines.append(
                f"  Object {i}: {d.label} (confidence: {d.confidence:.0%})\n"
                f"    - Bounding box: ({d.x1}, {d.y1}) to ({d.x2}, {d.y2})\n"
                f"    - Center: ({cx:.0f}, {cy:.0f})\n"
                f"    - Position: {v_pos}-{h_pos} of frame\n"
                f"    - Size: {box_w}x{box_h}px = {area_pct:.1f}% of frame\n"
                f"    - On rule-of-thirds line: {'yes' if on_thirds else 'no'}"
            )

        objects_text = "\n".join(lines)

        # Spatial relationships between objects
        relationships = []
        for i, a in enumerate(detections):
            for j, b in enumerate(detections):
                if j <= i:
                    continue
                a_cx = (a.x1 + a.x2) / 2
                a_cy = (a.y1 + a.y2) / 2
                b_cx = (b.x1 + b.x2) / 2
                b_cy = (b.y1 + b.y2) / 2

                dx = b_cx - a_cx
                dy = b_cy - a_cy
                dist = math.sqrt(dx ** 2 + dy ** 2)
                dist_pct = dist / math.sqrt(frame_w ** 2 + frame_h ** 2) * 100

                if abs(dx) > abs(dy):
                    direction = "to the right of" if dx > 0 else "to the left of"
                else:
                    direction = "below" if dy > 0 else "above"

                # Check overlap
                overlap_x = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
                overlap_y = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
                overlapping = overlap_x > 0 and overlap_y > 0

                rel = f"  - {b.label} is {direction} {a.label} (distance: {dist_pct:.1f}% of diagonal)"
                if overlapping:
                    rel += " [OVERLAPPING]"
                relationships.append(rel)

        rel_text = "\n".join(relationships) if relationships else "  - Only one object detected, no relationships."

        # Frame coverage analysis
        total_coverage = sum(
            (d.x2 - d.x1) * (d.y2 - d.y1) for d in detections
        ) / frame_area * 100

        # Balance analysis — are objects concentrated on one side?
        if detections:
            avg_cx = sum((d.x1 + d.x2) / 2 for d in detections) / len(detections)
            balance_ratio = avg_cx / frame_w
            if balance_ratio < 0.35:
                balance = "objects clustered toward the LEFT side"
            elif balance_ratio > 0.65:
                balance = "objects clustered toward the RIGHT side"
            else:
                balance = "objects roughly centered horizontally"
        else:
            balance = "no objects to analyze"

        return (
            f"Detected {len(detections)} object(s):\n"
            f"{objects_text}\n\n"
            f"Spatial relationships:\n{rel_text}\n\n"
            f"Frame coverage: {total_coverage:.1f}% of frame occupied by detected objects\n"
            f"Balance: {balance}"
        )

    # ------------------------------------------------------------------
    # User message builder
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        intent: str,
        scene_description: str,
        frame_width: int,
        frame_height: int,
        nima_score: float = 0.0,
        nima_std: float = 0.0,
        nima_rating: str = "unknown",
        history: Optional[list[str]] = None,
    ) -> str:
        """
        Build only the DYNAMIC part of the prompt (scene data, intent,
        scores, move history). All static instructions live in
        SYSTEM_PROMPT so they form a cacheable prefix.
        """
        nima_context = ""
        if nima_score > 0:
            nima_context = (
                f"\n--- AESTHETIC SCORE (NIMA Neural Image Assessment) ---\n"
                f"Score: {nima_score:.2f}/10 ({nima_rating})\n"
                f"Confidence interval: ±{nima_std:.2f}\n"
                f"Interpretation: scores below 5.0 need significant improvement, "
                f"above 7.0 indicates excellent aesthetics.\n"
            )

        # Move history gives the model feedback on its own past advice
        # so it converges instead of wandering
        history_context = ""
        if history:
            history_lines = "\n".join(f"  {line}" for line in history)
            history_context = (
                f"\n--- MOVE HISTORY (previous iterations of this session) ---\n"
                f"{history_lines}\n"
                f"Use this to avoid repeating moves that did not improve the score.\n"
            )

        return (
            f"--- USER INTENT ---\n"
            f"The photographer wants to: {intent}\n\n"
            f"--- FRAME METADATA ---\n"
            f"Resolution: {frame_width}x{frame_height} pixels\n"
            f"Aspect ratio: {frame_width / frame_height:.2f}:1\n\n"
            f"--- SCENE DATA (from YOLO object detection) ---\n"
            f"{scene_description}\n"
            f"{nima_context}"
            f"{history_context}"
        )

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_response(self, raw_text: str) -> CompositionAdvice:
        """
        Parse the LLM's raw text response into a CompositionAdvice.

        LLMs sometimes wrap JSON in markdown code fences like:
            ```json
            { ... }
            ```
        This method strips that wrapper before parsing (kept as a fallback
        even though response_format=json_object should prevent it).

        RAISES ValueError on unparseable responses instead of returning
        SAFE_DEFAULT. Why: swallowing the error here meant a garbage
        response consumed a full loop iteration with quality 0.0. Raising
        lets evaluate()'s retry loop actually retry the API call.
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

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.warning(f"Raw response was: {raw_text}")
            raise ValueError(f"Unparseable LLM response: {e}") from e