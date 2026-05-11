from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional
import logging
import cv2

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """
    A single detected object.
    Using a dataclass gives clean typed objects instead of
    raw lists or dicts that are easy to misread or misuse.
    """
    label: str
    confidence: float
    x1: int                         # bounding box top-left x
    y1: int                         # bounding box top-left y
    x2: int                         # bounding box bottom-right x
    y2: int                         # bounding box bottom-right y
    mask: Optional[object] = None   # pixel mask, only present with -seg models


class YoloDetector:
    """
    Wraps YOLOv8 detection with per-class confidence thresholds.

    Per-class thresholds let you make YOLO more sensitive to
    important classes (lower threshold = detects easier) and
    stricter on classes you care less about (higher threshold).

    Supports both standard detection models (yolov8n.pt)
    and segmentation models (yolov8n-seg.pt).
    Swap the model_path in settings.yaml to switch between them.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        default_confidence: float = 0.5,
        class_confidence_overrides: Optional[dict[str, float]] = None,
    ) -> None:
        """
        model_path: which YOLO model file to load.
        default_confidence: fallback threshold for all classes not in overrides.
        class_confidence_overrides: per-class thresholds.
            Lower value = more sensitive (detects easier).
            Higher value = stricter (only reports when very sure).

        Example overrides:
            {
                "person": 0.25,   # always catch the person
                "trophy": 0.30,   # fairly sensitive for props
                "chair":  0.80,   # strict — background clutter
            }
        """
        self.default_confidence = default_confidence
        self.class_confidence_overrides: dict[str, float] = class_confidence_overrides or {}

        logger.info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        logger.info("YOLO model loaded.")

        if self.class_confidence_overrides:
            logger.info(f"Per-class confidence overrides active: {self.class_confidence_overrides}")

    def detect(self, frame: cv2.typing.MatLike) -> list[Detection]:
        """
        Run detection on a single frame.
        Returns a list of Detection objects.

        Why two-pass approach:
        YOLO's built-in conf parameter is a single global value.
        To support per-class thresholds, we run YOLO at the lowest
        threshold so nothing gets missed, then filter each detection
        against its own class-specific threshold in a second pass.
        """
        # Run at the lowest threshold so no detection gets cut too early
        all_thresholds = list(self.class_confidence_overrides.values()) + [self.default_confidence]
        run_threshold = min(all_thresholds)

        results = self.model(frame, conf=run_threshold, verbose=False)
        detections: list[Detection] = []

        for result in results:
            for i, box in enumerate(result.boxes):
                label = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                # Use this class's specific threshold, or fall back to default
                threshold_for_this_class = self.class_confidence_overrides.get(
                    label,
                    self.default_confidence
                )

                # Filter out detections below their class threshold
                if confidence < threshold_for_this_class:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Grab segmentation mask if this is a -seg model
                mask = None
                if result.masks is not None:
                    mask = result.masks.data[i].cpu().numpy()

                detections.append(Detection(
                    label=label,
                    confidence=confidence,
                    x1=x1, y1=y1,
                    x2=x2, y2=y2,
                    mask=mask
                ))

        return detections

    def draw_boxes(self, frame: cv2.typing.MatLike, detections: list[Detection]) -> cv2.typing.MatLike:
        """
        Draw bounding boxes and labels onto the frame.
        Returns a copy of the frame with boxes drawn.
        Display logic lives here since it is specific to detections.
        """
        output = frame.copy()

        for det in detections:
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)

            label_text = f"{det.label} {det.confidence:.2f}"
            cv2.putText(
                output, label_text,
                (det.x1, det.y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

        return output