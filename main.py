import cv2
import os
import yaml
import logging
from typing import Optional

from camera.webcam import Webcam
from detection.yolo_detector import YoloDetector
from scoring.nima_scorer import NimaScorer, AestheticScore
from composition.vlm_client import VlmClient, CompositionAdvice, SceneChangeDetector
from composition.frame_simulator import FrameSimulator
from composition.composition_loop import CompositionLoop

# ---------------------------------------------------------------------------
# Logging setup
# Use logging instead of print so you can control verbosity levels
# and see timestamps when debugging on the robot later.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Display helper for live monitoring mode
# ---------------------------------------------------------------------------

def draw_monitor_overlay(
    frame: cv2.typing.MatLike,
    advice: Optional[CompositionAdvice],
    nima: Optional[AestheticScore] = None
) -> cv2.typing.MatLike:
    """
    Draws the composition score, NIMA aesthetic score, and advice
    onto the frame for the live monitoring mode.

    Color coding:
    - Green  = good composition (above 0.75)
    - Yellow = needs adjustment (0.50 to 0.75)
    - Red    = poor composition (below 0.50)
    """
    output = frame.copy()

    if advice is None:
        # Show NIMA score even without VLM advice
        hint_text = "Monitoring... press SPACE to run composition loop"
        if nima:
            hint_text = f"NIMA: {nima.mean_score:.1f}/10 ({nima.rating})  |  SPACE = composition loop"
        cv2.putText(
            output, hint_text,
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
        )
        return output

    if advice.framing_quality >= 0.75:
        color = (0, 200, 0)
        status = "Good composition"
    elif advice.framing_quality >= 0.50:
        color = (0, 200, 255)
        status = "Needs adjustment"
    else:
        color = (0, 0, 220)
        status = "Poor composition"

    # Semi-transparent top bar (taller to fit NIMA line)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, output, 0.45, 0, output)

    # Quality score line with NIMA
    score_text = f"Composition: {advice.framing_quality:.0%}  |  {status}"
    cv2.putText(output, score_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # NIMA aesthetic score line
    if nima:
        nima_color = (0, 200, 0) if nima.mean_score >= 6.0 else (0, 200, 255) if nima.mean_score >= 4.5 else (0, 0, 220)
        nima_text = f"NIMA Aesthetic: {nima.mean_score:.2f}/10 ({nima.rating})  [confidence: +/-{nima.std_score:.1f}]"
        cv2.putText(output, nima_text, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, nima_color, 1)

    # Movement suggestion
    if not advice.is_good_enough and advice.move_direction != "none":
        suggestion = f"Suggestion: move {advice.move_direction} {advice.move_distance_cm}cm"
        if advice.tilt_direction != "none":
            suggestion += f"  |  tilt {advice.tilt_direction} {advice.tilt_degrees}deg"
        cv2.putText(
            output, suggestion,
            (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )

    # Reasoning
    cv2.putText(
        output, advice.reasoning[:90],
        (15, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1
    )

    # Missing subjects warning at the bottom
    if advice.subjects_missing:
        missing_text = f"Missing: {', '.join(advice.subjects_missing)}"
        cv2.putText(
            output, missing_text,
            (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 1
        )

    # Controls reminder
    hint = "SPACE = run loop   Q = quit"
    cv2.putText(
        output, hint,
        (frame.shape[1] - 240, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1
    )

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load all config from yaml — nothing is hardcoded in the code
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Build each module independently so any one can be swapped later
    webcam = Webcam(
        device_index=config["camera"]["device_index"]
    )
    detector = YoloDetector(
        model_path=config["yolo"]["model_path"],
        default_confidence=config["yolo"]["confidence_threshold"],
        class_confidence_overrides=config["yolo"].get("class_confidence_overrides")
    )
    nima = NimaScorer(
        weights_path=config["nima"].get("weights_path"),
        input_size=config["nima"].get("input_size", 224),
        device=config["nima"].get("device"),
    )
    # Key lives in an environment variable, not in the tracked yaml file —
    # keeps secrets out of git history. Config value kept as a fallback
    # for local-only overrides.
    api_key = os.environ.get("DASHSCOPE_API_KEY") or config["vlm"].get("api_key")
    if not api_key:
        logger.error(
            "No API key found. Set the DASHSCOPE_API_KEY environment variable "
            "(PowerShell: setx DASHSCOPE_API_KEY \"your-key\" then restart the terminal)."
        )
        return

    vlm = VlmClient(
        api_key=api_key,
        model=config["vlm"]["model"],
        max_retries=config["vlm"]["max_retries"]
    )
    scene_detector = SceneChangeDetector(
        cooldown_seconds=config["composition"]["cooldown_seconds"],
        movement_threshold_px=config["composition"]["movement_threshold_px"]
    )
    simulator = FrameSimulator(
        pixel_cm=config["composition"].get("pixels_cm", 5.0),
        pixel_degree=config["composition"].get("pixels_degree", 3.0)
    )
    loop = CompositionLoop(
        webcam=webcam,
        detector=detector,
        vlm=vlm,
        simulator=simulator,
        nima=nima,
        max_iterations=config["composition"].get("max_iterations", 8),
        frame_callback=None
    )

    if not webcam.open():
        logger.error("Could not open webcam. Check device_index in config/settings.yaml.")
        return

    intent = input("What do you want to photograph? > ").strip()
    if not intent:
        intent = "take a well-composed photo of the scene"

    logger.info(f"Intent: '{intent}'")
    logger.info("Live monitoring active. Press SPACE to run the composition loop. Press Q to quit.")

    last_advice: Optional[CompositionAdvice] = None
    last_nima: Optional[AestheticScore] = None
    nima_score_threshold: float = config["nima"].get("score_threshold", 5.0)
    capture_count: int = 0

    # NIMA every frame tanks FPS on CPU (50-150ms/frame). For the live
    # display overlay, a score a few frames old is indistinguishable —
    # so throttle it. A FRESH score is still computed right before every
    # VLM call below, so composition accuracy is unaffected.
    nima_interval: int = config["nima"].get("monitor_interval", 5)
    frame_count: int = 0

    # ---------------------------------------------------------------------------
    # Main loop — two modes:
    #
    # MONITORING mode (default):
    #   YOLO runs every frame. VLM called whenever scene changes.
    #   Shows live composition score on screen.
    #   Cheap — VLM only called when needed.
    #
    # COMPOSITION LOOP mode (triggered by SPACE):
    #   Runs the full iterative loop with frame simulator.
    #   Keeps evaluating and adjusting until good enough or max attempts.
    #   Shows result and waits for S to save or any key to discard.
    # ---------------------------------------------------------------------------

    while True:
        frame = webcam.read_frame()
        if frame is None:
            continue

        # YOLO runs every frame — fast and free
        detections = detector.detect(frame)

        # NIMA throttled to every Nth frame — display only needs a rough
        # live score, and this keeps monitoring FPS high on CPU
        frame_count += 1
        if last_nima is None or frame_count % nima_interval == 0:
            last_nima = nima.score_frame(frame)

        # VLM only when scene changes and cooldown passed — keeps cost low
        if scene_detector.should_call_vlm(detections):
            # Fresh NIMA on the exact frame the VLM will reason about —
            # never feed the VLM a stale throttled score
            last_nima = nima.score_frame(frame)
            logger.info(
                f"Scene changed — calling LLM (data-only). "
                f"NIMA: {last_nima.mean_score:.2f}/10 ({last_nima.rating})"
            )
            h, w = frame.shape[:2]
            last_advice = vlm.evaluate(
                intent=intent,
                detections=detections,
                frame_width=w,
                frame_height=h,
                nima_score=last_nima.mean_score,
                nima_std=last_nima.std_score,
                nima_rating=last_nima.rating,
            )
            scene_detector.mark_called(detections)

        # Draw YOLO boxes then composition overlay with NIMA score
        display_frame = detector.draw_boxes(frame, detections)
        display_frame = draw_monitor_overlay(display_frame, last_advice, last_nima)

        cv2.imshow("Tabby Zero", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            logger.info("Quit.")
            break

        if key == ord(" "):
            # Trigger the full composition loop
            logger.info("Composition loop triggered by user.")
            cv2.destroyWindow("Tabby Zero")

            result = loop.run(intent=intent)

            # Show result and let user decide to save or discard
            save_path = f"tabby_capture_{capture_count:03d}.jpg"
            saved = loop.show_result_and_wait_for_save(result, save_path=save_path)

            if saved:
                capture_count += 1
                print(
                    f"\nCapture {capture_count} saved to {save_path}"
                    f"  |  Quality: {result.best_advice.framing_quality:.0%}"
                    f"  |  NIMA: {result.best_nima_score:.2f}/10"
                    f"  |  Iterations: {result.total_iterations}"
                )

            # Return to monitoring mode
            logger.info("Returning to monitoring mode.")

    webcam.close()
    cv2.destroyAllWindows()
    logger.info("Tabby Zero shut down cleanly.")


if __name__ == "__main__":
    main()