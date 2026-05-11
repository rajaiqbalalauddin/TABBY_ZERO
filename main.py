import cv2
import yaml
import logging
from typing import Optional

from camera.webcam import Webcam
from detection.yolo_detector import YoloDetector
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
    advice: Optional[CompositionAdvice]
) -> cv2.typing.MatLike:
    """
    Draws the composition score and advice onto the frame for
    the live monitoring mode (before the loop is triggered).

    Color coding:
    - Green  = good composition (above 0.75)
    - Yellow = needs adjustment (0.50 to 0.75)
    - Red    = poor composition (below 0.50)
    """
    output = frame.copy()

    if advice is None:
        cv2.putText(
            output, "Monitoring... press SPACE to run composition loop",
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

    # Semi-transparent top bar
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, output, 0.45, 0, output)

    # Quality score line
    score_text = f"Composition: {advice.framing_quality:.0%}  |  {status}"
    cv2.putText(output, score_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Movement suggestion
    if not advice.is_good_enough and advice.move_direction != "none":
        suggestion = f"Suggestion: move {advice.move_direction} {advice.move_distance_cm}cm"
        if advice.tilt_direction != "none":
            suggestion += f"  |  tilt {advice.tilt_direction} {advice.tilt_degrees}deg"
        cv2.putText(
            output, suggestion,
            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

    # Reasoning
    cv2.putText(
        output, advice.reasoning[:80],
        (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
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
    vlm = VlmClient(
        api_key=config["vlm"]["api_key"],
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
        max_iterations=config["composition"].get("max_iterations", 8),
        frame_callback= None
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
    capture_count: int = 0

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

        # VLM only when scene changes and cooldown passed — keeps cost low
        if scene_detector.should_call_vlm(detections):
            logger.info("Scene changed — calling VLM.")
            last_advice = vlm.evaluate(frame, intent, detections)
            scene_detector.mark_called(detections)

        # Draw YOLO boxes then composition overlay
        display_frame = detector.draw_boxes(frame, detections)
        display_frame = draw_monitor_overlay(display_frame, last_advice)

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
                    f"  |  Iterations: {result.total_iterations}"
                )

            # Return to monitoring mode
            logger.info("Returning to monitoring mode.")

    webcam.close()
    cv2.destroyAllWindows()
    logger.info("Tabby Zero shut down cleanly.")


if __name__ == "__main__":
    main()