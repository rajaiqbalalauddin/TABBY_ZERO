import cv2 
import logging
import time
from dataclasses import dataclass
from typing import Optional,Callable

from camera.webcam import Webcam
from detection.yolo_detector import YoloDetector
from composition.vlm_client import VlmClient,CompositionAdvice,SAFE_DEFAULT
from composition.frame_simulator import FrameSimulator

logger = logging.getLogger(__name__)

@dataclass
class LoopResult:
    best_frame:cv2.typing.MatLike
    best_advice:CompositionAdvice
    total_iterations:int
    success:bool
    
def draw_loop_overlay(
        frame:cv2.typing.MatLike,
        advice: Optional,
        iteration: int,
        max_iterations: int,
        simulator_state:str,
    ) -> cv2.typing.MatLike:
        
        output = frame.copy() 
        overlay = output.copy()
        cv2.rectangle(overlay,(0,0),(frame.shape[1],110),(0,0,0,),-1)
        
        cv2.addWeighted(overlay,0.6,output,0.4,0,output)
        
        overlay2 = output.copy()
        cv2.rectangle(overlay2, (0,frame.shape[0] -30),(frame.shape[1],frame.shape[0]),(0,0,0) , -1)
        cv2.addWeighted(overlay2,0.50,output,0.5,0,output)
        
        if advice is None:
            cv2.putText(
                output,f"Iteration {iteration}/{max_iterations} - Evalutating...",(15,35),cv2.FONT_HERSHEY_COMPLEX,0.7,(20,0,200,200),2)
            
        return output    
    
class CompositionLoop:
    
    def __init__(self,webcam:Webcam,detector:YoloDetector,vlm:VlmClient,simulator:FrameSimulator,max_iterations:int,frame_callback:Optional[Callable[[cv2.typing.MatLike],None]])-> None:
        
        self.webcam = webcam
        self.detector = detector
        self.vlm = vlm
        self.simulator = simulator
        self.max_iterations = max_iterations
        self.frame_callback = frame_callback 
        
    def run(self,intent:str) -> LoopResult:
        logger.info(f"Starting composition loop. Intent: '{intent}' | Max iterations: {self.max_iterations}")
        self.simulator.reset()
        
        best_frame : Optional[cv2.typing.MatLike] = None
        best_advice : CompositionAdvice = SAFE_DEFAULT
        best_quality:float = 0.0
        
        for iteration in range(1,self.max_iterations + 1 ):
            logger.info(f"--- Iteration {iteration}/{self.max_iterations} ---")
            
            raw_frame = self._capture_stable_frame()
            if raw_frame is None:
                logger.warning(f"Could not capture frame on iteration {iteration}. Skipping.")
                continue
        
            simulated_frame = self.simulator.apply_move(
                raw_frame,
                move_direction="none",
                move_distance= 0,
                tilt_direction="none",
                tilt_degrees=0
            )
            
            detections = self.detector.detect(simulated_frame)
            logger.info(
                f"YOLO detected: {[d.label for d in detections] or 'nothing'}"
            )
            
            logger.info("Calling VLM for composition evaluation...")
            
            advice = self.vlm.evaluate(simulated_frame,intent,detections)
            logger.info(
                f"Quality: {advice.framing_quality:.2f} | "
                f"Good enough: {advice.is_good_enough} | "
                f"Reason: {advice.reasoning}"
            )
            
            if advice.framing_quality > best_quality:
                best_quality = advice.framing_quality
                best_frame = simulated_frame.copy()
                best_advice = advice
                
            display = draw_loop_overlay(
                frame = simulated_frame,
                advice = advice,
                iteration = iteration,
                max_iterations = self.max_iterations,
                simulator_state = self.simulator.get_state_text()
            )
            cv2.imshow("Tabby Zero - Composition Loop",display)
            
            if self.frame_callback:
                self.frame_callback(display)
                
            cv2.waitKey(1)
            
            if advice.is_good_enough:
                logger.info(f"Composition accepted at iteration {iteration}.")
                break
            
            if iteration < self.max_iterations:
                logger.info(
                    f"Applying move: {advice.move_direction} {advice.move_distance_cm}cm | "
                    f"Tilt: {advice.tilt_direction} {advice.tilt_degrees}deg"
                )
            
            self.simulator.apply_move(
                raw_frame,
                move_direction= advice.move_direction,
                move_distance=advice.move_distance_cm,
                tilt_degrees= advice.tilt_degrees,
                tilt_direction= advice.tilt_direction
            )
            
            time.sleep(0.3)
            
            if best_frame is None:
                logger.warning("No valid frame captured during loop. Using last raw frame.")
                best_frame = raw_frame if raw_frame is not None else self.capture_stable_frame()
                
            total_iterations = min(iteration,self.max_iterations)
            success = best_advice.is_good_enough

            logger.info(
            f"Loop finished. Iterations: {total_iterations} | "
            f"Best quality: {best_quality:.2f} | Success: {success}"
        )
            
            return LoopResult(
            best_frame=best_frame,
            best_advice=best_advice,
            total_iterations=total_iterations,
            success=success
        )
    
    def show_result_and_wait_for_save(
        self,
        result:LoopResult,
        save_path:str = "tabby_capture.jpg"
        )-> bool:
        
        if result.best_frame is None:
            logger.error("No frame to display.")
            return False
        
        display = result.best_frame.copy()
        
        h,w = display.shape[:2]
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
        
        status = "Composition accepted" if result.success else "Best effort result"
        color = (0, 220, 0) if result.success else (0, 180, 255)
        
        cv2.putText(
            display,
            f"{status}  |  Quality: {result.best_advice.framing_quality:.0%}  |  "
            f"Iterations: {result.total_iterations}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
        )
        cv2.putText(
            display, result.best_advice.reasoning[:85],
            (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1
        )
        cv2.putText(
            display, "Press S to save   |   Press any other key to discard",
            (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        cv2.imshow("Tabby Zero - Result" , display)
        logger.info("Showing result. Press S to save, any other key to discard.")
        
        while True:
            key = cv2.waitKey(0) & 0xff
            
            if key == ord("s") or key == ord("S"):
                cv2.imwrite(save_path,result.best_frame)
                logger.info(f"Frame saved to: {save_path}")
                print(f"\nSaved to {save_path}")
                cv2.destroyWindow("Tabby Zero - Result")
                return True
            else:
                logger.info("Frame discarded by user.")
                cv2.destroyWindow("Tabby Zero - Result")
                return False
            
    def _capture_stable_frame(self, warmup_frames: int = 3) -> Optional[cv2.typing.MatLike]:
        
        for _ in range(warmup_frames):
            self.webcam.read_frame()
        return self.webcam.read_frame()