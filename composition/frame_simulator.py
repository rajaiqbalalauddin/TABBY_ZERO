import cv2
import numpy as np
import logging
from typing import Literal

logger = logging.getLogger(__name__)
# "forward" — must match the exact string the VLM prompt asks for,
# otherwise zoom moves get silently ignored (this was a real bug)
Direction = Literal["forward","backward","left","right", "none"]
TiltDirection = Literal["up","down","none"]
class FrameSimulator:
    #simulate camera movement by shifting the frame
    def __init__(self,pixel_cm:float = 5.0,pixel_degree:float = 3.0,min_crop_ratio: float = 0.5,max_crop_ratio:float = 1.0) -> None:
        
        self.pixel_cm = pixel_cm
        self.pixel_degree = pixel_degree
        self.min_crop = min_crop_ratio
        self.max_crop = max_crop_ratio
        self.offset_x : int = 0
        self.offset_y : int = 0
        self.zoom_ratio:float = 1.0
        
    def reset(self) -> None :
        
        self.offset_x  = 0
        self.offset_y  = 0
        self.zoom_ratio = 1.0
        logger.debug("Frame simulator reset to origin.")
        
    def update_state(self,
                move_direction:Direction,
                move_distance:int,
                tilt_direction:TiltDirection,
                tilt_degrees:float,
                )-> None:
        # State update only, no rendering. Exists so the composition loop
        # can record a move without paying for a render it will discard —
        # the next iteration renders from fresh webcam pixels anyway.
        move_px = int(move_distance * self.pixel_cm)
        tilt_px = int(tilt_degrees * self.pixel_degree)

        if move_direction == "left":
            self.offset_x = self.offset_x - move_px
        elif move_direction == "right":
            self.offset_x = self.offset_x + move_px

        if tilt_direction == "down":
            self.offset_y = self.offset_y - tilt_px
        elif tilt_direction == "up":
            self.offset_y = self.offset_y + tilt_px

        if move_direction == "forward":
            zoom_step = move_distance*0.01
            self.zoom_ratio = max(self.min_crop,self.zoom_ratio - zoom_step)
        elif move_direction == "backward":
            zoom_step = move_distance*0.01
            self.zoom_ratio = min(self.max_crop,self.zoom_ratio + zoom_step)

        logger.debug(
            f"Simulator state: offset=({self.offset_x}, {self.offset_y}), "
            f"zoom={self.zoom_ratio:.2f}"
        )

    def apply_move(self,
                frame : cv2.typing.MatLike,
                move_direction:Direction,
                move_distance:int,
                tilt_direction:TiltDirection,
                tilt_degrees:float,
                )-> cv2.typing.MatLike:
        # Kept as the convenience path: update state then render in one call
        self.update_state(move_direction, move_distance, tilt_direction, tilt_degrees)
        return self.render(frame)
    
    def render(self,frame:cv2.typing.MatLike) -> cv2.typing.MatLike:
        
        h , w = frame.shape[:2]
        
        crop_w = int (w*self.zoom_ratio)
        crop_h = int (h*self.zoom_ratio)
        
        center_x = w // 2 + self.offset_x
        center_y = h // 2 + self.offset_y
        
        x1 = center_x - crop_w // 2
        y1 = center_y - crop_h // 2

        # Clamp crop inside the frame. The old code used max(x1, w - crop_w),
        # which PINNED the crop to the bottom-right corner whenever zoomed —
        # so simulated moves never matched what the VLM asked for.
        x1 = min(max(x1, 0), w - crop_w)
        y1 = min(max(y1, 0), h - crop_h)
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        cropped = frame[y1:y2,x1:x2]
        rendered = cv2.resize(cropped,(w,h),interpolation=cv2.INTER_LINEAR)
        
        return rendered
    
    def get_state_text(self) -> str:
        
            return (
            f"Sim: offset=({self.offset_x}px, {self.offset_y}px) "
            f"zoom={self.zoom_ratio:.2f}x"
        )