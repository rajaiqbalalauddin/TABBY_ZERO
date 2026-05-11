import cv2
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Webcam:
    """
    Handles opening the webcam and capturing frames.
    Keeping this separate means swapping to a different
    camera source later (like Orange Pi CSI) is a one-line change.
    """

    def __init__(self, device_index: int = 0) -> None:
        # device_index 0 = built-in webcam
        # device_index 1 = external webcam if you have one
        self.device_index = device_index
        self.capture: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open the webcam. Returns True if successful."""
        self.capture = cv2.VideoCapture(self.device_index)
        if not self.capture.isOpened():
            logger.error(f"Could not open webcam at device index {self.device_index}")
            return False
        logger.info(f"Webcam opened at device index {self.device_index}")
        return True

    def read_frame(self) -> Optional[cv2.typing.MatLike]:
        """
        Capture one frame from the webcam.
        Returns the frame, or None if something went wrong.
        """
        if self.capture is None:
            logger.error("Webcam is not open. Call open() first.")
            return None

        success, frame = self.capture.read()
        if not success:
            logger.warning("Failed to read frame from webcam.")
            return None

        return frame

    def close(self) -> None:
        """Release the webcam so other apps can use it."""
        if self.capture is not None:
            self.capture.release()
            logger.info("Webcam closed.")