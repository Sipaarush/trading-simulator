"""
Preprocessor Module — Applies CLAHE, GaussianBlur, resize, and normalization
to raw captured frames for YOLOv8 inference input.

Contract (Layer 1 → Layer 2):
    processed: np.ndarray  — shape (640, 640, 3), float32 [0.0–1.0]
    original_crop: np.ndarray — raw frame at captured resolution (for overlay)
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess(
    frame: np.ndarray,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a raw captured frame for detection.

    Pipeline:
        BGR → Grayscale → CLAHE(clipLimit=2.0, tileGrid=8×8) → BGR
        → GaussianBlur(3,3) → resize to target_size × target_size
        → normalize [0, 1] float32

    Args:
        frame: Raw BGR frame from screen capture, shape (H, W, 3), dtype uint8.
        config: Application config dict. Uses 'model.img_size' for target
                resolution (default 640).

    Returns:
        Tuple of:
            processed — shape (target_size, target_size, 3), dtype float32, [0-1]
            original_crop — unmodified copy of input frame (for overlay later)
    """
    if frame is None or frame.size == 0:
        logger.error("Received empty frame for preprocessing")
        raise ValueError("Empty frame received")

    # Keep original for overlay
    original_crop = frame.copy()

    # Get target image size from config
    model_cfg = config.get("model", {})
    target_size: int = model_cfg.get("img_size", 640)

    try:
        # Step 1: Convert to grayscale for CLAHE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Step 3: Convert back to BGR (3-channel)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Step 4: Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced_bgr, (3, 3), 0)

        # Step 5: Resize to target dimensions
        resized = cv2.resize(
            blurred,
            (target_size, target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # Step 6: Normalize to [0, 1] float32
        processed = resized.astype(np.float32) / 255.0

        logger.debug(
            "Preprocessed frame: %s → (%d, %d, 3) float32",
            frame.shape,
            target_size,
            target_size,
        )

        return processed, original_crop

    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        raise
