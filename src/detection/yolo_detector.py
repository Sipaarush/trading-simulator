"""
YOLOv8 Detector Module — Runs YOLOv8 inference on preprocessed chart frames
to detect classical chart patterns. Uses GPU (CUDA) when available with FP16
for performance. Falls back to CPU on OOM errors.

Contract:
    Input:  frame (np.ndarray) — preprocessed (640x640x3 float32)
    Output: List[PatternResult]
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.core.hardware_profile import HardwareProfile
from src.detection.pattern_registry import PatternResult, get_pattern

logger = logging.getLogger(__name__)

# Mapping from YOLOv8 class names to pattern registry keys.
# The trained model's class names map to our registry.
YOLO_CLASS_MAP: dict[str, str] = {
    "head_and_shoulders": "HEAD_AND_SHOULDERS",
    "inv_head_shoulders": "INV_HEAD_SHOULDERS",
    "double_top": "DOUBLE_TOP",
    "double_bottom": "DOUBLE_BOTTOM",
    "ascending_triangle": "ASCENDING_TRIANGLE",
    "descending_triangle": "DESCENDING_TRIANGLE",
    "symmetrical_triangle": "SYMMETRICAL_TRIANGLE",
    "bull_flag": "BULL_FLAG",
    "bear_flag": "BEAR_FLAG",
    "rising_wedge": "RISING_WEDGE",
    "falling_wedge": "FALLING_WEDGE",
    "cup_and_handle": "CUP_AND_HANDLE",
    "pennant": "PENNANT",
    # Additional class name variations
    "head_shoulders": "HEAD_AND_SHOULDERS",
    "inverse_head_shoulders": "INV_HEAD_SHOULDERS",
    "bullish_flag": "BULL_FLAG",
    "bearish_flag": "BEAR_FLAG",
}


class YOLODetector:
    """
    YOLOv8-based chart pattern detector.

    Loads the model from config, runs inference with hardware-appropriate
    settings, and returns List[PatternResult].
    """

    def __init__(self, config: dict, profile: HardwareProfile) -> None:
        """
        Initialize YOLODetector.

        Args:
            config: Application config dict. Uses config['model']['model_path'].
            profile: HardwareProfile with device, img_size, and half_precision.
        """
        self._config = config
        self._profile = profile
        self._model = None
        self._model_loaded = False

        model_cfg = config.get("model", {})
        self._model_path: str = model_cfg.get("model_path", "models/best.pt")
        self._confidence_threshold: float = model_cfg.get("confidence_threshold", 0.65)
        self._img_size: int = profile.yolo_img_size
        self._half_precision: bool = profile.half_precision
        self._device: str = profile.device

        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLOv8 model. Downloads if not present."""
        try:
            from ultralytics import YOLO

            model_path = Path(self._model_path)

            if model_path.exists():
                logger.info("Loading YOLOv8 model from %s", model_path)
                self._model = YOLO(str(model_path))
            else:
                # Use a pretrained model as fallback (will need fine-tuning)
                logger.warning(
                    "Model not found at %s — loading pretrained %s",
                    model_path,
                    self._profile.yolo_model,
                )
                self._model = YOLO(f"{self._profile.yolo_model}.pt")

            # Move to appropriate device
            if self._device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        logger.info("Model loaded on CUDA (FP16: %s)", self._half_precision)
                    else:
                        self._device = "cpu"
                        self._half_precision = False
                        logger.info("CUDA not available — using CPU")
                except ImportError:
                    self._device = "cpu"
                    self._half_precision = False

            self._model_loaded = True
            logger.info(
                "YOLODetector ready — Device: %s | ImgSize: %d | FP16: %s | Threshold: %.2f",
                self._device,
                self._img_size,
                self._half_precision,
                self._confidence_threshold,
            )

        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            self._model_loaded = False

    def detect(self, frame: np.ndarray) -> list[PatternResult]:
        """
        Run YOLOv8 inference on a frame.

        Args:
            frame: Preprocessed frame, shape (H, W, 3). Can be float32 [0-1]
                   or uint8 [0-255]. YOLOv8 handles both.

        Returns:
            List[PatternResult] — detected patterns with confidence and bbox.
            Returns empty list on error.
        """
        if not self._model_loaded or self._model is None:
            logger.warning("YOLO model not loaded — skipping detection")
            return []

        if frame is None or frame.size == 0:
            logger.warning("Empty frame — skipping YOLO detection")
            return []

        try:
            # Convert float32 [0-1] to uint8 [0-255] if needed
            if frame.dtype == np.float32:
                inference_frame = (frame * 255).astype(np.uint8)
            else:
                inference_frame = frame

            # Run inference
            results = self._model.predict(
                source=inference_frame,
                imgsz=self._img_size,
                conf=self._confidence_threshold,
                device=self._device,
                half=self._half_precision,
                verbose=False,
            )

            # Parse results into PatternResult objects
            patterns = self._parse_results(results)
            logger.debug("YOLOv8 detected %d patterns", len(patterns))
            return patterns

        except Exception as e:
            error_str = str(e).lower()

            # Handle CUDA OOM
            if "out of memory" in error_str or "cuda" in error_str:
                logger.warning("CUDA OOM — clearing cache and retrying on CPU")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                # Retry on CPU
                try:
                    results = self._model.predict(
                        source=inference_frame if 'inference_frame' in dir() else frame,
                        imgsz=self._img_size,
                        conf=self._confidence_threshold,
                        device="cpu",
                        half=False,
                        verbose=False,
                    )
                    return self._parse_results(results)
                except Exception as cpu_err:
                    logger.error("CPU fallback also failed: %s", cpu_err)
                    return []
            else:
                logger.error("YOLO detection error: %s", e)
                return []

    def _parse_results(self, results) -> list[PatternResult]:
        """
        Parse ultralytics results into List[PatternResult].

        Args:
            results: ultralytics prediction results.

        Returns:
            List[PatternResult].
        """
        patterns = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                try:
                    # Get class name and confidence
                    cls_id = int(boxes.cls[i].item())
                    confidence = float(boxes.conf[i].item())

                    # Get class name from model
                    if hasattr(result, "names") and cls_id in result.names:
                        class_name = result.names[cls_id].lower().replace(" ", "_")
                    else:
                        continue

                    # Map to registry key
                    registry_key = YOLO_CLASS_MAP.get(class_name)
                    if registry_key is None:
                        # Try direct uppercase match
                        registry_key = class_name.upper()

                    # Get pattern metadata
                    meta = get_pattern(registry_key)
                    if meta is None:
                        logger.debug("Unmapped YOLO class: %s", class_name)
                        continue

                    # Get bounding box
                    bbox_tensor = boxes.xyxy[i]
                    bbox = (
                        int(bbox_tensor[0].item()),
                        int(bbox_tensor[1].item()),
                        int(bbox_tensor[2].item()),
                        int(bbox_tensor[3].item()),
                    )

                    pattern = PatternResult(
                        name=registry_key,
                        display_name=meta.display_name,
                        confidence=confidence,
                        bbox=bbox,
                        direction=meta.direction,
                        source="yolov8",
                    )
                    patterns.append(pattern)

                except Exception as e:
                    logger.debug("Error parsing detection %d: %s", i, e)
                    continue

        return patterns

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready."""
        return self._model_loaded
