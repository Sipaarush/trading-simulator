"""
Screen Capture Module — Captures frames from the user's screen using mss.
Provides a ScreenCapture class with start/stop lifecycle, background daemon
thread, thread-safe queue, and an interactive region selector via OpenCV.
"""

import logging
import threading
import time
from queue import Queue, Full
from typing import Optional

import cv2
import mss
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Default capture region (full HD left portion)
DEFAULT_REGION = {"top": 0, "left": 0, "width": 1280, "height": 720}


class ScreenCapture:
    """
    Captures screen frames in a background daemon thread.

    Usage:
        capture = ScreenCapture(config)
        capture.start()
        frame = capture.get_frame(timeout=1.0)
        capture.stop()
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize ScreenCapture.

        Args:
            config: Application config dict. Expects 'capture.region' and
                    'capture.interval_ms' keys.
        """
        capture_cfg = config.get("capture", {})
        region = capture_cfg.get("region", DEFAULT_REGION)
        self._region = {
            "top": region.get("top", 0),
            "left": region.get("left", 0),
            "width": region.get("width", 1280),
            "height": region.get("height", 720),
        }
        self._interval_ms: int = capture_cfg.get("interval_ms", 500)
        self._queue: Queue = Queue(maxsize=3)
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        logger.info(
            "ScreenCapture initialized — Region: %s | Interval: %dms",
            self._region,
            self._interval_ms,
        )

    def _capture_loop(self) -> None:
        """Background loop that continuously captures screen frames."""
        with mss.mss() as sct:
            while self._running:
                try:
                    start_time = time.perf_counter()

                    # Capture the region
                    screenshot = sct.grab(self._region)

                    # Convert to numpy BGR array (mss returns BGRA)
                    frame = np.array(screenshot, dtype=np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Put into queue (drop oldest if full)
                    try:
                        self._queue.put_nowait(frame)
                    except Full:
                        try:
                            self._queue.get_nowait()
                        except Exception:
                            pass
                        self._queue.put_nowait(frame)

                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    sleep_ms = max(0, self._interval_ms - elapsed_ms)
                    if sleep_ms > 0:
                        time.sleep(sleep_ms / 1000.0)

                except Exception as e:
                    logger.error("Capture error: %s", e)
                    time.sleep(0.1)

    def start(self) -> None:
        """Start the background capture thread."""
        if self._running:
            logger.warning("ScreenCapture already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="ScreenCaptureThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("ScreenCapture started")

    def stop(self) -> None:
        """Stop the background capture thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("ScreenCapture stopped")

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the latest captured frame.

        Args:
            timeout: Max seconds to wait for a frame.

        Returns:
            np.ndarray of shape (H, W, 3), dtype uint8, BGR format.
            Returns None if no frame available within timeout.
        """
        try:
            frame = self._queue.get(timeout=timeout)
            return frame
        except Exception:
            logger.warning("No frame available within %.1fs", timeout)
            return None

    @property
    def region(self) -> dict:
        """Current capture region."""
        return self._region.copy()


def draw_region_selector(config_path: str = "config.yaml") -> dict:
    """
    Interactive region selector — user draws a rectangle on a screenshot.
    Saves the selected region to config.yaml.

    Args:
        config_path: Path to config file to update.

    Returns:
        dict with top, left, width, height of the selected region.
    """
    logger.info("Opening region selector — drag to select chart area")

    # Capture full screen
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = sct.grab(monitor)
        screen = np.array(screenshot, dtype=np.uint8)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

    # Resize for display if needed
    h, w = screen.shape[:2]
    scale = min(1.0, 1920 / w, 1080 / h)
    if scale < 1.0:
        display = cv2.resize(screen, (int(w * scale), int(h * scale)))
    else:
        display = screen.copy()
        scale = 1.0

    # Let user select ROI
    window_name = "Select Chart Region — Press ENTER to confirm, ESC to cancel"
    roi = cv2.selectROI(window_name, display, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi[2] == 0 or roi[3] == 0:
        logger.warning("No region selected — using default")
        return DEFAULT_REGION

    # Scale back to original coordinates
    region = {
        "top": int(roi[1] / scale),
        "left": int(roi[0] / scale),
        "width": int(roi[2] / scale),
        "height": int(roi[3] / scale),
    }

    # Save to config.yaml
    try:
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            config = {}

        if "capture" not in config:
            config["capture"] = {}
        config["capture"]["region"] = region

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info("Region saved to %s: %s", config_path, region)
    except Exception as e:
        logger.error("Failed to save region to config: %s", e)

    return region
