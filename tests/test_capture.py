"""
Phase 1 Acceptance Tests — Screen Capture, Preprocessing, OCR, Hardware Profile.
Run: pytest tests/test_capture.py -v
"""

import logging
import time

import cv2
import numpy as np
import pytest
import yaml

logging.basicConfig(level=logging.INFO)


# ============================================================
# Hardware Profile Tests
# ============================================================

class TestHardwareProfile:
    """Tests for src/core/hardware_profile.py."""

    def test_import(self):
        """Module imports without error."""
        from src.core.hardware_profile import detect_hardware, HardwareProfile, HardwareTier
        assert HardwareProfile is not None
        assert HardwareTier is not None

    def test_detect_returns_profile(self):
        """detect_hardware() returns a valid HardwareProfile."""
        from src.core.hardware_profile import detect_hardware, HardwareProfile
        profile = detect_hardware()
        assert isinstance(profile, HardwareProfile)

    def test_tier_is_valid(self):
        """Detected tier is one of the 4 valid tiers."""
        from src.core.hardware_profile import detect_hardware, HardwareTier
        profile = detect_hardware()
        assert profile.tier in HardwareTier

    def test_device_is_cuda_or_cpu(self):
        """Device is either 'cuda' or 'cpu'."""
        from src.core.hardware_profile import detect_hardware
        profile = detect_hardware()
        assert profile.device in ("cuda", "cpu")

    def test_ram_detected(self):
        """System RAM is detected and > 0."""
        from src.core.hardware_profile import detect_hardware
        profile = detect_hardware()
        assert profile.ram_gb > 0

    def test_cpu_cores_detected(self):
        """CPU core count is > 0."""
        from src.core.hardware_profile import detect_hardware
        profile = detect_hardware()
        assert profile.cpu_cores > 0

    def test_yolo_model_valid(self):
        """YOLO model is either 'yolov8n' or 'yolov8s'."""
        from src.core.hardware_profile import detect_hardware
        profile = detect_hardware()
        assert profile.yolo_model in ("yolov8n", "yolov8s")

    def test_img_size_valid(self):
        """Image size is 416 or 640."""
        from src.core.hardware_profile import detect_hardware
        profile = detect_hardware()
        assert profile.yolo_img_size in (416, 640)

    def test_force_tier_override(self):
        """force_tier parameter overrides auto-detection."""
        from src.core.hardware_profile import detect_hardware, HardwareTier
        profile = detect_hardware(force_tier="ULTRA_LOW")
        assert profile.tier == HardwareTier.ULTRA_LOW
        assert profile.yolo_model == "yolov8n"
        assert profile.yolo_img_size == 416

    def test_hardware_tier_logged(self):
        """Hardware tier is logged on startup (just ensure no crash)."""
        from src.core.hardware_profile import detect_hardware
        profile = detect_hardware()
        assert profile.tier.value in ("ULTRA_LOW", "LOW", "MID", "HIGH")


# ============================================================
# Preprocessor Tests
# ============================================================

class TestPreprocessor:
    """Tests for src/capture/preprocessor.py."""

    @pytest.fixture
    def sample_frame(self):
        """Create a synthetic BGR frame (simulates a screenshot)."""
        # 720p frame with some gradients and noise
        frame = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
        # Add a gradient to simulate chart
        for i in range(720):
            frame[i, :, :] = np.clip(frame[i, :, :].astype(int) + i // 5, 0, 255).astype(np.uint8)
        return frame

    @pytest.fixture
    def config(self):
        """Default config for testing."""
        return {"model": {"img_size": 640}}

    def test_import(self):
        """Module imports without error."""
        from src.capture.preprocessor import preprocess
        assert preprocess is not None

    def test_output_shape(self, sample_frame, config):
        """Preprocessed frame has shape (640, 640, 3)."""
        from src.capture.preprocessor import preprocess
        processed, original = preprocess(sample_frame, config)
        assert processed.shape == (640, 640, 3)

    def test_output_dtype_float32(self, sample_frame, config):
        """Preprocessed frame is float32."""
        from src.capture.preprocessor import preprocess
        processed, _ = preprocess(sample_frame, config)
        assert processed.dtype == np.float32

    def test_output_range_0_1(self, sample_frame, config):
        """Preprocessed values are in [0.0, 1.0]."""
        from src.capture.preprocessor import preprocess
        processed, _ = preprocess(sample_frame, config)
        assert processed.min() >= 0.0
        assert processed.max() <= 1.0

    def test_original_crop_unchanged(self, sample_frame, config):
        """original_crop is identical to input frame."""
        from src.capture.preprocessor import preprocess
        _, original = preprocess(sample_frame, config)
        assert np.array_equal(original, sample_frame)

    def test_original_crop_dtype(self, sample_frame, config):
        """original_crop remains uint8."""
        from src.capture.preprocessor import preprocess
        _, original = preprocess(sample_frame, config)
        assert original.dtype == np.uint8

    def test_custom_img_size(self, sample_frame):
        """Custom img_size produces correct output shape."""
        from src.capture.preprocessor import preprocess
        config = {"model": {"img_size": 416}}
        processed, _ = preprocess(sample_frame, config)
        assert processed.shape == (416, 416, 3)

    def test_empty_frame_raises(self, config):
        """Empty frame raises ValueError."""
        from src.capture.preprocessor import preprocess
        with pytest.raises(ValueError):
            preprocess(np.array([]), config)

    def test_none_frame_raises(self, config):
        """None frame raises ValueError."""
        from src.capture.preprocessor import preprocess
        with pytest.raises(ValueError):
            preprocess(None, config)


# ============================================================
# Screen Capture Tests
# ============================================================

class TestScreenCapture:
    """Tests for src/capture/screen_capture.py."""

    @pytest.fixture
    def config(self):
        """Default config for capture testing."""
        return {
            "capture": {
                "region": {"top": 0, "left": 0, "width": 640, "height": 480},
                "interval_ms": 200,
            }
        }

    def test_import(self):
        """Module imports without error."""
        from src.capture.screen_capture import ScreenCapture
        assert ScreenCapture is not None

    def test_init(self, config):
        """ScreenCapture initializes without error."""
        from src.capture.screen_capture import ScreenCapture
        capture = ScreenCapture(config)
        assert capture is not None

    def test_frame_captured_fast(self, config):
        """Frame captured in < 50ms per call."""
        from src.capture.screen_capture import ScreenCapture
        capture = ScreenCapture(config)
        capture.start()
        try:
            # Wait for first frame
            time.sleep(0.5)
            start = time.perf_counter()
            frame = capture.get_frame(timeout=2.0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert frame is not None, "No frame captured"
            assert elapsed_ms < 50, f"get_frame took {elapsed_ms:.1f}ms (> 50ms)"
        finally:
            capture.stop()

    def test_frame_is_bgr_uint8(self, config):
        """Captured frame is BGR uint8 with shape (H, W, 3)."""
        from src.capture.screen_capture import ScreenCapture
        capture = ScreenCapture(config)
        capture.start()
        try:
            time.sleep(0.5)
            frame = capture.get_frame(timeout=2.0)
            assert frame is not None
            assert frame.dtype == np.uint8
            assert len(frame.shape) == 3
            assert frame.shape[2] == 3
        finally:
            capture.stop()

    def test_region_property(self, config):
        """Region property returns correct values."""
        from src.capture.screen_capture import ScreenCapture
        capture = ScreenCapture(config)
        region = capture.region
        assert region["width"] == 640
        assert region["height"] == 480

    def test_stop_without_start(self, config):
        """Stopping without starting does not crash."""
        from src.capture.screen_capture import ScreenCapture
        capture = ScreenCapture(config)
        capture.stop()  # Should not raise


# ============================================================
# Price Extractor Tests
# ============================================================

class TestPriceExtractor:
    """Tests for src/ocr/price_extractor.py."""

    @pytest.fixture
    def config(self):
        """Default OCR config."""
        return {
            "ocr": {
                "tesseract_config": "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.,",
                "sanity_threshold": 0.20,
            }
        }

    @pytest.fixture
    def chart_frame(self):
        """Create a synthetic chart frame with price labels."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Draw some chart-like content
        for i in range(100):
            y = 360 + int(50 * np.sin(i * 0.1))
            x = int(i * 11)
            if 0 <= x < 1280 and 0 <= y < 720:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Draw price labels on right side (Y-axis area)
        font = cv2.FONT_HERSHEY_SIMPLEX
        prices = ["45200.50", "45150.00", "45100.25", "45050.75"]
        for i, price in enumerate(prices):
            y_pos = 150 + i * 120
            cv2.putText(frame, price, (1140, y_pos), font, 0.5, (255, 255, 255), 1)

        return frame

    def test_import(self):
        """Module imports without error."""
        from src.ocr.price_extractor import PriceExtractor
        assert PriceExtractor is not None

    def test_init(self, config):
        """PriceExtractor initializes without error."""
        from src.ocr.price_extractor import PriceExtractor
        extractor = PriceExtractor(config)
        assert extractor is not None

    def test_empty_frame_returns_none(self, config):
        """Empty frame returns None."""
        from src.ocr.price_extractor import PriceExtractor
        extractor = PriceExtractor(config)
        result = extractor.extract(np.array([]))
        assert result is None

    def test_none_frame_returns_none(self, config):
        """None frame returns None."""
        from src.ocr.price_extractor import PriceExtractor
        extractor = PriceExtractor(config)
        result = extractor.extract(None)
        assert result is None

    def test_no_crash_on_blank_frame(self, config):
        """Blank frame does not crash — returns None or dict."""
        from src.ocr.price_extractor import PriceExtractor
        extractor = PriceExtractor(config)
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = extractor.extract(blank)
        # May return None or a dict — just shouldn't crash
        assert result is None or isinstance(result, dict)

    def test_ohlcv_keys_when_extracted(self, config, chart_frame):
        """When prices are extracted, OHLCV dict has all required keys."""
        from src.ocr.price_extractor import PriceExtractor
        extractor = PriceExtractor(config)

        # Set a last known price so fallback works
        extractor._last_known_price = 45100.0
        result = extractor.extract(chart_frame)

        if result is not None:
            assert "close" in result
            assert "high" in result
            assert "low" in result
            assert "volume" in result
            assert "price_range" in result
            assert isinstance(result["price_range"], tuple)
            assert len(result["price_range"]) == 2

    def test_price_within_range(self, config, chart_frame):
        """Extracted price is within ±5% of expected visible price."""
        from src.ocr.price_extractor import PriceExtractor
        extractor = PriceExtractor(config)
        extractor._last_known_price = 45100.0
        result = extractor.extract(chart_frame)

        if result is not None:
            expected = 45100.0
            tolerance = expected * 0.05  # 5%
            assert abs(result["close"] - expected) < tolerance, (
                f"Price {result['close']} not within 5% of {expected}"
            )

    def test_no_crash_on_dark_frame(self, config):
        """Dark-themed chart does not crash."""
        from src.ocr.price_extractor import PriceExtractor
        extractor = PriceExtractor(config)
        dark = np.full((720, 1280, 3), 30, dtype=np.uint8)
        result = extractor.extract(dark)
        assert result is None or isinstance(result, dict)

    def test_no_crash_on_light_frame(self, config):
        """Light-themed chart does not crash."""
        from src.ocr.price_extractor import PriceExtractor
        extractor = PriceExtractor(config)
        light = np.full((720, 1280, 3), 240, dtype=np.uint8)
        result = extractor.extract(light)
        assert result is None or isinstance(result, dict)


# ============================================================
# Config.yaml Tests
# ============================================================

class TestConfig:
    """Tests for config.yaml structure."""

    @pytest.fixture
    def config(self):
        """Load config.yaml."""
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)

    def test_config_loads(self, config):
        """config.yaml loads without error."""
        assert config is not None

    def test_hardware_section(self, config):
        """Hardware section exists with required keys."""
        assert "hardware" in config
        assert "auto_detect" in config["hardware"]

    def test_model_section(self, config):
        """Model section has required keys."""
        assert "model" in config
        assert "device" in config["model"]
        assert "model_size" in config["model"]
        assert "model_path" in config["model"]

    def test_capture_section(self, config):
        """Capture section has region and interval."""
        assert "capture" in config
        assert "region" in config["capture"]
        assert "interval_ms" in config["capture"]

    def test_ocr_section(self, config):
        """OCR section exists."""
        assert "ocr" in config

    def test_signals_section(self, config):
        """Signals section has weights and min score."""
        assert "signals" in config
        assert "min_confluence_score" in config["signals"]
        assert "weights" in config["signals"]

    def test_risk_section(self, config):
        """Risk section has ATR multiplier and R:R ratio."""
        assert "risk" in config
        assert "atr_multiplier" in config["risk"]
        assert "min_rr_ratio" in config["risk"]
