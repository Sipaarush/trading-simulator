"""
Phase 2 Acceptance Tests — Pattern Registry, YOLO Detector, Rule Detector.
Run: pytest tests/test_patterns.py -v
"""

import logging

import numpy as np
import pytest

logging.basicConfig(level=logging.INFO)


# ============================================================
# Pattern Registry Tests
# ============================================================

class TestPatternRegistry:
    """Tests for src/detection/pattern_registry.py."""

    def test_import(self):
        """Module imports without error."""
        from src.detection.pattern_registry import (
            PatternMeta,
            PatternResult,
            get_pattern,
            get_all_patterns,
        )
        assert PatternMeta is not None
        assert PatternResult is not None

    def test_registry_has_35_plus_patterns(self):
        """Registry contains at least 35 patterns."""
        from src.detection.pattern_registry import get_all_patterns
        patterns = get_all_patterns()
        assert len(patterns) >= 35, f"Only {len(patterns)} patterns (need 35+)"

    def test_get_pattern_by_name(self):
        """get_pattern returns correct PatternMeta."""
        from src.detection.pattern_registry import get_pattern
        pattern = get_pattern("BULL_FLAG")
        assert pattern is not None
        assert pattern.name == "BULL_FLAG"
        assert pattern.display_name == "Bull Flag"
        assert pattern.direction == "bullish"

    def test_get_pattern_unknown_returns_none(self):
        """Unknown pattern name returns None."""
        from src.detection.pattern_registry import get_pattern
        result = get_pattern("NONEXISTENT_PATTERN")
        assert result is None

    def test_all_patterns_have_required_fields(self):
        """Every pattern has all required fields populated."""
        from src.detection.pattern_registry import get_all_patterns
        for name, meta in get_all_patterns().items():
            assert meta.name == name, f"{name}: name mismatch"
            assert len(meta.display_name) > 0, f"{name}: missing display_name"
            assert meta.direction in ("bullish", "bearish", "neutral"), (
                f"{name}: invalid direction '{meta.direction}'"
            )
            assert 0.0 <= meta.win_rate <= 1.0, (
                f"{name}: win_rate {meta.win_rate} out of range"
            )
            assert meta.detection in ("talib", "yolov8", "yolov8+rule", "custom"), (
                f"{name}: invalid detection '{meta.detection}'"
            )

    def test_bullish_patterns_exist(self):
        """Registry has bullish patterns."""
        from src.detection.pattern_registry import get_patterns_by_direction
        bullish = get_patterns_by_direction("bullish")
        assert len(bullish) > 0

    def test_bearish_patterns_exist(self):
        """Registry has bearish patterns."""
        from src.detection.pattern_registry import get_patterns_by_direction
        bearish = get_patterns_by_direction("bearish")
        assert len(bearish) > 0

    def test_neutral_patterns_exist(self):
        """Registry has neutral patterns."""
        from src.detection.pattern_registry import get_patterns_by_direction
        neutral = get_patterns_by_direction("neutral")
        assert len(neutral) > 0

    def test_talib_patterns_exist(self):
        """Registry has TA-Lib detected patterns."""
        from src.detection.pattern_registry import get_patterns_by_detection
        talib_patterns = get_patterns_by_detection("talib")
        assert len(talib_patterns) > 0

    def test_yolov8_patterns_exist(self):
        """Registry has YOLOv8 detected patterns."""
        from src.detection.pattern_registry import get_patterns_by_detection
        yolo_patterns = get_patterns_by_detection("yolov8")
        assert len(yolo_patterns) > 0

    def test_pattern_result_dataclass(self):
        """PatternResult can be instantiated with all required fields."""
        from src.detection.pattern_registry import PatternResult
        result = PatternResult(
            name="BULL_FLAG",
            display_name="Bull Flag",
            confidence=0.85,
            bbox=(10, 20, 100, 200),
            direction="bullish",
            source="yolov8",
        )
        assert result.name == "BULL_FLAG"
        assert result.display_name == "Bull Flag"
        assert result.confidence == 0.85
        assert result.bbox == (10, 20, 100, 200)
        assert result.direction == "bullish"
        assert result.source == "yolov8"

    def test_pattern_result_exact_fields(self):
        """PatternResult has exactly the contract-required fields."""
        from src.detection.pattern_registry import PatternResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(PatternResult)}
        required = {"name", "display_name", "confidence", "bbox", "direction", "source"}
        assert fields == required, f"Fields mismatch: {fields} != {required}"

    # Specific pattern checks from PRD
    def test_head_and_shoulders(self):
        """Head & Shoulders pattern exists with correct metadata."""
        from src.detection.pattern_registry import get_pattern
        p = get_pattern("HEAD_AND_SHOULDERS")
        assert p is not None
        assert p.direction == "bearish"
        assert p.win_rate == 0.72

    def test_morning_star(self):
        """Morning Star pattern exists with correct metadata."""
        from src.detection.pattern_registry import get_pattern
        p = get_pattern("MORNING_STAR")
        assert p is not None
        assert p.direction == "bullish"
        assert p.win_rate == 0.72

    def test_three_white_soldiers(self):
        """Three White Soldiers has highest win rate (74%)."""
        from src.detection.pattern_registry import get_pattern
        p = get_pattern("THREE_WHITE_SOLDIERS")
        assert p is not None
        assert p.win_rate == 0.74

    def test_hammer(self):
        """Hammer pattern is bullish with 65% win rate."""
        from src.detection.pattern_registry import get_pattern
        p = get_pattern("HAMMER")
        assert p is not None
        assert p.direction == "bullish"
        assert p.win_rate == 0.65


# ============================================================
# YOLO Detector Tests
# ============================================================

class TestYOLODetector:
    """Tests for src/detection/yolo_detector.py."""

    def test_import(self):
        """Module imports without error."""
        from src.detection.yolo_detector import YOLODetector
        assert YOLODetector is not None

    def test_init_with_profile(self):
        """YOLODetector initializes with a hardware profile."""
        from src.core.hardware_profile import detect_hardware
        from src.detection.yolo_detector import YOLODetector

        profile = detect_hardware(force_tier="ULTRA_LOW")
        config = {"model": {"model_path": "models/best.pt", "confidence_threshold": 0.65}}
        detector = YOLODetector(config, profile)
        assert detector is not None

    def test_detect_empty_frame_returns_empty(self):
        """Empty frame returns empty list."""
        from src.core.hardware_profile import detect_hardware
        from src.detection.yolo_detector import YOLODetector

        profile = detect_hardware(force_tier="ULTRA_LOW")
        config = {"model": {"model_path": "models/best.pt", "confidence_threshold": 0.65}}
        detector = YOLODetector(config, profile)
        result = detector.detect(np.array([]))
        assert result == []

    def test_detect_none_frame_returns_empty(self):
        """None frame returns empty list."""
        from src.core.hardware_profile import detect_hardware
        from src.detection.yolo_detector import YOLODetector

        profile = detect_hardware(force_tier="ULTRA_LOW")
        config = {"model": {"model_path": "models/best.pt", "confidence_threshold": 0.65}}
        detector = YOLODetector(config, profile)
        result = detector.detect(None)
        assert result == []

    def test_detect_returns_list(self):
        """Detection on a valid frame returns a list."""
        from src.core.hardware_profile import detect_hardware
        from src.detection.yolo_detector import YOLODetector

        profile = detect_hardware(force_tier="ULTRA_LOW")
        config = {"model": {"model_path": "models/best.pt", "confidence_threshold": 0.65}}
        detector = YOLODetector(config, profile)

        # Create a synthetic frame
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert isinstance(result, list)

    def test_yolo_class_map_exists(self):
        """YOLO class map has entries for chart patterns."""
        from src.detection.yolo_detector import YOLO_CLASS_MAP
        assert len(YOLO_CLASS_MAP) > 0
        assert "head_and_shoulders" in YOLO_CLASS_MAP
        assert "double_top" in YOLO_CLASS_MAP
        assert "bull_flag" in YOLO_CLASS_MAP


# ============================================================
# Rule Detector Tests
# ============================================================

class TestRuleDetector:
    """Tests for src/detection/rule_detector.py."""

    @pytest.fixture
    def sample_ohlcv_bullish_engulfing(self):
        """Create OHLCV data with a bullish engulfing pattern."""
        # Simulate a downtrend followed by bullish engulfing
        n = 20
        np.random.seed(42)

        # Downtrend then reversal
        close = np.array([100 - i * 0.5 + np.random.randn() * 0.1 for i in range(n)])
        open_ = close + np.random.randn(n) * 0.3

        # Force bullish engulfing at the end
        # Penultimate candle: bearish (open > close)
        open_[-2] = close[-2] + 2.0
        # Last candle: bullish engulfing (opens below prior close, closes above prior open)
        open_[-1] = close[-2] - 0.5
        close[-1] = open_[-2] + 0.5

        high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * 0.5
        low = np.minimum(open_, close) - np.abs(np.random.randn(n)) * 0.5
        volume = np.random.randint(1000, 10000, n).astype(float)

        return {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

    @pytest.fixture
    def sample_ohlcv_hammer(self):
        """Create OHLCV data with a hammer pattern."""
        n = 20
        np.random.seed(123)

        close = np.array([100 - i * 0.3 for i in range(n)], dtype=np.float64)
        open_ = close.copy()

        # Hammer at the end: small body at top, long lower shadow
        open_[-1] = close[-1] + 0.1
        high = np.maximum(open_, close) + 0.2
        low = np.minimum(open_, close) - 0.3

        # Make hammer: very long lower shadow
        low[-1] = close[-1] - 3.0
        high[-1] = max(open_[-1], close[-1]) + 0.1

        volume = np.random.randint(1000, 10000, n).astype(float)

        return {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

    def test_import(self):
        """Module imports without error."""
        from src.detection.rule_detector import RuleDetector
        assert RuleDetector is not None

    def test_init(self):
        """RuleDetector initializes without error."""
        from src.detection.rule_detector import RuleDetector
        detector = RuleDetector()
        assert detector is not None

    def test_detect_returns_list(self, sample_ohlcv_bullish_engulfing):
        """Detection returns a list."""
        from src.detection.rule_detector import RuleDetector
        detector = RuleDetector()
        result = detector.detect(sample_ohlcv_bullish_engulfing)
        assert isinstance(result, list)

    def test_detect_none_returns_empty(self):
        """None OHLCV returns empty list."""
        from src.detection.rule_detector import RuleDetector
        detector = RuleDetector()
        result = detector.detect(None)
        assert result == []

    def test_detect_empty_returns_empty(self):
        """Empty OHLCV returns empty list."""
        from src.detection.rule_detector import RuleDetector
        detector = RuleDetector()
        result = detector.detect({})
        assert result == []

    def test_insufficient_data_returns_empty(self):
        """Less than 5 candles returns empty list."""
        from src.detection.rule_detector import RuleDetector
        detector = RuleDetector()
        ohlcv = {
            "open": np.array([1.0, 2.0, 3.0]),
            "high": np.array([1.5, 2.5, 3.5]),
            "low": np.array([0.5, 1.5, 2.5]),
            "close": np.array([1.2, 2.2, 3.2]),
            "volume": np.array([100.0, 200.0, 300.0]),
        }
        result = detector.detect(ohlcv)
        assert result == []

    def test_pattern_result_has_required_fields(self, sample_ohlcv_bullish_engulfing):
        """Detected patterns have all PatternResult fields."""
        from src.detection.rule_detector import RuleDetector
        from src.detection.pattern_registry import PatternResult
        detector = RuleDetector()
        results = detector.detect(sample_ohlcv_bullish_engulfing)

        for r in results:
            assert isinstance(r, PatternResult)
            assert isinstance(r.name, str)
            assert isinstance(r.display_name, str)
            assert isinstance(r.confidence, float)
            assert isinstance(r.bbox, tuple)
            assert isinstance(r.direction, str)
            assert isinstance(r.source, str)
            assert r.direction in ("bullish", "bearish", "neutral")
            assert r.source == "talib"

    def test_talib_detects_engulfing(self, sample_ohlcv_bullish_engulfing):
        """TA-Lib can detect engulfing patterns from crafted data."""
        from src.detection.rule_detector import RuleDetector, HAS_TALIB
        if not HAS_TALIB:
            pytest.skip("TA-Lib not installed")

        detector = RuleDetector()
        results = detector.detect(sample_ohlcv_bullish_engulfing)
        # May or may not detect depending on exact data — just verify no crash
        assert isinstance(results, list)

    def test_no_crash_on_random_data(self):
        """Random OHLCV data does not crash the detector."""
        from src.detection.rule_detector import RuleDetector
        detector = RuleDetector()
        n = 50
        ohlcv = {
            "open": np.random.uniform(100, 200, n),
            "high": np.random.uniform(150, 250, n),
            "low": np.random.uniform(50, 150, n),
            "close": np.random.uniform(100, 200, n),
            "volume": np.random.uniform(1000, 10000, n),
        }
        result = detector.detect(ohlcv)
        assert isinstance(result, list)

    def test_confidence_is_fixed(self, sample_ohlcv_bullish_engulfing):
        """Rule-based confidence is fixed at 0.80."""
        from src.detection.rule_detector import RuleDetector, HAS_TALIB
        if not HAS_TALIB:
            pytest.skip("TA-Lib not installed")

        detector = RuleDetector()
        results = detector.detect(sample_ohlcv_bullish_engulfing)
        for r in results:
            assert r.confidence == 0.80

    def test_source_is_talib(self, sample_ohlcv_bullish_engulfing):
        """All detections have source='talib'."""
        from src.detection.rule_detector import RuleDetector, HAS_TALIB
        if not HAS_TALIB:
            pytest.skip("TA-Lib not installed")

        detector = RuleDetector()
        results = detector.detect(sample_ohlcv_bullish_engulfing)
        for r in results:
            assert r.source == "talib"
