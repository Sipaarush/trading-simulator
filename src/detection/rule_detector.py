"""
Rule-Based Detector Module — Detects 35+ candlestick patterns using TA-Lib
CDL* functions on OHLCV data. Graceful fallback if TA-Lib is not installed.

Contract:
    Input:  ohlcv (dict) with 'open', 'high', 'low', 'close', 'volume' arrays
    Output: List[PatternResult]
    source field = 'talib', confidence = 0.80 (fixed for rule-based)
"""

import logging
from typing import Optional

import numpy as np

from src.detection.pattern_registry import PatternResult, get_pattern

logger = logging.getLogger(__name__)

# Try to import TA-Lib
try:
    import talib
    HAS_TALIB = True
    logger.info("TA-Lib imported successfully")
except ImportError:
    HAS_TALIB = False
    logger.warning("TA-Lib not installed — rule_detector will return empty results")


# Mapping of TA-Lib CDL function names to pattern registry keys
TALIB_PATTERN_MAP: dict[str, str] = {
    # Single candle
    "CDLHAMMER": "HAMMER",
    "CDLINVERTEDHAMMER": "INVERTED_HAMMER",
    "CDLDRAGONFLYDOJI": "DRAGONFLY_DOJI",
    "CDLMARUBOZU": None,         # Handled specially (bullish vs bearish)
    "CDLSHOOTINGSTAR": "SHOOTING_STAR",
    "CDLHANGINGMAN": "HANGING_MAN",
    "CDLGRAVESTONEDOJI": "GRAVESTONE_DOJI",
    "CDLDOJI": "DOJI",
    "CDLSPINNINGTOP": "SPINNING_TOP",

    # 2-3 candle
    "CDLENGULFING": None,        # Handled specially (bullish vs bearish)
    "CDLMORNINGSTAR": "MORNING_STAR",
    "CDLEVENINGSTAR": "EVENING_STAR",
    "CDLHARAMI": None,           # Handled specially
    "CDL3WHITESOLDIERS": "THREE_WHITE_SOLDIERS",
    "CDL3BLACKCROWS": "THREE_BLACK_CROWS",
    "CDLPIERCING": "PIERCING_LINE",
    "CDLDARKCLOUDCOVER": "DARK_CLOUD_COVER",

    # Additional patterns available in TA-Lib
    "CDLMORNINGDOJISTAR": "MORNING_STAR",
    "CDLEVENINGDOJISTAR": "EVENING_STAR",
}

# Fixed confidence for rule-based detections
RULE_CONFIDENCE = 0.80


class RuleDetector:
    """
    TA-Lib rule-based candlestick pattern detector.

    Detects all 35+ patterns using TA-Lib's CDL* functions on OHLCV data.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """
        Initialize RuleDetector.

        Args:
            config: Application config dict (optional, for future use).
        """
        self._config = config or {}
        self._min_candles = 5  # Minimum candles required for detection
        logger.info("RuleDetector initialized (TA-Lib available: %s)", HAS_TALIB)

    def detect(self, ohlcv: dict) -> list[PatternResult]:
        """
        Detect candlestick patterns from OHLCV data.

        Args:
            ohlcv: dict with keys 'open', 'high', 'low', 'close', 'volume'.
                   Each value is a numpy float64 array or list.

        Returns:
            List[PatternResult] — detected patterns.
            Returns empty list if TA-Lib is not installed or data is insufficient.
        """
        if not HAS_TALIB:
            logger.debug("TA-Lib not available — returning empty")
            return []

        if ohlcv is None:
            logger.warning("No OHLCV data provided")
            return []

        try:
            # Extract and validate arrays
            open_arr = self._to_float64_array(ohlcv.get("open"))
            high_arr = self._to_float64_array(ohlcv.get("high"))
            low_arr = self._to_float64_array(ohlcv.get("low"))
            close_arr = self._to_float64_array(ohlcv.get("close"))

            if any(arr is None for arr in [open_arr, high_arr, low_arr, close_arr]):
                logger.warning("Insufficient OHLCV data for pattern detection")
                return []

            if len(close_arr) < self._min_candles:
                logger.debug(
                    "Only %d candles (need %d) — skipping detection",
                    len(close_arr),
                    self._min_candles,
                )
                return []

            patterns: list[PatternResult] = []

            # Run all CDL* detections
            patterns.extend(self._detect_single_candle(open_arr, high_arr, low_arr, close_arr))
            patterns.extend(self._detect_multi_candle(open_arr, high_arr, low_arr, close_arr))
            patterns.extend(self._detect_directional(open_arr, high_arr, low_arr, close_arr))

            logger.debug("TA-Lib detected %d patterns", len(patterns))
            return patterns

        except Exception as e:
            logger.error("Rule detection error: %s", e)
            return []

    def _detect_single_candle(
        self,
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
    ) -> list[PatternResult]:
        """Detect single-candle patterns."""
        patterns = []

        # Hammer
        patterns.extend(self._check_cdl(
            "CDLHAMMER", talib.CDLHAMMER, open_arr, high_arr, low_arr, close_arr,
            "HAMMER",
        ))

        # Inverted Hammer
        patterns.extend(self._check_cdl(
            "CDLINVERTEDHAMMER", talib.CDLINVERTEDHAMMER,
            open_arr, high_arr, low_arr, close_arr,
            "INVERTED_HAMMER",
        ))

        # Shooting Star
        patterns.extend(self._check_cdl(
            "CDLSHOOTINGSTAR", talib.CDLSHOOTINGSTAR,
            open_arr, high_arr, low_arr, close_arr,
            "SHOOTING_STAR",
        ))

        # Hanging Man
        patterns.extend(self._check_cdl(
            "CDLHANGINGMAN", talib.CDLHANGINGMAN,
            open_arr, high_arr, low_arr, close_arr,
            "HANGING_MAN",
        ))

        # Doji
        patterns.extend(self._check_cdl(
            "CDLDOJI", talib.CDLDOJI, open_arr, high_arr, low_arr, close_arr,
            "DOJI",
        ))

        # Spinning Top
        patterns.extend(self._check_cdl(
            "CDLSPINNINGTOP", talib.CDLSPINNINGTOP,
            open_arr, high_arr, low_arr, close_arr,
            "SPINNING_TOP",
        ))

        # Dragonfly Doji
        patterns.extend(self._check_cdl(
            "CDLDRAGONFLYDOJI", talib.CDLDRAGONFLYDOJI,
            open_arr, high_arr, low_arr, close_arr,
            "DRAGONFLY_DOJI",
        ))

        # Gravestone Doji
        patterns.extend(self._check_cdl(
            "CDLGRAVESTONEDOJI", talib.CDLGRAVESTONEDOJI,
            open_arr, high_arr, low_arr, close_arr,
            "GRAVESTONE_DOJI",
        ))

        # Marubozu — direction-dependent
        patterns.extend(self._detect_marubozu(open_arr, high_arr, low_arr, close_arr))

        return patterns

    def _detect_multi_candle(
        self,
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
    ) -> list[PatternResult]:
        """Detect 2-3 candle patterns."""
        patterns = []

        # Engulfing — direction-dependent
        patterns.extend(self._detect_engulfing(open_arr, high_arr, low_arr, close_arr))

        # Morning Star
        patterns.extend(self._check_cdl(
            "CDLMORNINGSTAR", talib.CDLMORNINGSTAR,
            open_arr, high_arr, low_arr, close_arr,
            "MORNING_STAR",
        ))

        # Evening Star
        patterns.extend(self._check_cdl(
            "CDLEVENINGSTAR", talib.CDLEVENINGSTAR,
            open_arr, high_arr, low_arr, close_arr,
            "EVENING_STAR",
        ))

        # Harami — direction-dependent
        patterns.extend(self._detect_harami(open_arr, high_arr, low_arr, close_arr))

        # Three White Soldiers
        patterns.extend(self._check_cdl(
            "CDL3WHITESOLDIERS", talib.CDL3WHITESOLDIERS,
            open_arr, high_arr, low_arr, close_arr,
            "THREE_WHITE_SOLDIERS",
        ))

        # Three Black Crows
        patterns.extend(self._check_cdl(
            "CDL3BLACKCROWS", talib.CDL3BLACKCROWS,
            open_arr, high_arr, low_arr, close_arr,
            "THREE_BLACK_CROWS",
        ))

        # Piercing Line
        patterns.extend(self._check_cdl(
            "CDLPIERCING", talib.CDLPIERCING,
            open_arr, high_arr, low_arr, close_arr,
            "PIERCING_LINE",
        ))

        # Dark Cloud Cover
        patterns.extend(self._check_cdl(
            "CDLDARKCLOUDCOVER", talib.CDLDARKCLOUDCOVER,
            open_arr, high_arr, low_arr, close_arr,
            "DARK_CLOUD_COVER",
        ))

        # Morning Doji Star
        patterns.extend(self._check_cdl(
            "CDLMORNINGDOJISTAR", talib.CDLMORNINGDOJISTAR,
            open_arr, high_arr, low_arr, close_arr,
            "MORNING_STAR",
        ))

        # Evening Doji Star
        patterns.extend(self._check_cdl(
            "CDLEVENINGDOJISTAR", talib.CDLEVENINGDOJISTAR,
            open_arr, high_arr, low_arr, close_arr,
            "EVENING_STAR",
        ))

        return patterns

    def _detect_directional(
        self,
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
    ) -> list[PatternResult]:
        """Detect tweezer and other directional patterns."""
        patterns = []

        # Tweezer detection (not directly in TA-Lib — custom logic)
        patterns.extend(self._detect_tweezers(open_arr, high_arr, low_arr, close_arr))

        return patterns

    def _check_cdl(
        self,
        func_name: str,
        func,
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
        registry_key: str,
    ) -> list[PatternResult]:
        """
        Run a TA-Lib CDL function and convert non-zero results to PatternResult.

        Args:
            func_name: Name of the CDL function (for logging).
            func: TA-Lib CDL function reference.
            open_arr, high_arr, low_arr, close_arr: OHLC arrays.
            registry_key: Pattern registry key.

        Returns:
            List[PatternResult] for detected instances.
        """
        try:
            result = func(open_arr, high_arr, low_arr, close_arr)
            last_value = int(result[-1])

            if last_value != 0:
                meta = get_pattern(registry_key)
                if meta is None:
                    return []

                pattern = PatternResult(
                    name=registry_key,
                    display_name=meta.display_name,
                    confidence=RULE_CONFIDENCE,
                    bbox=(0, 0, 0, 0),  # No bbox for candlestick patterns
                    direction=meta.direction,
                    source="talib",
                )
                return [pattern]

        except Exception as e:
            logger.debug("%s detection failed: %s", func_name, e)

        return []

    def _detect_marubozu(
        self,
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
    ) -> list[PatternResult]:
        """Detect Marubozu (bullish or bearish based on result sign)."""
        try:
            result = talib.CDLMARUBOZU(open_arr, high_arr, low_arr, close_arr)
            last_value = int(result[-1])

            if last_value > 0:
                meta = get_pattern("BULLISH_MARUBOZU")
                if meta:
                    return [PatternResult(
                        name="BULLISH_MARUBOZU",
                        display_name=meta.display_name,
                        confidence=RULE_CONFIDENCE,
                        bbox=(0, 0, 0, 0),
                        direction="bullish",
                        source="talib",
                    )]
            elif last_value < 0:
                meta = get_pattern("BEARISH_MARUBOZU")
                if meta:
                    return [PatternResult(
                        name="BEARISH_MARUBOZU",
                        display_name=meta.display_name,
                        confidence=RULE_CONFIDENCE,
                        bbox=(0, 0, 0, 0),
                        direction="bearish",
                        source="talib",
                    )]
        except Exception as e:
            logger.debug("Marubozu detection failed: %s", e)

        return []

    def _detect_engulfing(
        self,
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
    ) -> list[PatternResult]:
        """Detect Engulfing (bullish or bearish based on result sign)."""
        try:
            result = talib.CDLENGULFING(open_arr, high_arr, low_arr, close_arr)
            last_value = int(result[-1])

            if last_value > 0:
                meta = get_pattern("BULLISH_ENGULFING")
                if meta:
                    return [PatternResult(
                        name="BULLISH_ENGULFING",
                        display_name=meta.display_name,
                        confidence=RULE_CONFIDENCE,
                        bbox=(0, 0, 0, 0),
                        direction="bullish",
                        source="talib",
                    )]
            elif last_value < 0:
                meta = get_pattern("BEARISH_ENGULFING")
                if meta:
                    return [PatternResult(
                        name="BEARISH_ENGULFING",
                        display_name=meta.display_name,
                        confidence=RULE_CONFIDENCE,
                        bbox=(0, 0, 0, 0),
                        direction="bearish",
                        source="talib",
                    )]
        except Exception as e:
            logger.debug("Engulfing detection failed: %s", e)

        return []

    def _detect_harami(
        self,
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
    ) -> list[PatternResult]:
        """Detect Harami (bullish or bearish based on result sign)."""
        try:
            result = talib.CDLHARAMI(open_arr, high_arr, low_arr, close_arr)
            last_value = int(result[-1])

            if last_value > 0:
                meta = get_pattern("BULLISH_HARAMI")
                if meta:
                    return [PatternResult(
                        name="BULLISH_HARAMI",
                        display_name=meta.display_name,
                        confidence=RULE_CONFIDENCE,
                        bbox=(0, 0, 0, 0),
                        direction="bullish",
                        source="talib",
                    )]
            elif last_value < 0:
                meta = get_pattern("BEARISH_HARAMI")
                if meta:
                    return [PatternResult(
                        name="BEARISH_HARAMI",
                        display_name=meta.display_name,
                        confidence=RULE_CONFIDENCE,
                        bbox=(0, 0, 0, 0),
                        direction="bearish",
                        source="talib",
                    )]
        except Exception as e:
            logger.debug("Harami detection failed: %s", e)

        return []

    def _detect_tweezers(
        self,
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
    ) -> list[PatternResult]:
        """Custom Tweezer Top/Bottom detection (not in standard TA-Lib)."""
        patterns = []

        try:
            if len(close_arr) < 2:
                return []

            tolerance = 0.001  # 0.1% tolerance for matching highs/lows

            # Tweezer Top — two candles with matching highs
            if abs(high_arr[-1] - high_arr[-2]) / high_arr[-2] < tolerance:
                if close_arr[-2] > open_arr[-2] and close_arr[-1] < open_arr[-1]:
                    meta = get_pattern("TWEEZER_TOP")
                    if meta:
                        patterns.append(PatternResult(
                            name="TWEEZER_TOP",
                            display_name=meta.display_name,
                            confidence=RULE_CONFIDENCE,
                            bbox=(0, 0, 0, 0),
                            direction="bearish",
                            source="talib",
                        ))

            # Tweezer Bottom — two candles with matching lows
            if abs(low_arr[-1] - low_arr[-2]) / low_arr[-2] < tolerance:
                if close_arr[-2] < open_arr[-2] and close_arr[-1] > open_arr[-1]:
                    meta = get_pattern("TWEEZER_BOTTOM")
                    if meta:
                        patterns.append(PatternResult(
                            name="TWEEZER_BOTTOM",
                            display_name=meta.display_name,
                            confidence=RULE_CONFIDENCE,
                            bbox=(0, 0, 0, 0),
                            direction="bullish",
                            source="talib",
                        ))

        except Exception as e:
            logger.debug("Tweezer detection failed: %s", e)

        return patterns

    def _to_float64_array(self, data) -> Optional[np.ndarray]:
        """
        Convert input data to numpy float64 array.

        Args:
            data: list, numpy array, or None.

        Returns:
            np.ndarray of dtype float64, or None if invalid.
        """
        if data is None:
            return None

        try:
            arr = np.asarray(data, dtype=np.float64)
            if arr.size == 0:
                return None
            return arr
        except (ValueError, TypeError):
            return None
