"""
Pattern Registry — Complete library of 35+ chart patterns with metadata.
Each pattern has a name, display_name, direction, win_rate, detection method,
and description. Used by both yolo_detector.py and rule_detector.py.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PatternMeta:
    """Metadata for a single chart pattern."""
    name: str               # Registry key, e.g. 'BULL_FLAG'
    display_name: str       # Human-readable, e.g. 'Bull Flag'
    direction: str          # 'bullish', 'bearish', or 'neutral'
    win_rate: float         # Historical win rate (0.0–1.0)
    detection: str          # 'talib', 'yolov8', 'yolov8+rule', 'custom'
    description: str        # Brief description


@dataclass
class PatternResult:
    """
    Detection result for a single pattern instance.

    Contract (Layer 2 → Layer 3):
        Both yolo_detector and rule_detector return List[PatternResult].
    """
    name: str               # e.g. 'BULL_FLAG' — key from pattern_registry
    display_name: str       # e.g. 'Bull Flag'
    confidence: float       # 0.0 – 1.0
    bbox: tuple             # (x1, y1, x2, y2) in original frame pixels
    direction: str          # 'bullish' | 'bearish' | 'neutral'
    source: str             # 'yolov8' | 'talib'


# ============================================================
# Complete Pattern Library (35+ patterns)
# ============================================================

_PATTERN_REGISTRY: dict[str, PatternMeta] = {}


def _register(
    name: str,
    display_name: str,
    direction: str,
    win_rate: float,
    detection: str,
    description: str,
) -> None:
    """Register a pattern in the global registry."""
    _PATTERN_REGISTRY[name] = PatternMeta(
        name=name,
        display_name=display_name,
        direction=direction,
        win_rate=win_rate,
        detection=detection,
        description=description,
    )


# ── SINGLE CANDLE ──

_register(
    "HAMMER", "Hammer", "bullish", 0.65, "talib",
    "Small body at top, long lower shadow. Bullish reversal after downtrend.",
)
_register(
    "INVERTED_HAMMER", "Inverted Hammer", "bullish", 0.62, "talib",
    "Small body at bottom, long upper shadow. Potential bullish reversal.",
)
_register(
    "DRAGONFLY_DOJI", "Dragonfly Doji", "bullish", 0.63, "talib",
    "Open and close at high, long lower shadow. Bullish signal at support.",
)
_register(
    "BULLISH_MARUBOZU", "Bullish Marubozu", "bullish", 0.67, "talib",
    "Long bullish body with no shadows. Strong buying pressure.",
)
_register(
    "SHOOTING_STAR", "Shooting Star", "bearish", 0.66, "talib",
    "Small body at bottom, long upper shadow. Bearish reversal after uptrend.",
)
_register(
    "HANGING_MAN", "Hanging Man", "bearish", 0.64, "talib",
    "Small body at top, long lower shadow in uptrend. Bearish reversal signal.",
)
_register(
    "GRAVESTONE_DOJI", "Gravestone Doji", "bearish", 0.62, "talib",
    "Open and close at low, long upper shadow. Bearish signal at resistance.",
)
_register(
    "BEARISH_MARUBOZU", "Bearish Marubozu", "bearish", 0.68, "talib",
    "Long bearish body with no shadows. Strong selling pressure.",
)
_register(
    "DOJI", "Doji", "neutral", 0.50, "talib",
    "Open equals close. Indecision — trend reversal or continuation.",
)
_register(
    "SPINNING_TOP", "Spinning Top", "neutral", 0.50, "talib",
    "Small body with upper and lower shadows. Market indecision.",
)

# ── 2–3 CANDLE ──

_register(
    "BULLISH_ENGULFING", "Bullish Engulfing", "bullish", 0.68, "talib",
    "Bullish candle fully engulfs prior bearish candle. Strong reversal.",
)
_register(
    "BEARISH_ENGULFING", "Bearish Engulfing", "bearish", 0.67, "talib",
    "Bearish candle fully engulfs prior bullish candle. Strong reversal.",
)
_register(
    "MORNING_STAR", "Morning Star", "bullish", 0.72, "talib",
    "3-candle bullish reversal: bearish, small body, bullish.",
)
_register(
    "EVENING_STAR", "Evening Star", "bearish", 0.71, "talib",
    "3-candle bearish reversal: bullish, small body, bearish.",
)
_register(
    "BULLISH_HARAMI", "Bullish Harami", "bullish", 0.61, "talib",
    "Small bullish candle inside prior bearish candle. Potential reversal.",
)
_register(
    "BEARISH_HARAMI", "Bearish Harami", "bearish", 0.60, "talib",
    "Small bearish candle inside prior bullish candle. Potential reversal.",
)
_register(
    "THREE_WHITE_SOLDIERS", "Three White Soldiers", "bullish", 0.74, "talib",
    "Three consecutive bullish candles with higher closes. Strong uptrend.",
)
_register(
    "THREE_BLACK_CROWS", "Three Black Crows", "bearish", 0.73, "talib",
    "Three consecutive bearish candles with lower closes. Strong downtrend.",
)
_register(
    "PIERCING_LINE", "Piercing Line", "bullish", 0.63, "talib",
    "Bullish candle opens below and closes above midpoint of prior bearish.",
)
_register(
    "DARK_CLOUD_COVER", "Dark Cloud Cover", "bearish", 0.63, "talib",
    "Bearish candle opens above and closes below midpoint of prior bullish.",
)
_register(
    "TWEEZER_BOTTOM", "Tweezer Bottom", "bullish", 0.62, "talib",
    "Two candles with matching lows. Bullish reversal at support.",
)
_register(
    "TWEEZER_TOP", "Tweezer Top", "bearish", 0.62, "talib",
    "Two candles with matching highs. Bearish reversal at resistance.",
)

# ── CLASSICAL CHART PATTERNS ──

_register(
    "HEAD_AND_SHOULDERS", "Head & Shoulders Top", "bearish", 0.72, "yolov8+rule",
    "Three peaks: middle highest. Bearish reversal pattern.",
)
_register(
    "INV_HEAD_SHOULDERS", "Inverse Head & Shoulders", "bullish", 0.73, "yolov8+rule",
    "Three troughs: middle deepest. Bullish reversal pattern.",
)
_register(
    "DOUBLE_TOP", "Double Top (M Pattern)", "bearish", 0.69, "yolov8+rule",
    "Two peaks at similar levels. Bearish reversal.",
)
_register(
    "DOUBLE_BOTTOM", "Double Bottom (W Pattern)", "bullish", 0.68, "yolov8+rule",
    "Two troughs at similar levels. Bullish reversal.",
)
_register(
    "CUP_AND_HANDLE", "Cup & Handle", "bullish", 0.70, "custom",
    "U-shaped cup followed by small handle. Bullish continuation.",
)
_register(
    "ASCENDING_TRIANGLE", "Ascending Triangle", "bullish", 0.66, "yolov8+rule",
    "Flat resistance with rising support. Bullish breakout likely.",
)
_register(
    "DESCENDING_TRIANGLE", "Descending Triangle", "bearish", 0.65, "yolov8+rule",
    "Flat support with falling resistance. Bearish breakdown likely.",
)
_register(
    "SYMMETRICAL_TRIANGLE", "Symmetrical Triangle", "neutral", 0.58, "yolov8",
    "Converging trendlines. Breakout direction uncertain.",
)
_register(
    "BULL_FLAG", "Bull Flag", "bullish", 0.67, "yolov8+rule",
    "Sharp rise followed by slight downward consolidation. Continuation.",
)
_register(
    "BEAR_FLAG", "Bear Flag", "bearish", 0.66, "yolov8+rule",
    "Sharp decline followed by slight upward consolidation. Continuation.",
)
_register(
    "PENNANT", "Pennant", "neutral", 0.62, "custom",
    "Small symmetrical triangle after strong move. Continuation pattern.",
)
_register(
    "RISING_WEDGE", "Rising Wedge", "bearish", 0.61, "yolov8",
    "Converging trendlines both rising. Bearish reversal.",
)
_register(
    "FALLING_WEDGE", "Falling Wedge", "bullish", 0.62, "yolov8",
    "Converging trendlines both falling. Bullish reversal.",
)


def get_pattern(name: str) -> Optional[PatternMeta]:
    """
    Retrieve metadata for a pattern by registry key.

    Args:
        name: Pattern key string (e.g. 'BULL_FLAG').

    Returns:
        PatternMeta if found, None otherwise.
    """
    pattern = _PATTERN_REGISTRY.get(name)
    if pattern is None:
        logger.warning("Unknown pattern: %s", name)
    return pattern


def get_all_patterns() -> dict[str, PatternMeta]:
    """
    Return the complete pattern registry.

    Returns:
        Dict mapping pattern names to PatternMeta objects.
    """
    return _PATTERN_REGISTRY.copy()


def get_patterns_by_direction(direction: str) -> list[PatternMeta]:
    """
    Get all patterns with a given direction.

    Args:
        direction: 'bullish', 'bearish', or 'neutral'.

    Returns:
        List of matching PatternMeta objects.
    """
    return [p for p in _PATTERN_REGISTRY.values() if p.direction == direction]


def get_patterns_by_detection(detection: str) -> list[PatternMeta]:
    """
    Get all patterns detected by a given method.

    Args:
        detection: 'talib', 'yolov8', 'yolov8+rule', or 'custom'.

    Returns:
        List of matching PatternMeta objects.
    """
    return [p for p in _PATTERN_REGISTRY.values() if p.detection == detection]


# Log registry size on import
logger.info("Pattern registry loaded: %d patterns", len(_PATTERN_REGISTRY))
