"""
Confluence Voting Engine — Phase 4.

WeightedVoter combines all signal sources using weighted voting.
Maximum possible score is 17.5.
A signal is only generated when |score| >= min_confluence_score (default 5.0).

Signal Sources & Weights (from PRD Section 8):
    YOLOv8 Pattern Detection:  ×3.0
    TA-Lib Rule Pattern:       ×2.5
    Volume Confirmation:       ×2.0
    RSI Extreme:               ×2.0
    MACD Cross:                ×2.0
    MA Stack Alignment:        ×1.5
    Fibonacci Zone:            ×1.5
    Bollinger Band Position:   ×1.0
    ADX Trend Guard:           ×1.0

ADX Guard: if ADX < 25, multiply total score × 0.5
Direction = 'wait' if abs(score) < min_confluence_score
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Result of confluence voting."""
    direction: str          # 'buy', 'sell', or 'wait'
    score: float            # Raw confluence score
    confidence: float       # 0–100
    contributing: list       # List of signal source names that fired
    patterns: list           # List of PatternResult objects
    indicators: dict         # Full indicator dict


class WeightedVoter:
    """
    Weighted voting engine for trade signal generation.

    Combines 9 signal sources with configurable weights.
    Applies ADX guard to filter choppy market signals.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the voter with configuration.

        Args:
            config: Full config dict (from config.yaml).
        """
        signals_cfg = config.get('signals', {})
        weights = signals_cfg.get('weights', {})

        self.min_score = signals_cfg.get('min_confluence_score', 5.0)

        self.w_yolov8 = weights.get('yolov8', 3.0)
        self.w_talib = weights.get('talib', 2.5)
        self.w_volume = weights.get('volume', 2.0)
        self.w_rsi = weights.get('rsi_extreme', 2.0)
        self.w_macd = weights.get('macd', 2.0)
        self.w_ma_align = weights.get('ma_align', 1.5)
        self.w_fibonacci = weights.get('fibonacci', 1.5)
        self.w_bb = weights.get('bb_position', 1.0)
        self.w_adx = weights.get('adx_trend', 1.0)

        self.confidence_threshold = config.get('model', {}).get(
            'confidence_threshold', 0.65
        )

        logger.info(
            "WeightedVoter initialized — min_score=%.1f, weights=%s",
            self.min_score, weights,
        )

    def vote(
        self,
        ohlcv: dict,
        patterns: list,
        indicators: Optional[dict] = None,
    ) -> SignalResult:
        """
        Run confluence voting across all signal sources.

        Args:
            ohlcv: OHLCV dict with price arrays.
            patterns: List[PatternResult] from detection layer.
            indicators: Combined IndicatorSet dict from indicator layer.

        Returns:
            SignalResult with direction, score, confidence, and metadata.
        """
        if indicators is None:
            indicators = {}

        score = 0.0
        contributing: list[str] = []

        # --- 1. YOLOv8 Pattern Detection (weight ×3.0) ---
        yolo_vote = self._vote_yolov8(patterns)
        if yolo_vote != 0:
            score += yolo_vote * self.w_yolov8
            contributing.append('yolov8')

        # --- 2. TA-Lib Rule Pattern (weight ×2.5) ---
        talib_vote = self._vote_talib(patterns)
        if talib_vote != 0:
            score += talib_vote * self.w_talib
            contributing.append('talib')

        # --- 3. Volume Confirmation (weight ×2.0) ---
        vol_vote = self._vote_volume(indicators)
        if vol_vote != 0:
            score += vol_vote * self.w_volume
            contributing.append('volume')

        # --- 4. RSI Extreme (weight ×2.0) ---
        rsi_vote = self._vote_rsi(indicators)
        if rsi_vote != 0:
            score += rsi_vote * self.w_rsi
            contributing.append('rsi_extreme')

        # --- 5. MACD Cross (weight ×2.0) ---
        macd_vote = self._vote_macd(indicators)
        if macd_vote != 0:
            score += macd_vote * self.w_macd
            contributing.append('macd')

        # --- 6. MA Stack Alignment (weight ×1.5) ---
        ma_vote = self._vote_ma_alignment(indicators)
        if ma_vote != 0:
            score += ma_vote * self.w_ma_align
            contributing.append('ma_align')

        # --- 7. Fibonacci Zone (weight ×1.5) ---
        fib_vote = self._vote_fibonacci(indicators, ohlcv)
        if fib_vote != 0:
            score += fib_vote * self.w_fibonacci
            contributing.append('fibonacci')

        # --- 8. Bollinger Band Position (weight ×1.0) ---
        bb_vote = self._vote_bb_position(indicators)
        if bb_vote != 0:
            score += bb_vote * self.w_bb
            contributing.append('bb_position')

        # --- 9. ADX Trend Guard (weight ×1.0) ---
        adx_val = indicators.get('adx', float('nan'))
        if not math.isnan(adx_val):
            if adx_val > 25:
                # Trending — add +1.0 in the direction of DI
                di_plus = indicators.get('di_plus', float('nan'))
                di_minus = indicators.get('di_minus', float('nan'))
                if not math.isnan(di_plus) and not math.isnan(di_minus):
                    if di_plus > di_minus:
                        score += self.w_adx
                        contributing.append('adx_trend')
                    elif di_minus > di_plus:
                        score -= self.w_adx
                        contributing.append('adx_trend')
            else:
                # Choppy market — halve ALL scores
                score *= 0.5
                logger.debug("ADX guard: score halved (ADX=%.1f < 25)", adx_val)

        # --- Determine direction ---
        if abs(score) < self.min_score:
            direction = 'wait'
        elif score > 0:
            direction = 'buy'
        else:
            direction = 'sell'

        # --- Confidence (0–100) ---
        max_possible = 17.5
        confidence = min(abs(score) / max_possible * 100, 100.0)

        return SignalResult(
            direction=direction,
            score=score,
            confidence=round(confidence, 1),
            contributing=contributing,
            patterns=patterns,
            indicators=indicators,
        )

    # ──────────────────────────────────────────────────────────
    # Individual voting functions
    # ──────────────────────────────────────────────────────────

    def _vote_yolov8(self, patterns: list) -> int:
        """
        YOLOv8 pattern vote.
        +1 if bullish pattern with confidence >= threshold,
        -1 if bearish, 0 otherwise.
        """
        for p in patterns:
            if getattr(p, 'source', '') == 'yolov8':
                confidence = getattr(p, 'confidence', 0)
                if confidence >= self.confidence_threshold:
                    direction = getattr(p, 'direction', 'neutral')
                    if direction == 'bullish':
                        return 1
                    elif direction == 'bearish':
                        return -1
        return 0

    def _vote_talib(self, patterns: list) -> int:
        """
        TA-Lib candlestick pattern vote.
        +1 if bullish, -1 if bearish, 0 otherwise.
        """
        for p in patterns:
            if getattr(p, 'source', '') == 'talib':
                direction = getattr(p, 'direction', 'neutral')
                if direction == 'bullish':
                    return 1
                elif direction == 'bearish':
                    return -1
        return 0

    def _vote_volume(self, indicators: dict) -> int:
        """
        Volume confirmation vote.
        +1 if volume > 1.5× 20-period MA (bullish volume surge).
        """
        vol_ratio = indicators.get('volume_ratio', float('nan'))
        if not math.isnan(vol_ratio) and vol_ratio > 1.5:
            return 1
        return 0

    def _vote_rsi(self, indicators: dict) -> int:
        """
        RSI extreme vote.
        +1 if RSI < 30 (oversold → buy), -1 if RSI > 70 (overbought → sell).
        """
        rsi = indicators.get('rsi', float('nan'))
        if math.isnan(rsi):
            return 0
        if rsi < 30:
            return 1
        elif rsi > 70:
            return -1
        return 0

    def _vote_macd(self, indicators: dict) -> int:
        """
        MACD cross vote.
        +1 if MACD histogram is positive (bullish),
        -1 if negative (bearish).
        """
        macd_hist = indicators.get('macd_hist', float('nan'))
        if math.isnan(macd_hist):
            return 0
        if macd_hist > 0:
            return 1
        elif macd_hist < 0:
            return -1
        return 0

    def _vote_ma_alignment(self, indicators: dict) -> int:
        """
        MA stack alignment vote.
        Bull: close > EMA9 > EMA21 > SMA50 → +1
        Bear: close < EMA9 < EMA21 < SMA50 → -1
        """
        ema9 = indicators.get('ema9', float('nan'))
        ema21 = indicators.get('ema21', float('nan'))
        sma50 = indicators.get('sma50', float('nan'))

        if any(math.isnan(v) for v in [ema9, ema21, sma50]):
            return 0

        if ema9 > ema21 > sma50:
            return 1
        elif ema9 < ema21 < sma50:
            return -1
        return 0

    def _vote_fibonacci(self, indicators: dict, ohlcv: dict) -> int:
        """
        Fibonacci zone vote.
        +1 if price is within 0.3% of a key Fibonacci level (0.382, 0.5, 0.618).
        """
        try:
            close_arr = ohlcv.get('close', [])
            high_arr = ohlcv.get('high', [])
            low_arr = ohlcv.get('low', [])

            if len(close_arr) < 20 or len(high_arr) < 20:
                return 0

            current_price = float(close_arr[-1]) if hasattr(close_arr, '__len__') else float(close_arr)
            recent_high = float(np.max(high_arr[-50:])) if len(high_arr) >= 50 else float(np.max(high_arr))
            recent_low = float(np.min(low_arr[-50:])) if len(low_arr) >= 50 else float(np.min(low_arr))

            price_range = recent_high - recent_low
            if price_range <= 0:
                return 0

            fib_levels = [0.382, 0.5, 0.618]
            for level in fib_levels:
                fib_price = recent_low + price_range * level
                if abs(current_price - fib_price) / current_price < 0.003:
                    # Near a Fibonacci level — could be support or resistance
                    return 1

        except Exception:
            pass
        return 0

    def _vote_bb_position(self, indicators: dict) -> int:
        """
        Bollinger Band position vote.
        +1 if price touches/crosses lower band (buy signal).
        -1 if price touches/crosses upper band (sell signal).
        """
        bb_upper = indicators.get('bb_upper', float('nan'))
        bb_lower = indicators.get('bb_lower', float('nan'))

        if math.isnan(bb_upper) or math.isnan(bb_lower):
            return 0

        # We need the current close — check if it's in indicators
        # Use SMA20 as a proxy for the close relative to bands
        bb_middle = indicators.get('bb_middle', float('nan'))
        sma20 = indicators.get('sma20', float('nan'))

        if math.isnan(bb_middle):
            return 0

        # Check EMA9 as proxy for current price level
        ema9 = indicators.get('ema9', float('nan'))
        if math.isnan(ema9):
            return 0

        if ema9 <= bb_lower:
            return 1  # Buy near lower band
        elif ema9 >= bb_upper:
            return -1  # Sell near upper band
        return 0
