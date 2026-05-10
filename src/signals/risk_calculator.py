"""
Risk Calculator Module — Phase 4.

Calculates stop-loss, take-profit, risk %, R:R ratio, position size,
and success probability for each trade signal.

Key formulas (from PRD):
    stop_loss = entry ± (ATR × config.risk.atr_multiplier)
    target1 = entry ± (risk × config.risk.min_rr_ratio)
    success_pct = base_win_rate × (1 + score/17.5×0.35) × vol_factor, cap 85

TradeSignal dataclass matches Layer 4→5 contract exactly.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """
    Complete trade signal — Layer 4 → Layer 5 contract.

    All fields must be populated before passing to output layer.
    """
    direction: str              # 'buy' | 'sell'
    confidence: float           # 0–100
    entry: float                # current price or breakout level
    stop_loss: float            # entry ± (ATR × 1.5)
    target1: float              # minimum 1.5:1 R/R
    target2: float              # extended target
    risk_pct: float             # |entry - SL| / entry × 100
    rr_ratio: float             # |TP1 - entry| / |entry - SL|
    position_size: float        # units to trade
    success_pct: float          # 0–85 (capped)
    pattern_name: str
    contributing: List[str]     # which signals fired
    indicators: dict
    atr: float


class RiskCalculator:
    """
    Calculate risk parameters and construct TradeSignal.

    Uses ATR-based stop-loss, configurable R:R ratio,
    and probability-based position sizing.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize with config settings.

        Args:
            config: Full config dict from config.yaml.
        """
        risk_cfg = config.get('risk', {})
        self.atr_multiplier = risk_cfg.get('atr_multiplier', 1.5)
        self.min_rr_ratio = risk_cfg.get('min_rr_ratio', 1.5)
        self.base_win_rate = risk_cfg.get('base_win_rate', 0.55)
        self.max_success_pct = risk_cfg.get('max_success_pct', 85)
        self.max_risk_pct = risk_cfg.get('max_risk_pct', 2.0)
        self.account_balance = risk_cfg.get('account_balance', 100000)

        logger.info(
            "RiskCalculator initialized — ATR mult=%.1f, min R:R=%.1f, "
            "base_win=%.2f, max_success=%d",
            self.atr_multiplier, self.min_rr_ratio,
            self.base_win_rate, self.max_success_pct,
        )

    def calculate(
        self,
        signal: Any,
        ohlcv: dict,
    ) -> Optional[TradeSignal]:
        """
        Calculate risk parameters and build TradeSignal.

        Args:
            signal: SignalResult from confluence voting.
            ohlcv: OHLCV dict with price arrays.

        Returns:
            TradeSignal if valid trade, None if signal is 'wait' or invalid.
        """
        try:
            # Skip 'wait' signals
            if signal.direction == 'wait':
                return None

            indicators = signal.indicators if hasattr(signal, 'indicators') else {}
            atr = indicators.get('atr', float('nan'))

            # Get current price
            close_arr = ohlcv.get('close', [])
            if len(close_arr) == 0:
                logger.warning("No close data for risk calculation")
                return None

            import numpy as np
            entry = float(np.asarray(close_arr)[-1])

            # Must have valid ATR
            if math.isnan(atr) or atr <= 0:
                # Fallback: use 1% of entry price
                atr = entry * 0.01
                logger.debug("ATR unavailable, using fallback: %.4f", atr)

            # --- Stop Loss ---
            risk_amount = atr * self.atr_multiplier
            if signal.direction == 'buy':
                stop_loss = entry - risk_amount
            else:  # sell
                stop_loss = entry + risk_amount

            # --- Targets ---
            reward_amount = risk_amount * self.min_rr_ratio
            if signal.direction == 'buy':
                target1 = entry + reward_amount
                target2 = entry + reward_amount * 2.0  # Extended target
            else:  # sell
                target1 = entry - reward_amount
                target2 = entry - reward_amount * 2.0

            # --- Risk % ---
            risk_pct = abs(entry - stop_loss) / entry * 100
            if risk_pct > self.max_risk_pct:
                # Cap risk — adjust stop loss
                if signal.direction == 'buy':
                    stop_loss = entry * (1 - self.max_risk_pct / 100)
                else:
                    stop_loss = entry * (1 + self.max_risk_pct / 100)
                risk_pct = self.max_risk_pct

            # --- R:R Ratio ---
            sl_distance = abs(entry - stop_loss)
            tp_distance = abs(target1 - entry)
            rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0

            # Ensure minimum R:R
            if rr_ratio < self.min_rr_ratio:
                rr_ratio = self.min_rr_ratio
                if signal.direction == 'buy':
                    target1 = entry + sl_distance * self.min_rr_ratio
                else:
                    target1 = entry - sl_distance * self.min_rr_ratio

            # --- Success Probability ---
            score = abs(signal.score)
            vol_factor = self._volume_factor(indicators)
            success_pct = self.base_win_rate * (1 + score / 17.5 * 0.35) * vol_factor * 100
            success_pct = min(success_pct, float(self.max_success_pct))
            success_pct = max(success_pct, 0.0)

            # --- Position Size (risk-based) ---
            risk_dollars = self.account_balance * (risk_pct / 100)
            position_size = risk_dollars / sl_distance if sl_distance > 0 else 0.0

            # --- Pattern Name ---
            pattern_name = "Unknown"
            patterns = getattr(signal, 'patterns', [])
            if patterns and len(patterns) > 0:
                pattern_name = getattr(patterns[0], 'display_name', 'Unknown')

            # --- Build TradeSignal ---
            trade = TradeSignal(
                direction=signal.direction,
                confidence=signal.confidence,
                entry=round(entry, 2),
                stop_loss=round(stop_loss, 2),
                target1=round(target1, 2),
                target2=round(target2, 2),
                risk_pct=round(risk_pct, 2),
                rr_ratio=round(rr_ratio, 2),
                position_size=round(position_size, 2),
                success_pct=round(success_pct, 1),
                pattern_name=pattern_name,
                contributing=getattr(signal, 'contributing', []),
                indicators=indicators,
                atr=round(atr, 4),
            )

            # --- Validation ---
            if not self._validate(trade):
                logger.warning("Trade signal failed validation: %s", trade)
                return None

            logger.info(
                "TradeSignal → %s | Entry=%.2f | SL=%.2f | TP1=%.2f | "
                "Risk=%.2f%% | R:R=%.1f | Success=%.1f%%",
                trade.direction.upper(), trade.entry, trade.stop_loss,
                trade.target1, trade.risk_pct, trade.rr_ratio, trade.success_pct,
            )

            return trade

        except Exception as e:
            logger.error("Risk calculation failed: %s", e)
            return None

    def _volume_factor(self, indicators: dict) -> float:
        """
        Calculate volume factor for success probability adjustment.

        Higher volume = higher confidence in the signal.
        """
        vol_ratio = indicators.get('volume_ratio', float('nan'))
        if math.isnan(vol_ratio):
            return 1.0

        if vol_ratio > 2.0:
            return 1.2
        elif vol_ratio > 1.5:
            return 1.1
        elif vol_ratio < 0.5:
            return 0.8
        return 1.0

    def _validate(self, trade: TradeSignal) -> bool:
        """
        Validate trade signal meets all requirements.

        Checks from PRD acceptance criteria:
            - SL always on correct side of entry
            - R:R ratio always >= 1.5
            - Success % never exceeds 85
        """
        # SL on correct side
        if trade.direction == 'buy' and trade.stop_loss >= trade.entry:
            logger.error("BUY signal but SL >= entry")
            return False
        if trade.direction == 'sell' and trade.stop_loss <= trade.entry:
            logger.error("SELL signal but SL <= entry")
            return False

        # R:R minimum
        if trade.rr_ratio < self.min_rr_ratio:
            logger.error("R:R ratio %.2f < minimum %.2f", trade.rr_ratio, self.min_rr_ratio)
            return False

        # Success cap
        if trade.success_pct > self.max_success_pct:
            logger.error("Success %% %.1f > max %d", trade.success_pct, self.max_success_pct)
            return False

        return True
