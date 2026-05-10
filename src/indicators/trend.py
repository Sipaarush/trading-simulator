"""
Trend Indicators Module — Phase 3.

Calculates trend-following indicators from OHLCV data using TA-Lib.
Each indicator file exposes: calculate(ohlcv: dict) -> dict

Returns a subset of the IndicatorSet with keys:
    sma20, sma50, sma200, ema9, ema21,
    macd, macd_signal, macd_hist,
    adx, di_plus, di_minus,
    supertrend, ichi_above, parabolic_sar

IMPORTANT:
    - Use float('nan') for indicators with insufficient data.
    - NEVER return None or raise exceptions — caller expects a dict always.
"""

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def calculate(ohlcv: dict) -> dict:
    """
    Calculate all trend indicators from OHLCV data.

    Args:
        ohlcv: dict with keys 'open', 'high', 'low', 'close', 'volume'
               — all numpy float64 arrays.

    Returns:
        dict with trend indicator values (floats or NaN).
    """
    result: dict[str, Any] = {
        'sma20': float('nan'),
        'sma50': float('nan'),
        'sma200': float('nan'),
        'ema9': float('nan'),
        'ema21': float('nan'),
        'macd': float('nan'),
        'macd_signal': float('nan'),
        'macd_hist': float('nan'),
        'adx': float('nan'),
        'di_plus': float('nan'),
        'di_minus': float('nan'),
        'supertrend': float('nan'),
        'ichi_above': float('nan'),
        'parabolic_sar': float('nan'),
    }

    try:
        close = np.asarray(ohlcv.get('close', []), dtype=np.float64)
        high = np.asarray(ohlcv.get('high', []), dtype=np.float64)
        low = np.asarray(ohlcv.get('low', []), dtype=np.float64)

        if len(close) < 5:
            return result

        import talib

        # --- SMA ---
        try:
            sma20 = talib.SMA(close, timeperiod=20)
            if len(sma20) > 0 and not np.isnan(sma20[-1]):
                result['sma20'] = float(sma20[-1])
        except Exception:
            pass

        try:
            sma50 = talib.SMA(close, timeperiod=50)
            if len(sma50) > 0 and not np.isnan(sma50[-1]):
                result['sma50'] = float(sma50[-1])
        except Exception:
            pass

        try:
            sma200 = talib.SMA(close, timeperiod=200)
            if len(sma200) > 0 and not np.isnan(sma200[-1]):
                result['sma200'] = float(sma200[-1])
        except Exception:
            pass

        # --- EMA ---
        try:
            ema9 = talib.EMA(close, timeperiod=9)
            if len(ema9) > 0 and not np.isnan(ema9[-1]):
                result['ema9'] = float(ema9[-1])
        except Exception:
            pass

        try:
            ema21 = talib.EMA(close, timeperiod=21)
            if len(ema21) > 0 and not np.isnan(ema21[-1]):
                result['ema21'] = float(ema21[-1])
        except Exception:
            pass

        # --- MACD ---
        try:
            macd, macd_signal, macd_hist = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            if len(macd) > 0 and not np.isnan(macd[-1]):
                result['macd'] = float(macd[-1])
            if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]):
                result['macd_signal'] = float(macd_signal[-1])
            if len(macd_hist) > 0 and not np.isnan(macd_hist[-1]):
                result['macd_hist'] = float(macd_hist[-1])
        except Exception:
            pass

        # --- ADX / DI ---
        try:
            adx = talib.ADX(high, low, close, timeperiod=14)
            if len(adx) > 0 and not np.isnan(adx[-1]):
                result['adx'] = float(adx[-1])
        except Exception:
            pass

        try:
            di_plus = talib.PLUS_DI(high, low, close, timeperiod=14)
            if len(di_plus) > 0 and not np.isnan(di_plus[-1]):
                result['di_plus'] = float(di_plus[-1])
        except Exception:
            pass

        try:
            di_minus = talib.MINUS_DI(high, low, close, timeperiod=14)
            if len(di_minus) > 0 and not np.isnan(di_minus[-1]):
                result['di_minus'] = float(di_minus[-1])
        except Exception:
            pass

        # --- Supertrend (ATR-based, custom) ---
        try:
            result['supertrend'] = _calc_supertrend(high, low, close)
        except Exception:
            pass

        # --- Ichimoku (is close above the cloud?) ---
        try:
            result['ichi_above'] = _calc_ichimoku_above(high, low, close)
        except Exception:
            pass

        # --- Parabolic SAR ---
        try:
            sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            if len(sar) > 0 and not np.isnan(sar[-1]):
                result['parabolic_sar'] = float(sar[-1])
        except Exception:
            pass

    except ImportError:
        logger.error("TA-Lib not installed — trend indicators unavailable")
    except Exception as e:
        logger.error("Trend indicator calculation failed: %s", e)

    return result


def _calc_supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> float:
    """
    Calculate Supertrend value.

    Returns the current supertrend level as a float, or NaN.
    """
    if len(close) < period + 1:
        return float('nan')

    try:
        import talib
        atr = talib.ATR(high, low, close, timeperiod=period)
    except Exception:
        return float('nan')

    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = np.full_like(close, float('nan'))
    direction = np.ones(len(close))  # 1 = up, -1 = down

    # Find first valid index
    first_valid = period
    for i in range(period, len(close)):
        if not np.isnan(atr[i]):
            first_valid = i
            break

    if first_valid >= len(close):
        return float('nan')

    supertrend[first_valid] = upper_band[first_valid]

    for i in range(first_valid + 1, len(close)):
        if np.isnan(atr[i]):
            supertrend[i] = supertrend[i - 1]
            continue

        # Adjust bands
        if close[i - 1] > upper_band[i - 1]:
            upper_band[i] = max(upper_band[i], upper_band[i - 1])

        if close[i - 1] < lower_band[i - 1]:
            lower_band[i] = min(lower_band[i], lower_band[i - 1])

        if supertrend[i - 1] == upper_band[i - 1]:
            if close[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
        else:
            if close[i] >= lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1

    val = supertrend[-1]
    return float('nan') if (isinstance(val, float) and math.isnan(val)) else float(val)


def _calc_ichimoku_above(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> float:
    """
    Determine if close is above the Ichimoku cloud.

    Returns:
        1.0 if above, 0.0 if below, NaN if insufficient data.
    """
    if len(close) < 52:
        return float('nan')

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = (np.max(high[-9:]) + np.min(low[-9:])) / 2.0

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = (np.max(high[-26:]) + np.min(low[-26:])) / 2.0

    # Senkou Span A: (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0

    # Senkou Span B: (52-period high + 52-period low) / 2
    senkou_b = (np.max(high[-52:]) + np.min(low[-52:])) / 2.0

    cloud_top = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)

    current_close = float(close[-1])

    if current_close > cloud_top:
        return 1.0
    elif current_close < cloud_bottom:
        return 0.0
    else:
        return 0.5  # Inside the cloud
