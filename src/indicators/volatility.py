"""
Volatility Indicators Module — Phase 3.

Calculates volatility indicators from OHLCV data using TA-Lib.
Each indicator file exposes: calculate(ohlcv: dict) -> dict

Returns a subset of the IndicatorSet with keys:
    bb_upper, bb_middle, bb_lower, bb_width,
    atr, kc_upper, kc_lower, dc_upper, dc_lower

IMPORTANT:
    - Use float('nan') for indicators with insufficient data.
    - NEVER return None or raise exceptions — caller expects a dict always.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def calculate(ohlcv: dict) -> dict:
    """
    Calculate all volatility indicators from OHLCV data.

    Args:
        ohlcv: dict with keys 'open', 'high', 'low', 'close', 'volume'
               — all numpy float64 arrays.

    Returns:
        dict with volatility indicator values (floats or NaN).
    """
    result: dict[str, Any] = {
        'bb_upper': float('nan'),
        'bb_middle': float('nan'),
        'bb_lower': float('nan'),
        'bb_width': float('nan'),
        'atr': float('nan'),
        'kc_upper': float('nan'),
        'kc_lower': float('nan'),
        'dc_upper': float('nan'),
        'dc_lower': float('nan'),
    }

    try:
        close = np.asarray(ohlcv.get('close', []), dtype=np.float64)
        high = np.asarray(ohlcv.get('high', []), dtype=np.float64)
        low = np.asarray(ohlcv.get('low', []), dtype=np.float64)

        if len(close) < 5:
            return result

        import talib

        # --- Bollinger Bands (20, 2) ---
        try:
            upper, middle, lower = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            if len(upper) > 0 and not np.isnan(upper[-1]):
                result['bb_upper'] = float(upper[-1])
            if len(middle) > 0 and not np.isnan(middle[-1]):
                result['bb_middle'] = float(middle[-1])
            if len(lower) > 0 and not np.isnan(lower[-1]):
                result['bb_lower'] = float(lower[-1])

            # BB Width = (upper - lower) / middle
            if (
                not np.isnan(result['bb_upper'])
                and not np.isnan(result['bb_lower'])
                and not np.isnan(result['bb_middle'])
                and result['bb_middle'] != 0
            ):
                result['bb_width'] = (
                    (result['bb_upper'] - result['bb_lower'])
                    / result['bb_middle']
                )
        except Exception:
            pass

        # --- ATR (14) ---
        try:
            atr = talib.ATR(high, low, close, timeperiod=14)
            if len(atr) > 0 and not np.isnan(atr[-1]):
                result['atr'] = float(atr[-1])
        except Exception:
            pass

        # --- Keltner Channel (20 EMA ± 2 × ATR) ---
        try:
            ema20 = talib.EMA(close, timeperiod=20)
            atr_kc = talib.ATR(high, low, close, timeperiod=14)
            if (
                len(ema20) > 0
                and not np.isnan(ema20[-1])
                and len(atr_kc) > 0
                and not np.isnan(atr_kc[-1])
            ):
                result['kc_upper'] = float(ema20[-1] + 2.0 * atr_kc[-1])
                result['kc_lower'] = float(ema20[-1] - 2.0 * atr_kc[-1])
        except Exception:
            pass

        # --- Donchian Channel (20-period high/low) ---
        try:
            if len(high) >= 20 and len(low) >= 20:
                result['dc_upper'] = float(np.max(high[-20:]))
                result['dc_lower'] = float(np.min(low[-20:]))
        except Exception:
            pass

    except ImportError:
        logger.error("TA-Lib not installed — volatility indicators unavailable")
    except Exception as e:
        logger.error("Volatility indicator calculation failed: %s", e)

    return result
