"""
Momentum Indicators Module — Phase 3.

Calculates momentum-based indicators from OHLCV data using TA-Lib.
Each indicator file exposes: calculate(ohlcv: dict) -> dict

Returns a subset of the IndicatorSet with keys:
    rsi, stoch_k, stoch_d, cci, willr, roc, mfi

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
    Calculate all momentum indicators from OHLCV data.

    Args:
        ohlcv: dict with keys 'open', 'high', 'low', 'close', 'volume'
               — all numpy float64 arrays.

    Returns:
        dict with momentum indicator values (floats or NaN).
    """
    result: dict[str, Any] = {
        'rsi': float('nan'),
        'stoch_k': float('nan'),
        'stoch_d': float('nan'),
        'cci': float('nan'),
        'willr': float('nan'),
        'roc': float('nan'),
        'mfi': float('nan'),
    }

    try:
        close = np.asarray(ohlcv.get('close', []), dtype=np.float64)
        high = np.asarray(ohlcv.get('high', []), dtype=np.float64)
        low = np.asarray(ohlcv.get('low', []), dtype=np.float64)
        volume = np.asarray(ohlcv.get('volume', []), dtype=np.float64)

        if len(close) < 5:
            return result

        import talib

        # --- RSI (14) ---
        try:
            rsi = talib.RSI(close, timeperiod=14)
            if len(rsi) > 0 and not np.isnan(rsi[-1]):
                result['rsi'] = float(rsi[-1])
        except Exception:
            pass

        # --- Stochastic (14, 3, 3) ---
        try:
            slowk, slowd = talib.STOCH(
                high, low, close,
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0,
            )
            if len(slowk) > 0 and not np.isnan(slowk[-1]):
                result['stoch_k'] = float(slowk[-1])
            if len(slowd) > 0 and not np.isnan(slowd[-1]):
                result['stoch_d'] = float(slowd[-1])
        except Exception:
            pass

        # --- CCI (14) ---
        try:
            cci = talib.CCI(high, low, close, timeperiod=14)
            if len(cci) > 0 and not np.isnan(cci[-1]):
                result['cci'] = float(cci[-1])
        except Exception:
            pass

        # --- Williams %R (14) ---
        try:
            willr = talib.WILLR(high, low, close, timeperiod=14)
            if len(willr) > 0 and not np.isnan(willr[-1]):
                result['willr'] = float(willr[-1])
        except Exception:
            pass

        # --- ROC (10) ---
        try:
            roc = talib.ROC(close, timeperiod=10)
            if len(roc) > 0 and not np.isnan(roc[-1]):
                result['roc'] = float(roc[-1])
        except Exception:
            pass

        # --- MFI (14) ---
        try:
            if len(volume) >= len(close) and np.any(volume > 0):
                mfi = talib.MFI(high, low, close, volume, timeperiod=14)
                if len(mfi) > 0 and not np.isnan(mfi[-1]):
                    result['mfi'] = float(mfi[-1])
        except Exception:
            pass

    except ImportError:
        logger.error("TA-Lib not installed — momentum indicators unavailable")
    except Exception as e:
        logger.error("Momentum indicator calculation failed: %s", e)

    return result
