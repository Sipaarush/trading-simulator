"""
Volume Indicators Module — Phase 3.

Calculates volume-based indicators from OHLCV data using TA-Lib.
Each indicator file exposes: calculate(ohlcv: dict) -> dict

Returns a subset of the IndicatorSet with keys:
    volume_ma, volume_ratio, obv, vwap, acc_dist, cmf

IMPORTANT:
    - Use float('nan') for indicators with insufficient data.
    - NEVER return None or raise exceptions — caller expects a dict always.
    - VWAP resets to NaN at start of new day.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def calculate(ohlcv: dict) -> dict:
    """
    Calculate all volume indicators from OHLCV data.

    Args:
        ohlcv: dict with keys 'open', 'high', 'low', 'close', 'volume'
               — all numpy float64 arrays.

    Returns:
        dict with volume indicator values (floats or NaN).
    """
    result: dict[str, Any] = {
        'volume_ma': float('nan'),
        'volume_ratio': float('nan'),
        'obv': float('nan'),
        'vwap': float('nan'),
        'acc_dist': float('nan'),
        'cmf': float('nan'),
    }

    try:
        close = np.asarray(ohlcv.get('close', []), dtype=np.float64)
        high = np.asarray(ohlcv.get('high', []), dtype=np.float64)
        low = np.asarray(ohlcv.get('low', []), dtype=np.float64)
        volume = np.asarray(ohlcv.get('volume', []), dtype=np.float64)

        if len(close) < 5 or len(volume) < 5:
            return result

        import talib

        # --- Volume MA (20-period SMA of volume) ---
        try:
            vol_ma = talib.SMA(volume, timeperiod=20)
            if len(vol_ma) > 0 and not np.isnan(vol_ma[-1]):
                result['volume_ma'] = float(vol_ma[-1])

                # Volume ratio = current volume / 20-period MA
                if vol_ma[-1] != 0 and not np.isnan(volume[-1]):
                    result['volume_ratio'] = float(volume[-1] / vol_ma[-1])
        except Exception:
            pass

        # --- OBV (On-Balance Volume) ---
        try:
            obv = talib.OBV(close, volume)
            if len(obv) > 0 and not np.isnan(obv[-1]):
                result['obv'] = float(obv[-1])
        except Exception:
            pass

        # --- VWAP (Volume Weighted Average Price — intraday) ---
        try:
            result['vwap'] = _calc_vwap(high, low, close, volume)
        except Exception:
            pass

        # --- Accumulation/Distribution ---
        try:
            ad = talib.AD(high, low, close, volume)
            if len(ad) > 0 and not np.isnan(ad[-1]):
                result['acc_dist'] = float(ad[-1])
        except Exception:
            pass

        # --- Chaikin Money Flow (20-period) ---
        try:
            result['cmf'] = _calc_cmf(high, low, close, volume, period=20)
        except Exception:
            pass

    except ImportError:
        logger.error("TA-Lib not installed — volume indicators unavailable")
    except Exception as e:
        logger.error("Volume indicator calculation failed: %s", e)

    return result


def _calc_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> float:
    """
    Calculate VWAP (Volume Weighted Average Price).

    Uses typical price × volume / cumulative volume.

    Returns:
        VWAP value as float, or NaN if insufficient data.
    """
    if len(close) < 1 or np.sum(volume) == 0:
        return float('nan')

    typical_price = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)

    if cum_vol[-1] == 0:
        return float('nan')

    vwap = cum_tp_vol[-1] / cum_vol[-1]
    return float(vwap) if not np.isnan(vwap) else float('nan')


def _calc_cmf(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 20,
) -> float:
    """
    Calculate Chaikin Money Flow.

    CMF = Sum(MF Volume, period) / Sum(Volume, period)
    MF Multiplier = ((close - low) - (high - close)) / (high - low)
    MF Volume = MF Multiplier × Volume

    Returns:
        CMF value as float, or NaN if insufficient data.
    """
    if len(close) < period:
        return float('nan')

    hl_range = high - low
    # Avoid division by zero
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)

    mf_multiplier = ((close - low) - (high - close)) / hl_range
    mf_volume = mf_multiplier * volume

    # Use last 'period' bars
    sum_mf_vol = np.sum(mf_volume[-period:])
    sum_vol = np.sum(volume[-period:])

    if sum_vol == 0:
        return float('nan')

    cmf = sum_mf_vol / sum_vol
    return float(cmf) if not np.isnan(cmf) else float('nan')
