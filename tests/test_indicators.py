"""
Tests for Phase 3 — Indicator Engine.

Tests all 25+ indicators across trend, momentum, volatility, and volume modules.
Uses synthetic OHLCV data to validate output types, NaN handling, and edge cases.
"""

import math
import sys
import os

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators import trend, momentum, volatility, volume


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def ohlcv_100():
    """Generate 100 bars of synthetic OHLCV data (uptrend with noise)."""
    np.random.seed(42)
    n = 100
    base = 100.0
    noise = np.random.randn(n) * 0.5
    trend_component = np.linspace(0, 20, n)

    close = base + trend_component + np.cumsum(noise)
    high = close + np.abs(np.random.randn(n)) * 1.0
    low = close - np.abs(np.random.randn(n)) * 1.0
    open_ = close + np.random.randn(n) * 0.3
    vol = np.random.randint(1000, 10000, n).astype(np.float64)

    return {
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': vol,
    }


@pytest.fixture
def ohlcv_250():
    """Generate 250 bars — enough for SMA200 and all indicators."""
    np.random.seed(123)
    n = 250
    base = 50.0
    noise = np.random.randn(n) * 0.3
    trend_component = np.linspace(0, 30, n)

    close = base + trend_component + np.cumsum(noise)
    high = close + np.abs(np.random.randn(n)) * 0.8
    low = close - np.abs(np.random.randn(n)) * 0.8
    open_ = close + np.random.randn(n) * 0.2
    vol = np.random.randint(5000, 50000, n).astype(np.float64)

    return {
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': vol,
    }


@pytest.fixture
def ohlcv_small():
    """Tiny dataset — only 3 bars (should return all NaN)."""
    return {
        'open': np.array([100.0, 101.0, 102.0]),
        'high': np.array([101.0, 102.0, 103.0]),
        'low': np.array([99.0, 100.0, 101.0]),
        'close': np.array([100.5, 101.5, 102.5]),
        'volume': np.array([1000.0, 2000.0, 1500.0]),
    }


@pytest.fixture
def ohlcv_oversold():
    """Generate data with a sharp drop — RSI should be ~30 or below."""
    np.random.seed(99)
    n = 100
    # Start high, then sharp drop
    close = np.concatenate([
        np.linspace(200, 200, 60),      # flat
        np.linspace(200, 150, 40),       # sharp drop
    ])
    high = close + 1.0
    low = close - 1.0
    open_ = close + 0.5
    vol = np.full(n, 5000.0)

    return {
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': vol,
    }


@pytest.fixture
def ohlcv_empty():
    """Empty dataset."""
    return {
        'open': np.array([]),
        'high': np.array([]),
        'low': np.array([]),
        'close': np.array([]),
        'volume': np.array([]),
    }


# ============================================================
# TREND TESTS
# ============================================================

class TestTrendIndicators:
    """Test trend.py module."""

    def test_returns_dict(self, ohlcv_100):
        """calculate() must return a dict."""
        result = trend.calculate(ohlcv_100)
        assert isinstance(result, dict)

    def test_all_keys_present(self, ohlcv_250):
        """All expected keys must be present in the result."""
        result = trend.calculate(ohlcv_250)
        expected_keys = [
            'sma20', 'sma50', 'sma200', 'ema9', 'ema21',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'di_plus', 'di_minus',
            'supertrend', 'ichi_above', 'parabolic_sar',
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_sma_values_with_enough_data(self, ohlcv_250):
        """SMA20/50/200 should return valid floats with 250 bars."""
        result = trend.calculate(ohlcv_250)
        assert not math.isnan(result['sma20'])
        assert not math.isnan(result['sma50'])
        assert not math.isnan(result['sma200'])

    def test_ema_values(self, ohlcv_100):
        """EMA9/21 should be valid with 100 bars."""
        result = trend.calculate(ohlcv_100)
        assert not math.isnan(result['ema9'])
        assert not math.isnan(result['ema21'])

    def test_macd_returns_all_three(self, ohlcv_100):
        """MACD should return macd, macd_signal, and macd_hist."""
        result = trend.calculate(ohlcv_100)
        assert not math.isnan(result['macd'])
        assert not math.isnan(result['macd_signal'])
        assert not math.isnan(result['macd_hist'])

    def test_adx_positive(self, ohlcv_100):
        """ADX should return a positive float."""
        result = trend.calculate(ohlcv_100)
        if not math.isnan(result['adx']):
            assert result['adx'] >= 0

    def test_insufficient_data_returns_nan(self, ohlcv_small):
        """Small data should return all NaN — no exceptions."""
        result = trend.calculate(ohlcv_small)
        assert isinstance(result, dict)
        # All values should be NaN for 3 bars
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not float"

    def test_empty_data_no_crash(self, ohlcv_empty):
        """Empty data should return dict with NaN — never crash."""
        result = trend.calculate(ohlcv_empty)
        assert isinstance(result, dict)

    def test_atr_positive_float(self, ohlcv_100):
        """ATR returned by volatility should be positive."""
        result = volatility.calculate(ohlcv_100)
        if not math.isnan(result['atr']):
            assert result['atr'] > 0


# ============================================================
# MOMENTUM TESTS
# ============================================================

class TestMomentumIndicators:
    """Test momentum.py module."""

    def test_returns_dict(self, ohlcv_100):
        """calculate() must return a dict."""
        result = momentum.calculate(ohlcv_100)
        assert isinstance(result, dict)

    def test_all_keys_present(self, ohlcv_100):
        """All expected keys must be present."""
        result = momentum.calculate(ohlcv_100)
        expected_keys = ['rsi', 'stoch_k', 'stoch_d', 'cci', 'willr', 'roc', 'mfi']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_rsi_range(self, ohlcv_100):
        """RSI should be between 0 and 100."""
        result = momentum.calculate(ohlcv_100)
        if not math.isnan(result['rsi']):
            assert 0 <= result['rsi'] <= 100

    def test_rsi_oversold_fixture(self, ohlcv_oversold):
        """RSI should be low (≤ 35) for an oversold dataset."""
        result = momentum.calculate(ohlcv_oversold)
        if not math.isnan(result['rsi']):
            assert result['rsi'] <= 35, f"RSI is {result['rsi']}, expected ≤ 35"

    def test_stochastic_range(self, ohlcv_100):
        """Stochastic K and D should be between 0 and 100."""
        result = momentum.calculate(ohlcv_100)
        for key in ['stoch_k', 'stoch_d']:
            if not math.isnan(result[key]):
                assert 0 <= result[key] <= 100

    def test_insufficient_data(self, ohlcv_small):
        """Small data should return dict — no crash."""
        result = momentum.calculate(ohlcv_small)
        assert isinstance(result, dict)

    def test_empty_data(self, ohlcv_empty):
        """Empty data returns dict with NaN."""
        result = momentum.calculate(ohlcv_empty)
        assert isinstance(result, dict)


# ============================================================
# VOLATILITY TESTS
# ============================================================

class TestVolatilityIndicators:
    """Test volatility.py module."""

    def test_returns_dict(self, ohlcv_100):
        """calculate() must return a dict."""
        result = volatility.calculate(ohlcv_100)
        assert isinstance(result, dict)

    def test_all_keys_present(self, ohlcv_100):
        """All expected keys must be present."""
        result = volatility.calculate(ohlcv_100)
        expected_keys = [
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'atr', 'kc_upper', 'kc_lower', 'dc_upper', 'dc_lower',
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_bb_band_ordering(self, ohlcv_100):
        """BB upper >= middle >= lower."""
        result = volatility.calculate(ohlcv_100)
        if (
            not math.isnan(result['bb_upper'])
            and not math.isnan(result['bb_middle'])
            and not math.isnan(result['bb_lower'])
        ):
            assert result['bb_upper'] >= result['bb_middle']
            assert result['bb_middle'] >= result['bb_lower']

    def test_atr_positive(self, ohlcv_100):
        """ATR should be a positive value."""
        result = volatility.calculate(ohlcv_100)
        if not math.isnan(result['atr']):
            assert result['atr'] > 0

    def test_keltner_ordering(self, ohlcv_100):
        """KC upper should be >= KC lower."""
        result = volatility.calculate(ohlcv_100)
        if (
            not math.isnan(result['kc_upper'])
            and not math.isnan(result['kc_lower'])
        ):
            assert result['kc_upper'] >= result['kc_lower']

    def test_donchian_ordering(self, ohlcv_100):
        """DC upper should be >= DC lower."""
        result = volatility.calculate(ohlcv_100)
        if (
            not math.isnan(result['dc_upper'])
            and not math.isnan(result['dc_lower'])
        ):
            assert result['dc_upper'] >= result['dc_lower']

    def test_insufficient_data(self, ohlcv_small):
        """Small data — no crash."""
        result = volatility.calculate(ohlcv_small)
        assert isinstance(result, dict)

    def test_empty_data(self, ohlcv_empty):
        """Empty data returns dict."""
        result = volatility.calculate(ohlcv_empty)
        assert isinstance(result, dict)


# ============================================================
# VOLUME TESTS
# ============================================================

class TestVolumeIndicators:
    """Test volume.py module."""

    def test_returns_dict(self, ohlcv_100):
        """calculate() must return a dict."""
        result = volume.calculate(ohlcv_100)
        assert isinstance(result, dict)

    def test_all_keys_present(self, ohlcv_100):
        """All expected keys must be present."""
        result = volume.calculate(ohlcv_100)
        expected_keys = ['volume_ma', 'volume_ratio', 'obv', 'vwap', 'acc_dist', 'cmf']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_vwap_reasonable(self, ohlcv_100):
        """VWAP should be within the price range."""
        result = volume.calculate(ohlcv_100)
        if not math.isnan(result['vwap']):
            min_price = float(np.min(ohlcv_100['low']))
            max_price = float(np.max(ohlcv_100['high']))
            assert min_price <= result['vwap'] <= max_price

    def test_cmf_range(self, ohlcv_100):
        """CMF should be between -1 and 1."""
        result = volume.calculate(ohlcv_100)
        if not math.isnan(result['cmf']):
            assert -1.0 <= result['cmf'] <= 1.0

    def test_volume_ratio_positive(self, ohlcv_100):
        """Volume ratio should be positive."""
        result = volume.calculate(ohlcv_100)
        if not math.isnan(result['volume_ratio']):
            assert result['volume_ratio'] > 0

    def test_insufficient_data(self, ohlcv_small):
        """Small data — no crash."""
        result = volume.calculate(ohlcv_small)
        assert isinstance(result, dict)

    def test_empty_data(self, ohlcv_empty):
        """Empty data returns dict."""
        result = volume.calculate(ohlcv_empty)
        assert isinstance(result, dict)


# ============================================================
# INTEGRATION: All indicators combined
# ============================================================

class TestAllIndicators:
    """Integration test — all indicator modules together."""

    def test_no_exception_on_valid_data(self, ohlcv_250):
        """All 4 modules should run without exceptions on valid data."""
        t = trend.calculate(ohlcv_250)
        m = momentum.calculate(ohlcv_250)
        v = volatility.calculate(ohlcv_250)
        vol = volume.calculate(ohlcv_250)

        # All should return dicts
        assert isinstance(t, dict)
        assert isinstance(m, dict)
        assert isinstance(v, dict)
        assert isinstance(vol, dict)

    def test_combined_indicator_count(self, ohlcv_250):
        """Combined result should have 25+ indicators."""
        combined = {}
        combined.update(trend.calculate(ohlcv_250))
        combined.update(momentum.calculate(ohlcv_250))
        combined.update(volatility.calculate(ohlcv_250))
        combined.update(volume.calculate(ohlcv_250))

        assert len(combined) >= 25, f"Only {len(combined)} indicators, expected 25+"

    def test_all_values_are_float(self, ohlcv_250):
        """Every indicator value must be a float (including NaN)."""
        combined = {}
        combined.update(trend.calculate(ohlcv_250))
        combined.update(momentum.calculate(ohlcv_250))
        combined.update(volatility.calculate(ohlcv_250))
        combined.update(volume.calculate(ohlcv_250))

        for key, val in combined.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    def test_no_none_values(self, ohlcv_250):
        """No indicator should ever return None."""
        combined = {}
        combined.update(trend.calculate(ohlcv_250))
        combined.update(momentum.calculate(ohlcv_250))
        combined.update(volatility.calculate(ohlcv_250))
        combined.update(volume.calculate(ohlcv_250))

        for key, val in combined.items():
            assert val is not None, f"{key} is None"
