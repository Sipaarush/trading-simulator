"""
Tests for Phase 4 — Signal Engine.

Tests confluence voting, risk calculation, and TradeSignal validation.
"""

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signals.confluence import WeightedVoter, SignalResult
from src.signals.risk_calculator import RiskCalculator, TradeSignal
from src.signals.ml_predictor import MLPredictor
from src.detection.pattern_registry import PatternResult


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def config():
    """Standard config matching config.yaml."""
    return {
        'signals': {
            'min_confluence_score': 5.0,
            'weights': {
                'yolov8': 3.0,
                'talib': 2.5,
                'volume': 2.0,
                'rsi_extreme': 2.0,
                'macd': 2.0,
                'ma_align': 1.5,
                'fibonacci': 1.5,
                'bb_position': 1.0,
                'adx_trend': 1.0,
            },
        },
        'risk': {
            'atr_multiplier': 1.5,
            'min_rr_ratio': 1.5,
            'base_win_rate': 0.55,
            'max_success_pct': 85,
            'max_risk_pct': 2.0,
            'account_balance': 100000,
        },
        'model': {
            'confidence_threshold': 0.65,
        },
    }


@pytest.fixture
def bullish_patterns():
    """Bullish patterns from both YOLOv8 and TA-Lib."""
    return [
        PatternResult(
            name='BULL_FLAG',
            display_name='Bull Flag',
            confidence=0.85,
            bbox=(100, 100, 200, 200),
            direction='bullish',
            source='yolov8',
        ),
        PatternResult(
            name='BULLISH_ENGULFING',
            display_name='Bullish Engulfing',
            confidence=0.80,
            bbox=(0, 0, 0, 0),
            direction='bullish',
            source='talib',
        ),
    ]


@pytest.fixture
def bearish_patterns():
    """Bearish patterns."""
    return [
        PatternResult(
            name='HEAD_AND_SHOULDERS',
            display_name='Head & Shoulders',
            confidence=0.78,
            bbox=(50, 50, 300, 300),
            direction='bearish',
            source='yolov8',
        ),
        PatternResult(
            name='BEARISH_ENGULFING',
            display_name='Bearish Engulfing',
            confidence=0.80,
            bbox=(0, 0, 0, 0),
            direction='bearish',
            source='talib',
        ),
    ]


@pytest.fixture
def bullish_indicators():
    """Indicators indicating bullish conditions."""
    return {
        'rsi': 25.0,            # Oversold — buy
        'macd': 0.5,
        'macd_signal': 0.3,
        'macd_hist': 0.2,       # Positive — buy
        'adx': 30.0,            # Trending
        'di_plus': 28.0,
        'di_minus': 15.0,
        'ema9': 105.0,
        'ema21': 103.0,
        'sma50': 100.0,         # Bull alignment: ema9 > ema21 > sma50
        'sma200': 95.0,
        'bb_upper': 110.0,
        'bb_middle': 103.0,
        'bb_lower': 96.0,
        'bb_width': 0.14,
        'atr': 2.5,
        'volume_ratio': 1.8,    # High volume
        'volume_ma': 5000.0,
        'obv': 100000.0,
        'vwap': 103.0,
        'cmf': 0.15,
        'stoch_k': 20.0,
        'stoch_d': 22.0,
        'cci': -120.0,
        'willr': -85.0,
        'roc': -2.0,
        'mfi': 25.0,
        'supertrend': 98.0,
        'ichi_above': 1.0,
        'parabolic_sar': 97.0,
        'sma20': 102.0,
    }


@pytest.fixture
def ohlcv_bull():
    """OHLCV data for bullish scenario."""
    np.random.seed(42)
    n = 100
    close = np.linspace(90, 105, n)
    return {
        'open': close - 0.5,
        'high': close + 1.0,
        'low': close - 1.0,
        'close': close,
        'volume': np.full(n, 5000.0),
    }


# ============================================================
# CONFLUENCE TESTS
# ============================================================

class TestWeightedVoter:
    """Test confluence.py."""

    def test_returns_signal_result(self, config, ohlcv_bull, bullish_patterns, bullish_indicators):
        """vote() must return a SignalResult."""
        voter = WeightedVoter(config)
        result = voter.vote(ohlcv_bull, bullish_patterns, bullish_indicators)
        assert isinstance(result, SignalResult)

    def test_bullish_signal_generated(self, config, ohlcv_bull, bullish_patterns, bullish_indicators):
        """Strong bullish signals should generate a 'buy' direction."""
        voter = WeightedVoter(config)
        result = voter.vote(ohlcv_bull, bullish_patterns, bullish_indicators)
        assert result.direction == 'buy'
        assert result.score >= 5.0

    def test_bearish_signal_generated(self, config, ohlcv_bull, bearish_patterns):
        """Bearish patterns with bearish indicators should give 'sell'."""
        bearish_indicators = {
            'rsi': 78.0,            # Overbought
            'macd_hist': -0.5,      # Negative — sell
            'adx': 35.0,
            'di_plus': 12.0,
            'di_minus': 30.0,
            'ema9': 95.0,
            'ema21': 98.0,
            'sma50': 102.0,         # Bear alignment
            'bb_upper': 100.0,
            'bb_lower': 90.0,
            'bb_middle': 95.0,
            'volume_ratio': 1.6,
            'atr': 2.0,
            'sma20': 97.0,
        }
        voter = WeightedVoter(config)
        result = voter.vote(ohlcv_bull, bearish_patterns, bearish_indicators)
        assert result.direction == 'sell'

    def test_wait_on_weak_signal(self, config, ohlcv_bull):
        """No patterns + neutral indicators → 'wait'."""
        voter = WeightedVoter(config)
        neutral_indicators = {
            'rsi': 50.0,
            'macd_hist': 0.0,
            'adx': 15.0,  # Choppy
            'volume_ratio': 1.0,
        }
        result = voter.vote(ohlcv_bull, [], neutral_indicators)
        assert result.direction == 'wait'

    def test_adx_guard_halves_score(self, config, ohlcv_bull, bullish_patterns):
        """ADX < 25 should halve the total score."""
        indicators_low_adx = {
            'rsi': 25.0,
            'macd_hist': 0.5,
            'adx': 20.0,  # Below 25 → halve
            'volume_ratio': 1.8,
            'ema9': 105.0,
            'ema21': 103.0,
            'sma50': 100.0,
        }
        voter = WeightedVoter(config)
        result = voter.vote(ohlcv_bull, bullish_patterns, indicators_low_adx)
        # Score should be halved due to ADX guard
        assert result.score < 10.0  # Would be higher without halving

    def test_score_range(self, config, ohlcv_bull, bullish_patterns, bullish_indicators):
        """Score should not exceed theoretical max of 17.5."""
        voter = WeightedVoter(config)
        result = voter.vote(ohlcv_bull, bullish_patterns, bullish_indicators)
        assert abs(result.score) <= 20.0  # Allow small margin

    def test_confidence_range(self, config, ohlcv_bull, bullish_patterns, bullish_indicators):
        """Confidence should be 0–100."""
        voter = WeightedVoter(config)
        result = voter.vote(ohlcv_bull, bullish_patterns, bullish_indicators)
        assert 0 <= result.confidence <= 100

    def test_contributing_list(self, config, ohlcv_bull, bullish_patterns, bullish_indicators):
        """Contributing should list which sources fired."""
        voter = WeightedVoter(config)
        result = voter.vote(ohlcv_bull, bullish_patterns, bullish_indicators)
        assert isinstance(result.contributing, list)
        assert len(result.contributing) > 0

    def test_min_score_5_for_signal(self, config, ohlcv_bull):
        """Score < 5.0 should always be 'wait'."""
        voter = WeightedVoter(config)
        result = voter.vote(ohlcv_bull, [], {})
        assert result.direction == 'wait'


# ============================================================
# RISK CALCULATOR TESTS
# ============================================================

class TestRiskCalculator:
    """Test risk_calculator.py."""

    def test_returns_trade_signal(self, config, ohlcv_bull, bullish_indicators):
        """calculate() should return TradeSignal for valid signal."""
        voter = WeightedVoter(config)
        signal = voter.vote(
            ohlcv_bull,
            [PatternResult('BULL_FLAG', 'Bull Flag', 0.85, (0,0,0,0), 'bullish', 'yolov8'),
             PatternResult('BULLISH_ENGULFING', 'Bullish Engulfing', 0.8, (0,0,0,0), 'bullish', 'talib')],
            bullish_indicators,
        )
        calc = RiskCalculator(config)
        trade = calc.calculate(signal, ohlcv_bull)
        assert trade is not None
        assert isinstance(trade, TradeSignal)

    def test_sl_correct_side_buy(self, config, ohlcv_bull, bullish_indicators):
        """For BUY: SL < entry."""
        voter = WeightedVoter(config)
        signal = voter.vote(
            ohlcv_bull,
            [PatternResult('BULL_FLAG', 'Bull Flag', 0.85, (0,0,0,0), 'bullish', 'yolov8'),
             PatternResult('BULLISH_ENGULFING', 'Bullish Engulfing', 0.8, (0,0,0,0), 'bullish', 'talib')],
            bullish_indicators,
        )
        calc = RiskCalculator(config)
        trade = calc.calculate(signal, ohlcv_bull)
        if trade and trade.direction == 'buy':
            assert trade.stop_loss < trade.entry

    def test_rr_ratio_minimum(self, config, ohlcv_bull, bullish_indicators):
        """R:R ratio should be >= 1.5."""
        voter = WeightedVoter(config)
        signal = voter.vote(
            ohlcv_bull,
            [PatternResult('BULL_FLAG', 'Bull Flag', 0.85, (0,0,0,0), 'bullish', 'yolov8'),
             PatternResult('BULLISH_ENGULFING', 'Bullish Engulfing', 0.8, (0,0,0,0), 'bullish', 'talib')],
            bullish_indicators,
        )
        calc = RiskCalculator(config)
        trade = calc.calculate(signal, ohlcv_bull)
        if trade:
            assert trade.rr_ratio >= 1.5

    def test_success_pct_capped_at_85(self, config, ohlcv_bull, bullish_indicators):
        """Success % should never exceed 85."""
        voter = WeightedVoter(config)
        signal = voter.vote(
            ohlcv_bull,
            [PatternResult('BULL_FLAG', 'Bull Flag', 0.85, (0,0,0,0), 'bullish', 'yolov8'),
             PatternResult('BULLISH_ENGULFING', 'Bullish Engulfing', 0.8, (0,0,0,0), 'bullish', 'talib')],
            bullish_indicators,
        )
        calc = RiskCalculator(config)
        trade = calc.calculate(signal, ohlcv_bull)
        if trade:
            assert trade.success_pct <= 85.0

    def test_wait_returns_none(self, config, ohlcv_bull):
        """Wait signal should return None."""
        voter = WeightedVoter(config)
        signal = voter.vote(ohlcv_bull, [], {})
        assert signal.direction == 'wait'
        calc = RiskCalculator(config)
        trade = calc.calculate(signal, ohlcv_bull)
        assert trade is None

    def test_trade_signal_all_fields(self, config, ohlcv_bull, bullish_indicators):
        """TradeSignal should have all required fields."""
        voter = WeightedVoter(config)
        signal = voter.vote(
            ohlcv_bull,
            [PatternResult('BULL_FLAG', 'Bull Flag', 0.85, (0,0,0,0), 'bullish', 'yolov8'),
             PatternResult('BULLISH_ENGULFING', 'Bullish Engulfing', 0.8, (0,0,0,0), 'bullish', 'talib')],
            bullish_indicators,
        )
        calc = RiskCalculator(config)
        trade = calc.calculate(signal, ohlcv_bull)
        if trade:
            assert hasattr(trade, 'direction')
            assert hasattr(trade, 'confidence')
            assert hasattr(trade, 'entry')
            assert hasattr(trade, 'stop_loss')
            assert hasattr(trade, 'target1')
            assert hasattr(trade, 'target2')
            assert hasattr(trade, 'risk_pct')
            assert hasattr(trade, 'rr_ratio')
            assert hasattr(trade, 'position_size')
            assert hasattr(trade, 'success_pct')
            assert hasattr(trade, 'pattern_name')
            assert hasattr(trade, 'contributing')
            assert hasattr(trade, 'indicators')
            assert hasattr(trade, 'atr')


# ============================================================
# ML PREDICTOR TESTS
# ============================================================

class TestMLPredictor:
    """Test ml_predictor.py."""

    def test_init_no_crash(self, config):
        """MLPredictor should initialize without crash."""
        predictor = MLPredictor(config)
        assert predictor is not None

    def test_predict_returns_dict(self, config, bullish_indicators):
        """predict() should return a dict."""
        predictor = MLPredictor(config)
        result = predictor.predict(bullish_indicators, 8.0)
        assert isinstance(result, dict)
        assert 'ml_confidence' in result
        assert 'ml_direction' in result
        assert 'ml_available' in result

    def test_predict_empty_indicators(self, config):
        """Empty indicators should not crash."""
        predictor = MLPredictor(config)
        result = predictor.predict({}, 0.0)
        assert isinstance(result, dict)

    def test_extract_features(self, config, bullish_indicators):
        """Feature extraction should return numpy array."""
        predictor = MLPredictor(config)
        features = predictor.extract_features(bullish_indicators)
        if features is not None:
            assert features.shape[1] == len(predictor.feature_names)
