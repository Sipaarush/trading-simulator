"""
Indicators Package — Phase 3.

Provides 25+ technical indicators across 4 modules:
    - trend: SMA, EMA, MACD, ADX, Supertrend, Ichimoku, Parabolic SAR
    - momentum: RSI, Stochastic, CCI, Williams %R, ROC, MFI
    - volatility: Bollinger Bands, ATR, Keltner Channel, Donchian Channel
    - volume: Volume MA, Volume Ratio, OBV, VWAP, A/D, CMF

Each module exposes: calculate(ohlcv: dict) -> dict
"""

from src.indicators import trend, momentum, volatility, volume

__all__ = ['trend', 'momentum', 'volatility', 'volume']
