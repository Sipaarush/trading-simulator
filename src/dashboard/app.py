"""
Streamlit Dashboard — Phase 5.

Real-time trading simulator dashboard with:
    - Wide layout
    - Hardware tier badge in header
    - Live annotated chart image (left 70%)
    - Signal card with BUY/SELL badge, metrics (right 30%)
    - Indicator panel with RSI gauge, ADX, MACD, volume ratio
    - Signal history table from SQLite (last 50)
    - Auto-refresh based on hardware tier
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

logger = logging.getLogger(__name__)

try:
    import streamlit as st
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.error("Streamlit not installed — dashboard unavailable")


def run_dashboard() -> None:
    """Launch the Streamlit dashboard."""
    if not ST_AVAILABLE:
        logger.error("Cannot run dashboard — Streamlit not available")
        return

    st.set_page_config(
        page_title="Real-Time AI Trading Simulator",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # --- CSS Styling ---
    st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1a1a2e; border-radius: 10px; padding: 10px; }
    .buy-badge {
        background-color: #006400;
        color: white;
        padding: 8px 20px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
    }
    .sell-badge {
        background-color: #8B0000;
        color: white;
        padding: 8px 20px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
    }
    .wait-badge {
        background-color: #444;
        color: #aaa;
        padding: 8px 20px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
    }
    .tier-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 14px;
    }
    .tier-ultra-low { background-color: #555; color: #ccc; }
    .tier-low { background-color: #1e3a5f; color: #7eb8da; }
    .tier-mid { background-color: #2d5016; color: #8bc34a; }
    .tier-high { background-color: #4a0e4e; color: #ce93d8; }
    </style>
    """, unsafe_allow_html=True)

    # --- Initialize session state ---
    if 'pipeline_data' not in st.session_state:
        st.session_state.pipeline_data = None
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []

    # --- Header ---
    col_title, col_tier = st.columns([4, 1])
    with col_title:
        st.title("📈 Real-Time AI Trading Simulator")
    with col_tier:
        tier = _get_tier()
        tier_class = f"tier-{tier.lower().replace('_', '-')}"
        st.markdown(
            f'<span class="tier-badge {tier_class}">{tier}</span>',
            unsafe_allow_html=True,
        )

    # --- Main Layout ---
    col_chart, col_signal = st.columns([7, 3])

    with col_chart:
        st.subheader("📊 Live Chart")
        chart_placeholder = st.empty()

        # Show latest annotated frame or placeholder
        pipeline = st.session_state.get('pipeline_data')
        if pipeline and 'annotated_frame' in pipeline:
            frame = pipeline['annotated_frame']
            if frame is not None and frame.size > 0:
                import cv2
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                chart_placeholder.image(frame_rgb, use_container_width=True)
            else:
                chart_placeholder.info("⏳ Waiting for chart capture...")
        else:
            chart_placeholder.info("⏳ Waiting for chart capture... Start main.py to begin.")

    with col_signal:
        st.subheader("📡 Signal")

        if pipeline and 'trade' in pipeline and pipeline['trade']:
            trade = pipeline['trade']
            direction = trade.get('direction', 'wait') if isinstance(trade, dict) else getattr(trade, 'direction', 'wait')

            # Direction badge
            badge_class = f"{direction}-badge"
            st.markdown(
                f'<div class="{badge_class}">{direction.upper()}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # Trade metrics
            if isinstance(trade, dict):
                entry = trade.get('entry', 0)
                sl = trade.get('stop_loss', 0)
                tp1 = trade.get('target1', 0)
                conf = trade.get('confidence', 0)
                success = trade.get('success_pct', 0)
                rr = trade.get('rr_ratio', 0)
                pattern = trade.get('pattern_name', 'N/A')
            else:
                entry = getattr(trade, 'entry', 0)
                sl = getattr(trade, 'stop_loss', 0)
                tp1 = getattr(trade, 'target1', 0)
                conf = getattr(trade, 'confidence', 0)
                success = getattr(trade, 'success_pct', 0)
                rr = getattr(trade, 'rr_ratio', 0)
                pattern = getattr(trade, 'pattern_name', 'N/A')

            st.metric("Pattern", pattern)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Entry", f"₹{entry:.2f}")
                st.metric("Stop Loss", f"₹{sl:.2f}")
            with c2:
                st.metric("Target 1", f"₹{tp1:.2f}")
                st.metric("R:R Ratio", f"{rr:.1f}")

            st.metric("Confidence", f"{conf:.0f}%")
            st.metric("Success %", f"{success:.0f}%")
        else:
            st.markdown(
                '<div class="wait-badge">WAIT</div>',
                unsafe_allow_html=True,
            )
            st.info("No active signal — monitoring...")

    # --- Indicator Panel ---
    st.markdown("---")
    st.subheader("📊 Indicators")

    if pipeline and 'indicators' in pipeline:
        ind = pipeline['indicators']
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            rsi = ind.get('rsi', float('nan'))
            rsi_str = f"{rsi:.1f}" if not (isinstance(rsi, float) and rsi != rsi) else "N/A"
            st.metric("RSI (14)", rsi_str)
            adx = ind.get('adx', float('nan'))
            adx_str = f"{adx:.1f}" if not (isinstance(adx, float) and adx != adx) else "N/A"
            st.metric("ADX", adx_str)
        with c2:
            macd = ind.get('macd', float('nan'))
            macd_str = f"{macd:.4f}" if not (isinstance(macd, float) and macd != macd) else "N/A"
            st.metric("MACD", macd_str)
            macd_hist = ind.get('macd_hist', float('nan'))
            hist_str = f"{macd_hist:.4f}" if not (isinstance(macd_hist, float) and macd_hist != macd_hist) else "N/A"
            st.metric("MACD Hist", hist_str)
        with c3:
            vol_ratio = ind.get('volume_ratio', float('nan'))
            vol_str = f"{vol_ratio:.2f}x" if not (isinstance(vol_ratio, float) and vol_ratio != vol_ratio) else "N/A"
            st.metric("Volume Ratio", vol_str)
            atr = ind.get('atr', float('nan'))
            atr_str = f"{atr:.4f}" if not (isinstance(atr, float) and atr != atr) else "N/A"
            st.metric("ATR (14)", atr_str)
        with c4:
            ema9 = ind.get('ema9', float('nan'))
            ema_str = f"{ema9:.2f}" if not (isinstance(ema9, float) and ema9 != ema9) else "N/A"
            st.metric("EMA 9", ema_str)
            bb_w = ind.get('bb_width', float('nan'))
            bb_str = f"{bb_w:.4f}" if not (isinstance(bb_w, float) and bb_w != bb_w) else "N/A"
            st.metric("BB Width", bb_str)
    else:
        st.info("Indicators will appear when pipeline is running.")

    # --- Signal History ---
    st.markdown("---")
    st.subheader("📋 Signal History (Last 50)")

    try:
        from src.database.signal_logger import SignalLogger
        db_logger = SignalLogger({
            'database': {'path': 'signals.db', 'table_name': 'trade_signals'}
        })
        recent = db_logger.get_recent(50)
        if recent:
            import pandas as pd
            df = pd.DataFrame(recent)
            # Color by direction
            st.dataframe(df, use_container_width=True, height=300)
        else:
            st.info("No signal history yet.")
    except Exception:
        st.info("Signal history will appear after first signal is generated.")

    # --- Auto-refresh ---
    refresh_ms = _get_refresh_ms()
    time.sleep(refresh_ms / 1000.0)
    st.rerun()


def _get_tier() -> str:
    """Get the current hardware tier."""
    try:
        from src.core.hardware_profile import detect_hardware
        profile = detect_hardware()
        return profile.tier.value
    except Exception:
        return "UNKNOWN"


def _get_refresh_ms() -> int:
    """Get refresh interval based on hardware tier."""
    tier = _get_tier()
    refresh_map = {
        'MID': 600,
        'HIGH': 600,
        'LOW': 1000,
        'ULTRA_LOW': 1500,
    }
    return refresh_map.get(tier, 1000)


# Entry point for Streamlit
if __name__ == "__main__" or ST_AVAILABLE:
    # Only run if executed via `streamlit run`
    import inspect
    if any('streamlit' in str(f.filename) for f in inspect.stack()):
        run_dashboard()
