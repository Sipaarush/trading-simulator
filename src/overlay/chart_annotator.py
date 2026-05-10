"""
Chart Annotator Module — Phase 5.

Draws overlay on captured chart frames with trade signal information:
    - Green solid line = entry price
    - Red solid line = stop loss
    - Green dashed lines = TP1, TP2 with price labels
    - Top-right badge: BUY (dark green bg) or SELL (dark red bg)
    - Badge content: direction, pattern name, success%, R:R ratio
    - Bottom bar: entry/SL/TP prices + signal sources (semi-transparent bg)
    - All text has filled dark background rectangle for readability
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available — chart annotation disabled")


def annotate(
    frame: np.ndarray,
    trade: Optional[object] = None,
) -> np.ndarray:
    """
    Annotate a chart frame with trade signal information.

    Args:
        frame: np.ndarray — the original captured frame (BGR).
        trade: TradeSignal dataclass, or None for no annotation.

    Returns:
        np.ndarray — annotated frame (new copy).
    """
    if not CV2_AVAILABLE:
        return frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    if frame is None or frame.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    annotated = frame.copy()

    if trade is None:
        return annotated

    try:
        h, w = annotated.shape[:2]

        # Extract trade info
        direction = getattr(trade, 'direction', 'unknown')
        entry = getattr(trade, 'entry', 0)
        stop_loss = getattr(trade, 'stop_loss', 0)
        target1 = getattr(trade, 'target1', 0)
        target2 = getattr(trade, 'target2', 0)
        success_pct = getattr(trade, 'success_pct', 0)
        rr_ratio = getattr(trade, 'rr_ratio', 0)
        pattern_name = getattr(trade, 'pattern_name', 'Unknown')
        contributing = getattr(trade, 'contributing', [])

        # --- Color scheme ---
        if direction == 'buy':
            badge_color = (0, 100, 0)       # Dark green
            entry_color = (0, 255, 0)        # Green
            sl_color = (0, 0, 255)           # Red
            tp_color = (0, 200, 0)           # Light green
        else:
            badge_color = (0, 0, 139)        # Dark red
            entry_color = (0, 0, 255)        # Red
            sl_color = (0, 255, 0)           # Green
            tp_color = (0, 0, 200)           # Light red

        # --- Price level lines ---
        # Map prices to Y coordinates (approximate)
        prices = [entry, stop_loss, target1, target2]
        min_price = min(prices) * 0.998
        max_price = max(prices) * 1.002
        price_range = max_price - min_price if max_price > min_price else 1.0

        def price_to_y(price: float) -> int:
            """Convert price to Y coordinate (inverted — high price = low Y)."""
            ratio = (price - min_price) / price_range
            return int(h - ratio * h * 0.6 - h * 0.15)

        # Entry line (solid green/red)
        y_entry = price_to_y(entry)
        cv2.line(annotated, (0, y_entry), (w, y_entry), entry_color, 2)
        _draw_label(annotated, f"Entry: {entry:.2f}", (w - 200, y_entry - 5), entry_color)

        # Stop Loss line (solid red/green)
        y_sl = price_to_y(stop_loss)
        cv2.line(annotated, (0, y_sl), (w, y_sl), sl_color, 2)
        _draw_label(annotated, f"SL: {stop_loss:.2f}", (w - 200, y_sl - 5), sl_color)

        # TP1 line (dashed green)
        y_tp1 = price_to_y(target1)
        _draw_dashed_line(annotated, (0, y_tp1), (w, y_tp1), tp_color, 2)
        _draw_label(annotated, f"TP1: {target1:.2f}", (w - 200, y_tp1 - 5), tp_color)

        # TP2 line (dashed green)
        y_tp2 = price_to_y(target2)
        _draw_dashed_line(annotated, (0, y_tp2), (w, y_tp2), tp_color, 1)
        _draw_label(annotated, f"TP2: {target2:.2f}", (w - 200, y_tp2 - 5), tp_color)

        # --- Top-right badge ---
        badge_text = direction.upper()
        badge_w, badge_h = 120, 50
        badge_x = w - badge_w - 10
        badge_y = 10

        # Semi-transparent badge background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (badge_x, badge_y),
                      (badge_x + badge_w, badge_y + badge_h),
                      badge_color, -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        # Badge text
        cv2.putText(
            annotated, badge_text,
            (badge_x + 15, badge_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
        )

        # --- Info badge (below main badge) ---
        info_lines = [
            f"{pattern_name}",
            f"Success: {success_pct:.0f}%",
            f"R:R {rr_ratio:.1f}",
        ]
        info_y = badge_y + badge_h + 10
        for i, line in enumerate(info_lines):
            _draw_label(
                annotated, line,
                (badge_x, info_y + i * 25),
                (200, 200, 200),
                bg_color=(40, 40, 40),
            )

        # --- Bottom bar with trade details ---
        bar_h = 35
        bar_y = h - bar_h

        # Semi-transparent background
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, bar_y), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

        # Bottom bar text
        sources = ", ".join(contributing) if contributing else "N/A"
        bottom_text = (
            f"Entry: {entry:.2f} | SL: {stop_loss:.2f} | "
            f"TP1: {target1:.2f} | TP2: {target2:.2f} | "
            f"Sources: {sources}"
        )
        cv2.putText(
            annotated, bottom_text,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1,
        )

    except Exception as e:
        logger.error("Chart annotation failed: %s", e)

    return annotated


def _draw_label(
    frame: np.ndarray,
    text: str,
    position: tuple,
    color: tuple,
    bg_color: tuple = (30, 30, 30),
    font_scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw text with a filled dark background rectangle for readability."""
    if not CV2_AVAILABLE:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    pad = 4

    # Background rectangle
    cv2.rectangle(
        frame,
        (x - pad, y - text_h - pad),
        (x + text_w + pad, y + baseline + pad),
        bg_color, -1,
    )

    # Text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def _draw_dashed_line(
    frame: np.ndarray,
    pt1: tuple,
    pt2: tuple,
    color: tuple,
    thickness: int = 1,
    dash_length: int = 10,
    gap_length: int = 10,
) -> None:
    """Draw a dashed line between two points."""
    if not CV2_AVAILABLE:
        return

    x1, y1 = pt1
    x2, y2 = pt2

    # For horizontal lines
    if y1 == y2:
        x = x1
        while x < x2:
            end_x = min(x + dash_length, x2)
            cv2.line(frame, (x, y1), (end_x, y1), color, thickness)
            x = end_x + gap_length
    else:
        # General case
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if dist == 0:
            return
        dx = (x2 - x1) / dist
        dy = (y2 - y1) / dist

        d = 0
        while d < dist:
            start = (int(x1 + dx * d), int(y1 + dy * d))
            end_d = min(d + dash_length, dist)
            end = (int(x1 + dx * end_d), int(y1 + dy * end_d))
            cv2.line(frame, start, end, color, thickness)
            d = end_d + gap_length
