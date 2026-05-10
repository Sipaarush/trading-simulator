"""
Telegram Bot Module — Phase 5.

Sends trade signal notifications with annotated chart screenshots.

Rules:
    - Token from os.getenv('TELEGRAM_TOKEN') — never from config.yaml
    - Message: emoji + direction + all TradeSignal fields formatted clearly
    - Attach annotated frame as JPEG (cv2 encode, quality 85)
    - Disabled silently if token/chat_id not set
"""

import io
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Telegram notification bot for trade signals.

    Sends formatted messages and annotated chart screenshots
    to a configured Telegram chat.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize Telegram bot.

        Args:
            config: Full config dict (token loaded from .env, NOT config).
        """
        from dotenv import load_dotenv
        load_dotenv()

        self.token = os.getenv('TELEGRAM_TOKEN', '').strip()
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '').strip()
        self.enabled = bool(self.token and self.chat_id)

        alerts_cfg = config.get('alerts', {})
        self.telegram_enabled = alerts_cfg.get('telegram_enabled', True)

        if not self.enabled:
            logger.info("Telegram bot disabled — token or chat_id not set")
        elif not self.telegram_enabled:
            logger.info("Telegram alerts disabled in config")
            self.enabled = False
        else:
            logger.info("Telegram bot initialized — chat_id=%s", self.chat_id)

    def send(
        self,
        trade: object,
        annotated_frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Send trade signal notification to Telegram.

        Args:
            trade: TradeSignal dataclass.
            annotated_frame: Annotated chart frame (BGR, np.ndarray).

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.enabled:
            return False

        try:
            import requests

            # --- Build message ---
            message = self._format_message(trade)

            # --- Send message ---
            msg_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            msg_payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML',
            }
            msg_response = requests.post(msg_url, data=msg_payload, timeout=10)

            if msg_response.status_code != 200:
                logger.error(
                    "Telegram message failed: %s %s",
                    msg_response.status_code, msg_response.text,
                )
                return False

            logger.info("Telegram message sent successfully")

            # --- Send screenshot if available ---
            if annotated_frame is not None and annotated_frame.size > 0:
                self._send_photo(annotated_frame)

            return True

        except ImportError:
            logger.warning("requests library not installed — cannot send Telegram")
            return False
        except Exception as e:
            logger.error("Telegram send failed: %s", e)
            return False

    def _format_message(self, trade: object) -> str:
        """Format TradeSignal into a readable Telegram message."""
        direction = getattr(trade, 'direction', 'unknown')
        entry = getattr(trade, 'entry', 0)
        stop_loss = getattr(trade, 'stop_loss', 0)
        target1 = getattr(trade, 'target1', 0)
        target2 = getattr(trade, 'target2', 0)
        risk_pct = getattr(trade, 'risk_pct', 0)
        rr_ratio = getattr(trade, 'rr_ratio', 0)
        success_pct = getattr(trade, 'success_pct', 0)
        confidence = getattr(trade, 'confidence', 0)
        pattern_name = getattr(trade, 'pattern_name', 'N/A')
        contributing = getattr(trade, 'contributing', [])
        position_size = getattr(trade, 'position_size', 0)

        # Emoji
        if direction == 'buy':
            emoji = "🟢🔼"
            dir_label = "BUY"
        else:
            emoji = "🔴🔽"
            dir_label = "SELL"

        sources = ", ".join(contributing) if contributing else "N/A"

        message = (
            f"{emoji} <b>{dir_label} SIGNAL</b> {emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📊 <b>Pattern:</b> {pattern_name}\n"
            f"💰 <b>Entry:</b> ₹{entry:.2f}\n"
            f"🛑 <b>Stop Loss:</b> ₹{stop_loss:.2f}\n"
            f"🎯 <b>Target 1:</b> ₹{target1:.2f}\n"
            f"🎯 <b>Target 2:</b> ₹{target2:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ <b>Risk:</b> {risk_pct:.2f}%\n"
            f"📐 <b>R:R Ratio:</b> {rr_ratio:.1f}\n"
            f"✅ <b>Success:</b> {success_pct:.0f}%\n"
            f"🔒 <b>Confidence:</b> {confidence:.0f}%\n"
            f"📦 <b>Position:</b> {position_size:.0f} units\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📡 <b>Sources:</b> {sources}\n"
            f"🤖 <i>AI Trading Simulator v1.0</i>"
        )

        return message

    def _send_photo(self, frame: np.ndarray) -> bool:
        """Send annotated chart as photo to Telegram."""
        try:
            import cv2
            import requests

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            photo_bytes = io.BytesIO(buffer.tobytes())
            photo_bytes.name = 'chart.jpg'

            photo_url = f"https://api.telegram.org/bot{self.token}/sendPhoto"
            files = {'photo': photo_bytes}
            data = {
                'chat_id': self.chat_id,
                'caption': '📊 Annotated Chart',
            }
            response = requests.post(photo_url, files=files, data=data, timeout=15)

            if response.status_code == 200:
                logger.info("Telegram photo sent successfully")
                return True
            else:
                logger.error(
                    "Telegram photo failed: %s %s",
                    response.status_code, response.text,
                )
                return False

        except Exception as e:
            logger.error("Telegram photo send failed: %s", e)
            return False
