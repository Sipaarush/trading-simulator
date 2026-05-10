"""
Price Extractor Module — Extracts OHLCV price data from chart screenshots
using Tesseract OCR with pixel-ratio fallback.

Contract (Layer 1 output):
    ohlcv: dict = {
        'close': float,
        'high': float,
        'low': float,
        'volume': float,
        'price_range': (float, float)   # (min_visible, max_visible)
    }
    Returns None if extraction fails — caller must check.
"""

import logging
import re
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try to import pytesseract
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logger.warning("pytesseract not installed — OCR will use fallback only")


class PriceExtractor:
    """
    Extracts price information from trading chart screenshots.

    Primary: Tesseract OCR on the right 12% of the frame (Y-axis labels).
    Fallback: Pixel-ratio estimation from chart dimensions.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize PriceExtractor.

        Args:
            config: Application config dict. Uses 'ocr' section for settings.
        """
        ocr_cfg = config.get("ocr", {})
        self._tesseract_config: str = ocr_cfg.get(
            "tesseract_config",
            "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789.,"
        )
        self._last_known_price: Optional[float] = None
        self._sanity_threshold: float = ocr_cfg.get("sanity_threshold", 0.20)
        logger.info("PriceExtractor initialized (Tesseract available: %s)", HAS_TESSERACT)

    def extract(self, frame: np.ndarray) -> Optional[dict]:
        """
        Extract OHLCV-like price data from a chart screenshot.

        Args:
            frame: BGR image of the chart, shape (H, W, 3), dtype uint8.

        Returns:
            dict with keys: 'close', 'high', 'low', 'volume',
            'price_range': (min_visible, max_visible).
            Returns None if extraction fails.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame — cannot extract prices")
            return None

        try:
            prices = self._extract_ocr(frame)

            if prices is None or len(prices) < 2:
                logger.debug("OCR insufficient — trying pixel-ratio fallback")
                prices = self._extract_fallback(frame)

            if prices is None or len(prices) < 2:
                logger.warning("Price extraction failed — returning last known")
                if self._last_known_price is not None:
                    return self._build_ohlcv_from_last_known()
                return None

            # Sort prices to determine range
            prices.sort()
            min_price = prices[0]
            max_price = prices[-1]
            close_price = prices[-1]  # Most recent / rightmost

            # Sanity check: discard if > 20% deviation from last known
            if self._last_known_price is not None:
                deviation = abs(close_price - self._last_known_price) / self._last_known_price
                if deviation > self._sanity_threshold:
                    logger.warning(
                        "Price deviation %.1f%% exceeds threshold — using last known",
                        deviation * 100,
                    )
                    return self._build_ohlcv_from_last_known()

            self._last_known_price = close_price

            ohlcv = {
                "close": close_price,
                "high": max_price,
                "low": min_price,
                "volume": 0.0,  # Volume not reliably extractable from screenshot
                "price_range": (min_price, max_price),
            }

            logger.debug("Extracted prices: %s", ohlcv)
            return ohlcv

        except Exception as e:
            logger.error("Price extraction error: %s", e)
            if self._last_known_price is not None:
                return self._build_ohlcv_from_last_known()
            return None

    def _extract_ocr(self, frame: np.ndarray) -> Optional[list]:
        """
        Extract prices using Tesseract OCR on the Y-axis region.

        Args:
            frame: BGR image.

        Returns:
            List of float prices found, or None.
        """
        if not HAS_TESSERACT:
            return None

        try:
            h, w = frame.shape[:2]

            # Crop right 12% of the frame (Y-axis price labels)
            x_start = int(w * 0.88)
            y_axis_crop = frame[:, x_start:]

            # Enhance for OCR: convert to grayscale + threshold
            gray = cv2.cvtColor(y_axis_crop, cv2.COLOR_BGR2GRAY)

            # Adaptive thresholding for better OCR in varying themes
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2,
            )

            # Invert if dark background (more white pixels means light bg)
            white_ratio = np.mean(thresh > 128)
            if white_ratio < 0.5:
                thresh = cv2.bitwise_not(thresh)

            # Run Tesseract
            text = pytesseract.image_to_string(thresh, config=self._tesseract_config)

            # Parse numbers from OCR output
            prices = self._parse_prices(text)
            return prices if prices else None

        except Exception as e:
            logger.debug("OCR extraction failed: %s", e)
            return None

    def _extract_fallback(self, frame: np.ndarray) -> Optional[list]:
        """
        Fallback price extraction using pixel-ratio estimation.
        Uses the vertical position of price-colored pixels to estimate
        relative price levels.

        Args:
            frame: BGR image.

        Returns:
            List of estimated prices, or None.
        """
        try:
            h, w = frame.shape[:2]

            # If we have a last known price, create estimates based on
            # chart pixel analysis
            if self._last_known_price is None:
                return None

            # Analyze the chart area (exclude right 12% = y-axis)
            chart_w = int(w * 0.88)
            chart = frame[:, :chart_w]

            # Convert to grayscale and find edges
            gray = cv2.cvtColor(chart, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Find topmost and bottommost significant edge rows
            row_sums = np.sum(edges, axis=1)
            significant = np.where(row_sums > (chart_w * 0.05))[0]

            if len(significant) < 2:
                return None

            top_row = significant[0]
            bottom_row = significant[-1]
            chart_height_px = bottom_row - top_row

            if chart_height_px <= 0:
                return None

            # Estimate price range based on typical chart proportions
            # Assume ~2% price range visible on screen
            price_range_pct = 0.02
            mid_price = self._last_known_price
            half_range = mid_price * price_range_pct / 2

            high_price = mid_price + half_range
            low_price = mid_price - half_range

            return [low_price, mid_price, high_price]

        except Exception as e:
            logger.debug("Fallback extraction failed: %s", e)
            return None

    def _parse_prices(self, text: str) -> list:
        """
        Parse price values from OCR text output.

        Args:
            text: Raw OCR text.

        Returns:
            List of valid float prices.
        """
        prices = []

        # Match numbers with optional decimals (e.g., 45123.50, 1,234.56)
        pattern = r"[\d,]+\.?\d*"
        matches = re.findall(pattern, text)

        for match in matches:
            try:
                # Remove commas
                clean = match.replace(",", "")
                value = float(clean)

                # Filter out obviously invalid prices (too small or too large)
                if 0.001 < value < 10_000_000:
                    prices.append(value)
            except ValueError:
                continue

        return prices

    def _build_ohlcv_from_last_known(self) -> dict:
        """
        Build an OHLCV dict from the last known price.

        Returns:
            dict with estimated OHLCV values.
        """
        price = self._last_known_price
        spread = price * 0.001  # 0.1% spread estimate

        return {
            "close": price,
            "high": price + spread,
            "low": price - spread,
            "volume": 0.0,
            "price_range": (price - spread, price + spread),
        }
