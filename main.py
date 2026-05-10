"""
Main Entry Point — Real-Time AI Trading Simulator.

Orchestrates the entire pipeline:
    1. Load config → detect hardware → download model if missing
    2. draw_region_selector() if region is default (0,0,1280,720)
    3. Start FastAPI on port 8000 (daemon thread)
    4. Start Streamlit (subprocess.Popen)
    5. Run pipeline loop:
       capture → preprocess → detect → indicators → signal → overlay → alert

Raises RuntimeError if Python >= 3.12 (TA-Lib wheel constraint).
"""

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── Python version check ──
if sys.version_info >= (3, 12):
    raise RuntimeError(
        f"Python {sys.version_info.major}.{sys.version_info.minor} detected. "
        "This project requires Python 3.11 due to TA-Lib wheel constraints. "
        "Please use Python 3.11."
    )

# ── Setup logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Add project root to path ──
PROJECT_ROOT = str(Path(__file__).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_config() -> dict:
    """Load configuration from config.yaml."""
    import yaml

    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    if not os.path.exists(config_path):
        logger.error("config.yaml not found at %s", config_path)
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded from %s", config_path)
    return config


def setup_hardware(config: dict):
    """Detect hardware and return profile."""
    from src.core.hardware_profile import detect_hardware

    force_tier = config.get('hardware', {}).get('force_tier', '')
    profile = detect_hardware(force_tier=force_tier if force_tier else None)
    return profile


def ensure_model(config: dict) -> None:
    """Download YOLOv8 model if not present."""
    model_path = config.get('model', {}).get('model_path', 'models/best.pt')

    # Check for yolov8n.pt in root (used as fallback)
    if os.path.exists('yolov8n.pt'):
        logger.info("YOLOv8 model found: yolov8n.pt")
        return

    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)

    if os.path.exists(os.path.join(PROJECT_ROOT, model_path)):
        logger.info("Model found at %s", model_path)
        return

    logger.info("Downloading YOLOv8n model...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        logger.info("YOLOv8n model downloaded successfully")
    except Exception as e:
        logger.warning("Model download failed: %s — will use CPU inference", e)


def start_fastapi(config: dict) -> None:
    """Start FastAPI server on port 8000 as daemon thread."""
    def _run_api():
        try:
            import uvicorn
            from fastapi import FastAPI

            app = FastAPI(title="Trading Simulator API", version="1.0")

            @app.get("/")
            def root():
                return {"status": "running", "version": "1.0"}

            @app.get("/health")
            def health():
                return {"status": "healthy"}

            @app.get("/signal")
            def get_signal():
                return {"signal": "No active signal"}

            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
        except Exception as e:
            logger.error("FastAPI failed to start: %s", e)

    api_thread = threading.Thread(target=_run_api, daemon=True)
    api_thread.start()
    logger.info("FastAPI started on port 8000 (daemon thread)")


def start_streamlit() -> subprocess.Popen:
    """Start Streamlit dashboard as subprocess."""
    try:
        dashboard_path = os.path.join(PROJECT_ROOT, "src", "dashboard", "app.py")
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", dashboard_path,
             "--server.port", "8501",
             "--server.headless", "true",
             "--browser.gatherUsageStats", "false"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info("Streamlit dashboard started — http://localhost:8501")
        return process
    except Exception as e:
        logger.error("Failed to start Streamlit: %s", e)
        return None


def run_pipeline(config: dict, profile) -> None:
    """
    Main pipeline loop.

    capture → preprocess → detect → indicators → signal → overlay → alert → log
    """
    # ── Import all modules ──
    from src.capture.screen_capture import ScreenCapture
    from src.capture.preprocessor import preprocess
    from src.ocr.price_extractor import PriceExtractor
    from src.detection.yolo_detector import YOLODetector
    from src.detection.rule_detector import RuleDetector
    from src.indicators import trend, momentum, volatility, volume
    from src.signals.confluence import WeightedVoter
    from src.signals.risk_calculator import RiskCalculator
    from src.signals.ml_predictor import MLPredictor
    from src.overlay.chart_annotator import annotate
    from src.alerts.telegram_bot import TelegramBot
    from src.database.signal_logger import SignalLogger

    # ── Initialize components ──
    capture = ScreenCapture(config)
    price_extractor = PriceExtractor(config)
    yolo_detector = YOLODetector(config, profile)
    rule_detector = RuleDetector(config)
    voter = WeightedVoter(config)
    risk_calc = RiskCalculator(config)
    ml_predictor = MLPredictor(config)
    telegram = TelegramBot(config)
    db_logger = SignalLogger(config)

    logger.info("All pipeline components initialized")
    logger.info("Starting pipeline loop — interval=%dms", profile.capture_interval_ms)

    # ── Start capture ──
    capture.start()

    try:
        while True:
            loop_start = time.time()

            try:
                # --- 1. CAPTURE ---
                frame = capture.get_frame(timeout=2.0)
                if frame is None:
                    time.sleep(0.1)
                    continue

                # --- 2. PREPROCESS ---
                processed, original_crop = preprocess(frame, config)

                # --- 3. OCR (price extraction) ---
                ohlcv = price_extractor.extract(frame)
                if ohlcv is None:
                    logger.debug("OCR failed — skipping frame")
                    time.sleep(0.1)
                    continue

                # --- 4. DETECT ---
                # YOLOv8 (GPU thread)
                yolo_patterns = yolo_detector.detect(processed)

                # TA-Lib rules (CPU thread)
                rule_patterns = rule_detector.detect(ohlcv)

                # Combine patterns
                all_patterns = yolo_patterns + rule_patterns

                # --- 5. INDICATORS ---
                indicators = {}
                indicators.update(trend.calculate(ohlcv))
                indicators.update(momentum.calculate(ohlcv))
                indicators.update(volatility.calculate(ohlcv))
                indicators.update(volume.calculate(ohlcv))

                # --- 6. CONFLUENCE VOTING ---
                signal = voter.vote(ohlcv, all_patterns, indicators)

                # --- 7. RISK CALCULATION ---
                trade = risk_calc.calculate(signal, ohlcv)

                # --- 8. ML PREDICTION (optional enhancement) ---
                if trade:
                    ml_result = ml_predictor.predict(indicators, signal.score)
                    # Log ML prediction alongside trade
                    logger.debug("ML prediction: %s", ml_result)

                # --- 9. OVERLAY ---
                annotated_frame = annotate(original_crop, trade)

                # --- 10. ALERTS & LOGGING ---
                if trade:
                    # Log to database
                    signal_id = db_logger.log(trade)

                    # Send Telegram alert
                    telegram.send(trade, annotated_frame)

                    logger.info(
                        "🔔 SIGNAL: %s %s | Entry=%.2f | SL=%.2f | TP=%.2f | "
                        "Conf=%.0f%% | Success=%.0f%%",
                        trade.direction.upper(),
                        trade.pattern_name,
                        trade.entry,
                        trade.stop_loss,
                        trade.target1,
                        trade.confidence,
                        trade.success_pct,
                    )

                # --- Update Streamlit state ---
                try:
                    import streamlit as st
                    if hasattr(st, 'session_state'):
                        st.session_state.pipeline_data = {
                            'annotated_frame': annotated_frame,
                            'trade': trade,
                            'indicators': indicators,
                            'patterns': all_patterns,
                        }
                except Exception:
                    pass  # Streamlit not in this thread

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error("Pipeline iteration error: %s", e)

            # --- Timing ---
            elapsed_ms = (time.time() - loop_start) * 1000
            sleep_ms = max(0, profile.capture_interval_ms - elapsed_ms)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

            logger.debug("Pipeline loop: %.0fms (target: %dms)", elapsed_ms, profile.capture_interval_ms)

    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user (Ctrl+C)")
    finally:
        capture.stop()
        logger.info("Capture stopped")


def main() -> None:
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Real-Time AI Trading Simulator v1.0")
    logger.info("=" * 60)

    # 1. Load config
    config = load_config()
    if not config:
        logger.error("Failed to load config — exiting")
        sys.exit(1)

    # 2. Detect hardware
    profile = setup_hardware(config)
    logger.info("Hardware tier: %s", profile.tier.value)

    # 3. Ensure model exists
    ensure_model(config)

    # 4. Region selector (if default)
    capture_cfg = config.get('capture', {}).get('region', {})
    if (
        capture_cfg.get('top', 0) == 0
        and capture_cfg.get('left', 0) == 0
        and capture_cfg.get('width', 1280) == 1280
        and capture_cfg.get('height', 720) == 720
    ):
        logger.info("Using default capture region: (0, 0, 1280, 720)")
        logger.info("To change, run region selector or edit config.yaml")

    # 5. Start FastAPI
    start_fastapi(config)

    # 6. Start Streamlit
    streamlit_proc = start_streamlit()

    # 7. Run pipeline
    try:
        run_pipeline(config, profile)
    except Exception as e:
        logger.error("Pipeline crashed: %s", e)
    finally:
        # Cleanup
        if streamlit_proc:
            streamlit_proc.terminate()
            logger.info("Streamlit process terminated")

        logger.info("Simulator shut down")


if __name__ == "__main__":
    main()
