# 🤖 Real-Time AI Trading Simulator v1.0

A local desktop application that reads a live trading chart directly from your screen, detects chart patterns using computer vision (YOLOv8 + TA-Lib), calculates 25+ technical indicators, and provides instant actionable trade signals — entry, stop-loss, take-profit, risk %, and success probability.

**All processing is 100% local — no cloud, no API keys needed for core functionality.**

---

## 🚀 How to Run (Every Time)

### Step 1: Open Terminal in the project folder
```bash
cd C:\Users\Priyanshu\Downloads\github\trading-simulator
```

### Step 2: Activate the virtual environment
```bash
venv\Scripts\activate
```

### Step 3: Open your trading chart
Open **TradingView**, **MT4**, **Zerodha Kite**, or any chart platform in your browser/app.

### Step 4: Run the simulator
```bash
python main.py
```

### What happens when you run it:
1. ⚙️ **Hardware auto-detected** → picks the right model for your GPU
2. 📸 **Screen capture starts** → reads the chart region every 500ms
3. 🔍 **Pattern detection** → YOLOv8 (GPU) + TA-Lib (CPU) scan for 35+ patterns
4. 📊 **Indicators calculated** → RSI, MACD, Bollinger Bands, etc.
5. 🎯 **Signal generated** → BUY/SELL with entry, SL, TP, confidence
6. 📱 **Telegram alert sent** → signal + annotated screenshot
7. 🖥️ **Dashboard opens** → http://localhost:8501

### Step 5: View the dashboard
Open your browser to **http://localhost:8501** to see:
- Live annotated chart
- BUY/SELL signal card
- Indicator gauges (RSI, ADX, MACD)
- Signal history table

### Step 6: Stop the simulator
Press **Ctrl+C** in the terminal.

---

## 📁 Quick Reference

| What | Where |
|------|-------|
| Run the app | `python main.py` |
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| Config | `config.yaml` |
| Telegram setup | `.env` file |
| Run tests | `pytest tests/ -v` |
| Signal database | `signals.db` |

---

## ⚙️ Configuration (`config.yaml`)

### Change capture region
Edit `config.yaml` → `capture.region`:
```yaml
capture:
  region:
    top: 0
    left: 0
    width: 1280
    height: 720
```
Or delete the region values and re-run — it will open a region selector.

### Force a hardware tier
```yaml
hardware:
  force_tier: "MID"   # ULTRA_LOW, LOW, MID, HIGH
```

### Adjust signal sensitivity
```yaml
signals:
  min_confluence_score: 5.0   # Lower = more signals, Higher = fewer but stronger
```

---

## 🖥️ Hardware Tiers (Auto-Detected)

| Tier | GPU | Model | Speed |
|------|-----|-------|-------|
| ULTRA_LOW | No GPU / Intel HD | YOLOv8n 416px (CPU) | ~1800ms |
| LOW | GTX 1650 | YOLOv8n 416px (CUDA FP16) | ~600ms |
| **MID ★** | **RTX 2050** | **YOLOv8s 640px (CUDA FP16)** | **~380ms** |
| HIGH | RTX 3060+ | YOLOv8s 640px (CUDA FP16) | ~250ms |

Your system: **MID tier** (RTX 2050, 4GB VRAM, 16GB RAM)

---

## 📱 Telegram Alerts

Signals are auto-sent to your Telegram. To change:
1. Edit `.env`:
   ```
   TELEGRAM_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```
2. If `.env` is missing or empty, Telegram is silently disabled (no crash).

---

## 🧪 Run Tests

```bash
venv\Scripts\activate
pytest tests/ -v
```

Expected: **128 passed** ✅

---

## 📦 First-Time Setup (Already Done)

```bash
git clone https://github.com/Sipaarush/trading-simulator.git
cd trading-simulator
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install TA-Lib --find-links https://github.com/cgohlke/talib-build/releases
pip install streamlit sqlalchemy uvicorn fastapi
```

---

## 🏗️ Project Structure

```
trading-simulator/
├── main.py                    # Entry point — starts everything
├── config.yaml                # All settings (auto-adapts to hardware)
├── .env                       # Telegram token (private)
├── requirements.txt
├── models/                    # YOLOv8 weights
├── src/
│   ├── core/hardware_profile.py       # GPU/RAM detect → tier
│   ├── capture/screen_capture.py      # mss screen grab
│   ├── capture/preprocessor.py        # CLAHE + blur + resize
│   ├── ocr/price_extractor.py         # Tesseract OCR
│   ├── detection/pattern_registry.py  # 35+ pattern metadata
│   ├── detection/yolo_detector.py     # YOLOv8 GPU inference
│   ├── detection/rule_detector.py     # TA-Lib candlestick rules
│   ├── indicators/trend.py            # SMA/EMA/MACD/ADX
│   ├── indicators/momentum.py         # RSI/Stoch/CCI
│   ├── indicators/volatility.py       # BB/ATR/Keltner
│   ├── indicators/volume.py           # OBV/VWAP/CMF
│   ├── signals/confluence.py          # Weighted voting engine
│   ├── signals/risk_calculator.py     # SL/TP/risk%/success%
│   ├── signals/ml_predictor.py        # XGBoost + LSTM (optional)
│   ├── overlay/chart_annotator.py     # OpenCV draw lines/badges
│   ├── alerts/telegram_bot.py         # Send signal + screenshot
│   ├── database/signal_logger.py      # SQLite CRUD
│   └── dashboard/app.py              # Streamlit real-time UI
└── tests/                            # 128 acceptance tests
```

---

**Team:** Priyanshu (Phase 1-2) + Janvi (Phase 3-5) + Antigravity AI