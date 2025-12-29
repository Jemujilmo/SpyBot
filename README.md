# Market Copilot for SPY

A Python-based decision support system for SPY options trading that reconstructs technical context from raw market data—no screen scraping, no broker automation.

## ⚡ Quick Start

**New users: Get up and running in 3 steps!**

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd spytradebot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the system
python market_copilot.py
```
# terminal command is "cd "C:\Users\Kryst\Code Stuff\spytradebot"
.venv\Scripts\python.exe flask_app.py --port=5050"
---

### Web dashboard (Flask + Plotly) — quick commands

If you want the interactive browser dashboard with Plotly charts, run the Flask app below.

**Two Dashboard Modes Available:**

1. **Multi-Timeframe View** (Default - http://localhost:5050/)
   - Shows 1-minute, 5-minute, and 15-minute charts simultaneously
   - Best for active trading and quick timeframe comparison
   - All charts update in real-time

2. **Single-Chart Indicator View** (http://localhost:5050/indicator)
   - Focused single chart view with persistent zoom/pan
   - Chart state saved in browser (survives page refresh)
   - Cleaner interface for monitoring one timeframe

Using a virtualenv (recommended):

```bash
# create and activate a venv (unix/mac)
python3 -m venv .venv
source .venv/bin/activate

# install requirements
pip install -r requirements.txt

# run the dashboard on the default port (5000)
python flask_app.py

# or run on a custom port (example 5050)
python flask_app.py --port=5050

# or set via env var
FLASK_RUN_PORT=5050 python flask_app.py
```

If you already have a virtualenv created by the project, run using its python explicitly:

```bash
./.venv/bin/python flask_app.py --port=5050
```

**Web Dashboard Features:**
- ✅ **Multi-ticker support** - Track any US stock or ETF (SPY, QQQ, AAPL, TSLA, etc.)
  - Library of 30+ popular tickers (major ETFs, FAANG+, sector funds)
  - Optional: Fetch complete Nasdaq/NYSE ticker lists (3000+ symbols)
  - Switch tickers without additional API requests
- ✅ **Real-time candlestick charts** with VWAP, EMA9, EMA21 overlays
- ✅ **Volume bars** (color-coded green/red)
- ✅ **Market status indicator** (🟢 Open / 🔴 Closed)
- ✅ **Auto-refresh every 2 seconds** with time-since-update counter
- ✅ **Real options data** from live options chains:
  - Options walls (support/resistance from open interest)
  - IV Rank & IV Percentile
  - Put/Call ratio (volume & open interest based)
  - Gamma exposure levels (GEX)
- ✅ **Algorithmic entry signals** with multi-factor analysis
  - 🟢 BUY signals (call entry points) - green upward triangles
  - 🔴 SELL signals (put entry points) - red downward triangles
  - 4-factor scoring: Bias + VWAP + EMA + RSI (0-100 points)
  - Time-filtered: Only signals during prime hours (9:45 AM - 3:45 PM ET)
  - Strength percentage displayed (60%+ threshold)
- ✅ **Gamma squeeze indicator** (volume + volatility + momentum)
- ✅ **5m and 15m bias** with confidence scoring
- ✅ **Interactive Plotly charts** (zoom, pan, hover for details)
- ✅ **12-hour time format** with AM/PM (standard, not military)

Notes:
- The app prints the full URL it is serving (e.g. http://localhost:5050). If port 5000 is busy on your machine, use --port to choose another port.
- The API endpoints the UI uses are `/api/analysis` and `/test` for quick connectivity checks.

<!-- Important notes callout -->
<div style="background: #fff9db; border-left: 4px solid #ffd54a; padding: 12px; margin: 12px 0;">
    <strong>⚠️ Important notes — things to consider / be aware of</strong>
    <ul>
        <li><strong>Cold-start behaviour</strong>: the server may return a transient HTTP 500 on the very first API request while background data is being built. The UI now uses safe fallbacks, but expect a short delay on the first load.</li>
        <li><strong>Development server only</strong>: this project uses Flask's development server (debug + reloader). It's fine for local use only — do not expose it to production networks.</li>
        <li><strong>Background builder</strong>: a background thread builds cached payloads. Check <code>dashboard.log</code> for any tracebacks; uncaught exceptions will appear there.</li>
        <li><strong>Port & resource conflicts</strong>: if the dashboard cannot start, another process may be using the port. The helper script kills stale listeners on the configured port before starting; you can also use <code>lsof -iTCP:5050 -sTCP:LISTEN</code> to inspect.</li>
        <li><strong>Logs</strong>: <code>dashboard.log</code> captures server output. Rotate or trim the file if you run the dashboard long-term.</li>
        <li><strong>Virtualenv & background starts</strong>: avoid <code>source .venv/bin/activate</code> in backgrounded scripts — use the venv python directly (the project scripts do this for you) to prevent suspended jobs.</li>
        <li><strong>Dependencies & reproducibility</strong>: pin exact versions in <code>requirements.txt</code> for reproducible runs; the helper installs requirements if missing.</li>
        <li><strong>Rate limiting</strong>: the default request delay is 2s (configurable in <code>config.py</code>). Respect external API limits when changing it.</li>
    <li><strong>Automated checks</strong>: a GitHub Actions smoke-test workflow has been added at <code>.github/workflows/smoke-test.yml</code>. It starts the Flask app, waits for <code>/api/analysis</code> to respond, validates basic JSON keys, and then stops the server. To run a local smoke test, use the helper script and then curl the API (example shown below).</li>
    </ul>
</div>

To run the smoke test locally (quick):

```bash
# start the dashboard and wait for the API to be ready
./scripts/start_dashboard_detach.sh 5050

# check the API response
curl -sS http://127.0.0.1:5050/api/analysis | jq . | head -n 40

# stop the dashboard
if [ -f dashboard.pid ]; then kill "$(cat dashboard.pid)" && rm -f dashboard.pid; fi
```


**That's it!** The system will fetch live SPY data and provide bias/confidence signals.

> 💡 **Want more?** See the [How to Use](#-how-to-use) section below for interactive dashboards, live charts, and other interfaces.

---

## 🎯 Purpose

Market Copilot analyzes SPY across multiple timeframes to provide structured signals for options trading decisions. It helps determine whether to use:
- **Directional strategies** (calls/puts, spreads) when bias is strong
- **Theta decay strategies** (iron condors, strangles) when neutral

**This is decision support only** - no automated trading, no broker integration.

## 🏗️ Architecture

### Modular Design
- **data_fetcher.py**: Abstracted data layer (currently Yahoo Finance, easily swappable)
  - **Built-in rate limiting**: Automatic 2-second delays (~1800 req/hour, safely under Yahoo's limit)
- **market_hours.py**: Market hours detection and data freshness validation
  - **Automatic warnings** when market is closed or data is stale
- **indicators.py**: Technical indicator calculations (EMA, RSI, ATR, VWAP)
- **bias_classifier.py**: Market bias determination with confidence scoring
- **signal_generator.py**: Structured signal generation and synthesis
- **options_data.py**: Real options chain analysis (IV Rank, P/C Ratio, GEX, Walls)
- **analyzers.py**: Lightweight fallback analyzers for Flask UI
- **flask_app.py**: Web server with real-time charting API
- **market_copilot.py**: Main orchestration and output formatting
- **config.py**: Centralized configuration

### Analysis Pipeline

1. **Data Fetching**: Pulls OHLCV data for 5m and 15m timeframes
2. **Indicator Calculation**: Computes EMA(9), EMA(21), RSI(14), ATR(14), Session VWAP
3. **Bias Classification**: Analyzes three conditions:
   - Price vs VWAP
   - EMA9 vs EMA21
   - RSI regime (>55 bullish, <45 bearish)
4. **Confidence Scoring**: Based on alignment of conditions (0.0 to 1.0)
5. **Volatility Detection**: ATR trend analysis (Expansion/Compression/Neutral)
6. **Signal Synthesis**: Combines timeframes with plain-English recommendations

## 📦 Installation & Setup

### System Requirements
- **Python 3.8 or higher**
- **Internet connection** (for live market data from Yahoo Finance)
- **Terminal/Command Prompt** (PowerShell, cmd, bash, etc.)

### Step-by-Step Installation

**1. Clone or Download the Repository**
```bash
git clone <your-repo-url>
cd spytradebot
```

**2. Install Required Packages**
```bash
pip install -r requirements.txt
```

This installs:
- `yfinance` - Market data from Yahoo Finance
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `ta` - Technical analysis indicators
- `pytz` - Timezone handling
- `rich` - Terminal UI formatting
- `plotext` - ASCII charts in terminal

**3. Verify Installation (Optional)**
```bash
# Run demo mode to test without live data
python demo_mode.py

# Run system tests
python test_system.py
```

### First Run

```bash
# Simple analysis output
python market_copilot.py
```

You should see output like:
```
================================================================================
  MARKET COPILOT - SPY Analysis
  2025-12-24 14:30:00
================================================================================

📊 5M Timeframe:
   Bias: Bullish (Strong - 83.3%)
   ...
```

---

## 🚀 How to Use

### Available Interfaces

Market Copilot offers **5 different ways** to view your analysis:

#### 1️⃣ **Web Dashboard** (Recommended - Interactive Browser UI)
```bash
python flask_app.py --port=5050
```
**Features:**
- Real-time Plotly candlestick charts (5m, 15m, optional 1m)
- Market status indicator (🟢 Open / 🔴 Closed)
- Live options data: IV Rank, Put/Call Ratio, Options Walls (OI-based)
- Gamma exposure levels and gamma squeeze indicator
- Buy/sell signals with sentiment analysis
- Auto-refresh every 2 seconds
- Interactive zoom, pan, hover details
- Two view modes:
  - **Main Dashboard** (`http://localhost:5050/`) - Multi-timeframe view
  - **Indicator View** (`http://localhost:5050/indicator`) - Single chart with persistent zoom

#### 2️⃣ **Basic Analysis** (Terminal - Quick Snapshot)
```bash
python market_copilot.py
```
- Simple, clean terminal output
- Shows bias, confidence, and indicators
- One-time analysis snapshot

#### 3️⃣ **Terminal Dashboard** (Interactive HUD)
```bash
python terminal_dashboard.py
```
- Live updating dashboard with ASCII charts
- Displays 5m and 15m timeframes
- ASCII candlestick charts
- Auto-refreshes every 60 seconds
- Press `Ctrl+C` to exit

#### 4️⃣ **Chart View** (Live Charts with Auto-Refresh)
```bash
# One-time chart display
python chart_view.py

# Live mode with 60-second refresh
python chart_view.py --live

# Live mode with custom refresh (e.g., 30 seconds)
python chart_view.py --live 30
```
- Full-screen ASCII candlestick charts
- Shows EMA9, EMA21, and VWAP overlays
- Connection and market status indicators
- Gamma squeeze indicators

#### 5️⃣ **Demo Mode** (No installation required!)
```bash
# Random realistic scenario
python demo_mode.py

# Specific scenarios
python demo_mode.py bullish
python demo_mode.py bearish
python demo_mode.py neutral
```
- Uses sample data (no Yahoo Finance needed)
- Great for testing or demonstrations
- Shows how the system works without API calls

---

## 🧪 Testing (No Installation Required!)

**Try the system immediately without installing dependencies:**

```bash
# Run demo mode with sample data
python demo_mode.py

# Test specific scenarios
python demo_mode.py bullish
python demo_mode.py bearish
python demo_mode.py neutral

# Run system tests
python test_system.py
```

The demo mode shows exactly how the system works with realistic sample data - no Yahoo Finance API or packages needed!

---

## 💻 Programmatic Usage (For Developers)

### Basic Python Integration

```python
from market_copilot import MarketCopilot

# Create and run the copilot
copilot = MarketCopilot()
signal = copilot.analyze(verbose=True)

# Automatically warns if market is closed or data is stale
```

**Note**: The system works when the market is closed, but will warn you that the data is from the last trading session.

### Example Output

```
================================================================================
  MARKET COPILOT - SPY Analysis
  2025-12-23 14:30:00
================================================================================

📊 5M Timeframe:
   Bias: Bullish (Strong - 83.3%)
   Volatility: Expansion

   Indicators:
      Close: $578.45
      VWAP:  $577.20
      EMA9:  $578.10  |  EMA21: $577.50
      RSI:   62.3  |  ATR: $1.25

   Analysis:
      • Bias confidence: 3/3 bullish, 0/3 bearish
      • Price above VWAP (578.45 > 577.20)
      • EMA9 above EMA21 (578.10 > 577.50)
      • RSI bullish regime (62.3 > 55)

📊 15M Timeframe:
   Bias: Bullish (Moderate - 66.7%)
   Volatility: Expansion
   ...

--------------------------------------------------------------------------------
📈 SYNTHESIS:
   Overall Bias: Bullish (Avg Confidence: 75.0%)
   Alignment: 2/2 timeframes agree - Strong

💡 RECOMMENDATIONS:
   • Strong bullish setup - consider directional call options or bullish spreads
   • Volatility expanding - directional strategies may benefit from increased movement
```

### Advanced Usage

```python
from market_copilot import MarketCopilot

# Initialize with custom ticker
copilot = MarketCopilot(ticker="SPY")

# Customize rate limiting (default: 2 seconds between requests)
copilot = MarketCopilot(ticker="SPY", request_delay=3.0)  # Slower, safer
copilot = MarketCopilot(ticker="SPY", request_delay=1.0)  # Faster, riskier

# Run analysis
signal = copilot.analyze(verbose=True)

# Export to JSON
copilot.export_to_json(signal, "spy_signal.json")

# Access structured data
print(f"Overall bias: {signal['synthesis']['overall_bias']}")
print(f"Confidence: {signal['synthesis']['average_confidence']}")

# Get specific timeframe data
tf_5m = signal['timeframes']['5m']
print(f"5m RSI: {tf_5m['indicators']['rsi']}")
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Ticker
DEFAULT_TICKER = "SPY"

# Timeframes
TIMEFRAMES = {
    "short": "5m",   # Short-term execution
    "medium": "15m"  # Structural bias
}

# Indicator parameters
INDICATORS = {
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi_period": 14,
    "atr_period": 14
}

# Bias thresholds
BIAS_THRESHOLDS = {
    "rsi_bullish": 55,
    "rsi_bearish": 45
}

# Rate limiting
REQUEST_DELAY = 2.0  # Seconds between API requests (~1800 req/hour)
MAX_REQUESTS_PER_HOUR = 1800
```

## 📊 Signal Structure

Each signal is a dictionary containing:

```python
{
    "ticker": "SPY",
    "analysis_timestamp": "2025-12-23 14:30:00",
    "timeframes": {
        "5m": {
            "bias": "Bullish",
            "confidence": 0.833,
            "confidence_label": "Strong",
            "volatility_regime": "Expansion",
            "indicators": {
                "close": 578.45,
                "ema_9": 578.10,
                "ema_21": 577.50,
                "rsi": 62.3,
                "atr": 1.25,
                "vwap": 577.20
            },
            "analysis_notes": [...]
        },
        "15m": {...}
    },
    "synthesis": {
        "overall_bias": "Bullish",
        "average_confidence": 0.750,
        "timeframe_alignment": "2/2 timeframes agree",
        "alignment_strength": "Strong",
        "recommendations": [...]
    }
}
```

## 🎓 How It Works

### Bias Classification Logic

The system evaluates three conditions:

1. **Price vs VWAP**: Above = bullish signal, Below = bearish signal
2. **EMA Crossover**: EMA9 > EMA21 = bullish, EMA9 < EMA21 = bearish
3. **RSI Regime**: >55 = bullish, <45 = bearish, between = neutral

**Confidence Score** = (Aligned signals) / (Total signals)

### Volatility Regime Detection

Analyzes ATR trend over recent periods:
- **Expansion**: ATR rising (good for directional plays)
- **Compression**: ATR falling (good for theta strategies)
- **Neutral**: Flat ATR

### Multi-Timeframe Synthesis

Combines 5m (execution) and 15m (structure) analysis:
- Strong alignment → Higher conviction trades
- Divergence → Reduce size or wait
- Expansion + Strong bias → Directional options
- Compression + Neutral → Theta strategies

## 🔮 Future Expansion Ideas

The modular design supports easy addition of:

- **More timeframes** (1m, 30m, 1h, daily)
- **Additional indicators** (MACD, Bollinger Bands, Volume Profile)
- **Options-specific metrics** (IV Rank, IV Percentile, Put/Call Ratio)
- **Alerts system** (Discord, Telegram, email)
- **Backtesting engine** (historical signal performance)
- **Alternative data sources** (Alpha Vantage, Polygon.io, IEX Cloud)
- **Multiple tickers** (QQQ, IWM, sector ETFs)

### Example: Swapping Data Sources

```python
# In data_fetcher.py, add a new class:
class AlphaVantageDataFetcher(DataFetcher):
    def fetch_data(self, interval, period):
        # Implementation here
        pass

# Then in market_copilot.py:
copilot = MarketCopilot(ticker="SPY", data_source="alphavantage")
```

## ⚠️ Disclaimers

- **Not financial advice**: This is educational software for decision support only
- **No guarantees**: Past performance does not indicate future results
- **Market data limitations**: Yahoo Finance has rate limits (~2000/hour) - system auto-throttles to ~1800/hour
- **Use at your own risk**: Always validate signals and manage risk appropriately

## 📚 Additional Documentation

- **[WEB_DASHBOARD_GUIDE.md](WEB_DASHBOARD_GUIDE.md)**: Complete web dashboard reference - features, indicators, API endpoints, troubleshooting
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**: Technical summary of latest features (options data, IV metrics, P/C ratio, GEX)
- **[MARKET_HOURS.md](MARKET_HOURS.md)**: Market hours detection, data freshness, and trading session awareness
- **[RATE_LIMITING.md](RATE_LIMITING.md)**: Detailed guide on API rate limiting, monitoring, and customization
- **[QUICKREF.md](QUICKREF.md)**: Quick reference for common operations
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture and data flow diagrams
- **[INSTALL.md](INSTALL.md)**: Installation and troubleshooting guide

## 📝 License

MIT License - Feel free to modify and extend for your own use.

## 🤝 Contributing

This is a personal project, but suggestions for improvements are welcome:
- Better indicator calculations
- Additional bias classification rules
- Improved volatility detection
- Performance optimizations

---

**Remember**: This tool provides decision support. Always:
- Validate signals manually
- Manage position sizing
- Use appropriate risk management
- Understand the strategies you're deploying
