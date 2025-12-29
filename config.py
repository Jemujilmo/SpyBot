"""
Configuration settings for Market Copilot
"""

# Ticker configuration
DEFAULT_TICKER = "SPY"

# Timeframe settings
TIMEFRAMES = {
    "short": "5m",   # Short-term execution bias
    "medium": "15m"  # Higher-timeframe structural bias
}

# Indicator parameters
INDICATORS = {
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi_period": 14,
    "atr_period": 14
}

# Bias classification thresholds
BIAS_THRESHOLDS = {
    "rsi_bullish": 55,
    "rsi_bearish": 45
}

# Data fetching settings
DATA_PERIOD = "5d"  # Fetch last 5 days to ensure enough data for calculations

# Rate limiting (to avoid Yahoo Finance API limits)
REQUEST_DELAY = 0.5  # Seconds between requests (with parallel fetching, this is less critical)
MAX_REQUESTS_PER_HOUR = 1800  # Safety margin below Yahoo's ~2000/hour limit
