"""Flask Web Interface for Market Copilot
Provides interactive charts with:
- Candlestick charts with VWAP, EMA9, EMA21 overlays
- Gamma indicators
- Options walls (support/resistance levels)
- Sentiment-based buy/sell signals

This module builds three independent Plotly figures (1m / 5m / 15m),
serializes them for client-side Plotly.js rendering, and exposes a small
JSON API used by the dashboard front-end.
"""

from flask import Flask, render_template, jsonify, request
import threading
import time
import sys
import os
from datetime import datetime
import pytz

import pandas as pd

from market_copilot import MarketCopilot
from ticker_list import get_ticker_list
from indicators import calculate_all_indicators
from config import REQUEST_DELAY, INDICATORS, DISPLAY_TIMEZONE, CACHE_DURATION
from market_hours import MarketHours
from analyzers import OptionsWallAnalyzer
from signal_backtester import SignalBacktester
from options_data import OptionsDataFetcher
from new_signal_logic import generate_multi_timeframe_signals
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Test mode for CI/CD - returns mock data instead of fetching from APIs
TEST_MODE = os.environ.get('FLASK_TEST_MODE', '').lower() in ('1', 'true', 'yes')

# Initialize display timezone
MarketHours.set_display_timezone(DISPLAY_TIMEZONE)

app = Flask(__name__)

# Lightweight cache for API payloads (per-ticker)
_cache_lock = threading.Lock()
_ticker_cache = {}  # {ticker: {'payload': {...}, 'cached_at': timestamp, 'is_building': bool}}
CACHE_TTL = CACHE_DURATION  # Use configured cache duration (60 seconds)


def _serializable_value(v):
    try:
        if isinstance(v, pd.Timestamp):
            return v.strftime('%Y-%m-%dT%H:%M:%S%z')
    except Exception:
        pass
    try:
        from datetime import datetime as _dt
        if isinstance(v, _dt):
            return v.strftime('%Y-%m-%dT%H:%M:%S%z')
    except Exception:
        pass
    try:
        import numpy as _np
        if isinstance(v, (_np.generic,)):
            return v.item()
    except Exception:
        pass
    return v


def _serialize_fig(fig):
    if fig is None:
        return None
    chart_data = []
    for t in fig.data:
        td = t.to_plotly_json()
        for k in ('x', 'y', 'open', 'high', 'low', 'close', 'customdata'):
            if k in td and td[k] is not None:
                try:
                    arr = list(td[k])
                    arr = [
                        [_serializable_value(v2) for v2 in v]
                        if isinstance(v, (list, tuple))
                        else _serializable_value(v)
                        for v in arr
                    ]
                    td[k] = arr
                except Exception:
                    pass
        chart_data.append(td)
    chart_layout = fig.layout.to_plotly_json() if hasattr(fig.layout, 'to_plotly_json') else dict(fig.layout)
    return {'data': chart_data, 'layout': chart_layout}


def _build_price_volume_figure(data, indicators, title, timeframe_label, ticker='SPY', signals=None):
    """Build a two-row (price + volume) Plotly figure for one timeframe.

    Auto-fits Y-axis to recent price action for better visibility.
    
    Args:
        ticker: Stock ticker symbol (used in candlestick trace name)
        signals: List of signal dicts with 'timestamp', 'type', 'price', 'strength'
    """
    if data is None or data.empty:
        return None

    # Determine how many recent candles to use for Y-axis calculation
    if timeframe_label == '1m':
        recent_candles = 120  # 2 hours
    elif timeframe_label == '5m':
        recent_candles = 78   # ~6.5 hours
    else:  # 15m
        recent_candles = 52   # ~13 hours (1 trading day)
    
    # Get recent data for Y-axis range calculation
    recent_data = data.tail(recent_candles) if len(data) > recent_candles else data
    recent_indicators = indicators.tail(recent_candles) if len(indicators) > recent_candles else indicators
    
    # Calculate Y-axis range including price AND indicators (VWAP, EMAs)
    price_min = float(recent_data['Low'].min())
    price_max = float(recent_data['High'].max())
    
    # Also consider indicator values to ensure they're visible
    indicator_values = []
    if 'VWAP' in recent_indicators and not recent_indicators['VWAP'].isna().all():
        indicator_values.extend(recent_indicators['VWAP'].dropna().tolist())
    if 'EMA_fast' in recent_indicators and not recent_indicators['EMA_fast'].isna().all():
        indicator_values.extend(recent_indicators['EMA_fast'].dropna().tolist())
    if 'EMA_slow' in recent_indicators and not recent_indicators['EMA_slow'].isna().all():
        indicator_values.extend(recent_indicators['EMA_slow'].dropna().tolist())
    
    # Expand range to include indicators
    if indicator_values:
        price_min = min(price_min, min(indicator_values))
        price_max = max(price_max, max(indicator_values))
    
    # Add minimal padding (1% on each side) for visual clarity
    price_range = price_max - price_min
    padding = price_range * 0.01 if price_range > 0 else 0.25
    
    y0 = price_min - padding
    y1 = price_max + padding

    # Create 3 subplots: main chart, volume, MACD
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,
        row_heights=[0.65, 0.15, 0.20],
        subplot_titles=(title, f'{timeframe_label} Volume', 'MACD')
    )

    # Format times for hover as 12-hour with AM/PM in display timezone (CST)
    try:
        formatted_times = []
        for dt in data.index:
            if hasattr(dt, 'strftime'):
                # Convert to display timezone (Central Time)
                dt_display = MarketHours.to_display_time(dt)
                formatted_times.append(dt_display.strftime('%b %d, %Y, %I:%M %p CT'))
            else:
                formatted_times.append(str(dt))
    except Exception:
        formatted_times = [str(dt) for dt in data.index]

    try:
        custom = list(zip(
            indicators.get('VWAP', [None] * len(data)),
            indicators.get('EMA_fast', [None] * len(data)),
            indicators.get('EMA_slow', [None] * len(data)),
            data['Volume'].tolist(),
        ))
    except Exception:
        custom = [[None, None, None, None] for _ in range(len(data))]

    # Calculate price change for each candle
    price_changes = []
    percent_changes = []
    for i in range(len(data)):
        try:
            open_price = float(data['Open'].iloc[i])
            close_price = float(data['Close'].iloc[i])
            change = close_price - open_price
            pct_change = (change / open_price * 100) if open_price != 0 else 0
            price_changes.append(f"{change:+.2f}")
            percent_changes.append(f"{pct_change:+.2f}%")
        except:
            price_changes.append("N/A")
            percent_changes.append("N/A")

    fig.add_trace(go.Candlestick(
        x=data.index, 
        open=data['Open'], 
        high=data['High'], 
        low=data['Low'], 
        close=data['Close'],
        name=f'{ticker} {timeframe_label}',
        text=formatted_times,
        customdata=list(zip(price_changes, percent_changes)),
        increasing_line_color='#00FF41',
        decreasing_line_color='#FF0000',
        increasing_fillcolor='rgba(0, 255, 65, 0.7)',
        decreasing_fillcolor='rgba(255, 0, 0, 0.7)',
        line=dict(width=1),
        hovertemplate='<b style="font-size:14px">%{text}</b><br>' +
                      '<span style="color:#00BFFF">━━━━━━━━━━━━━━━━</span><br>' +
                      '<b>Open:</b> $%{open:.2f}<br>' +
                      '<b>High:</b> $%{high:.2f}<br>' +
                      '<b>Low:</b> $%{low:.2f}<br>' +
                      '<b>Close:</b> $%{close:.2f}<br>' +
                      '<b>Change:</b> %{customdata[0]} (%{customdata[1]})<br>' +
                      '<extra></extra>',
        hoverlabel=dict(
            bgcolor='rgba(0, 0, 0, 0.9)',
            bordercolor='#00FF41',
            font=dict(family='Courier New', size=13, color='#00FF41')
        )
    ), row=1, col=1)

    if 'VWAP' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=indicators['VWAP'], name=f'VWAP ({timeframe_label})',
                                  line=dict(color='purple', width=2, dash='dot')), row=1, col=1)
    if 'EMA_fast' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=indicators['EMA_fast'], name=f'EMA 9 ({timeframe_label})',
                                  line=dict(color='#2196F3', width=1.5)), row=1, col=1)
    if 'EMA_slow' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=indicators['EMA_slow'], name=f'EMA 21 ({timeframe_label})',
                                  line=dict(color='#FF9800', width=1.5)), row=1, col=1)

    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(data['Close'], data['Open'])]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], marker_color=colors, showlegend=False), row=2, col=1)

    # Add MACD subplot
    if 'MACD' in indicators and 'MACD_signal' in indicators and 'MACD_histogram' in indicators:
        # MACD Line and Signal Line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=indicators['MACD'],
            name='MACD',
            line=dict(color='#2196F3', width=1.5)
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=indicators['MACD_signal'],
            name='Signal',
            line=dict(color='#FF9800', width=1.5)
        ), row=3, col=1)
        
        # MACD Histogram (color-coded: green for positive, red for negative)
        colors_macd = ['#26a69a' if h >= 0 else '#ef5350' for h in indicators['MACD_histogram']]
        fig.add_trace(go.Bar(
            x=data.index,
            y=indicators['MACD_histogram'],
            name='Histogram',
            marker_color=colors_macd,
            showlegend=True
        ), row=3, col=1)
        
        # Add zero line for reference
        fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dash'), row=3, col=1)

    # Add buy/sell signals as markers
    if signals:
        buy_signals = [s for s in signals if s['type'] == 'buy']
        sell_signals = [s for s in signals if s['type'] == 'sell']
        
        print(f"📍 Chart {timeframe_label}: {len(buy_signals)} buy, {len(sell_signals)} sell signals")
        
        if buy_signals:
            buy_times = [s['timestamp'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            buy_strength = [s.get('strength', 50) for s in buy_signals]
            
            fig.add_trace(go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=14,
                    color='#00E676',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>BUY</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>',
                showlegend=True
            ), row=1, col=1)
        
        if sell_signals:
            sell_times = [s['timestamp'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            sell_strength = [s.get('strength', 50) for s in sell_signals]
            
            fig.add_trace(go.Scatter(
                x=sell_times,
                y=sell_prices,
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=14,
                    color='#FF1744',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>SELL</b><br>Price: $%{y:.2f}<br>Time: %{x}<extra></extra>',
                showlegend=True
            ), row=1, col=1)

    fig.update_layout(
        template='plotly_dark', 
        hovermode='x unified',
        showlegend=True,
        margin=dict(l=60, r=30, t=40, b=40), 
        autosize=True, 
        height=900,
        dragmode='zoom',
        modebar_add=['v1hovermode', 'toggleSpikelines'],
        # Better grid and tick behavior
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickmode='auto',
            nticks=20,
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='#00FF41',
            spikethickness=1
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            tickmode='auto',
            nticks=15,
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='#00FF41',
            spikethickness=1
        )
    )
    # Set tight Y-axis range for price chart based on recent data
    fig.update_yaxes(
        range=[y0, y1],
        row=1, col=1, 
        title_text='Price ($)',
        fixedrange=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    fig.update_yaxes(
        title_text='Volume', 
        row=2, col=1, 
        fixedrange=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    fig.update_yaxes(
        title_text='MACD', 
        row=3, col=1, 
        fixedrange=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)'
    )
    fig.update_xaxes(
        title_text='Time', 
        row=3, col=1, 
        tickformat='%I:%M %p',
        fixedrange=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        tickmode='auto',
        nticks=20
    )

    return fig


def create_chart(copilot_data, data_15m, indicators_15m, ticker="SPY", data_1m=None, indicators_1m=None):
    data_5m = copilot_data['data_5m']
    indicators_5m = copilot_data['indicators_5m']
    bias_5m = copilot_data.get('bias_5m')
    bias_15m = copilot_data.get('bias_15m')

    current_price = float(data_5m['Close'].iloc[-1])
    
    # Try real options data first, fallback to synthetic
    try:
        options_fetcher = OptionsDataFetcher(ticker)
        walls = options_fetcher.get_options_walls(current_price)
        iv_metrics = options_fetcher.get_iv_metrics()
        pcr = options_fetcher.get_put_call_ratio()
        gex = options_fetcher.get_gamma_exposure(current_price)
    except Exception as e:
        # Fallback to synthetic data
        print(f"Options data error: {e}, using fallback")
        walls = OptionsWallAnalyzer().get_options_walls(current_price)
        iv_metrics = None
        pcr = None
        gex = None
    
    # Use new multi-timeframe signal logic
    signals = generate_multi_timeframe_signals(
        data_1m, data_5m, data_15m,
        indicators_1m, indicators_5m, indicators_15m
    )
    
    # Run backtest analysis
    backtester = SignalBacktester(lookforward_candles=5)
    backtest_report = backtester.generate_report(data_5m, indicators_5m, signals)
    print("\n" + backtest_report)

    fig_1m = None
    if data_1m is not None and indicators_1m is not None:
        print(f"[create_chart] Creating 1m chart with {len(data_1m)} candles")
        fig_1m = _build_price_volume_figure(data_1m, indicators_1m, f'{ticker} 1-Minute Chart', '1m', ticker, signals)
        print(f"[create_chart] 1m chart created: {fig_1m is not None}")
    else:
        print(f"[create_chart] Skipping 1m chart - data_1m: {data_1m is not None}, indicators_1m: {indicators_1m is not None}")

    fig_5m = _build_price_volume_figure(data_5m, indicators_5m, f'{ticker} 5-Minute Chart', '5m', ticker, signals)
    fig_15m = _build_price_volume_figure(data_15m, indicators_15m, f'{ticker} 15-Minute Chart', '15m', ticker, signals)

    return {'1m': fig_1m, '5m': fig_5m, '15m': fig_15m}, walls, signals, iv_metrics, pcr, gex


def build_and_cache_payload(ticker="SPY"):
    print(f"[build_and_cache_payload] Building payload for ticker: {ticker}")
    global _ticker_cache
    
    # Initialize ticker cache entry if needed
    with _cache_lock:
        if ticker not in _ticker_cache:
            _ticker_cache[ticker] = {'payload': None, 'cached_at': 0, 'is_building': False}
        
        if _ticker_cache[ticker]['is_building']:
            print(f"[build_and_cache_payload] Already building for {ticker}, skipping")
            return
        _ticker_cache[ticker]['is_building'] = True
    
    try:
        copilot = MarketCopilot(ticker=ticker, request_delay=REQUEST_DELAY)
        
        # Fetch data SEQUENTIALLY to respect rate limits
        # Parallel fetching bypasses rate limiting and causes 429 errors
        print(f"[{ticker}] Fetching 5m data...")
        try:
            data_5m = copilot.data_fetcher.fetch_data('5m', '5d')
        except Exception as e:
            print(f"[ERROR] Error fetching 5m data for {ticker}: {e}")
            data_5m = None
        
        time.sleep(REQUEST_DELAY)  # Explicit delay between requests
        
        print(f"[{ticker}] Fetching 15m data...")
        try:
            data_15m = copilot.data_fetcher.fetch_data('15m', '1mo')
        except Exception:
            try:
                time.sleep(REQUEST_DELAY)
                data_15m = copilot.data_fetcher.fetch_data('15m', '5d')
            except Exception as e:
                print(f"[ERROR] Error fetching 15m data for {ticker}: {e}")
                data_15m = None
        
        time.sleep(REQUEST_DELAY)  # Explicit delay between requests
        
        print(f"[{ticker}] Fetching 1m data...")
        try:
            data_1m = copilot.data_fetcher.fetch_data('1m', '1d')
        except Exception as e:
            print(f"[ERROR] Error fetching 1m data for {ticker}: {e}")
            data_1m = None
        # Check if we have critical data (5m minimum required)
        if data_5m is None or data_5m.empty:
            print(f"[ERROR] No 5m data available for {ticker} - cannot build payload")
            error_msg = "Rate limited or no data available. Using cached data if available."
            with _cache_lock:
                # Keep existing cached data if available, just update timestamp
                if _ticker_cache[ticker]['payload'] is None or 'error' in _ticker_cache[ticker]['payload']:
                    _ticker_cache[ticker]['payload'] = {
                        'error': error_msg,
                        'ticker': ticker,
                        'retry_after': 60  # Suggest waiting 60 seconds
                    }
                _ticker_cache[ticker]['cached_at'] = time.time()
            return
        
        if data_15m is None or data_15m.empty:
            print(f"No 15m data available for {ticker}, using 5m data for both timeframes")
            data_15m = data_5m.copy()

        data_5m = data_5m.tail(78)
        data_15m = data_15m.tail(100)

        indicators_5m = calculate_all_indicators(data_5m, INDICATORS)
        indicators_15m = calculate_all_indicators(data_15m, INDICATORS)

        bias_5m, conf_5m, _ = copilot.bias_classifier.classify_bias(indicators_5m.iloc[-1])
        bias_15m, conf_15m, _ = copilot.bias_classifier.classify_bias(indicators_15m.iloc[-1])

        copilot_data = {'data_5m': data_5m, 'indicators_5m': indicators_5m, 'bias_5m': bias_5m.value, 'bias_15m': bias_15m.value}

        # Process 1m data if available
        indicators_1m = None
        if data_1m is not None and not data_1m.empty:
            print(f"[{ticker}] 1m data: {len(data_1m)} candles")
            data_1m = data_1m.tail(240)
            indicators_1m = calculate_all_indicators(data_1m, INDICATORS)
        else:
            print(f"[{ticker}] WARNING: No 1m data available")

        figs, walls, signals, iv_metrics, pcr, gex = create_chart(copilot_data, data_15m, indicators_15m, ticker=ticker, data_1m=data_1m, indicators_1m=indicators_1m)

        # Provide a small indicators summary for the frontend which expects
        # data.indicators.close and data.indicators.rsi (and a gamma_score).
        try:
            current_price = float(data_5m['Close'].iloc[-1])
        except Exception:
            current_price = None

        try:
            rsi_val = float(indicators_5m['RSI'].iloc[-1]) if 'RSI' in indicators_5m else None
        except Exception:
            rsi_val = None

        # Compute a compact gamma_score similar to chart_view.py so the UI can display a gauge
        try:
            recent_volume = data_5m['Volume'].iloc[-5:].mean()
            avg_volume = data_5m['Volume'].mean()
            volume_ratio = (recent_volume / avg_volume) if avg_volume and avg_volume > 0 else 1.0
        except Exception:
            volume_ratio = 1.0

        try:
            if 'ATR' in indicators_5m:
                current_atr = float(indicators_5m['ATR'].iloc[-1])
                avg_atr = float(indicators_5m['ATR'].mean())
                volatility_ratio = (current_atr / avg_atr) if avg_atr and avg_atr > 0 else 1.0
            else:
                volatility_ratio = 1.0
        except Exception:
            volatility_ratio = 1.0

        try:
            price_change_5m = ((data_5m['Close'].iloc[-1] / data_5m['Close'].iloc[-5]) - 1) * 100 if len(data_5m) >= 5 else 0
        except Exception:
            price_change_5m = 0

        try:
            gamma_score = min(100, int((volume_ratio * 30 + volatility_ratio * 30 + abs(price_change_5m) * 10)))
        except Exception:
            gamma_score = 0

        # Get market status
        market_status_info = MarketHours.get_market_status()
        
        # Get latest candle times for each timeframe
        latest_times = {}
        try:
            if data_1m is not None and not data_1m.empty:
                latest_times['1m'] = MarketHours.to_display_time(data_1m.index[-1]).strftime('%I:%M %p CT')
        except Exception:
            pass
        
        try:
            latest_times['5m'] = MarketHours.to_display_time(data_5m.index[-1]).strftime('%I:%M %p CT')
        except Exception:
            pass
        
        try:
            latest_times['15m'] = MarketHours.to_display_time(data_15m.index[-1]).strftime('%I:%M %p CT')
        except Exception:
            pass

        payload = {
            'chart_1m': _serialize_fig(figs.get('1m')) if figs.get('1m') is not None else None,
            'chart_5m': _serialize_fig(figs.get('5m')),
            'chart_15m': _serialize_fig(figs.get('15m')),
            'bias_5m': {'bias': bias_5m.value, 'confidence': conf_5m},
            'bias_15m': {'bias': bias_15m.value, 'confidence': conf_15m},
            'walls': walls[:5],
            'signals': [{'timestamp': MarketHours.to_display_time(s['timestamp']).strftime('%Y-%m-%d %I:%M:%S %p CT'), 'price': s['price'], 'type': s['type'], 'strength': s['strength']} for s in signals[-10:]],
            'timestamp': MarketHours.to_display_time(datetime.now(pytz.UTC)).strftime('%Y-%m-%d %I:%M:%S %p CT'),
            'indicators': {
                'close': current_price,
                'rsi': rsi_val,
            },
            'gamma_score': gamma_score,
            'iv_metrics': iv_metrics,
            'put_call_ratio': pcr,
            'gamma_exposure': gex,
            'market_status': {
                'status': market_status_info['status'],
                'is_open': market_status_info['is_open']
            },
            'latest': latest_times,
        }

        with _cache_lock:
            _ticker_cache[ticker]['payload'] = payload
            _ticker_cache[ticker]['cached_at'] = time.time()

    except Exception as e:
        print(f"ERROR building payload for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        # Store error payload so frontend knows what happened
        with _cache_lock:
            _ticker_cache[ticker]['payload'] = {
                'error': f'Failed to fetch data for {ticker}: {str(e)}',
                'ticker': ticker
            }
            _ticker_cache[ticker]['cached_at'] = time.time()
    finally:
        with _cache_lock:
            _ticker_cache[ticker]['is_building'] = False


def periodic_refresh():
    """Background thread to refresh default ticker (SPY) data"""
    failed_attempts = 0
    while True:
        try:
            # Check if we're rate limited (have an error payload)
            with _cache_lock:
                current_payload = _ticker_cache.get('SPY', {}).get('payload')
                if current_payload and 'error' in current_payload:
                    # We're rate limited, wait longer
                    wait_time = min(300, CACHE_TTL * (2 ** failed_attempts))  # Max 5 minutes
                    print(f"[periodic_refresh] Rate limited, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    failed_attempts += 1
                else:
                    # Normal refresh
                    failed_attempts = 0
                    time.sleep(CACHE_TTL)
            
            build_and_cache_payload()
        except Exception as e:
            print(f"[periodic_refresh] Error: {e}")
            failed_attempts += 1
            time.sleep(CACHE_TTL * 2)  # Wait longer on error


@app.route('/')
def index():
    """Main dashboard with multi-timeframe charts (1m/5m/15m)"""
    return render_template('index.html')


@app.route('/indicator')
def indicator():
    """Simplified single-chart indicator view with persistent zoom"""
    return render_template('indicator.html')


@app.route('/health')
def health():
    """Simple health check endpoint - no data fetching required"""
    return jsonify({'status': 'ok', 'timestamp': time.time()})


@app.route('/api/tickers')
def get_tickers():
    """Return list of available tickers for dropdown"""
    tickers = get_ticker_list()
    return jsonify({'tickers': [{'symbol': s, 'name': n} for s, n in tickers]})


@app.route('/api/analysis/debug')
def get_analysis_debug():
    try:
        copilot = MarketCopilot()
        data_5m = copilot.data_fetcher.fetch_data('5m', '5d')
        time.sleep(REQUEST_DELAY)
        if data_5m is None or data_5m.empty:
            return jsonify({'error': 'No 5m data available'}), 500

        data_5m = data_5m.tail(78)
        indicators_5m = calculate_all_indicators(data_5m, INDICATORS)

        N = 10
        rows = []
        start = max(0, len(data_5m) - N)
        for i in range(start, len(data_5m)):
            ts = data_5m.index[i]
            ts_display = MarketHours.to_display_time(ts)
            rows.append({
                'timestamp': ts_display.strftime('%Y-%m-%d %I:%M:%S %p CT'),
                'open': float(data_5m['Open'].iloc[i]),
                'high': float(data_5m['High'].iloc[i]),
                'low': float(data_5m['Low'].iloc[i]),
                'close': float(data_5m['Close'].iloc[i]),
                'vwap': float(indicators_5m['VWAP'].iloc[i]) if 'VWAP' in indicators_5m else None,
                'ema9': float(indicators_5m['EMA_fast'].iloc[i]) if 'EMA_fast' in indicators_5m else None,
                'ema21': float(indicators_5m['EMA_slow'].iloc[i]) if 'EMA_slow' in indicators_5m else None,
                'volume': int(data_5m['Volume'].iloc[i])
            })

        return jsonify({'sample': rows, 'count': len(rows)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis')
def get_analysis():
    try:
        # Return mock data in test mode (for CI/CD)
        if TEST_MODE:
            return jsonify({
                'ticker': 'SPY',
                'bias_5m': 'BULLISH',
                'bias_15m': 'BULLISH',
                'test_mode': True,
                'timestamp': time.time()
            })
        
        # Get ticker from query parameter (default to SPY)
        ticker = request.args.get('ticker', 'SPY').upper()
        
        # Check if cache exists and is fresh for this ticker
        with _cache_lock:
            if ticker in _ticker_cache:
                cache_entry = _ticker_cache[ticker]
                payload = cache_entry['payload']
                cache_age = time.time() - cache_entry['cached_at']
                
                # If we have an error payload (rate limited), return it with retry hint
                if payload and 'error' in payload:
                    payload['cache_age'] = int(cache_age)
                    return jsonify(payload), 429  # Return 429 Too Many Requests
                
                # Return fresh cached data
                if payload and cache_age < CACHE_TTL:
                    return jsonify(payload)
                
                # Cache exists but stale - if we're currently building, return stale data
                if payload and cache_entry['is_building']:
                    payload['stale'] = True
                    payload['cache_age'] = int(cache_age)
                    return jsonify(payload)

        # Build on-demand if cache empty or stale
        build_and_cache_payload(ticker)

        # Wait for build to complete (up to 20 seconds to account for rate limits)
        wait_start = time.time()
        while True:
            with _cache_lock:
                if ticker in _ticker_cache:
                    payload = _ticker_cache[ticker]['payload']
                    if payload:
                        # Return even if error - frontend will handle it
                        if 'error' in payload:
                            return jsonify(payload), 429
                        return jsonify(payload)
            if time.time() - wait_start > 20:
                break
            time.sleep(0.5)

        # Timeout waiting for data
        return jsonify({
            'error': 'Timeout waiting for data. Server may be rate limited.',
            'retry_after': 60
        }), 503
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import os

    port = int(os.environ.get('FLASK_RUN_PORT') or os.environ.get('FLASK_PORT') or 5000)
    for arg in sys.argv[1:]:
        if arg.startswith('--port=') or arg.startswith('-p='):
            try:
                port = int(arg.split('=', 1)[1])
            except Exception:
                pass

    print('=' * 80)
    print('  MARKET COPILOT - Web Interface')
    print('  Starting Flask server...')
    print('=' * 80)
    print(f"\nAccess the dashboard at: http://localhost:{port}\n")
    
    # Pre-cache SPY data on startup to avoid slow first load
    print("[STARTUP] Pre-caching SPY data...")
    try:
        cache_thread = threading.Thread(target=build_and_cache_payload, args=('SPY',), daemon=True)
        cache_thread.start()
        print("[STARTUP] Cache warming started in background")
    except Exception as e:
        print(f"[WARNING] Could not start cache warming: {e}")

    try:
        refresher = threading.Thread(target=periodic_refresh, daemon=True)
        refresher.start()
        print("[OK] Background refresh thread started")
    except Exception as e:
        print(f"[ERROR] Error starting background thread: {e}")
        import traceback
        traceback.print_exc()

    try:
        app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
    except Exception as e:
        print(f"[ERROR] Flask server error: {e}")
        import traceback
        traceback.print_exc()
