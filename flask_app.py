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

from flask import Flask, render_template, jsonify
import threading
import time
import sys
from datetime import datetime

import pandas as pd

from market_copilot import MarketCopilot
from indicators import calculate_all_indicators
from config import REQUEST_DELAY, INDICATORS
from analyzers import OptionsWallAnalyzer, SentimentAnalyzer
from plotly.subplots import make_subplots
import plotly.graph_objects as go

app = Flask(__name__)

# Lightweight cache for API payloads
_cache_lock = threading.Lock()
_cached_payload = None
_cached_at = 0
_is_building = False
CACHE_TTL = 5  # seconds


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


def _build_price_volume_figure(data, indicators, title, timeframe_label):
    """Build a two-row (price + volume) Plotly figure for one timeframe.

    Uses ATR and a small baseline to compute a display window so SPY's
    typically-small intraday moves remain visible.
    """
    if data is None or data.empty:
        return None

    price_min = float(data['Low'].min())
    price_max = float(data['High'].max())
    last_close = float(data['Close'].iloc[-1])
    data_range = price_max - price_min if price_max > price_min else 0.0

    atr = None
    try:
        atr = float(indicators['ATR'].iloc[-1]) if 'ATR' in indicators else None
    except Exception:
        atr = None

    # baseline choices tuned for SPY intraday moves
    if timeframe_label == '1m':
        base = 0.8
    elif timeframe_label == '5m':
        base = 2.5
    else:
        base = 4.0

    if atr and atr > 0:
        display_range = max(data_range, atr * 6, base)
    else:
        display_range = max(data_range, base)

    half = display_range / 2.0
    y0 = last_close - half
    y1 = last_close + half

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.75, 0.25], subplot_titles=(title, f'{timeframe_label} Volume'))

    try:
        custom = list(zip(
            indicators.get('VWAP', [None] * len(data)),
            indicators.get('EMA_fast', [None] * len(data)),
            indicators.get('EMA_slow', [None] * len(data)),
            data['Volume'].tolist(),
        ))
    except Exception:
        custom = [[None, None, None, None] for _ in range(len(data))]

    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
        name=f'SPY {timeframe_label}', customdata=custom
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

    fig.update_layout(template='plotly_dark', hovermode='closest', showlegend=True,
                      margin=dict(l=60, r=30, t=40, b=40), autosize=True)
    fig.update_yaxes(range=[y0, y1], row=1, col=1, title_text='Price ($)')
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    fig.update_xaxes(title_text='Time', row=2, col=1)

    return fig


def create_chart(copilot_data, data_15m, indicators_15m, data_1m=None, indicators_1m=None):
    data_5m = copilot_data['data_5m']
    indicators_5m = copilot_data['indicators_5m']
    bias_5m = copilot_data.get('bias_5m')
    bias_15m = copilot_data.get('bias_15m')

    current_price = float(data_5m['Close'].iloc[-1])
    walls = OptionsWallAnalyzer().get_options_walls(current_price)
    signals = SentimentAnalyzer().analyze_sentiment(data_5m, indicators_5m, bias_5m, bias_15m)

    fig_1m = None
    if data_1m is not None and indicators_1m is not None:
        fig_1m = _build_price_volume_figure(data_1m, indicators_1m, 'SPY 1-Minute Chart', '1m')

    fig_5m = _build_price_volume_figure(data_5m, indicators_5m, 'SPY 5-Minute Chart', '5m')
    fig_15m = _build_price_volume_figure(data_15m, indicators_15m, 'SPY 15-Minute Chart', '15m')

    return {'1m': fig_1m, '5m': fig_5m, '15m': fig_15m}, walls, signals


def build_and_cache_payload():
    global _cached_payload, _cached_at, _is_building
    if _is_building:
        return
    _is_building = True
    try:
        copilot = MarketCopilot()
        data_5m = copilot.data_fetcher.fetch_data('5m', '5d')
        time.sleep(REQUEST_DELAY)
        data_15m = copilot.data_fetcher.fetch_data('15m', '1mo')

        if data_5m is None or data_15m is None or data_5m.empty or data_15m.empty:
            return

        data_5m = data_5m.tail(78)
        data_15m = data_15m.tail(100)

        indicators_5m = calculate_all_indicators(data_5m, INDICATORS)
        indicators_15m = calculate_all_indicators(data_15m, INDICATORS)

        bias_5m, conf_5m, _ = copilot.bias_classifier.classify_bias(indicators_5m.iloc[-1])
        bias_15m, conf_15m, _ = copilot.bias_classifier.classify_bias(indicators_15m.iloc[-1])

        copilot_data = {'data_5m': data_5m, 'indicators_5m': indicators_5m, 'bias_5m': bias_5m.value, 'bias_15m': bias_15m.value}

        # optional 1m
        data_1m = None
        indicators_1m = None
        try:
            data_1m = copilot.data_fetcher.fetch_data('1m', '1d')
            time.sleep(REQUEST_DELAY)
            if data_1m is not None and not data_1m.empty:
                data_1m = data_1m.tail(240)
                indicators_1m = calculate_all_indicators(data_1m, INDICATORS)
        except Exception:
            data_1m = None
            indicators_1m = None

        figs, walls, signals = create_chart(copilot_data, data_15m, indicators_15m, data_1m=data_1m, indicators_1m=indicators_1m)

        payload = {
            'chart_1m': _serialize_fig(figs.get('1m')) if figs.get('1m') is not None else None,
            'chart_5m': _serialize_fig(figs.get('5m')),
            'chart_15m': _serialize_fig(figs.get('15m')),
            'bias_5m': {'bias': bias_5m.value, 'confidence': conf_5m},
            'bias_15m': {'bias': bias_15m.value, 'confidence': conf_15m},
            'walls': walls[:5],
            'signals': [{'timestamp': s['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), 'price': s['price'], 'type': s['type'], 'strength': s['strength']} for s in signals[-10:]],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with _cache_lock:
            _cached_payload = payload
            _cached_at = time.time()

    finally:
        _is_building = False


def periodic_refresh():
    while True:
        try:
            build_and_cache_payload()
        except Exception:
            pass
        time.sleep(CACHE_TTL)


@app.route('/')
def index():
    return render_template('index.html')


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
            rows.append({
                'timestamp': pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
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
        # If cache is fresh, return it
        with _cache_lock:
            if _cached_payload and (time.time() - _cached_at) < CACHE_TTL:
                return jsonify(_cached_payload)

        # Build on-demand if cache empty or stale
        build_and_cache_payload()

        with _cache_lock:
            if _cached_payload:
                resp = dict(_cached_payload)
            else:
                return jsonify({'error': 'No data available'}), 500

        # Enrich with market status / latest timestamps
        try:
            from market_hours import MarketHours
            resp['market_status'] = MarketHours.get_market_status()
        except Exception:
            resp['market_status'] = {'status': 'Unknown', 'is_open': False}

        return jsonify(resp)
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
    print(f"\n📊 Access the dashboard at: http://localhost:{port}")

    try:
        refresher = threading.Thread(target=periodic_refresh, daemon=True)
        refresher.start()
    except Exception:
        pass

    app.run(debug=True, host='0.0.0.0', port=port)
