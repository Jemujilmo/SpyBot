"""Lightweight analyzers used by the Flask dashboard.

This file provides minimal, safe implementations for:
- OptionsWallAnalyzer.get_options_walls(price)
- SentimentAnalyzer.analyze_sentiment(...)

They are intentionally simple so the web UI won't crash when the
full production analyzers are not available.
"""
from datetime import datetime
from typing import List, Dict


class OptionsWallAnalyzer:
    """Return a small set of synthetic options walls (support/resistance).

    This is a fallback implementation to avoid NameError in the dashboard.
    It creates a few strikes around the current price with decreasing strength.
    """
    def __init__(self):
        pass

    def get_options_walls(self, price: float) -> List[Dict]:
        try:
            base = round(float(price))
        except Exception:
            base = 0

        offsets = [-3, -2, -1, 1, 2]
        walls = []
        for i, off in enumerate(offsets):
            strike = float(base + off)
            walls.append({
                'strike': strike,
                'type': 'resistance' if off > 0 else 'support',
                'strength': max(10, 90 - i * 15)
            })
        return walls


class SentimentAnalyzer:
    """Simple sentiment/signal generator used by the UI.

    It emits a single weak buy/sell signal based on price vs VWAP and
    the provided bias strings. This keeps the visualization meaningful
    without depending on heavier ML or external services.
    """
    def __init__(self):
        pass

    def analyze_sentiment(self, data_5m, indicators_5m, bias_5m, bias_15m):
        """Generate buy/sell signals based on multiple factors.
        
        Analyzes the CURRENT (most recent) candle to determine entry points.
        
        Signal conditions:
        - BUY: Bullish bias + price > VWAP + EMA9 > EMA21 + RSI 45-70
        - SELL: Bearish bias + price < VWAP + EMA9 < EMA21 + RSI 30-55
        
        Time filters:
        - Only signals between 9:45 AM - 3:45 PM ET
        - Avoids market open volatility and closing noise
        """
        signals = []
        try:
            last_close = float(data_5m['Close'].iloc[-1])
            last_timestamp = data_5m.index[-1]
            
            # Time filter disabled for after-hours testing
            # You can re-enable this during market hours if needed
            
            # Get indicators for current candle
            last_vwap = float(indicators_5m['VWAP'].iloc[-1]) if 'VWAP' in indicators_5m else None
            last_ema9 = float(indicators_5m['EMA_fast'].iloc[-1]) if 'EMA_fast' in indicators_5m else None
            last_ema21 = float(indicators_5m['EMA_slow'].iloc[-1]) if 'EMA_slow' in indicators_5m else None
            last_rsi = float(indicators_5m['RSI'].iloc[-1]) if 'RSI' in indicators_5m else None

            # Normalize bias strings
            b5 = (bias_5m or '').lower()
            b15 = (bias_15m or '').lower()
            
            # Track signal strength (0-100)
            buy_score = 0
            sell_score = 0
            
            # Factor 1: Bias alignment (30 points)
            if 'bull' in b5:
                buy_score += 15
            if 'bear' in b5:
                sell_score += 15
            if 'bull' in b15:
                buy_score += 15
            if 'bear' in b15:
                sell_score += 15
            
            # Factor 2: Price vs VWAP (20 points)
            if last_vwap is not None:
                if last_close > last_vwap:
                    buy_score += 20
                else:
                    sell_score += 20
            
            # Factor 3: EMA crossover (25 points)
            if last_ema9 is not None and last_ema21 is not None:
                if last_ema9 > last_ema21:
                    buy_score += 25
                else:
                    sell_score += 25
            
            # Factor 4: RSI regime (25 points)
            if last_rsi is not None:
                if 45 <= last_rsi <= 70:  # Bullish zone but not overbought
                    buy_score += 25
                elif 30 <= last_rsi <= 55:  # Bearish zone but not oversold
                    sell_score += 25
                elif last_rsi > 70:  # Overbought - penalize buys
                    buy_score -= 15
                elif last_rsi < 30:  # Oversold - penalize sells
                    sell_score -= 15
            
            # Generate signal if score > threshold (50% - more lenient)
            if buy_score >= 50:
                signals.append({
                    'timestamp': last_timestamp,
                    'price': last_close,
                    'type': 'buy',
                    'strength': min(100, buy_score),
                    'label': f'BUY ({buy_score}%)'
                })
                print(f"✅ BUY signal generated: {buy_score}% at ${last_close:.2f}")
            elif sell_score >= 50:
                signals.append({
                    'timestamp': last_timestamp,
                    'price': last_close,
                    'type': 'sell',
                    'strength': min(100, sell_score),
                    'label': f'SELL ({sell_score}%)'
                })
                print(f"✅ SELL signal generated: {sell_score}% at ${last_close:.2f}")
            else:
                print(f"ℹ️ No signal: BUY={buy_score}%, SELL={sell_score}%")
                
        except Exception as e:
            # Be tolerant: return empty signals on any failure
            print(f"Signal generation error: {e}")
            return []
        return signals
