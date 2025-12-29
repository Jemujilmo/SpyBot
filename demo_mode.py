"""
Demo Mode - Run Market Copilot with sample data (no dependencies needed)
Perfect for testing the system without installing packages or making API calls
"""
import json
from datetime import datetime, timedelta


def generate_sample_data():
    """
    Generate sample market data for testing.
    
    ⚠️  WARNING: These are FICTIONAL EXAMPLE values for demonstration only!
    These do NOT represent actual SPY prices. They are designed to show
    how the system classifies different market scenarios (bullish/bearish/neutral).
    
    For real market data, use the live mode with Yahoo Finance API.
    """
    
    # Scenario 1: Strong Bullish Setup
    # FICTIONAL EXAMPLE: Price trending up, above VWAP, EMAs aligned bullish
    bullish_5m = {
        "timestamp": "EXAMPLE 14:30:00",
        "close": 500.25,
        "vwap": 499.10,
        "ema_9": 500.00,
        "ema_21": 499.35,
        "rsi": 62.3,
        "atr": 1.45
    }
    
    bullish_15m = {
        "timestamp": "EXAMPLE 14:30:00",
        "close": 500.25,
        "vwap": 498.80,
        "ema_9": 499.85,
        "ema_21": 498.90,
        "rsi": 58.7,
        "atr": 1.65
    }
    
    # Scenario 2: Strong Bearish Setup
    # FICTIONAL EXAMPLE: Price trending down, below VWAP, EMAs aligned bearish
    bearish_5m = {
        "timestamp": "EXAMPLE 14:30:00",
        "close": 496.40,
        "vwap": 497.75,
        "ema_9": 496.90,
        "ema_21": 497.60,
        "rsi": 38.5,
        "atr": 1.55
    }
    
    bearish_15m = {
        "timestamp": "EXAMPLE 14:30:00",
        "close": 496.40,
        "vwap": 498.00,
        "ema_9": 496.70,
        "ema_21": 498.10,
        "rsi": 41.2,
        "atr": 1.75
    }
    
    # Scenario 3: Neutral/Mixed Signals
    # FICTIONAL EXAMPLE: Choppy price action, mixed signals across indicators
    neutral_5m = {
        "timestamp": "EXAMPLE 14:30:00",
        "close": 498.50,
        "vwap": 498.70,
        "ema_9": 498.80,
        "ema_21": 498.60,
        "rsi": 51.5,
        "atr": 1.15
    }
    
    neutral_15m = {
        "timestamp": "EXAMPLE 14:30:00",
        "close": 498.50,
        "vwap": 498.00,
        "ema_9": 498.30,
        "ema_21": 498.70,
        "rsi": 48.3,
        "atr": 1.25
    }
    
    return {
        "bullish": {"5m": bullish_5m, "15m": bullish_15m},
        "bearish": {"5m": bearish_5m, "15m": bearish_15m},
        "neutral": {"5m": neutral_5m, "15m": neutral_15m}
    }


def classify_bias_demo(data, rsi_bullish=55, rsi_bearish=45):
    """
    Demo version of bias classification (without pandas dependency).
    """
    bullish_signals = 0
    bearish_signals = 0
    total_signals = 0
    notes = []
    
    # Condition 1: Close vs VWAP
    total_signals += 1
    if data['close'] > data['vwap']:
        bullish_signals += 1
        notes.append(f"Price above VWAP ({data['close']:.2f} > {data['vwap']:.2f})")
    else:
        bearish_signals += 1
        notes.append(f"Price below VWAP ({data['close']:.2f} < {data['vwap']:.2f})")
    
    # Condition 2: EMA9 vs EMA21
    total_signals += 1
    if data['ema_9'] > data['ema_21']:
        bullish_signals += 1
        notes.append(f"EMA9 above EMA21 ({data['ema_9']:.2f} > {data['ema_21']:.2f})")
    else:
        bearish_signals += 1
        notes.append(f"EMA9 below EMA21 ({data['ema_9']:.2f} < {data['ema_21']:.2f})")
    
    # Condition 3: RSI regime
    total_signals += 1
    if data['rsi'] > rsi_bullish:
        bullish_signals += 1
        notes.append(f"RSI bullish regime ({data['rsi']:.1f} > {rsi_bullish})")
    elif data['rsi'] < rsi_bearish:
        bearish_signals += 1
        notes.append(f"RSI bearish regime ({data['rsi']:.1f} < {rsi_bearish})")
    else:
        notes.append(f"RSI neutral zone ({data['rsi']:.1f})")
    
    # Determine bias
    if bullish_signals > bearish_signals:
        bias = "Bullish"
        confidence = bullish_signals / total_signals
    elif bearish_signals > bullish_signals:
        bias = "Bearish"
        confidence = bearish_signals / total_signals
    else:
        bias = "Neutral"
        confidence = 0.5
    
    # Confidence label
    if confidence >= 0.9:
        conf_label = "Very Strong"
    elif confidence >= 0.75:
        conf_label = "Strong"
    elif confidence >= 0.6:
        conf_label = "Moderate"
    else:
        conf_label = "Weak"
    
    notes.insert(0, f"Bias confidence: {bullish_signals}/{total_signals} bullish, {bearish_signals}/{total_signals} bearish")
    
    return bias, confidence, conf_label, notes


def generate_signal_demo(ticker, timeframe, data):
    """
    Generate a demo signal for one timeframe.
    """
    bias, confidence, conf_label, notes = classify_bias_demo(data)
    
    # Simple volatility regime (based on ATR value)
    if data['atr'] > 1.3:
        vol_regime = "Expansion"
    elif data['atr'] < 1.0:
        vol_regime = "Compression"
    else:
        vol_regime = "Neutral"
    
    return {
        "ticker": ticker,
        "timeframe": timeframe,
        "timestamp": data['timestamp'],
        "bias": bias,
        "confidence": round(confidence, 3),
        "confidence_label": conf_label,
        "volatility_regime": vol_regime,
        "indicators": {
            "close": data['close'],
            "ema_9": data['ema_9'],
            "ema_21": data['ema_21'],
            "rsi": data['rsi'],
            "atr": data['atr'],
            "vwap": data['vwap']
        },
        "analysis_notes": notes
    }


def synthesize_signals_demo(signals):
    """
    Demo version of signal synthesis.
    """
    biases = [s['bias'] for s in signals.values()]
    confidences = [s['confidence'] for s in signals.values()]
    
    bullish_count = sum(1 for b in biases if b == "Bullish")
    bearish_count = sum(1 for b in biases if b == "Bearish")
    
    if bullish_count > bearish_count:
        overall_bias = "Bullish"
    elif bearish_count > bullish_count:
        overall_bias = "Bearish"
    else:
        overall_bias = "Neutral"
    
    avg_confidence = sum(confidences) / len(confidences)
    
    total_tf = len(signals)
    max_alignment = max(bullish_count, bearish_count)
    alignment_ratio = max_alignment / total_tf
    alignment_strength = "Strong" if alignment_ratio >= 0.8 else "Moderate" if alignment_ratio >= 0.5 else "Weak"
    
    # Generate recommendations
    recommendations = []
    
    if overall_bias == "Bullish":
        if avg_confidence >= 0.75 and alignment_ratio >= 0.8:
            recommendations.append("Strong bullish setup - consider directional call options or bullish spreads")
        elif avg_confidence >= 0.5:
            recommendations.append("Moderate bullish bias - smaller directional positions or credit puts may be appropriate")
        else:
            recommendations.append("Weak bullish signal - consider waiting for stronger confirmation")
    elif overall_bias == "Bearish":
        if avg_confidence >= 0.75 and alignment_ratio >= 0.8:
            recommendations.append("Strong bearish setup - consider directional put options or bearish spreads")
        elif avg_confidence >= 0.5:
            recommendations.append("Moderate bearish bias - smaller directional positions or credit calls may be appropriate")
        else:
            recommendations.append("Weak bearish signal - consider waiting for stronger confirmation")
    else:
        recommendations.append("Neutral/mixed signals - theta strategies (iron condors, strangles) may be more appropriate")
    
    # Volatility recommendations
    vol_regimes = [s['volatility_regime'] for s in signals.values()]
    expansion_count = sum(1 for v in vol_regimes if v == "Expansion")
    
    if expansion_count >= len(vol_regimes) * 0.7:
        recommendations.append("Volatility expanding - directional strategies may benefit from increased movement")
    elif expansion_count <= len(vol_regimes) * 0.3:
        recommendations.append("Volatility compressing - theta decay strategies may be favorable")
    
    if alignment_ratio < 0.6:
        recommendations.append("Timeframes show divergence - reduce position size or wait for alignment")
    
    return {
        "overall_bias": overall_bias,
        "average_confidence": round(avg_confidence, 3),
        "timeframe_alignment": f"{max_alignment}/{total_tf} timeframes agree",
        "alignment_strength": alignment_strength,
        "recommendations": recommendations
    }


def print_signal_demo(signal):
    """
    Print the demo signal in formatted output.
    """
    # ⚠️  DEMO MODE: Using FICTIONAL example values (NOT real prices)
    print("\n" + "="*80)
    print(f"  MARKET COPILOT DEMO - {signal['ticker']} Analysis")
    print(f"  {signal['analysis_timestamp']}")
    print("  Mode: DEMO (Sample Data - NOT Real Market Prices)")
    print("="*80)
    
    # Print each timeframe
    for tf_name, tf_signal in signal['timeframes'].items():
        print(f"\n📊 {tf_name.upper()} Timeframe:")
        print(f"   Bias: {tf_signal['bias']} ({tf_signal['confidence_label']} - {tf_signal['confidence']:.1%})")
        print(f"   Volatility: {tf_signal['volatility_regime']}")
        
        ind = tf_signal['indicators']
        print(f"\n   Indicators:")
        print(f"      Close: ${ind['close']}")
        print(f"      VWAP:  ${ind['vwap']}")
        print(f"      EMA9:  ${ind['ema_9']}  |  EMA21: ${ind['ema_21']}")
        print(f"      RSI:   {ind['rsi']}  |  ATR: ${ind['atr']}")
        
        print(f"\n   Analysis:")
        for note in tf_signal['analysis_notes']:
            print(f"      • {note}")
    
    # Synthesis
    print("\n" + "-"*80)
    print("📈 SYNTHESIS:")
    synthesis = signal['synthesis']
    print(f"   Overall Bias: {synthesis['overall_bias']} (Avg Confidence: {synthesis['average_confidence']:.1%})")
    print(f"   Alignment: {synthesis['timeframe_alignment']} - {synthesis['alignment_strength']}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in synthesis['recommendations']:
        print(f"   • {rec}")
    
    print("\n" + "="*80 + "\n")


def run_demo(scenario="bullish"):
    """
    Run a complete demo analysis.
    
    Args:
        scenario: "bullish", "bearish", or "neutral"
    """
    print("\n" + "🎭 "*20)
    print(f"  RUNNING DEMO MODE - {scenario.upper()} SCENARIO")
    print(f"  ⚠️  FICTIONAL example values - NOT actual market prices!")
    print(f"  Purpose: Demonstrate how the system analyzes market conditions")
    print("🎭 "*20)
    
    # Generate sample data
    sample_data = generate_sample_data()
    data = sample_data[scenario]
    
    # Generate signals for each timeframe
    signals = {}
    for tf in ["5m", "15m"]:
        signals[tf] = generate_signal_demo("SPY", tf, data[tf])
    
    # Synthesize
    synthesis = synthesize_signals_demo(signals)
    
    # Create complete signal
    signal = {
        "ticker": "SPY",
        "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "DEMO",
        "scenario": scenario,
        "timeframes": signals,
        "synthesis": synthesis
    }
    
    # Print formatted output
    print_signal_demo(signal)
    
    return signal


def run_all_scenarios():
    """
    Run all three demo scenarios.
    """
    print("\n" + "="*80)
    print("  MARKET COPILOT - DEMO MODE")
    print("  Testing all scenarios with sample data")
    print("="*80)
    
    scenarios = ["bullish", "bearish", "neutral"]
    
    for scenario in scenarios:
        signal = run_demo(scenario)
        input(f"\nPress Enter to see next scenario ({scenario} → next)...")
    
    print("\n✅ Demo complete! All scenarios tested.\n")
    print("To run with LIVE data, install dependencies and use:")
    print("  python market_copilot.py\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        scenario = sys.argv[1].lower()
        if scenario in ["bullish", "bearish", "neutral"]:
            run_demo(scenario)
        elif scenario == "all":
            run_all_scenarios()
        else:
            print(f"Unknown scenario: {scenario}")
            print("Usage: python demo_mode.py [bullish|bearish|neutral|all]")
    else:
        # Default: run all scenarios
        run_all_scenarios()
