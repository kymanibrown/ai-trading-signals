import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import os

# Configure page
st.set_page_config(
    page_title="AI Trading Signals",
    page_icon="📈",
    layout="wide"
)

# Data Source Configuration
DATA_SOURCE = "Yahoo Finance"  # More reliable than Alpha Vantage

def generate_sample_data(symbol="EUR/USD", periods=100, interval="15min"):
    """Generate sample data for demo purposes"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate realistic price data
    base_price = 1.1000 if "EUR" in symbol else 1.3000 if "GBP" in symbol else 0.7500
    np.random.seed(42)  # For reproducible results
    
    # Calculate time interval in minutes
    interval_minutes = int(interval.replace('min', ''))
    
    dates = [datetime.now() - timedelta(minutes=interval_minutes*i) for i in range(periods)]
    dates.reverse()
    
    prices = []
    current_price = base_price
    
    for _ in range(periods):
        # Random walk with some trend
        change = np.random.normal(0, 0.001) + np.random.normal(0, 0.0005)
        current_price += change
        current_price = max(current_price, base_price * 0.95)  # Prevent negative prices
        
        # Generate OHLC data
        high = current_price + abs(np.random.normal(0, 0.0005))
        low = current_price - abs(np.random.normal(0, 0.0005))
        open_price = current_price + np.random.normal(0, 0.0003)
        close_price = current_price
        
        prices.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price
        })
    
    df = pd.DataFrame(prices, index=dates)
    return df

class TradingSignalGenerator:
    def __init__(self):
        self.data_source = "Yahoo Finance"
    
    def get_forex_data(self, from_symbol, to_symbol, interval="15min"):
        """Get forex data from Yahoo Finance"""
        try:
            # Convert interval to Yahoo Finance format
            interval_map = {
                "1min": "1m",
                "5min": "5m", 
                "15min": "15m",
                "30min": "30m",
                "60min": "1h"
            }
            yf_interval = interval_map.get(interval, "15m")
            
            # Yahoo Finance forex symbols
            symbol_map = {
                "EUR/USD": "EURUSD=X",
                "GBP/USD": "GBPUSD=X", 
                "USD/JPY": "USDJPY=X",
                "AUD/USD": "AUDUSD=X",
                "USD/CAD": "USDCAD=X"
            }
            
            symbol = symbol_map.get(f"{from_symbol}/{to_symbol}", f"{from_symbol}{to_symbol}=X")
            
            st.info(f"📊 Fetching {symbol} data from Yahoo Finance...")
            
            # Get data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval=yf_interval)
            
            if df.empty:
                st.error(f"❌ No data found for {from_symbol}/{to_symbol}")
                return None
            
            # Check the actual columns and handle them properly
            st.write(f"🔍 Debug: Actual columns = {list(df.columns)}")
            
            # Yahoo Finance columns are: Open, High, Low, Close, Volume, Dividends, Stock Splits
            # We need to rename them to match our expected format
            column_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            # Only rename columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Select only the columns we need
            available_cols = ['open', 'high', 'low', 'close']
            existing_cols = [col for col in available_cols if col in df.columns]
            
            if len(existing_cols) < 4:
                st.error(f"❌ Missing required columns. Available: {list(df.columns)}")
                return None
            
            df = df[existing_cols]
            
            st.success(f"✅ Successfully fetched {len(df)} data points for {from_symbol}/{to_symbol}")
            return df
            
        except Exception as e:
            st.error(f"❌ Error fetching forex data: {str(e)}")
            return None
    
    def get_crypto_data(self, symbol, market="USD", interval="15min"):
        """Get crypto data from Yahoo Finance"""
        try:
            # Convert interval to Yahoo Finance format
            interval_map = {
                "1min": "1m",
                "5min": "5m", 
                "15min": "15m",
                "30min": "30m",
                "60min": "1h"
            }
            yf_interval = interval_map.get(interval, "15m")
            
            # Yahoo Finance crypto symbols
            symbol_map = {
                "BTC": "BTC-USD",
                "ETH": "ETH-USD",
                "ADA": "ADA-USD", 
                "DOT": "DOT-USD",
                "LINK": "LINK-USD"
            }
            
            yf_symbol = symbol_map.get(symbol, f"{symbol}-USD")
            
            st.info(f"📊 Fetching {yf_symbol} data from Yahoo Finance...")
            
            # Get data from Yahoo Finance
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period="5d", interval=yf_interval)
            
            if df.empty:
                st.error(f"❌ No data found for {symbol}")
                return None
            
            # Check the actual columns and handle them properly
            st.write(f"🔍 Debug: Actual columns = {list(df.columns)}")
            
            # Yahoo Finance columns are: Open, High, Low, Close, Volume, Dividends, Stock Splits
            # We need to rename them to match our expected format
            column_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            # Only rename columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Select only the columns we need
            available_cols = ['open', 'high', 'low', 'close']
            existing_cols = [col for col in available_cols if col in df.columns]
            
            if len(existing_cols) < 4:
                st.error(f"❌ Missing required columns. Available: {list(df.columns)}")
                return None
            
            df = df[existing_cols]
            
            st.success(f"✅ Successfully fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            st.error(f"❌ Error fetching crypto data: {str(e)}")
            return None
    
    def test_data_source(self):
        """Test if Yahoo Finance is accessible"""
        try:
            ticker = yf.Ticker("AAPL")
            data = ticker.history(period="1d")
            if not data.empty:
                return True, "Yahoo Finance is working correctly"
            else:
                return False, "No data received from Yahoo Finance"
        except Exception as e:
            return False, f"Error accessing Yahoo Finance: {str(e)}"

class SignalAnalyzer:
    @staticmethod
    def calculate_rsi(prices, window=14):
        """Calculate RSI manually"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD manually"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, rolling_mean, lower_band
    
    def generate_signals(self, df, interval="15min"):
        """Generate trading signals based on multiple indicators"""
        if df is None or len(df) < 50:
            return None
        
        # Adjust sensitivity based on timeframe
        if interval == "1min":
            rsi_oversold = 25  # More sensitive for 1min
            rsi_overbought = 75
        elif interval == "5min":
            rsi_oversold = 28
            rsi_overbought = 72
        else:
            rsi_oversold = 30  # Standard for longer timeframes
            rsi_overbought = 70
        
        # Calculate indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
        
        # Calculate moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Generate individual signals
        signals = []
        
        # RSI signals
        if df['rsi'].iloc[-1] < rsi_oversold:
            signals.append(("RSI", "BUY", "Oversold", 0.7))
        elif df['rsi'].iloc[-1] > rsi_overbought:
            signals.append(("RSI", "SELL", "Overbought", 0.7))
        
        # MACD signals
        if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
            signals.append(("MACD", "BUY", "Bullish Crossover", 0.8))
        elif df['macd'].iloc[-1] < df['macd_signal'].iloc[-1] and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
            signals.append(("MACD", "SELL", "Bearish Crossover", 0.8))
        
        # Bollinger Bands signals
        if df['close'].iloc[-1] < df['bb_lower'].iloc[-1]:
            signals.append(("BB", "BUY", "Below Lower Band", 0.6))
        elif df['close'].iloc[-1] > df['bb_upper'].iloc[-1]:
            signals.append(("BB", "SELL", "Above Upper Band", 0.6))
        
        # Moving Average signals
        if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] and df['sma_20'].iloc[-2] <= df['sma_50'].iloc[-2]:
            signals.append(("MA", "BUY", "Golden Cross", 0.9))
        elif df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1] and df['sma_20'].iloc[-2] >= df['sma_50'].iloc[-2]:
            signals.append(("MA", "SELL", "Death Cross", 0.9))
        
        # Calculate overall signal strength
        buy_signals = [s for s in signals if s[1] == "BUY"]
        sell_signals = [s for s in signals if s[1] == "SELL"]
        
        buy_strength = sum([s[3] for s in buy_signals]) / len(buy_signals) if buy_signals else 0
        sell_strength = sum([s[3] for s in sell_signals]) / len(sell_signals) if sell_signals else 0
        
        overall_signal = "NEUTRAL"
        if buy_strength > sell_strength and buy_strength > 0.6:
            overall_signal = "BUY"
        elif sell_strength > buy_strength and sell_strength > 0.6:
            overall_signal = "SELL"
        
        return {
            'signals': signals,
            'overall': overall_signal,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'data': df
        }

def create_chart(df, signals_data):
    """Create interactive trading chart with signals"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=('Price & Indicators', 'RSI', 'MACD', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], line=dict(color='gray', dash='dash'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_middle'], line=dict(color='blue'), name='BB Middle'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], line=dict(color='gray', dash='dash'), name='BB Lower'), row=1, col=1)
    
    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], line=dict(color='orange'), name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], line=dict(color='red'), name='SMA 50'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], line=dict(color='blue'), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], line=dict(color='red'), name='Signal'), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='Histogram'), row=3, col=1)
    
    fig.update_layout(
        title="Trading Analysis Dashboard",
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    return fig

# Streamlit App
def main():
    st.title("🤖 AI Trading Signal Generator")
    st.markdown("### Real-time forex and crypto trading signals powered by technical analysis")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Market selection
    market_type = st.sidebar.selectbox("Market Type", ["Forex", "Crypto"])
    
    if market_type == "Forex":
        symbol = st.sidebar.selectbox("Currency Pair", 
                                     ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"])
        from_currency, to_currency = symbol.split("/")
    else:
        symbol = st.sidebar.selectbox("Cryptocurrency", 
                                     ["BTC", "ETH", "ADA", "DOT", "LINK"])
        from_currency = symbol
        to_currency = "USD"
    
    interval = st.sidebar.selectbox("Timeframe", ["1min", "5min", "15min", "30min", "60min"])
    
    # Add note about 1-minute data (after interval is defined)
    if interval == "1min":
        st.info("⚠️ **1-minute data**: High frequency signals may generate more noise. Consider using longer timeframes for more reliable signals.")
    
    # Show timeframe info
    timeframe_info = {
        "1min": "Ultra-short term, high frequency signals",
        "5min": "Short term, quick momentum changes", 
        "15min": "Medium term, balanced signals",
        "30min": "Medium-long term, trend following",
        "60min": "Long term, major trend analysis"
    }
    
    if st.sidebar.checkbox("Show timeframe info"):
        st.sidebar.info(f"**{interval}**: {timeframe_info[interval]}")
    
    # Initialize components
    if 'signal_generator' not in st.session_state:
        st.session_state.signal_generator = TradingSignalGenerator()
    
    signal_analyzer = SignalAnalyzer()
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)")
    
    # Test data source button
    if st.sidebar.button("🔑 Test Data Source"):
        with st.spinner("Testing Yahoo Finance..."):
            is_valid, message = st.session_state.signal_generator.test_data_source()
            if is_valid:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
    
    if st.sidebar.button("Generate Signals") or auto_refresh:
        with st.spinner("Fetching market data..."):
            # Get data based on market type
            if market_type == "Forex":
                df = st.session_state.signal_generator.get_forex_data(
                    from_currency, to_currency, interval
                )
            else:
                df = st.session_state.signal_generator.get_crypto_data(
                    from_currency, to_currency, interval
                )
            
            if df is not None:
                # Generate signals
                signals_data = signal_analyzer.generate_signals(df, interval)
                
                if signals_data:
                    # Display overall signal
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        signal_color = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}[signals_data['overall']]
                        st.metric("Overall Signal", 
                                f"{signal_color} {signals_data['overall']}")
                    
                    with col2:
                        st.metric("Buy Strength", f"{signals_data['buy_strength']:.2f}")
                    
                    with col3:
                        st.metric("Sell Strength", f"{signals_data['sell_strength']:.2f}")
                    
                    # Display individual signals
                    st.subheader("📊 Signal Details")
                    for indicator, signal, reason, strength in signals_data['signals']:
                        emoji = "🟢" if signal == "BUY" else "🔴"
                        st.write(f"{emoji} **{indicator}**: {signal} - {reason} (Strength: {strength:.1f})")
                    
                    # Display chart
                    st.subheader("📈 Technical Analysis Chart")
                    chart = create_chart(signals_data['data'], signals_data)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Display current price info
                    current_price = df['close'].iloc[-1]
                    price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
                    price_change_pct = (price_change / df['close'].iloc[-2]) * 100
                    
                    st.subheader("💰 Current Market Data")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Price", f"{current_price:.5f}")
                    with col2:
                        st.metric("Change", f"{price_change:.5f}", f"{price_change_pct:.2f}%")
                    with col3:
                        st.metric("RSI", f"{signals_data['data']['rsi'].iloc[-1]:.1f}")
                    with col4:
                        st.metric("Volume Trend", "📈" if len(signals_data['signals']) > 0 else "📊")
                    
                    # Risk management suggestions
                    st.subheader("⚠️ Risk Management")
                    if signals_data['overall'] != "NEUTRAL":
                        st.info(f"""
                        **Suggested Action**: {signals_data['overall']}
                        
                        **Entry Strategy**: Wait for confirmation on next candle
                        **Stop Loss**: Set 1-2% below/above entry point  
                        **Take Profit**: Target 2-3% gain for favorable risk/reward ratio
                        **Position Size**: Risk no more than 1-2% of portfolio
                        """)
                    else:
                        st.warning("Mixed signals detected. Consider waiting for clearer market direction.")
            
            else:
                # Use demo mode with sample data
                st.warning("⚠️ API data unavailable. Using demo mode with sample data.")
                st.info("💡 Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key) for real data.")
                
                # Generate sample data
                sample_df = generate_sample_data(symbol, 100, interval)
                signals_data = signal_analyzer.generate_signals(sample_df, interval)
                
                if signals_data:
                    # Display demo notice
                    st.success("🎯 Demo Mode Active - Showing sample data and signals")
                    
                    # Display overall signal
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        signal_color = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}[signals_data['overall']]
                        st.metric("Overall Signal", 
                                f"{signal_color} {signals_data['overall']}")
                    
                    with col2:
                        st.metric("Buy Strength", f"{signals_data['buy_strength']:.2f}")
                    
                    with col3:
                        st.metric("Sell Strength", f"{signals_data['sell_strength']:.2f}")
                    
                    # Display individual signals
                    st.subheader("📊 Signal Details")
                    for indicator, signal, reason, strength in signals_data['signals']:
                        emoji = "🟢" if signal == "BUY" else "🔴"
                        st.write(f"{emoji} **{indicator}**: {signal} - {reason} (Strength: {strength:.1f})")
                    
                    # Display chart
                    st.subheader("📈 Technical Analysis Chart (Demo Data)")
                    chart = create_chart(signals_data['data'], signals_data)
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Display current price info
                    current_price = sample_df['close'].iloc[-1]
                    price_change = sample_df['close'].iloc[-1] - sample_df['close'].iloc[-2]
                    price_change_pct = (price_change / sample_df['close'].iloc[-2]) * 100
                    
                    st.subheader("💰 Current Market Data (Demo)")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Price", f"{current_price:.5f}")
                    with col2:
                        st.metric("Change", f"{price_change:.5f}", f"{price_change_pct:.2f}%")
                    with col3:
                        st.metric("RSI", f"{signals_data['data']['rsi'].iloc[-1]:.1f}")
                    with col4:
                        st.metric("Volume Trend", "📈" if len(signals_data['signals']) > 0 else "📊")
                    
                    # Risk management suggestions
                    st.subheader("⚠️ Risk Management")
                    if signals_data['overall'] != "NEUTRAL":
                        st.info(f"""
                        **Suggested Action**: {signals_data['overall']}
                        
                        **Entry Strategy**: Wait for confirmation on next candle
                        **Stop Loss**: Set 1-2% below/above entry point  
                        **Take Profit**: Target 2-3% gain for favorable risk/reward ratio
                        **Position Size**: Risk no more than 1-2% of portfolio
                        """)
                    else:
                        st.warning("Mixed signals detected. Consider waiting for clearer market direction.")
        
        # Auto refresh logic
        if auto_refresh:
            time.sleep(30)
            st.rerun()

if __name__ == "__main__":
    main()