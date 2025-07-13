# 🤖 AI Trading Signal Generator

A real-time forex and crypto trading signal generator built with Streamlit, powered by technical analysis and Alpha Vantage API.

## Features

- **Real-time Market Data**: Fetch live forex and cryptocurrency data
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages
- **Interactive Charts**: Plotly-powered candlestick charts with indicators
- **Signal Generation**: AI-powered buy/sell signals based on multiple indicators
- **Risk Management**: Built-in risk management suggestions
- **Auto-refresh**: Automatic data updates every 30 seconds

## Installation

1. Install dependencies:
```bash
pip3 install -r requirements.txt
```

2. Get a free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)

3. Update the API key in `main.py`:
```python
ALPHA_VANTAGE_KEY = "YOUR_API_KEY_HERE"
```

## Usage

Run the application:
```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Select Market Type**: Choose between Forex or Crypto
2. **Choose Symbol**: Select your desired currency pair or cryptocurrency
3. **Set Timeframe**: Choose from 15min, 30min, or 60min intervals
4. **Generate Signals**: Click "Generate Signals" or enable auto-refresh
5. **Analyze Results**: View signals, charts, and risk management suggestions

## Technical Indicators

- **RSI (Relative Strength Index)**: Identifies overbought/oversold conditions
- **MACD**: Momentum indicator with signal line crossovers
- **Bollinger Bands**: Volatility indicator with price channel analysis
- **Moving Averages**: SMA and EMA for trend identification

## Signal Strength

- **BUY**: Multiple indicators suggest upward movement
- **SELL**: Multiple indicators suggest downward movement  
- **NEUTRAL**: Mixed or weak signals

## Risk Management

- Never risk more than 1-2% of your portfolio per trade
- Always use stop-loss orders
- Wait for confirmation before entering positions
- Consider market conditions and news events

## Disclaimer

This tool is for educational purposes only. Trading involves risk and you should never invest more than you can afford to lose. Always do your own research and consider consulting with a financial advisor.

## Files

- `main.py`: Main application file
- `requirements.txt`: Python dependencies
- `README.md`: This documentation file 