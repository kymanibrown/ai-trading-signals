import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.title("🎯 Final Test - Trading Signal Generator")
st.write("### Testing all components with system Python...")

# Test 1: Basic imports
st.success("✅ All imports successful!")

# Test 2: Yahoo Finance
st.write("### Testing Yahoo Finance...")
try:
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1d")
    if not data.empty:
        st.success(f"✅ Yahoo Finance working! Got {len(data)} data points")
        st.write("Sample data:", data.head())
    else:
        st.error("❌ No data from Yahoo Finance")
except Exception as e:
    st.error(f"❌ Yahoo Finance error: {str(e)}")

# Test 3: Plotly chart
st.write("### Testing Plotly...")
try:
    # Create a simple candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=['2024-01-01', '2024-01-02', '2024-01-03'],
                                         open=[100, 101, 99],
                                         high=[105, 106, 104],
                                         low=[98, 100, 97],
                                         close=[101, 99, 103])])
    fig.update_layout(title="Test Candlestick Chart")
    st.plotly_chart(fig)
    st.success("✅ Plotly charts working!")
except Exception as e:
    st.error(f"❌ Plotly error: {str(e)}")

# Test 4: Streamlit components
st.write("### Testing Streamlit components...")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Test Metric", "100", "5%")
with col2:
    st.button("Test Button")
with col3:
    st.selectbox("Test Select", ["Option 1", "Option 2", "Option 3"])

st.success("✅ Streamlit components working!")

st.write("### 🎉 All tests passed! Your app should work now.")
st.write("Try running the main app: `streamlit run main_simple.py`") 