import streamlit as st
import pandas as pd
import numpy as np

st.title("🔧 Simple Streamlit Test")

st.write("### Testing basic Streamlit functionality...")

# Test 1: Basic Streamlit
st.success("✅ Streamlit is working!")

# Test 2: Pandas
st.write("### Testing Pandas...")
try:
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [10, 20, 30, 40]
    })
    st.write("✅ Pandas working!")
    st.dataframe(df)
except Exception as e:
    st.error(f"❌ Pandas error: {str(e)}")

# Test 3: Simple chart
st.write("### Testing Plotly...")
try:
    import plotly.graph_objects as go
    
    # Create a simple line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[1, 4, 2, 3], mode='lines', name='Test'))
    fig.update_layout(title="Test Chart")
    
    st.plotly_chart(fig)
    st.success("✅ Plotly charts working!")
except Exception as e:
    st.error(f"❌ Plotly error: {str(e)}")

# Test 4: Try importing yfinance
st.write("### Testing yfinance import...")
try:
    import yfinance as yf
    st.success("✅ yfinance imported successfully!")
    
    # Test basic yfinance functionality
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1d")
    if not data.empty:
        st.success(f"✅ Yahoo Finance working! Got {len(data)} data points")
    else:
        st.warning("⚠️ No data from Yahoo Finance")
        
except Exception as e:
    st.error(f"❌ yfinance error: {str(e)}")

st.write("### If you can see this, Streamlit is working correctly!")
st.write("Try the main app now.") 