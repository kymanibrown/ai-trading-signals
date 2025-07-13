import streamlit as st
import yfinance as yf

st.title("🔧 Streamlit Test App")

st.write("### Testing basic functionality...")

# Test 1: Basic Streamlit
st.success("✅ Streamlit is working!")

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

st.write("### If you can see this, Streamlit is working correctly!")
st.write("Try clicking the 'Generate Signals' button in the main app.") 