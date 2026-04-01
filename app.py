import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import datetime as dt
import nltk
import xml.etree.ElementTree as ET
from nltk.sentiment import SentimentIntensityAnalyzer

# -----------------------------
# Setup & Resource Loading
# -----------------------------

# Download VADER lexicon
nltk.download("vader_lexicon", quiet=True)

st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")

@st.cache_resource
def get_vader():
    return SentimentIntensityAnalyzer()

# -----------------------------
# CSS Styling
# -----------------------------

st.markdown("""
<style>
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        grid-gap: 15px;
        margin-top: 20px;
    }
    .stock-card {
        border-radius: 10px;
        padding: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        font-family: sans-serif;
    }
    .ticker { font-size: 22px; font-weight: bold; }
    .signal-badge {
        background: rgba(255,255,255,0.2);
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 12px;
        font-weight: 600;
    }
    .sentiment-bg {
        background: rgba(0,0,0,0.1);
        height: 6px;
        border-radius: 3px;
        margin-top: 8px;
    }
    .sentiment-fill { height: 100%; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Data Processing Engine
# -----------------------------

@st.cache_data(ttl=3600)  # Cache results for 1 hour
def fetch_market_data(ticker_list):
    vader = get_vader()
    processed_data = []
    
    # Progress bar for the initial load
    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(ticker_list):
        try:
            # 1. Fetch Price (Fixing the Multi-Index Issue)
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)
            if df.empty: continue
            
            # Extract Close prices safely
            if isinstance(df.columns, pd.MultiIndex):
                close_prices = df['Close'][ticker].dropna()
            else:
                close_prices = df['Close'].dropna()

            # 2. Trend Logic
            start_p, end_p = close_prices.iloc[0], close_prices.iloc[-1]
            change = (end_p - start_p) / start_p
            trend = "Uptrend" if change > 0.03 else "Downtrend" if change < -0.03 else "Neutral"

            # 3. News & Sentiment Logic
            url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=5)
            
            sent_score = 0.0
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                titles = [item.find("title").text for item in root.iter("item")][:8]
                if titles:
                    scores = [vader.polarity_scores(t)["compound"] for t in titles]
                    sent_score = sum(scores) / len(scores)

            # 4. Final Classification
            if sent_score > 0.12 and trend == "Uptrend":
                signal = "Potential Buy"
                color = "#27AE60"
            elif sent_score < -0.12 and trend == "Downtrend":
                signal = "Avoid"
                color = "#C0392B"
            else:
                signal = "Watch"
                color = "#7F8C8D"

            processed_data.append({
                "ticker": ticker,
                "trend": trend,
                "sentiment": round(sent_score, 2),
                "signal": signal,
                "color": color,
                "icon": "▲" if trend == "Uptrend" else "▼" if trend == "Downtrend" else "•"
            })
        except Exception:
            continue
        
        progress_bar.progress((idx + 1) / len(ticker_list))
    
    progress_bar.empty()
    return processed_data

# -----------------------------
# UI Logic
# -----------------------------

sp100 = [
    "AAPL","ABBV","ABT","ACN","ADBE","AMD","AMGN","AMZN","AVGO","AXP",
    "BA","BAC","BLK","BMY","BRK-B","C","CAT","COST","CRM","CSCO",
    "CVS","CVX","DIS","GE","GOOGL","GS","HD","IBM","INTC","JNJ",
    "JPM","KO","LLY","LMT","MA","MCD","META","MMM","MSFT","NFLX",
    "NKE","NVDA","ORCL","PEP","PFE","PG","PYPL","QCOM","SBUX","T",
    "TGT","TSLA","UNH","V","VZ","WFC","WMT","XOM"
]

st.title("📈 S&P 100 Sentiment & Trend Scanner")
st.write("Real-time analysis of price action and news sentiment.")

if st.button("🔄 Refresh Market Scan"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Analyzing 60+ tickers... Please wait."):
    data = fetch_market_data(sp100)

# Build HTML Grid
html_grid = '<div class="card-grid">'
for item in data:
    # Sentiment bar math
    bar_width = min(abs(item['sentiment']) * 100, 100)
    bar_color = "#FFF" if abs(item['sentiment']) > 0.05 else "rgba(255,255,255,0.3)"
    
    html_grid += f"""
    <div class="stock-card" style="background-color: {item['color']};">
        <div>
            <div class="ticker">{item['ticker']} <span style="float:right; font-size:14px;">{item['icon']}</span></div>
            <div style="font-size:11px; opacity:0.8;">{item['trend']}</div>
        </div>
        <div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:10px;">Sent: {item['sentiment']}</span>
                <span class="signal-badge">{item['signal']}</span>
            </div>
            <div class="sentiment-bg">
                <div class="sentiment-fill" style="width: {bar_width}%; background: {bar_color};"></div>
            </div>
        </div>
    </div>
    """
html_grid += '</div>'

st.markdown(html_grid, unsafe_allow_html=True)

st.divider()
st.caption(f"Last Full Scan: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# Add the unsafe_allow_html=True parameter
st.markdown(html_grid, unsafe_allow_html=True)
