import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import datetime as dt
import nltk
import xml.etree.ElementTree as ET
import time
from nltk.sentiment import SentimentIntensityAnalyzer

# 1. Setup & Resource Loading
nltk.download("vader_lexicon", quiet=True)

st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")

@st.cache_resource
def get_vader():
    return SentimentIntensityAnalyzer()

# 2. CSS Styling (Modern Dark-ish Cards)
st.markdown("""
<style>
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        grid-gap: 15px;
        margin-top: 20px;
    }
    .stock-card {
        border-radius: 12px;
        padding: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        font-family: 'Inter', sans-serif;
    }
    .ticker { font-size: 24px; font-weight: 800; letter-spacing: -0.5px; }
    .signal-badge {
        background: rgba(255,255,255,0.25);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .sentiment-bg {
        background: rgba(0,0,0,0.15);
        height: 8px;
        border-radius: 4px;
        margin-top: 10px;
        overflow: hidden;
    }
    .sentiment-fill { height: 100%; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# 3. The "Engine" - Fetches and Processes Data
@st.cache_data(ttl=3600)
def fetch_market_data(ticker_list):
    vader = get_vader()
    processed_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(ticker_list):
        status_text.text(f"Scanning {ticker}...")
        try:
            # A. Fetch Price (Handles the Multi-Index bug)
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)
            if df.empty: continue
            
            if isinstance(df.columns, pd.MultiIndex):
                close_prices = df['Close'][ticker].dropna()
            else:
                close_prices = df['Close'].dropna()

            if len(close_prices) < 2: continue

            # B. Trend Logic
            start_p, end_p = float(close_prices.iloc[0]), float(close_prices.iloc[-1])
            change = (end_p - start_p) / start_p
            trend = "Uptrend" if change > 0.03 else "Downtrend" if change < -0.03 else "Neutral"

            # C. News & Sentiment
            url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            
            sent_score = 0.0
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                titles = [item.find("title").text for item in root.iter("item")][:8]
                if titles:
                    scores = [vader.polarity_scores(t)["compound"] for t in titles]
                    sent_score = sum(scores) / len(scores)
            
            # D. Signal Logic
            if sent_score > 0.12 and trend == "Uptrend":
                signal, color = "Potential Buy", "#27AE60"
            elif sent_score < -0.12 and trend == "Downtrend":
                signal, color = "Avoid", "#C0392B"
            else:
                signal, color = "Watch", "#7F8C8D"

            processed_data.append({
                "ticker": ticker,
                "trend": trend,
                "sentiment": round(sent_score, 2),
                "signal": signal,
                "color": color,
                "icon": "▲" if trend == "Uptrend" else "▼" if trend == "Downtrend" else "•"
            })
            time.sleep(0.05) # Prevent hitting rate limits
        except Exception:
            continue
        
        progress_bar.progress((idx + 1) / len(ticker_list))
    
    progress_bar.empty()
    status_text.empty()
    return processed_data # This MUST be outside the 'for' loop

# 4. Main UI App
st.title("📈 S&P 100 Sentiment Scanner")

# Tickers (Shortened for initial speed, add more as needed)
tickers = ["AAPL","AMZN","GOOGL","MSFT","NVDA","TSLA","META","NFLX","AMD","DIS","JPM","V","GS","WMT","KO","PEP"]

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Analyzing Market..."):
    results = fetch_market_data(tickers)

# Render results in the grid
if results:
    html_grid = '<div class="card-grid">'
    for item in results:
        bar_width = min(abs(item['sentiment']) * 100, 100)
        # Choose bar color based on positive/negative
        bar_color = "#2ecc71" if item['sentiment'] > 0 else "#e74c3c" if item['sentiment'] < 0 else "#bdc3c7"
        
        html_grid += f"""
        <div class="stock-card" style="background-color: {item['color']};">
            <div>
                <div class="ticker">{item['ticker']} <span style="float:right; font-size:16px;">{item['icon']}</span></div>
                <div style="font-size:12px; opacity:0.9;">{item['trend']}</div>
            </div>
            <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <span style="font-size:11px; font-weight:bold;">Sent: {item['sentiment']}</span>
                    <span class="signal-badge">{item['signal']}</span>
                </div>
                <div class="sentiment-bg">
                    <div class="sentiment-fill" style="width: {bar_width}%; background: white; opacity: 0.6;"></div>
                </div>
            </div>
        </div>
        """
    html_grid += '</div>'
    st.markdown(html_grid, unsafe_allow_html=True)
else:
    st.error("No data could be retrieved. Check your internet connection or ticker list.")

st.divider()
st.caption(f"Last Scan: {dt.datetime.now().strftime('%H:%M:%S')}")
