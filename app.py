import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# -----------------------------
# Helper Functions
# -----------------------------

def get_price_data(ticker):
    """Fetch 6 months of data and flatten columns to avoid MultiIndex errors."""
    data = yf.download(ticker, period="6mo", interval="1d", progress=False)
    if data.empty:
        return pd.DataFrame()
    
    # Standardize column names (fixes issues with newer yfinance versions)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    return data

def compute_trend(data):
    """Calculate MAs and determine trend."""
    if data.empty or len(data) < 50:
        return "No Data"

    # Create copies to avoid SettingWithCopy warnings
    df = data.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    current_ma20 = df["MA20"].iloc[-1]
    current_ma50 = df["MA50"].iloc[-1]

    if current_ma20 > current_ma50:
        return "Uptrend"
    elif current_ma20 < current_ma50:
        return "Downtrend"
    else:
        return "Neutral"

def get_news_headlines(ticker):
    """Fetch recent headlines from Google News RSS."""
    feed_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    return [entry.title for entry in feed.entries[:5]]

def compute_sentiment(headlines):
    """Analyze sentiment of headlines using VADER."""
    sia = SentimentIntensityAnalyzer()
    if not headlines:
        return 0
    scores = [sia.polarity_scores(h)["compound"] for h in headlines]
    return np.mean(scores)

def classify_signal(sentiment, trend):
    """Determine the final recommendation."""
    if trend == "Uptrend" and sentiment > 0.1:
        return "Potential Buy"
    elif trend == "Downtrend" and sentiment < -0.1:
        return "Avoid for Now"
    else:
        return "Watch"

# -----------------------------
# Page Layout
# -----------------------------

st.set_page_config(page_title="Stock Trend & Sentiment Dashboard", layout="wide")

st.title("📈 Stock Trend & Sentiment Dashboard (S&P 100)")
st.markdown("This dashboard scans S&P 100 stocks, analyzes their 20/50 Day MA cross, and checks recent news sentiment.")

# -----------------------------
# S&P 100 Universe
# -----------------------------

sp100 = [
    "AAPL","MSFT","AMZN","GOOGL","GOOG","NVDA","META","TSLA","UNH",
    "JNJ","V","XOM","JPM","PG","MA","HD","CVX","LLY","ABBV",
    "PEP","KO","PFE","BAC","AVGO","COST","MRK","TMO","DIS","WMT",
    "CSCO","ABT","ACN","DHR","MCD","ADBE","NFLX","CRM","TXN","LIN",
    "CMCSA","NKE","WFC","INTC","HON","PM","UNP","MS","AMGN","UPS",
    "QCOM","SCHW","RTX","LOW","NEE","IBM","SBUX","MDT","CAT","GS",
    "BLK","AMT","CVS","DE","SPGI","PLD","INTU","SYK","BKNG","ISRG",
    "MDLZ","T","ADI","ZTS","MO","GILD","LMT","AXP","NOW","MMC",
    "C","EL","ADP","REGN","BDX","CI","SO","DUK","CL","USB",
    "PNC","CB","TGT","FIS","EQIX","ICE","APD","CSX","NSC","FDX"
]

# -----------------------------
# Build the Dashboard
# -----------------------------

if st.button('🚀 Start Analysis'):
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []

    for i, ticker in enumerate(sp100):
        status_text.text(f"Analyzing {ticker} ({i+1}/{len(sp100)})...")
        try:
            data = get_price_data(ticker)
            trend = compute_trend(data)
            headlines = get_news_headlines(ticker)
            sentiment = compute_sentiment(headlines)
            label = classify_signal(sentiment, trend)

            results.append({
                "Ticker": ticker,
                "Trend": trend,
                "Sentiment": round(sentiment, 3),
                "Signal": label
            })
        except Exception as e:
            # Silently skip errors for specific tickers
            continue
        
        progress_bar.progress((i + 1) / len(sp100))

    status_text.success("Analysis Complete!")
    
    df = pd.DataFrame(results)

    # Styling function
    def color_signal(val):
        if val == "Potential Buy":
            return "background-color: #d4edda; color: #155724;" # Green
        elif val == "Avoid for Now":
            return "background-color: #f8d7da; color: #721c24;" # Red
        else:
            return ""

    # Display the final table
    st.subheader("Market Scan Results")
    st.dataframe(
        df.style.applymap(color_signal, subset=["Signal"]),
        use_container_width=True,
        height=600
    )
else:
    st.info("Click the button above to begin the scan.")
