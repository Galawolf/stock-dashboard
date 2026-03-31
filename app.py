import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser

# -----------------------------
# Helper Functions (must be ABOVE the search bar)
# -----------------------------

def get_price_data(ticker):
    """Fetch recent price data with fallback."""
    try:
        data = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if data is None or data.empty:
            data = yf.download(ticker, period="1y", interval="1d", progress=False)
        return data
    except Exception:
        return pd.DataFrame()

def compute_trend(data):
    """Compute simple trend score using moving averages."""
    if data is None or data.empty:
        return "No Data"

    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA50"] = data["Close"].rolling(50).mean()

    if len(data) < 50:
        return "No Data"

    if data["MA20"].iloc[-1] > data["MA50"].iloc[-1]:
        return "Uptrend"
    elif data["MA20"].iloc[-1] < data["MA50"].iloc[-1]:
        return "Downtrend"
    else:
        return "Neutral"

def get_news_headlines(ticker):
    """Pull free news headlines using RSS feeds."""
    feed_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    headlines = [entry.title for entry in feed.entries[:5]]
    return headlines

def compute_sentiment(headlines):
    """Compute average sentiment score from headlines."""
    sia = SentimentIntensityAnalyzer()
    if not headlines:
        return 0
    scores = [sia.polarity_scores(h)["compound"] for h in headlines]
    return np.mean(scores)

def classify(sentiment, trend):
    """Combine sentiment + trend into a simple label."""
    if trend == "Uptrend" and sentiment > 0.1:
        return "Potential Buy"
    elif trend == "Downtrend" and sentiment < -0.1:
        return "Avoid for Now"
    else:
        return "Watch"
        
# Download VADER lexicon if needed
import nltk
nltk.download('vader_lexicon')



# -----------------------------
# Page Layout
# -----------------------------

st.set_page_config(page_title="Stock Trend & Sentiment Dashboard", layout="wide")

st.title("📈 Stock Trend & Sentiment Dashboard (S&P 100)")

# -----------------------------
# Search Bar for Individual Stocks
# -----------------------------
st.subheader("🔍 Search for a Stock")

user_ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, NVDA):", "").upper()

if user_ticker:
    try:
        st.write(f"### Results for {user_ticker}")

        # Fetch data
        data = get_price_data(user_ticker)
        trend = compute_trend(data)
        headlines = get_news_headlines(user_ticker)
        sentiment = compute_sentiment(headlines)
        label = classify(sentiment, trend)

        # Display results
        st.write(f"**Trend:** {trend}")
        st.write(f"**Sentiment Score:** {round(sentiment, 3)}")
        st.write(f"**Signal:** {label}")

        # Show chart
        if not data.empty:
            st.line_chart(data["Close"])
        else:
            st.warning("No price data available for this ticker.")

        # Show headlines
        st.write("### Recent Headlines")
        for h in headlines:
            st.write(f"- {h}")

        st.write("---")

    except Exception as e:
        st.error("Could not fetch data for that ticker. Check the symbol and try again.")

# -----------------------------
# 1. Define the S&P 100 universe
# -----------------------------
sp100 = [
    "AAPL","MSFT","AMZN","GOOGL","GOOG","BRK-B","NVDA","META","TSLA","UNH",
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
# 3. Build the dashboard
# -----------------------------

results = []

st.write("Analyzing S&P 100... this may take ~20 seconds.")

for ticker in sp100:
    try:
        data = get_price_data(ticker)
        trend = compute_trend(data)
        headlines = get_news_headlines(ticker)
        sentiment = compute_sentiment(headlines)
        label = classify(sentiment, trend)

        results.append({
            "Ticker": ticker,
            "Trend": trend,
            "Sentiment": round(sentiment, 3),
            "Signal": label
        })
    except Exception:
        pass

df = pd.DataFrame(results)

# Color coding
def color_signal(val):
    if val == "Potential Buy":
        return "background-color: #b6f2b6"
    elif val == "Avoid for Now":
        return "background-color: #f7b6b6"
    else:
        return ""

st.dataframe(df.style.applymap(color_signal, subset=["Signal"]))
