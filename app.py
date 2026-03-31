import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
import nltk

nltk.download('vader_lexicon')

# -----------------------------
# Helper Functions
# -----------------------------

def get_price_data(ticker):
    data = yf.download(ticker, period="6mo", interval="1d", progress=False)
    return data

def compute_trend(data):
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
    feed_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(feed_url)
    return [entry.title for entry in feed.entries[:5]]

def compute_sentiment(headlines):
    sia = SentimentIntensityAnalyzer()
    if not headlines:
        return 0
    scores = [sia.polarity_scores(h)["compound"] for h in headlines]
    return np.mean(scores)

def classify(sentiment, trend):
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

# -----------------------------
# S&P 100 Universe
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
# Build the Dashboard
# -----------------------------

st.write("Analyzing S&P 100... this may take ~20 seconds.")

results = []

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

def color_signal(val):
    if val == "Potential Buy":
        return "background-color: #b6f2b6"
    elif val == "Avoid for Now":
        return "background-color: #f7b6b6"
    else:
        return ""

st.dataframe(df.style.applymap(color_signal, subset=["Signal"]))
