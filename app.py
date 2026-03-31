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
# Build the Card Grid Dashboard
# -----------------------------

st.write("Analyzing S&P 100... this may take ~20 seconds.")

results = []

for ticker in sp100:
    try:
        data = get_price_data(ticker)
        trend = compute_trend(data)
        headlines = get_news_headlines(ticker)
        sentiment = compute_sentiment(headlines)
        signal = classify(sentiment, trend)

        results.append({
            "Ticker": ticker,
            "Trend": trend,
            "Sentiment": round(sentiment, 3),
            "Signal": signal
        })
    except Exception:
        pass


# -----------------------------
# Card CSS (fixed + smaller)
# -----------------------------

st.markdown("""
<style>
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    grid-gap: 16px;
    margin-top: 20px;
}

.stock-card {
    border-radius: 12px;
    padding: 14px;
    color: white;
    box-shadow: 0 3px 8px rgba(0,0,0,0.08);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    height: 130px;
}

.stock-card:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 14px rgba(0,0,0,0.15);
    cursor: pointer;
}

.ticker {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 4px;
}

.company {
    font-size: 12px;
    opacity: 0.85;
    margin-bottom: 10px;
}

.bottom-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
}

.signal-badge {
    padding: 3px 6px;
    border-radius: 6px;
    background: rgba(255,255,255,0.25);
    font-size: 11px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Trend Icons
# -----------------------------

def trend_icon(trend):
    if trend == "Uptrend":
        return "▲"
    elif trend == "Downtrend":
        return "▼"
    else:
        return "•"


# -----------------------------
# Card Background Colors
# -----------------------------

def card_color(signal):
    if signal == "Potential Buy":
        return "#2ECC71"
    elif signal == "Avoid for Now":
        return "#E74C3C"
    else:
        return "#BDC3C7"


# -----------------------------
# Render the Card Grid
# -----------------------------

html_cards = '<div class="card-grid">'

for stock in results:
    bg = card_color(stock["Signal"])
    icon = trend_icon(stock["Trend"])

    html_cards += f"""
    <div class="stock-card" style="background:{bg}">
        <div class="ticker">{stock['Ticker']} <span style="float:right;">{icon}</span></div>
        <div class="company">S&P 100 Company</div>
        <div class="bottom-row">
            <div>Sent: {stock['Sentiment']}</div>
            <div class="signal-badge">{stock['Signal']}</div>
        </div>
    </div>
    """

html_cards += "</div>"

st.markdown(html_cards, unsafe_allow_html=True)
