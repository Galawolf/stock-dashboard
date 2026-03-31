import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import datetime as dt
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Make sure VADER is available
nltk.download("vader_lexicon", quiet=True)

st.set_page_config(page_title="Stock Trend & Sentiment Dashboard", layout="wide")

# -----------------------------
# S&P 100 Tickers (placeholder)
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
# If you already have a full sp100 list, keep yours instead of this.


# -----------------------------
# Caching & Models
# -----------------------------

@st.cache_resource
def load_finbert_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    nlp = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=device
    )
    return nlp

@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()


finbert_pipeline = load_finbert_pipeline()
vader = load_vader()


# -----------------------------
# Data Fetching (Cached)
# -----------------------------

@st.cache_data(show_spinner=False)
def get_price_data(ticker, period="6mo", interval="1d"):
    return yf.download(ticker, period=period, interval=interval, progress=False)

@st.cache_data(show_spinner=False)
def get_news_headlines(ticker, max_headlines=15):
    """
    Simple Google News RSS fetcher.
    """
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    resp = requests.get(url, timeout=10)
    headlines = []
    if resp.status_code == 200:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.content)
        for item in root.iter("item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                headlines.append(title_el.text)
            if len(headlines) >= max_headlines:
                break
    return headlines


# -----------------------------
# Sentiment Computation
# -----------------------------

def vader_sentiment(text: str) -> float:
    scores = vader.polarity_scores(text)
    return scores["compound"]  # -1 to 1

def finbert_sentiment(text: str) -> float:
    """
    Map FinBERT scores to a single value in [-1, 1].
    """
    outputs = finbert_pipeline(text)[0]  # list of dicts
    score_map = {}
    for out in outputs:
        label = out["label"].lower()
        score_map[label] = out["score"]

    pos = score_map.get("positive", 0.0)
    neg = score_map.get("negative", 0.0)
    neu = score_map.get("neutral", 0.0)

    # Simple expectation: positive - negative
    return pos - neg  # roughly in [-1, 1]


@st.cache_data(show_spinner=False)
def compute_combined_sentiment_for_ticker(ticker: str, max_headlines: int = 15) -> float:
    headlines = get_news_headlines(ticker, max_headlines=max_headlines)
    if not headlines:
        return 0.0

    vader_scores = []
    finbert_scores = []

    for h in headlines:
        vader_scores.append(vader_sentiment(h))
        finbert_scores.append(finbert_sentiment(h))

    vader_avg = sum(vader_scores) / len(vader_scores)
    finbert_avg = sum(finbert_scores) / len(finbert_scores)

    # Weighted combo: 40% VADER, 60% FinBERT
    combined = 0.4 * vader_avg + 0.6 * finbert_avg
    return combined


# -----------------------------
# Trend Computation
# -----------------------------

def compute_trend(price_df: pd.DataFrame) -> str:
    if price_df is None or price_df.empty:
        return "Neutral"
    close = price_df["Close"].dropna()
    if len(close) < 2:
        return "Neutral"
    start = close.iloc[0]
    end = close.iloc[-1]
    change = (end - start) / start
    if change > 0.03:
        return "Uptrend"
    elif change < -0.03:
        return "Downtrend"
    else:
        return "Neutral"


# -----------------------------
# Signal Classification
# -----------------------------

def classify_signal(sentiment: float, trend: str) -> str:
    if sentiment > 0.15 and trend == "Uptrend":
        return "Potential Buy"
    elif sentiment < -0.15 and trend == "Downtrend":
        return "Avoid for Now"
    else:
        return "Watch"


# -----------------------------
# UI: Styles
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
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
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
    margin-bottom: 6px;
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

/* Sentiment bar */
.sentiment-bar-container {
    width: 100%;
    height: 4px;
    border-radius: 999px;
    background: rgba(255,255,255,0.25);
    margin-top: 6px;
    overflow: hidden;
}

.sentiment-bar-fill {
    height: 100%;
    border-radius: 999px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Helpers for UI
# -----------------------------

def trend_icon(trend: str) -> str:
    if trend == "Uptrend":
        return "▲"
    elif trend == "Downtrend":
        return "▼"
    else:
        return "•"

def card_color(signal: str) -> str:
    if signal == "Potential Buy":
        return "#2ECC71"
    elif signal == "Avoid for Now":
        return "#E74C3C"
    else:
        return "#BDC3C7"

def sentiment_bar_style(sentiment: float) -> (str, float):
    """
    Returns (color, width_percent) for the sentiment bar.
    """
    magnitude = min(abs(sentiment), 1.0)
    width = magnitude * 100.0
    if sentiment > 0.05:
        color = "#27AE60"  # green
    elif sentiment < -0.05:
        color = "#C0392B"  # red
    else:
        color = "#7F8C8D"  # gray
    return color, width


# -----------------------------
# Layout
# -----------------------------

st.title("Stock Trend & Sentiment Dashboard (S&P 100)")

with st.spinner("Analyzing S&P 100... this may take ~20 seconds on first load."):
    results = []

    for ticker in sp100:
        try:
            prices = get_price_data(ticker)
            trend = compute_trend(prices)
            sentiment = compute_combined_sentiment_for_ticker(ticker, max_headlines=15)
            signal = classify_signal(sentiment, trend)

            results.append({
                "Ticker": ticker,
                "Trend": trend,
                "Sentiment": round(sentiment, 3),
                "Signal": signal
            })
        except Exception:
            # Skip tickers that fail
            continue

last_updated = dt.datetime.now().strftime("%Y-%m-%d %I:%M %p")

st.caption(f"Last updated: {last_updated}")
st.subheader("Market Overview")

# -----------------------------
# Render Card Grid
# -----------------------------

html_cards = '<div class="card-grid">'

for stock in results:
    bg = card_color(stock["Signal"])
    icon = trend_icon(stock["Trend"])
    sentiment_val = stock["Sentiment"]
    bar_color, bar_width = sentiment_bar_style(sentiment_val)

    html_cards += (
        f'<div class="stock-card" style="background:{bg}">'
        f'<div>'
        f'<div class="ticker">{stock["Ticker"]}<span style="float:right;">{icon}</span></div>'
        f'<div class="company">S&P 100 Company</div>'
        f'</div>'
        f'<div>'
        f'<div class="bottom-row">'
        f'<div>Sent: {sentiment_val}</div>'
        f'<div class="signal-badge">{stock["Signal"]}</div>'
        f'</div>'
        f'<div class="sentiment-bar-container">'
        f'<div class="sentiment-bar-fill" '
        f'style="width:{bar_width}%; background:{bar_color};"></div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

html_cards += "</div>"

st.markdown(html_cards, unsafe_allow_html=True)
