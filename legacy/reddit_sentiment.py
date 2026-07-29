"""LEGACY -- the original SignalSniper Reddit sentiment scraper.

Kept for reference and as an optional pre-market thesis generator. It is NOT in
the trading path. See docs/DATA_SOURCES.md for the full reasoning; the short
version is four things:

  1. `scrape_reddit()` runs once on startup and never again -- there is no loop.
  2. `if ticker in full_text.upper()` matches "AMC" inside "AMC theaters" and
     "ALL" inside "all of it". Ticker extraction now requires $TICKER or
     (NASDAQ: X) and filters against the configured universe.
  3. A general-purpose sentiment model on filing text answers the wrong question
     at 200ms a pass. The 8-K item code answers the right one in microseconds.
  4. Social sentiment lags price on anything liquid.

Where it still earns its keep: microcaps where retail flow is the marginal buyer,
and as a pre-market screen for what is about to get crowded. Run it separately,
before the bell, and treat the output as a watchlist -- never as a trigger.
"""

# main.py

import os
import praw
from transformers import pipeline
from datetime import datetime
from collections import defaultdict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# === CONFIG === #
TICKERS = ["GME", "AMC", "TSLA", "NVDA", "AAPL", "PLTR"]
SUBREDDITS = ["wallstreetbets", "stocks", "pennystocks"]
POST_LIMIT = 100

# === REDDIT + MODEL === #
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="signal-sniper-bot"
)
sentiment_analyzer = pipeline("sentiment-analysis")

# === FASTAPI SETUP === #
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentiment_data = defaultdict(list)

class SentimentResponse(BaseModel):
    ticker: str
    sentiment: str
    score: float
    time: str
    subreddit: str
    text: str

# === FUNCTIONS === #
def clean_text(text):
    return text.replace('\n', ' ').strip()

def analyze_post(text):
    result = sentiment_analyzer(text[:512])[0]
    return result['label'], float(result['score'])

def scrape_reddit():
    for sub in SUBREDDITS:
        subreddit = reddit.subreddit(sub)
        for post in subreddit.hot(limit=POST_LIMIT):
            title = clean_text(post.title)
            body = clean_text(post.selftext)
            full_text = f"{title} {body}"
            for ticker in TICKERS:
                if f"${ticker}" in full_text.upper() or ticker in full_text.upper():
                    label, score = analyze_post(full_text)
                    entry = {
                        "ticker": ticker,
                        "sentiment": label,
                        "score": score,
                        "time": datetime.utcnow().isoformat(),
                        "subreddit": sub,
                        "text": title
                    }
                    sentiment_data[ticker].append(entry)

@app.get("/sentiment/{ticker}", response_model=list[SentimentResponse])
def get_sentiment(ticker: str):
    return sentiment_data.get(ticker.upper(), [])

@app.get("/trending")
def get_trending():
    trending = sorted(sentiment_data.items(), key=lambda x: len(x[1]), reverse=True)
    return [{"ticker": t, "mentions": len(v)} for t, v in trending[:10]]

@app.on_event("startup")
def startup_event():
    scrape_reddit()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
