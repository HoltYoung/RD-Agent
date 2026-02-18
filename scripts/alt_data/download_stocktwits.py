"""
Download StockTwits sentiment data.
API: https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json
No API key needed. Rate limit: ~200 req/hour.
Saves recent messages with bullish/bearish sentiment.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("C:/Users/holty/projects/rd-agent/universe_data/alt_data/stocktwits")
TICKER_FILE = Path("C:/Users/holty/projects/rd-agent/universe_data/ticker_list.txt")
PROGRESS_FILE = DATA_DIR / "_progress.json"
API_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def load_tickers():
    with open(TICKER_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_tickers": [], "failed_tickers": {}, "last_run": None}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def fetch_stocktwits(ticker):
    """Fetch recent messages and sentiment for a ticker from StockTwits."""
    url = API_URL.format(ticker=ticker)
    resp = requests.get(url, headers=HEADERS, timeout=30)

    if resp.status_code == 429:
        return "rate_limited", None
    if resp.status_code == 404:
        return "not_found", None

    resp.raise_for_status()
    data = resp.json()

    messages = data.get("messages", [])
    if not messages:
        return "no_messages", None

    records = []
    for msg in messages:
        sentiment = msg.get("entities", {}).get("sentiment", {})
        records.append({
            "message_id": msg.get("id"),
            "created_at": msg.get("created_at"),
            "body": msg.get("body", "")[:500],  # truncate long messages
            "sentiment": sentiment.get("basic") if sentiment else None,
            "user_followers": msg.get("user", {}).get("followers", 0),
            "user_following": msg.get("user", {}).get("following", 0),
            "likes_total": msg.get("likes", {}).get("total", 0) if msg.get("likes") else 0,
        })

    df = pd.DataFrame(records)

    # Also extract symbol-level sentiment summary
    symbol_info = data.get("symbol", {})
    summary = {
        "ticker": ticker,
        "watchlist_count": symbol_info.get("watchlist_count", 0),
        "total_messages": len(messages),
        "bullish_count": sum(1 for r in records if r["sentiment"] == "Bullish"),
        "bearish_count": sum(1 for r in records if r["sentiment"] == "Bearish"),
        "neutral_count": sum(1 for r in records if r["sentiment"] is None),
    }

    return "ok", (df, summary)


def main():
    tickers = load_tickers()
    progress = load_progress()
    completed = set(progress["completed_tickers"])
    today = datetime.now().strftime("%Y-%m-%d")

    print(f"StockTwits Sentiment Download - {today}")
    print(f"Universe: {len(tickers)} tickers, {len(completed)} already completed")

    remaining = [t for t in tickers if t not in completed]
    print(f"Remaining: {len(remaining)} tickers")

    all_messages = []
    all_summaries = []
    rate_limit_hits = 0

    for idx, ticker in enumerate(remaining):
        print(f"  [{idx+1}/{len(remaining)}] {ticker}", end="")

        try:
            status, result = fetch_stocktwits(ticker)

            if status == "rate_limited":
                rate_limit_hits += 1
                print(f" -> RATE LIMITED (hit #{rate_limit_hits})")
                if rate_limit_hits >= 3:
                    print(f"\n  Too many rate limits. Waiting 5 minutes...")
                    time.sleep(300)
                    rate_limit_hits = 0
                    # Retry this ticker
                    status, result = fetch_stocktwits(ticker)
                    if status == "rate_limited":
                        print(f"  Still rate limited after wait. Stopping for now.")
                        break
                else:
                    time.sleep(30)
                    continue

            if status == "ok" and result:
                df_msgs, summary = result
                df_msgs["ticker"] = ticker
                df_msgs["fetch_date"] = today

                # Save per-ticker
                ticker_file = DATA_DIR / f"{ticker}.parquet"
                df_msgs.to_parquet(ticker_file, index=False)

                all_messages.append(df_msgs)
                all_summaries.append(summary)
                print(f" -> {len(df_msgs)} messages, {summary['bullish_count']}B/{summary['bearish_count']}Be")
            elif status == "not_found":
                print(f" -> not found on StockTwits")
                progress["failed_tickers"][ticker] = "not_found"
            else:
                print(f" -> {status}")
                progress["failed_tickers"][ticker] = status

            completed.add(ticker)
            progress["completed_tickers"] = list(completed)

        except Exception as e:
            print(f" -> error: {e}")
            progress["failed_tickers"][ticker] = str(e)
            completed.add(ticker)
            progress["completed_tickers"] = list(completed)

        # Save progress every 25 tickers
        if (idx + 1) % 25 == 0:
            save_progress(progress)
            print(f"  ... progress saved ({len(completed)} done)")

        # Delay to respect rate limits (~200 req/hr = 1 req every 18 seconds)
        # We'll use 18 seconds to be safe
        time.sleep(18)

    # Final save
    progress["last_run"] = today
    save_progress(progress)

    # Save combined data
    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        summary_path = DATA_DIR / f"stocktwits_summary_{today}.parquet"
        df_summary.to_parquet(summary_path, index=False)
        print(f"\nSaved summary: {len(df_summary)} tickers -> {summary_path}")

    if all_messages:
        combined = pd.concat(all_messages, ignore_index=True)
        combined_path = DATA_DIR / f"stocktwits_messages_{today}.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"Saved messages: {len(combined)} total -> {combined_path}")

    print(f"\nDone! Completed: {len(completed)}/{len(tickers)}")
    failed_count = len(progress["failed_tickers"])
    if failed_count:
        print(f"Failed: {failed_count}")


if __name__ == "__main__":
    main()
