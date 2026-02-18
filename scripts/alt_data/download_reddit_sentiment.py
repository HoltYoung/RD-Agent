"""
Download Reddit sentiment data from ApeWisdom API.
ApeWisdom aggregates mentions/sentiment from Reddit finance subs.
API: https://apewisdom.io/api/v1.0/filter/all-stocks/
No API key needed. Be polite: ~1 req/sec.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/"
DATA_DIR = Path("C:/Users/holty/projects/rd-agent/universe_data/alt_data/reddit_sentiment")
TICKER_FILE = Path("C:/Users/holty/projects/rd-agent/universe_data/ticker_list.txt")
PROGRESS_FILE = DATA_DIR / "_progress.json"


def load_tickers():
    with open(TICKER_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"last_fetch": None, "pages_fetched": 0}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def fetch_all_pages():
    """Fetch all pages from ApeWisdom API."""
    all_results = []
    page = 1
    while True:
        url = f"{BASE_URL}?page={page}"
        print(f"  Fetching page {page}...")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        all_results.extend(results)
        page += 1
        time.sleep(1)  # polite delay

        # Safety limit
        if page > 50:
            print("  Reached page limit (50), stopping.")
            break

    return all_results


def main():
    tickers = set(load_tickers())
    progress = load_progress()
    today = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching ApeWisdom Reddit sentiment data for {today}...")
    print(f"Universe: {len(tickers)} tickers")

    # Fetch all data from API
    all_results = fetch_all_pages()
    print(f"  Got {len(all_results)} total ticker mentions from API")

    if not all_results:
        print("No data returned from API.")
        return

    # Convert to DataFrame and deduplicate by ticker (pages overlap)
    df = pd.DataFrame(all_results)
    print(f"  Columns: {list(df.columns)}")
    if "ticker" in df.columns:
        df = df.drop_duplicates(subset=["ticker"], keep="first")
        print(f"  Unique tickers after dedup: {len(df)}")

    # Filter to our universe
    if "ticker" in df.columns:
        df_universe = df[df["ticker"].isin(tickers)].copy()
        print(f"  Matched {len(df_universe)} entries to our {len(tickers)}-ticker universe")
    else:
        print(f"  Warning: 'ticker' column not found. Available columns: {list(df.columns)}")
        df_universe = df

    # Add fetch date
    df_universe["fetch_date"] = today

    # Save full snapshot
    full_path = DATA_DIR / f"reddit_sentiment_{today}.parquet"
    df_universe.to_parquet(full_path, index=False)
    print(f"  Saved universe-filtered data to {full_path}")

    # Also save ALL tickers (useful for broader analysis)
    all_path = DATA_DIR / f"reddit_sentiment_all_{today}.parquet"
    df["fetch_date"] = today
    df.to_parquet(all_path, index=False)
    print(f"  Saved all tickers to {all_path}")

    # Update progress
    progress["last_fetch"] = today
    progress["pages_fetched"] = len(all_results)
    save_progress(progress)

    # Print top mentions from our universe
    if "mentions" in df_universe.columns and len(df_universe) > 0:
        top = df_universe.nlargest(20, "mentions")[["ticker", "mentions", "rank"]].to_string(index=False)
        print(f"\n  Top 20 mentions from our universe:\n{top}")

    print("\nDone!")


if __name__ == "__main__":
    main()
