"""
Download Google Trends search interest data for tickers.
Uses pytrends library. Very rate-limited - 1 request every 3-5 seconds.
Often blocks/429s. Uses try/except and saves progress incrementally.
Downloads 90-day lookback at daily resolution.
"""

import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path("C:/Users/holty/projects/rd-agent/universe_data/alt_data/google_trends")
TICKER_FILE = Path("C:/Users/holty/projects/rd-agent/universe_data/ticker_list.txt")
PROGRESS_FILE = DATA_DIR / "_progress.json"


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


def fetch_trends(ticker, timeframe):
    """Fetch Google Trends data for a single ticker."""
    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))

    # Search for the ticker as a stock keyword
    kw = f"{ticker} stock"
    pytrends.build_payload([kw], cat=0, timeframe=timeframe, geo="US")
    df = pytrends.interest_over_time()

    if df.empty:
        return None

    # Rename column
    df = df.rename(columns={kw: "interest"})
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    df = df.reset_index()
    df["ticker"] = ticker
    return df


def main():
    tickers = load_tickers()
    progress = load_progress()
    completed = set(progress["completed_tickers"])
    today = datetime.now().strftime("%Y-%m-%d")

    # 90-day lookback
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    print(f"Google Trends Download - {today}")
    print(f"Timeframe: {timeframe}")
    print(f"Universe: {len(tickers)} tickers, {len(completed)} already completed")

    remaining = [t for t in tickers if t not in completed]
    print(f"Remaining: {len(remaining)} tickers")
    print("NOTE: Google Trends is heavily rate-limited. This will take a long time.")
    print("      Script saves progress and can be resumed.\n")

    all_data = []
    consecutive_errors = 0
    max_consecutive_errors = 5

    for idx, ticker in enumerate(remaining):
        print(f"  [{idx+1}/{len(remaining)}] {ticker}", end="", flush=True)

        try:
            df = fetch_trends(ticker, timeframe)
            if df is not None and len(df) > 0:
                # Save per-ticker
                ticker_file = DATA_DIR / f"{ticker}.parquet"
                df.to_parquet(ticker_file, index=False)
                all_data.append(df)
                print(f" -> {len(df)} days of data")
                consecutive_errors = 0
            else:
                print(f" -> no data")
                progress["failed_tickers"][ticker] = "no_data"
                consecutive_errors = 0

            completed.add(ticker)
            progress["completed_tickers"] = list(completed)

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Too Many" in err_str.lower() or "rate" in err_str.lower():
                consecutive_errors += 1
                print(f" -> RATE LIMITED ({consecutive_errors}/{max_consecutive_errors})")

                if consecutive_errors >= max_consecutive_errors:
                    wait_time = 120
                    print(f"\n  Too many rate limits. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    consecutive_errors = 0
                    # Don't mark as completed - will retry next run
                    continue
                else:
                    time.sleep(30 + random.uniform(5, 15))
                    continue
            else:
                print(f" -> error: {err_str[:80]}")
                progress["failed_tickers"][ticker] = err_str[:200]
                completed.add(ticker)
                progress["completed_tickers"] = list(completed)
                consecutive_errors = 0

        # Save progress every 10 tickers
        if (idx + 1) % 10 == 0:
            save_progress(progress)
            print(f"  ... progress saved ({len(completed)} done)")

        # Random delay between 3-6 seconds to avoid rate limiting
        delay = random.uniform(3, 6)
        time.sleep(delay)

    # Final save
    progress["last_run"] = today
    save_progress(progress)

    # Save combined data
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined_path = DATA_DIR / f"google_trends_combined_{today}.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"\nSaved combined: {len(combined)} rows, {combined['ticker'].nunique()} tickers")
        print(f"  -> {combined_path}")

    print(f"\nDone! Completed: {len(completed)}/{len(tickers)}")
    failed_count = len(progress["failed_tickers"])
    if failed_count:
        print(f"Failed: {failed_count}")


if __name__ == "__main__":
    main()
