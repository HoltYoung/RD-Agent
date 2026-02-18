"""
Download Wikipedia pageview data for company pages.
Uses Wikimedia REST API (no key needed, generous rate limits).
URL: https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{TITLE}/daily/{START}/{END}
Downloads last 2 years of daily data.
Saves progress incrementally so it can resume if interrupted.
"""

import json
import os
import time
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("C:/Users/holty/projects/rd-agent/universe_data/alt_data/wikipedia_pageviews")
TICKER_FILE = Path("C:/Users/holty/projects/rd-agent/universe_data/ticker_list.txt")
PROGRESS_FILE = DATA_DIR / "_progress.json"
MAPPING_FILE = DATA_DIR / "_ticker_to_wiki.json"

# Mapping of tickers to Wikipedia article titles
# We'll use yfinance to get company names, then map to wiki titles
WIKI_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title}/daily/{start}/{end}"

# Headers required by Wikimedia API
HEADERS = {
    "User-Agent": "RDAgent/1.0 (research project; contact: research@example.com)",
    "Accept": "application/json",
}


def load_tickers():
    with open(TICKER_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_tickers": [], "failed_tickers": {}}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def get_ticker_to_wiki_mapping(tickers):
    """Build a mapping from ticker to Wikipedia article title using yfinance."""
    if MAPPING_FILE.exists():
        with open(MAPPING_FILE) as f:
            mapping = json.load(f)
        # Check if we have all tickers
        missing = [t for t in tickers if t not in mapping]
        if not missing:
            return mapping
        print(f"  Have mapping for {len(mapping)} tickers, fetching {len(missing)} more...")
    else:
        mapping = {}
        missing = tickers

    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed, using ticker-based fallback mapping")
        for t in missing:
            mapping[t] = t
        return mapping

    # Batch fetch info using yfinance
    batch_size = 50
    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        print(f"  Fetching company names for tickers {i+1}-{min(i+batch_size, len(missing))} of {len(missing)}...")

        for ticker in batch:
            try:
                info = yf.Ticker(ticker).info
                name = info.get("longName") or info.get("shortName") or ticker
                # Clean up for Wikipedia: remove Inc., Corp., etc. and use as search
                # For Wikipedia, the article title often IS the company name
                wiki_title = name
                # Some common cleanups for Wikipedia article titles
                for suffix in [", Inc.", ", Inc", " Inc.", " Inc", " Corporation", " Corp.", " Corp",
                               " Company", " Co.", " Ltd.", " Ltd", " PLC", " plc",
                               ", LLC", " LLC", " LP", " L.P.", " N.V.", " S.A.",
                               " Group", " Holdings", " Holding"]:
                    if wiki_title.endswith(suffix):
                        # Keep the base but also try with suffix
                        pass
                mapping[ticker] = wiki_title
            except Exception as e:
                # Fallback to ticker symbol
                mapping[ticker] = ticker

        # Save mapping incrementally
        with open(MAPPING_FILE, "w") as f:
            json.dump(mapping, f, indent=2)

        time.sleep(0.5)

    return mapping


def fetch_pageviews(title, start_date, end_date):
    """Fetch pageviews for a Wikipedia article."""
    encoded_title = urllib.parse.quote(title.replace(" ", "_"), safe="")
    start_str = start_date.strftime("%Y%m%d") + "00"
    end_str = end_date.strftime("%Y%m%d") + "00"

    url = WIKI_API.format(title=encoded_title, start=start_str, end=end_str)

    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        return None  # Article not found
    resp.raise_for_status()

    data = resp.json()
    items = data.get("items", [])
    if not items:
        return None

    records = []
    for item in items:
        records.append({
            "date": item.get("timestamp", "")[:8],
            "views": item.get("views", 0),
            "article": item.get("article", title),
        })

    return pd.DataFrame(records)


def main():
    tickers = load_tickers()
    progress = load_progress()
    completed = set(progress["completed_tickers"])

    print(f"Wikipedia Pageviews Download")
    print(f"Universe: {len(tickers)} tickers, {len(completed)} already completed")

    # Date range: last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Build ticker-to-wiki mapping
    print("\nBuilding ticker-to-Wikipedia mapping...")
    mapping = get_ticker_to_wiki_mapping(tickers)
    print(f"  Mapping ready for {len(mapping)} tickers")

    # Download pageviews
    remaining = [t for t in tickers if t not in completed]
    print(f"\nDownloading pageviews for {len(remaining)} remaining tickers...")

    all_data = []
    for idx, ticker in enumerate(remaining):
        wiki_title = mapping.get(ticker, ticker)
        print(f"  [{idx+1}/{len(remaining)}] {ticker} -> \"{wiki_title}\"", end="")

        try:
            df = fetch_pageviews(wiki_title, start_date, end_date)
            if df is not None and len(df) > 0:
                df["ticker"] = ticker
                df["wiki_title"] = wiki_title

                # Save individual ticker file
                ticker_file = DATA_DIR / f"{ticker}.parquet"
                df.to_parquet(ticker_file, index=False)
                all_data.append(df)
                print(f" -> {len(df)} days of data")
            else:
                print(f" -> no data found")
                progress["failed_tickers"][ticker] = "no_data"
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f" -> RATE LIMITED, waiting 60s...")
                time.sleep(60)
                # Retry once
                try:
                    df = fetch_pageviews(wiki_title, start_date, end_date)
                    if df is not None and len(df) > 0:
                        df["ticker"] = ticker
                        df["wiki_title"] = wiki_title
                        ticker_file = DATA_DIR / f"{ticker}.parquet"
                        df.to_parquet(ticker_file, index=False)
                        all_data.append(df)
                        print(f" -> {len(df)} days (retry)")
                    else:
                        print(f" -> no data (retry)")
                        progress["failed_tickers"][ticker] = "no_data_retry"
                except Exception as e2:
                    print(f" -> failed on retry: {e2}")
                    progress["failed_tickers"][ticker] = str(e2)
            else:
                print(f" -> error: {e}")
                progress["failed_tickers"][ticker] = str(e)
        except Exception as e:
            print(f" -> error: {e}")
            progress["failed_tickers"][ticker] = str(e)

        completed.add(ticker)
        progress["completed_tickers"] = list(completed)

        # Save progress every 20 tickers
        if (idx + 1) % 20 == 0:
            save_progress(progress)
            print(f"  ... progress saved ({len(completed)} done)")

        # Small delay to be respectful
        time.sleep(0.2)

    # Final save
    save_progress(progress)

    # Combine all data into a single file
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined_path = DATA_DIR / "wikipedia_pageviews_combined.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"\nSaved combined data: {len(combined)} rows, {combined['ticker'].nunique()} tickers")
        print(f"  -> {combined_path}")

    failed_count = len(progress["failed_tickers"])
    print(f"\nDone! Completed: {len(completed)}, Failed: {failed_count}")
    if failed_count > 0:
        print(f"  Failed tickers: {list(progress['failed_tickers'].keys())[:20]}...")


if __name__ == "__main__":
    main()
