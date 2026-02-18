"""
Download SEC Fails-to-Deliver (FTD) data for the past 2 years.

FTD data is published bimonthly as pipe-delimited text files at:
https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data

Downloads ZIP files, parses pipe-delimited data, filters to universe tickers,
and saves as parquet files.

Re-runnable: skips files already downloaded.
"""

import io
import os
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# Paths
BASE_DIR = Path(r"C:\Users\holty\projects\rd-agent")
TICKER_FILE = BASE_DIR / "universe_data" / "ticker_list.txt"
OUTPUT_DIR = BASE_DIR / "universe_data" / "alt_data" / "sec_ftd"

# SEC rate limit compliance
HEADERS = {"User-Agent": "RDAgent research@example.com"}
REQUEST_DELAY = 0.15  # ~6-7 requests/sec to stay well under 10/sec limit


def load_tickers():
    """Load ticker list from file."""
    with open(TICKER_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def generate_ftd_urls(years_back=2):
    """Generate FTD file URLs for the last N years.

    FTD files follow the pattern:
    https://www.sec.gov/files/data/fails-deliver-data/cnsfailsYYYYMM{a|b}.zip
    where a = first half of month, b = second half.
    """
    urls = []
    now = datetime.now()
    start = now - timedelta(days=years_back * 365)

    year = start.year
    month = start.month

    while (year, month) <= (now.year, now.month):
        for half in ["a", "b"]:
            ym = f"{year}{month:02d}"
            url = f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{ym}{half}.zip"
            urls.append((ym + half, url))
        month += 1
        if month > 12:
            month = 1
            year += 1

    return urls


def download_and_parse_ftd(url, tickers):
    """Download a single FTD ZIP file and parse it, filtering to our tickers."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    # FTD ZIPs contain a single pipe-delimited text file
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            return None
        with zf.open(names[0]) as f:
            # Read as text, handle encoding issues
            raw = f.read()
            # Try UTF-8 first, fall back to latin-1
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1")

    # Parse pipe-delimited data
    # Columns: SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return None

    # Use pandas to parse
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep="|",
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception:
        return None

    if df.empty:
        return None

    # Normalize column names
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

    # Filter to our tickers
    symbol_col = None
    for col in df.columns:
        if "SYMBOL" in col:
            symbol_col = col
            break

    if symbol_col is None:
        return None

    df[symbol_col] = df[symbol_col].str.strip()
    df = df[df[symbol_col].isin(tickers)]

    if df.empty:
        return None

    # Standardize column names
    rename_map = {}
    for col in df.columns:
        if "SETTLEMENT" in col and "DATE" in col:
            rename_map[col] = "settlement_date"
        elif "CUSIP" in col:
            rename_map[col] = "cusip"
        elif "SYMBOL" in col:
            rename_map[col] = "symbol"
        elif "QUANTITY" in col or "FAILS" in col:
            rename_map[col] = "quantity_fails"
        elif "DESCRIPTION" in col:
            rename_map[col] = "description"
        elif "PRICE" in col:
            rename_map[col] = "price"

    df = df.rename(columns=rename_map)

    # Convert types
    if "settlement_date" in df.columns:
        df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
    if "quantity_fails" in df.columns:
        df["quantity_fails"] = pd.to_numeric(df["quantity_fails"], errors="coerce")
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers from universe")

    urls = generate_ftd_urls(years_back=2)
    print(f"Generated {len(urls)} FTD file URLs to check")

    all_dfs = []
    downloaded = 0
    skipped = 0

    # Check if we already have a combined file and when it was last updated
    combined_file = OUTPUT_DIR / "ftd_combined.parquet"

    for period_id, url in urls:
        # Check if individual period file already exists
        period_file = OUTPUT_DIR / f"ftd_{period_id}.parquet"
        if period_file.exists():
            # Load existing data
            df = pd.read_parquet(period_file)
            all_dfs.append(df)
            skipped += 1
            continue

        print(f"  Downloading {period_id}...", end=" ", flush=True)
        try:
            df = download_and_parse_ftd(url, tickers)
            time.sleep(REQUEST_DELAY)

            if df is not None and not df.empty:
                df.to_parquet(period_file, index=False)
                all_dfs.append(df)
                print(f"OK ({len(df)} rows)")
                downloaded += 1
            else:
                print("no data/404")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDownloaded {downloaded} new files, skipped {skipped} existing")

    # Combine all data
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values(["symbol", "settlement_date"]).reset_index(drop=True)
        combined.to_parquet(combined_file, index=False)
        print(f"Combined FTD data: {len(combined)} rows, {combined['symbol'].nunique()} tickers")
        print(f"Date range: {combined['settlement_date'].min()} to {combined['settlement_date'].max()}")
        print(f"Saved to: {combined_file}")
    else:
        print("No FTD data collected")


if __name__ == "__main__":
    main()
