"""
Download FINRA short interest data for our ticker universe and save as parquet.
Tries the FINRA API (api.finra.org) first, then falls back to alternative sources.
Re-runnable: overwrites existing output.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path("C:/Users/holty/projects/rd-agent/universe_data/alt_data/finra_short_interest")
OUTPUT_FILE = OUTPUT_DIR / "finra_short_interest.parquet"
TICKER_FILE = Path("C:/Users/holty/projects/rd-agent/universe_data/ticker_list.txt")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Content-Type": "application/json",
}


def load_tickers() -> list:
    """Load our universe tickers."""
    return TICKER_FILE.read_text().strip().splitlines()


def try_finra_api(tickers: list) -> pd.DataFrame | None:
    """Try FINRA's public API for short interest data."""
    print("Attempting FINRA API (api.finra.org)...")

    # FINRA's public API endpoint for short interest
    url = "https://api.finra.org/data/group/otcMarket/name/shortInterest"

    all_records = []
    batch_size = 50
    ticker_batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    for batch_idx, batch in enumerate(ticker_batches):
        # FINRA API uses a specific query format
        payload = {
            "fields": ["symbolCode", "settlementDate", "currentShortPositionQuantity",
                        "previousShortPositionQuantity", "averageDailyVolumeQuantity",
                        "daysToCoverQuantity", "changePreviousNumber", "changePercent"],
            "dateRangeFilters": [
                {
                    "fieldName": "settlementDate",
                    "startDate": (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d"),
                    "endDate": datetime.now().strftime("%Y-%m-%d"),
                }
            ],
            "domainFilters": [
                {
                    "fieldName": "symbolCode",
                    "values": batch,
                }
            ],
            "limit": 5000,
            "offset": 0,
            "sortFields": ["-settlementDate"],
        }

        try:
            resp = requests.post(url, json=payload, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    all_records.extend(data)
                    print(f"  Batch {batch_idx + 1}/{len(ticker_batches)}: got {len(data)} records")
                else:
                    print(f"  Batch {batch_idx + 1}/{len(ticker_batches)}: empty response")
            elif resp.status_code == 403:
                print(f"  FINRA API returned 403 Forbidden (may require registration)")
                if batch_idx == 0:
                    return None
                break
            elif resp.status_code == 429:
                print(f"  Rate limited, waiting 5s...")
                time.sleep(5)
                continue
            else:
                print(f"  FINRA API returned {resp.status_code}: {resp.text[:200]}")
                if batch_idx == 0:
                    return None
                break
            time.sleep(0.3)
        except Exception as e:
            print(f"  Error: {e}")
            if batch_idx == 0:
                return None
            break

    if not all_records:
        return None

    df = pd.DataFrame(all_records)
    # Standardize column names
    col_map = {
        "symbolCode": "ticker",
        "settlementDate": "settlement_date",
        "currentShortPositionQuantity": "short_interest",
        "averageDailyVolumeQuantity": "avg_daily_volume",
        "daysToCoverQuantity": "days_to_cover",
        "previousShortPositionQuantity": "prev_short_interest",
        "changePreviousNumber": "change_prev",
        "changePercent": "change_pct",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return df


def try_finra_consolidated_api(tickers: list) -> pd.DataFrame | None:
    """Try the consolidated FINRA short interest endpoint."""
    print("Attempting FINRA consolidated short interest API...")

    url = "https://api.finra.org/data/group/consolidatedShortInterest/name/consolidatedShortInterestTable"

    all_records = []
    batch_size = 50
    ticker_batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    for batch_idx, batch in enumerate(ticker_batches):
        payload = {
            "fields": ["symbolCode", "settlementDate", "currentShortPositionQuantity",
                        "averageDailyVolumeQuantity", "daysToCoverQuantity"],
            "dateRangeFilters": [
                {
                    "fieldName": "settlementDate",
                    "startDate": (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d"),
                    "endDate": datetime.now().strftime("%Y-%m-%d"),
                }
            ],
            "domainFilters": [
                {
                    "fieldName": "symbolCode",
                    "values": batch,
                }
            ],
            "limit": 5000,
            "offset": 0,
            "sortFields": ["-settlementDate"],
        }

        try:
            resp = requests.post(url, json=payload, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    all_records.extend(data)
                    print(f"  Batch {batch_idx + 1}/{len(ticker_batches)}: got {len(data)} records")
                else:
                    print(f"  Batch {batch_idx + 1}/{len(ticker_batches)}: empty response")
            elif resp.status_code in (403, 401):
                print(f"  FINRA consolidated API returned {resp.status_code}")
                if batch_idx == 0:
                    return None
                break
            else:
                print(f"  FINRA consolidated API returned {resp.status_code}")
                if batch_idx == 0:
                    return None
                break
            time.sleep(0.3)
        except Exception as e:
            print(f"  Error: {e}")
            if batch_idx == 0:
                return None
            break

    if not all_records:
        return None

    df = pd.DataFrame(all_records)
    col_map = {
        "symbolCode": "ticker",
        "settlementDate": "settlement_date",
        "currentShortPositionQuantity": "short_interest",
        "averageDailyVolumeQuantity": "avg_daily_volume",
        "daysToCoverQuantity": "days_to_cover",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return df


def try_sec_short_interest(tickers: list) -> pd.DataFrame | None:
    """Try getting short volume data from FINRA's publicly published daily short volume files."""
    print("Attempting FINRA daily short volume reports...")

    all_records = []
    # FINRA publishes daily short volume data at this location
    base_url = "https://cdn.finra.org/equity/regsho/daily"

    # Try recent dates
    ticker_set = set(tickers)
    today = datetime.now()
    dates_tried = 0
    dates_found = 0

    for days_back in range(0, 90):  # Last 90 calendar days (~3 months)
        date = today - timedelta(days=days_back)
        if date.weekday() >= 5:  # Skip weekends
            continue

        date_str = date.strftime("%Y%m%d")
        # Try both consolidated and individual exchange files
        urls = [
            f"{base_url}/CNMSshvol{date_str}.txt",
            f"{base_url}/FNSQshvol{date_str}.txt",
            f"{base_url}/FNYXshvol{date_str}.txt",
            f"{base_url}/FORFshvol{date_str}.txt",
        ]

        dates_tried += 1
        found_data = False

        for file_url in urls:
            try:
                resp = requests.get(file_url, timeout=15)
                if resp.status_code == 200:
                    lines = resp.text.strip().split("\n")
                    if len(lines) < 2:
                        continue
                    # Parse pipe-delimited file
                    # Format: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
                    for line in lines[1:]:  # Skip header
                        parts = line.strip().split("|")
                        if len(parts) >= 5:
                            symbol = parts[1]
                            if symbol in ticker_set:
                                record = {
                                    "date": parts[0],
                                    "ticker": symbol,
                                    "short_volume": int(parts[2]) if parts[2] else 0,
                                    "short_exempt_volume": int(parts[3]) if parts[3] else 0,
                                    "total_volume": int(parts[4]) if parts[4] else 0,
                                    "market": parts[5] if len(parts) > 5 else "",
                                }
                                all_records.append(record)
                    found_data = True
            except Exception as e:
                pass  # File might not exist for this date

        if found_data:
            dates_found += 1
            print(f"  {date.strftime('%Y-%m-%d')}: found short volume data")
        elif dates_tried <= 3:
            print(f"  {date.strftime('%Y-%m-%d')}: no data available")

        time.sleep(0.1)

    print(f"  Tried {dates_tried} dates, found data for {dates_found}")

    if not all_records:
        return None

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    # Aggregate across markets for each ticker/date
    agg_df = df.groupby(["ticker", "date"]).agg(
        short_volume=("short_volume", "sum"),
        short_exempt_volume=("short_exempt_volume", "sum"),
        total_volume=("total_volume", "sum"),
    ).reset_index()

    agg_df["short_ratio"] = agg_df["short_volume"] / agg_df["total_volume"].replace(0, float("nan"))

    return agg_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_tickers()
    print(f"FINRA Short Interest Data Downloader")
    print(f"  Universe: {len(tickers)} tickers")
    print()

    # Try FINRA API first
    df = try_finra_api(tickers)

    if df is None:
        print()
        df = try_finra_consolidated_api(tickers)

    if df is None:
        print()
        # Fall back to daily short volume files (publicly available, no auth)
        df = try_sec_short_interest(tickers)

    if df is not None and len(df) > 0:
        # Ensure date column
        date_col = None
        for col in ["settlement_date", "date"]:
            if col in df.columns:
                date_col = col
                break

        if date_col and date_col != "settlement_date":
            df = df.rename(columns={date_col: "settlement_date"})

        if "settlement_date" in df.columns:
            df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
            df = df.sort_values(["ticker", "settlement_date"])

        print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        if "ticker" in df.columns:
            print(f"Unique tickers: {df['ticker'].nunique()}")
        if "settlement_date" in df.columns:
            print(f"Date range: {df['settlement_date'].min()} to {df['settlement_date'].max()}")

        df.to_parquet(OUTPUT_FILE, index=False)
        print(f"\nSaved to {OUTPUT_FILE}")
        print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    else:
        print("\n" + "=" * 70)
        print("WARNING: Could not download short interest data automatically.")
        print("FINRA API may require registration/authentication.")
        print()
        print("Manual options:")
        print("  1. Register at https://developer.finra.org/ for API access")
        print("  2. Download from https://www.finra.org/finra-data/browse-catalog/short-interest/data")
        print("  3. Use a broker's short interest data (e.g., Interactive Brokers)")
        print()
        print("Creating empty parquet with expected schema for downstream compatibility...")

        schema_df = pd.DataFrame(
            columns=["ticker", "settlement_date", "short_volume", "short_exempt_volume",
                      "total_volume", "short_ratio"]
        )
        schema_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"Saved empty schema to {OUTPUT_FILE}")
        print("=" * 70)


if __name__ == "__main__":
    main()
