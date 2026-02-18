"""
Download FINRA Daily Short Sale Volume data for universe tickers.
Fetches daily CNMS short volume files from FINRA CDN for the last 2 years,
filters to our ticker universe, and saves as parquet.

Re-runnable: skips dates already downloaded and appends new data.
"""

import datetime
import io
import os
import time

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TICKER_FILE = os.path.join(PROJECT_ROOT, "universe_data", "ticker_list.txt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "universe_data", "alt_data", "finra_short_volume")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "finra_short_volume.parquet")

FINRA_URL_TEMPLATE = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"

# Column names in the FINRA pipe-delimited files
FINRA_COLUMNS = ["Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market"]


def load_tickers():
    with open(TICKER_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def get_trading_dates(start_date, end_date):
    """Generate weekday dates between start and end (inclusive)."""
    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday-Friday
            dates.append(current)
        current += datetime.timedelta(days=1)
    return dates


def load_existing_dates():
    """Load dates already present in the output parquet to skip re-downloading."""
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_parquet(OUTPUT_FILE)
        if "Date" in df.columns and len(df) > 0:
            return set(pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d"))
    return set()


def download_single_day(date_str, tickers, session):
    """Download and parse a single day's FINRA short volume file."""
    url = FINRA_URL_TEMPLATE.format(date=date_str)
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 404:
            return None  # No data for this date (holiday, etc.)
        resp.raise_for_status()

        # Parse pipe-delimited text
        df = pd.read_csv(
            io.StringIO(resp.text),
            delimiter="|",
            header=0,
            dtype=str,
        )

        # Normalize column names
        df.columns = [c.strip() for c in df.columns]

        # Filter to our tickers
        if "Symbol" in df.columns:
            df = df[df["Symbol"].isin(tickers)]
        else:
            return None

        if len(df) == 0:
            return None

        # Convert numeric columns
        for col in ["ShortVolume", "ShortExemptVolume", "TotalVolume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching {date_str}: {e}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers from universe")

    existing_dates = load_existing_dates()
    if existing_dates:
        print(f"Found {len(existing_dates)} dates already downloaded")

    # Last 2 years of trading days
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=730)
    all_dates = get_trading_dates(start_date, end_date)
    print(f"Date range: {start_date} to {end_date} ({len(all_dates)} weekdays)")

    # Filter out already-downloaded dates
    dates_to_fetch = [d for d in all_dates if d.strftime("%Y%m%d") not in existing_dates]
    print(f"Dates to fetch: {len(dates_to_fetch)}")

    if not dates_to_fetch:
        print("All dates already downloaded. Nothing to do.")
        return

    session = requests.Session()
    all_frames = []
    success_count = 0
    skip_count = 0
    consecutive_failures = 0

    for i, date in enumerate(dates_to_fetch):
        date_str = date.strftime("%Y%m%d")
        df = download_single_day(date_str, tickers, session)

        if df is not None:
            all_frames.append(df)
            success_count += 1
            consecutive_failures = 0
        else:
            skip_count += 1
            consecutive_failures += 1

        if (i + 1) % 50 == 0 or i == len(dates_to_fetch) - 1:
            print(f"  Progress: {i + 1}/{len(dates_to_fetch)} dates processed ({success_count} success, {skip_count} skipped/failed)")

        # If we get many consecutive failures, we may be rate-limited or hitting bad dates
        if consecutive_failures > 20:
            print("  WARNING: 20+ consecutive failures, pausing 5 seconds...")
            time.sleep(5)
            consecutive_failures = 0

        # Small delay to be polite
        time.sleep(0.1)

    if not all_frames:
        print("No new data downloaded.")
        return

    new_df = pd.concat(all_frames, ignore_index=True)
    print(f"\nNew data: {len(new_df)} rows across {success_count} days")

    # Merge with existing data
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_parquet(OUTPUT_FILE)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    else:
        combined = new_df

    combined.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved {len(combined)} total rows to {OUTPUT_FILE}")

    # Print summary stats
    print(f"\nSummary:")
    print(f"  Unique dates: {combined['Date'].nunique()}")
    print(f"  Unique symbols: {combined['Symbol'].nunique()}")
    print(f"  Date range: {combined['Date'].min()} to {combined['Date'].max()}")


if __name__ == "__main__":
    main()
