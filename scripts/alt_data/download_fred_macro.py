"""
Download FRED macro data series and save as a single parquet file.
Uses fredapi if FRED_API_KEY is set, otherwise falls back to direct CSV download from FRED website.
Re-runnable: overwrites existing output.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SERIES_IDS = [
    "DGS10",       # 10-Year Treasury
    "DGS2",        # 2-Year Treasury
    "T10Y2Y",      # 10Y-2Y Spread
    "FEDFUNDS",    # Fed Funds Rate
    "CPIAUCSL",    # CPI
    "UNRATE",      # Unemployment Rate
    "ICSA",        # Initial Jobless Claims
    "INDPRO",      # Industrial Production
    "M2SL",        # M2 Money Supply
    "VIXCLS",      # VIX
    "DCOILWTICO",  # Crude Oil WTI
    "SP500",       # S&P 500
    "BAMLH0A0HYM2",  # High Yield Spread
    "UMCSENT",     # Consumer Sentiment
]

OUTPUT_DIR = Path("C:/Users/holty/projects/rd-agent/universe_data/alt_data/fred_macro")
OUTPUT_FILE = OUTPUT_DIR / "fred_macro.parquet"

# 5 years of history
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5 * 365)


def download_via_fredapi(api_key: str) -> pd.DataFrame:
    """Download using the fredapi package."""
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    frames = {}
    for sid in SERIES_IDS:
        print(f"  Downloading {sid} via fredapi...")
        try:
            s = fred.get_series(sid, observation_start=START_DATE, observation_end=END_DATE)
            s.name = sid
            frames[sid] = s
        except Exception as e:
            print(f"  WARNING: Failed to download {sid}: {e}")
    df = pd.DataFrame(frames)
    df.index.name = "date"
    return df


def download_via_csv() -> pd.DataFrame:
    """Fallback: download CSVs directly from FRED website (no API key needed)."""
    frames = {}
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")
    for sid in SERIES_IDS:
        url = (
            f"https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans"
            f"&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on"
            f"&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0"
            f"&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes"
            f"&id={sid}&scale=left&cosd={start_str}&coed={end_str}"
            f"&line_color=%234572a7&link_values=false&line_style=solid"
            f"&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0"
            f"&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01"
            f"&line_index=1&transformation=lin&vintage_date={end_str}"
            f"&revision_date={end_str}&nd={start_str}"
        )
        print(f"  Downloading {sid} via CSV...")
        try:
            s = pd.read_csv(url, index_col=0, parse_dates=True)
            # The CSV has column named after the series ID with '.' for missing
            col = s.columns[0]
            s[col] = pd.to_numeric(s[col], errors="coerce")
            frames[sid] = s[col].rename(sid)
        except Exception as e:
            print(f"  WARNING: Failed to download {sid}: {e}")
    df = pd.DataFrame(frames)
    df.index.name = "date"
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("FRED_API_KEY", "")

    print(f"FRED Macro Data Downloader")
    print(f"  Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"  Series: {len(SERIES_IDS)}")
    print()

    if api_key:
        print("Using fredapi (API key found)")
        df = download_via_fredapi(api_key)
    else:
        print("No FRED_API_KEY found. Using direct CSV download from FRED website.")
        print("(To use the fredapi, set FRED_API_KEY env var. Free key at https://fred.stlouisfed.org/docs/api/api_key.html)")
        print()
        df = download_via_csv()

    # Forward-fill monthly/weekly series to daily
    df = df.sort_index()
    # Drop rows that are all NaN
    df = df.dropna(how="all")

    print(f"\nResult: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Non-null counts:\n{df.count()}")

    df.to_parquet(OUTPUT_FILE, index=True)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
