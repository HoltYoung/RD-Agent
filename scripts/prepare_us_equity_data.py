"""
One-time data preparation script.
Converts per-ticker CSV files from universe_data/ into HDF5 format
that RD-Agent's factor code expects.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_SOURCE = PROJECT_ROOT / "universe_data" / "universe_data"
DATA_DEST = PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data"
DATA_DEST_DEBUG = PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data_debug"

DEBUG_TICKER_COUNT = 50


def load_prices(data_source: Path) -> pd.DataFrame:
    """Load all ticker price CSVs into a single DataFrame with MultiIndex [datetime, instrument]."""
    all_frames = []
    ticker_dirs = sorted([d for d in data_source.iterdir() if d.is_dir() and d.name.endswith("_Data")])
    print(f"Found {len(ticker_dirs)} ticker directories")

    for i, ticker_dir in enumerate(ticker_dirs):
        ticker = ticker_dir.name.replace("_Data", "")
        price_file = ticker_dir / f"{ticker}_prices.csv"
        if not price_file.exists():
            print(f"  Skipping {ticker}: no prices.csv found")
            continue

        df = pd.read_csv(price_file, parse_dates=["date"])
        df = df.rename(columns={
            "date": "datetime",
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
        })

        # Compute adjustment factor
        if "adj_close" in df.columns and "$close" in df.columns:
            df["$factor"] = df["adj_close"] / df["$close"]
            df["$factor"] = df["$factor"].fillna(1.0)
        else:
            df["$factor"] = 1.0

        df["instrument"] = ticker
        df = df[["datetime", "instrument", "$open", "$high", "$low", "$close", "$volume", "$factor"]]
        df = df.dropna(subset=["datetime"])
        all_frames.append(df)

        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(ticker_dirs)} tickers...")

    print(f"Concatenating {len(all_frames)} ticker DataFrames...")
    combined = pd.concat(all_frames, ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"])
    combined = combined.set_index(["datetime", "instrument"]).sort_index()
    print(f"Combined price data shape: {combined.shape}")
    return combined


def load_fundamentals(data_source: Path, price_index: pd.MultiIndex) -> pd.DataFrame:
    """Load XBRL fundamental data and forward-fill onto the daily price grid.

    Only keeps XBRL columns that appear in at least 30% of tickers to avoid
    a massive sparse matrix (different companies report different XBRL fields).
    """
    ticker_dirs = sorted([d for d in data_source.iterdir() if d.is_dir() and d.name.endswith("_Data")])

    # First pass: count column frequency across tickers
    from collections import Counter
    col_counts = Counter()
    valid_ticker_count = 0

    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name.replace("_Data", "")
        xbrl_file = ticker_dir / f"{ticker}_xbrl.csv"
        if not xbrl_file.exists():
            continue
        # Just read header to count columns
        try:
            header = pd.read_csv(xbrl_file, nrows=0)
        except Exception:
            continue
        xbrl_cols = [c for c in header.columns if c.startswith("xbrl_")]
        if xbrl_cols:
            col_counts.update(xbrl_cols)
            valid_ticker_count += 1

    if valid_ticker_count == 0:
        print("WARNING: No XBRL data found!")
        return pd.DataFrame()

    # Keep columns that appear in >= 30% of tickers
    min_count = int(valid_ticker_count * 0.3)
    common_cols = sorted([col for col, count in col_counts.items() if count >= min_count])
    print(f"  Found {len(col_counts)} unique XBRL columns, keeping {len(common_cols)} that appear in >= 30% of tickers")

    if not common_cols:
        print("WARNING: No common XBRL columns found!")
        return pd.DataFrame()

    # Second pass: load data with only common columns
    all_frames = []
    for i, ticker_dir in enumerate(ticker_dirs):
        ticker = ticker_dir.name.replace("_Data", "")
        xbrl_file = ticker_dir / f"{ticker}_xbrl.csv"
        if not xbrl_file.exists():
            continue

        df = pd.read_csv(xbrl_file)
        if "filing_date" not in df.columns or "ticker" not in df.columns:
            continue

        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df = df.dropna(subset=["filing_date"])
        if df.empty:
            continue

        # Keep only common XBRL columns present in this ticker
        available_cols = [c for c in common_cols if c in df.columns]
        if not available_cols:
            continue

        df = df[["filing_date", "ticker"] + available_cols].copy()
        df = df.rename(columns={"filing_date": "datetime", "ticker": "instrument"})
        df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
        df = df.set_index("datetime")

        # Get the date range for this ticker from the price data
        ticker_dates = price_index.get_level_values("instrument") == ticker
        if not ticker_dates.any():
            continue
        date_range = price_index[ticker_dates].get_level_values("datetime").unique()

        # Reindex to daily and forward-fill
        df = df.reindex(date_range, method="ffill")
        df["instrument"] = ticker
        df = df.reset_index().rename(columns={"index": "datetime"})
        df = df.set_index(["datetime", "instrument"])

        # Convert to numeric
        for col in available_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        all_frames.append(df)

        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(ticker_dirs)} XBRL files...")

    if not all_frames:
        print("WARNING: No XBRL data found!")
        return pd.DataFrame()

    print(f"Concatenating {len(all_frames)} XBRL DataFrames...")
    combined = pd.concat(all_frames)
    print(f"Combined fundamentals shape: {combined.shape}")
    return combined


def generate_readme(dest: Path, price_df: pd.DataFrame, fund_df: pd.DataFrame) -> None:
    """Generate README.md describing the data files."""
    tickers = price_df.index.get_level_values("instrument").unique()
    date_range = price_df.index.get_level_values("datetime")

    fund_cols_desc = ""
    if not fund_df.empty:
        fund_cols = fund_df.columns.tolist()
        bs_cols = [c for c in fund_cols if "xbrl_bs_" in c]
        is_cols = [c for c in fund_cols if "xbrl_is_" in c]
        cf_cols = [c for c in fund_cols if "xbrl_cf_" in c]
        fund_cols_desc = f"""
## fundamentals.h5

XBRL financial statement data forward-filled to daily frequency.
- Read with: `pd.read_hdf("fundamentals.h5", key="data")`
- Index: MultiIndex [datetime, instrument]
- Balance sheet columns ({len(bs_cols)}): prefixed with `xbrl_bs_`
- Income statement columns ({len(is_cols)}): prefixed with `xbrl_is_`
- Cash flow columns ({len(cf_cols)}): prefixed with `xbrl_cf_`
- Total columns: {len(fund_cols)}
- Values are forward-filled from quarterly SEC filings to daily frequency
"""

    readme = f"""# US Equity Factor Data

## daily_pv.h5

Daily price-volume data for {len(tickers)} US equities.
- Read with: `pd.read_hdf("daily_pv.h5", key="data")`
- Index: MultiIndex [datetime, instrument]
- Date range: {date_range.min().strftime('%Y-%m-%d')} to {date_range.max().strftime('%Y-%m-%d')}
- Columns:
  - `$open`: Opening price
  - `$high`: High price
  - `$low`: Low price
  - `$close`: Closing price
  - `$volume`: Trading volume
  - `$factor`: Adjustment factor (adj_close / close)
- Shape: {price_df.shape[0]} rows x {price_df.shape[1]} columns
{fund_cols_desc}
## Usage in factor code

```python
import pandas as pd

# Load price data
df = pd.read_hdf("daily_pv.h5", key="data")
# df.index has levels: ['datetime', 'instrument']
# Access: df['$close'], df['$volume'], etc.

# Load fundamentals (if needed)
fund = pd.read_hdf("fundamentals.h5", key="data")
# fund.index has levels: ['datetime', 'instrument']
# Access: fund['xbrl_is_RevenueFromContractWithCustomerExcludingAssessedTax'], etc.
```
"""
    (dest / "README.md").write_text(readme)


def main():
    print("=" * 60)
    print("US Equity Data Preparation")
    print("=" * 60)

    if not DATA_SOURCE.exists():
        print(f"ERROR: Data source not found at {DATA_SOURCE}")
        print("Make sure you've downloaded the data first.")
        sys.exit(1)

    # Load prices
    print("\n--- Loading price data ---")
    price_df = load_prices(DATA_SOURCE)

    # Load fundamentals
    print("\n--- Loading XBRL fundamental data ---")
    fund_df = load_fundamentals(DATA_SOURCE, price_df.index)

    # Save full dataset
    print(f"\n--- Saving full dataset to {DATA_DEST} ---")
    DATA_DEST.mkdir(parents=True, exist_ok=True)
    price_df.to_hdf(str(DATA_DEST / "daily_pv.h5"), key="data", mode="w")
    print(f"  Saved daily_pv.h5 ({price_df.shape})")
    if not fund_df.empty:
        fund_df.to_hdf(str(DATA_DEST / "fundamentals.h5"), key="data", mode="w")
        print(f"  Saved fundamentals.h5 ({fund_df.shape})")
    generate_readme(DATA_DEST, price_df, fund_df)
    print("  Saved README.md")

    # Save debug dataset (subset of tickers)
    print(f"\n--- Saving debug dataset to {DATA_DEST_DEBUG} ---")
    DATA_DEST_DEBUG.mkdir(parents=True, exist_ok=True)
    all_tickers = price_df.index.get_level_values("instrument").unique()
    debug_tickers = all_tickers[:DEBUG_TICKER_COUNT]
    debug_price = price_df.loc[price_df.index.get_level_values("instrument").isin(debug_tickers)]
    debug_price.to_hdf(str(DATA_DEST_DEBUG / "daily_pv.h5"), key="data", mode="w")
    print(f"  Saved daily_pv.h5 ({debug_price.shape}) - {len(debug_tickers)} tickers")
    if not fund_df.empty:
        debug_fund = fund_df.loc[fund_df.index.get_level_values("instrument").isin(debug_tickers)]
        debug_fund.to_hdf(str(DATA_DEST_DEBUG / "fundamentals.h5"), key="data", mode="w")
        print(f"  Saved fundamentals.h5 ({debug_fund.shape})")
    generate_readme(DATA_DEST_DEBUG, debug_price, fund_df)
    print("  Saved README.md")

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"  Full data: {DATA_DEST}")
    print(f"  Debug data: {DATA_DEST_DEBUG}")
    print("=" * 60)


if __name__ == "__main__":
    main()
