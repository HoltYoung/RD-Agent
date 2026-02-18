"""
Utility module to load alt-data sources into pandas DataFrames aligned
to the (datetime, instrument) MultiIndex used by daily_pv.h5.

Each load_*() function:
  - Reads from parquet files in universe_data/alt_data/
  - Handles missing files gracefully (returns empty DataFrame)
  - Forward-fills values to daily frequency where needed
  - Returns a DataFrame with (datetime, instrument) MultiIndex

Usage:
    from scripts.alt_data.load_alt_data import load_all_alt_data
    df = load_all_alt_data()
"""

from __future__ import annotations

import glob as _glob
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALT_DATA_DIR = PROJECT_ROOT / "universe_data" / "alt_data"
DAILY_PV_PATH = PROJECT_ROOT / "git_ignore_folder" / "factor_implementation_source_data" / "daily_pv.h5"


def _get_reference_index() -> pd.MultiIndex | None:
    """Load the (datetime, instrument) MultiIndex from daily_pv.h5 for alignment."""
    if not DAILY_PV_PATH.exists():
        return None
    try:
        with pd.HDFStore(str(DAILY_PV_PATH), "r") as store:
            keys = store.keys()
            if keys:
                return store[keys[0]].index
    except Exception:
        pass
    return None


def _align_to_reference(
    df: pd.DataFrame,
    ref_index: pd.MultiIndex | None,
    forward_fill: bool = True,
) -> pd.DataFrame:
    """
    Reindex *df* (which must have a (datetime, instrument) MultiIndex)
    to match the reference index from daily_pv.h5. Forward-fills gaps if requested.
    """
    if ref_index is None or df.empty:
        return df

    # Only keep instruments that exist in the reference
    ref_instruments = ref_index.get_level_values("instrument").unique()
    df_instruments = df.index.get_level_values("instrument").unique()
    common = ref_instruments.intersection(df_instruments)
    if common.empty:
        return pd.DataFrame(index=ref_index, columns=df.columns, dtype="float64")

    # Reindex to the reference grid
    df = df.reindex(ref_index)

    if forward_fill:
        # Forward-fill within each instrument
        df = df.groupby(level="instrument", group_keys=False).ffill()

    return df


def _safe_read_parquet(path: Path | str) -> pd.DataFrame:
    """Read a parquet file, returning an empty DataFrame if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"WARNING: Failed to read {path}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------


def load_insider_trading() -> pd.DataFrame:
    """
    Load SEC insider trading data.

    Returns DataFrame with (datetime, instrument) MultiIndex and columns:
      - net_insider_buys_30d: net shares acquired minus disposed in rolling 30d
      - insider_buy_ratio_90d: fraction of buy transactions in rolling 90d
      - insider_transaction_count_30d: total transactions in rolling 30d
    """
    path = ALT_DATA_DIR / "sec_insider" / "insider_combined.parquet"
    raw = _safe_read_parquet(path)
    if raw.empty:
        return pd.DataFrame()

    ref_index = _get_reference_index()

    # Normalize columns
    raw["transaction_date"] = pd.to_datetime(raw["transaction_date"], errors="coerce")
    raw = raw.dropna(subset=["transaction_date", "ticker"])
    raw["ticker"] = raw["ticker"].astype(str).str.strip()

    # Determine buy/sell direction
    # acquired_disposed: A = acquired (buy), D = disposed (sell)
    raw["signed_shares"] = raw["shares"].fillna(0)
    raw.loc[raw["acquired_disposed"] == "D", "signed_shares"] *= -1

    raw["is_buy"] = (raw["acquired_disposed"] == "A").astype(int)
    raw["tx_count"] = 1

    # Aggregate to daily per ticker
    daily = (
        raw.groupby([raw["transaction_date"].dt.normalize(), "ticker"])
        .agg(
            net_shares=("signed_shares", "sum"),
            buy_count=("is_buy", "sum"),
            tx_count=("tx_count", "sum"),
        )
        .reset_index()
    )
    daily = daily.rename(columns={"transaction_date": "datetime", "ticker": "instrument"})
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    daily = daily.set_index(["datetime", "instrument"]).sort_index()

    # Compute rolling features per instrument
    results = []
    for instrument, grp in daily.groupby(level="instrument"):
        # Ensure a proper date index for rolling
        g = grp.droplevel("instrument")
        # Resample to calendar daily and fill with 0 (no activity days)
        g = g.resample("D").sum().fillna(0)

        out = pd.DataFrame(index=g.index)
        out["net_insider_buys_30d"] = g["net_shares"].rolling(30, min_periods=1).sum()
        out["insider_transaction_count_30d"] = g["tx_count"].rolling(30, min_periods=1).sum()

        buy_sum_90 = g["buy_count"].rolling(90, min_periods=1).sum()
        tx_sum_90 = g["tx_count"].rolling(90, min_periods=1).sum()
        out["insider_buy_ratio_90d"] = (buy_sum_90 / tx_sum_90.replace(0, np.nan))

        out["instrument"] = instrument
        out.index.name = "datetime"
        out = out.reset_index().set_index(["datetime", "instrument"])
        results.append(out)

    if not results:
        return pd.DataFrame()

    df = pd.concat(results).sort_index()
    df = df[["net_insider_buys_30d", "insider_buy_ratio_90d", "insider_transaction_count_30d"]]
    return _align_to_reference(df, ref_index)


def load_ftd() -> pd.DataFrame:
    """
    Load SEC Failure-to-Deliver data.

    Returns DataFrame with (datetime, instrument) MultiIndex and columns:
      - ftd_quantity: quantity of fails-to-deliver
      - ftd_dollar_value: quantity * price
    Forward-filled from bimonthly to daily.
    """
    path = ALT_DATA_DIR / "sec_ftd" / "ftd_combined.parquet"
    raw = _safe_read_parquet(path)
    if raw.empty:
        return pd.DataFrame()

    ref_index = _get_reference_index()

    raw["settlement_date"] = pd.to_datetime(raw["settlement_date"], errors="coerce")
    raw = raw.dropna(subset=["settlement_date", "symbol"])
    raw["symbol"] = raw["symbol"].astype(str).str.strip()
    raw["ftd_dollar_value"] = raw["quantity_fails"] * raw["price"].fillna(0)

    # Aggregate per day per symbol (may have multiple entries)
    daily = (
        raw.groupby([raw["settlement_date"].dt.normalize(), "symbol"])
        .agg(
            ftd_quantity=("quantity_fails", "sum"),
            ftd_dollar_value=("ftd_dollar_value", "sum"),
        )
        .reset_index()
    )
    daily = daily.rename(columns={"settlement_date": "datetime", "symbol": "instrument"})
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    daily = daily.set_index(["datetime", "instrument"]).sort_index()

    return _align_to_reference(daily, ref_index, forward_fill=True)


def load_short_volume() -> pd.DataFrame:
    """
    Load FINRA daily short volume data.

    Returns DataFrame with (datetime, instrument) MultiIndex and columns:
      - short_ratio: ShortVolume / TotalVolume
      - short_volume_zscore_20d: z-score of short_ratio over 20-day window
    """
    path = ALT_DATA_DIR / "finra_short_volume" / "finra_short_volume.parquet"
    raw = _safe_read_parquet(path)
    if raw.empty:
        return pd.DataFrame()

    ref_index = _get_reference_index()

    # Date column is in YYYYMMDD string format
    raw["datetime"] = pd.to_datetime(raw["Date"], format="%Y%m%d", errors="coerce")
    raw = raw.dropna(subset=["datetime", "Symbol"])
    raw["Symbol"] = raw["Symbol"].astype(str).str.strip()

    raw["short_ratio"] = raw["ShortVolume"] / raw["TotalVolume"].replace(0, np.nan)

    daily = raw[["datetime", "Symbol", "short_ratio"]].copy()
    daily = daily.rename(columns={"Symbol": "instrument"})
    daily = daily.set_index(["datetime", "instrument"]).sort_index()

    # Remove duplicates (multiple markets aggregated)
    daily = daily.groupby(level=["datetime", "instrument"]).mean()

    # Compute z-score per instrument
    results = []
    for instrument, grp in daily.groupby(level="instrument"):
        g = grp.droplevel("instrument").copy()
        rolling_mean = g["short_ratio"].rolling(20, min_periods=5).mean()
        rolling_std = g["short_ratio"].rolling(20, min_periods=5).std()
        g["short_volume_zscore_20d"] = (g["short_ratio"] - rolling_mean) / rolling_std.replace(0, np.nan)
        g["instrument"] = instrument
        g.index.name = "datetime"
        g = g.reset_index().set_index(["datetime", "instrument"])
        results.append(g)

    if not results:
        return pd.DataFrame()

    df = pd.concat(results).sort_index()
    df = df[["short_ratio", "short_volume_zscore_20d"]]
    return _align_to_reference(df, ref_index, forward_fill=True)


def load_short_interest() -> pd.DataFrame:
    """
    Load FINRA short interest data.

    Returns DataFrame with (datetime, instrument) MultiIndex and columns:
      - short_interest_ratio: short ratio from FINRA data
    """
    path = ALT_DATA_DIR / "finra_short_interest" / "finra_short_interest.parquet"
    raw = _safe_read_parquet(path)
    if raw.empty:
        return pd.DataFrame()

    ref_index = _get_reference_index()

    raw["settlement_date"] = pd.to_datetime(raw["settlement_date"], errors="coerce")
    raw = raw.dropna(subset=["settlement_date", "ticker"])
    raw["ticker"] = raw["ticker"].astype(str).str.strip()

    daily = raw[["settlement_date", "ticker", "short_ratio"]].copy()
    daily = daily.rename(columns={"settlement_date": "datetime", "ticker": "instrument", "short_ratio": "short_interest_ratio"})
    daily["datetime"] = pd.to_datetime(daily["datetime"])
    daily = daily.set_index(["datetime", "instrument"]).sort_index()

    # Remove any duplicates
    daily = daily[~daily.index.duplicated(keep="last")]

    return _align_to_reference(daily, ref_index, forward_fill=True)


def load_options_metrics() -> pd.DataFrame:
    """
    Load Yahoo options metrics (snapshot data).

    Returns DataFrame with (datetime, instrument) MultiIndex and columns:
      - put_call_ratio
      - total_put_oi
      - total_call_oi
    Forward-filled (snapshot data, no date column -- uses today's date as snapshot date).
    """
    path = ALT_DATA_DIR / "yahoo_options" / "options_metrics.parquet"
    raw = _safe_read_parquet(path)
    if raw.empty:
        return pd.DataFrame()

    ref_index = _get_reference_index()

    raw = raw.dropna(subset=["ticker"])
    raw["ticker"] = raw["ticker"].astype(str).str.strip()

    # Options metrics is a snapshot file (no date column).
    # Use the file's modification time as the snapshot date.
    try:
        mtime = pd.Timestamp(path.stat().st_mtime, unit="s").normalize()
    except Exception:
        mtime = pd.Timestamp.now().normalize()

    df = raw[["ticker", "put_call_ratio", "total_put_oi", "total_call_oi"]].copy()
    df = df.rename(columns={"ticker": "instrument"})
    df["datetime"] = mtime
    df = df.set_index(["datetime", "instrument"]).sort_index()

    return _align_to_reference(df, ref_index, forward_fill=True)


def load_reddit_sentiment() -> pd.DataFrame:
    """
    Load Reddit sentiment data (ApeWisdom-style daily snapshots).

    Returns DataFrame with (datetime, instrument) MultiIndex and columns:
      - reddit_mentions: number of mentions
      - reddit_rank: rank on the subreddit
    Forward-filled from snapshot data.
    """
    pattern = str(ALT_DATA_DIR / "reddit_sentiment" / "reddit_sentiment_2*.parquet")
    files = sorted(_glob.glob(pattern))
    if not files:
        return pd.DataFrame()

    ref_index = _get_reference_index()

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna(subset=["ticker", "fetch_date"])
    raw["ticker"] = raw["ticker"].astype(str).str.strip()
    raw["datetime"] = pd.to_datetime(raw["fetch_date"], errors="coerce")
    raw = raw.dropna(subset=["datetime"])

    df = raw[["datetime", "ticker", "mentions", "rank"]].copy()
    df = df.rename(columns={"ticker": "instrument", "mentions": "reddit_mentions", "rank": "reddit_rank"})
    df["datetime"] = df["datetime"].dt.normalize()
    df = df.set_index(["datetime", "instrument"]).sort_index()

    # Remove duplicates (keep last)
    df = df[~df.index.duplicated(keep="last")]

    return _align_to_reference(df, ref_index, forward_fill=True)


def load_wikipedia_pageviews() -> pd.DataFrame:
    """
    Load Wikipedia pageviews data.

    Returns DataFrame with (datetime, instrument) MultiIndex and columns:
      - wiki_views_7d_avg: 7-day rolling average of daily views
      - wiki_views_change_7d: percent change in 7-day average vs prior 7 days
    """
    path = ALT_DATA_DIR / "wikipedia_pageviews" / "wikipedia_pageviews_combined.parquet"
    raw = _safe_read_parquet(path)
    if raw.empty:
        return pd.DataFrame()

    ref_index = _get_reference_index()

    # Date is in YYYYMMDD string format
    raw["datetime"] = pd.to_datetime(raw["date"], format="%Y%m%d", errors="coerce")
    raw = raw.dropna(subset=["datetime", "ticker"])
    raw["ticker"] = raw["ticker"].astype(str).str.strip()

    daily = raw[["datetime", "ticker", "views"]].copy()
    daily = daily.rename(columns={"ticker": "instrument"})
    daily = daily.set_index(["datetime", "instrument"]).sort_index()

    # Remove duplicates
    daily = daily.groupby(level=["datetime", "instrument"]).sum()

    # Compute rolling features per instrument
    results = []
    for instrument, grp in daily.groupby(level="instrument"):
        g = grp.droplevel("instrument").copy()
        # Resample to daily and fill missing days with 0
        g = g.resample("D").sum().fillna(0)

        out = pd.DataFrame(index=g.index)
        out["wiki_views_7d_avg"] = g["views"].rolling(7, min_periods=1).mean()
        # Percent change: current 7d avg vs prior 7d avg
        avg_7d = g["views"].rolling(7, min_periods=1).mean()
        avg_7d_prior = avg_7d.shift(7)
        out["wiki_views_change_7d"] = ((avg_7d - avg_7d_prior) / avg_7d_prior.replace(0, np.nan)) * 100

        out["instrument"] = instrument
        out.index.name = "datetime"
        out = out.reset_index().set_index(["datetime", "instrument"])
        results.append(out)

    if not results:
        return pd.DataFrame()

    df = pd.concat(results).sort_index()
    return _align_to_reference(df, ref_index, forward_fill=True)


def load_fred_macro() -> pd.DataFrame:
    """
    Load FRED macro economic data.

    Returns DataFrame with just date index (no instrument dimension).
    All 14 FRED series as columns, forward-filled to daily frequency.
    """
    path = ALT_DATA_DIR / "fred_macro" / "fred_macro.parquet"
    raw = _safe_read_parquet(path)
    if raw.empty:
        return pd.DataFrame()

    # Index is already 'date' as datetime
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    # Resample to daily and forward-fill
    daily = raw.resample("D").last()
    daily = daily.ffill()

    daily.index.name = "datetime"
    return daily


def load_congressional_trading() -> pd.DataFrame:
    """
    Load Congressional trading disclosure data.

    Note: This data has limited ticker-level granularity (it contains filer info
    but not always specific tickers). Returns the raw data with a date index.
    """
    path = ALT_DATA_DIR / "congressional_trading" / "congressional_trading.parquet"
    raw = _safe_read_parquet(path)
    if raw.empty:
        return pd.DataFrame()

    raw["FilingDate"] = pd.to_datetime(raw["FilingDate"], errors="coerce")
    raw = raw.dropna(subset=["FilingDate"])
    raw = raw.set_index("FilingDate").sort_index()
    raw.index.name = "datetime"
    return raw


def load_all_alt_data() -> pd.DataFrame:
    """
    Load and merge all alt-data sources into a single DataFrame
    with (datetime, instrument) MultiIndex.

    FRED macro data is broadcast across all instruments.
    Congressional trading data is excluded (no ticker mapping).
    """
    ref_index = _get_reference_index()

    # Load all instrument-level sources
    loaders = [
        ("insider", load_insider_trading),
        ("ftd", load_ftd),
        ("short_volume", load_short_volume),
        ("short_interest", load_short_interest),
        ("options", load_options_metrics),
        ("reddit", load_reddit_sentiment),
        ("wikipedia", load_wikipedia_pageviews),
    ]

    merged = None
    for name, loader_fn in loaders:
        print(f"Loading {name}...")
        try:
            df = loader_fn()
            if df.empty:
                print(f"  {name}: empty (no data or missing file)")
                continue
            print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} cols -> {list(df.columns)}")
            if merged is None:
                merged = df
            else:
                merged = merged.join(df, how="outer")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    # Add FRED macro data (broadcast across instruments)
    print("Loading fred_macro...")
    try:
        fred = load_fred_macro()
        if not fred.empty and merged is not None:
            # Join FRED on the datetime level only
            dates = merged.index.get_level_values("datetime")
            instruments = merged.index.get_level_values("instrument")

            fred_aligned = fred.reindex(dates)
            fred_aligned.index = merged.index
            merged = merged.join(fred_aligned, rsuffix="_fred")
            print(f"  fred_macro: {fred.shape[0]} rows, {fred.shape[1]} cols")
        elif not fred.empty:
            print(f"  fred_macro: {fred.shape[0]} rows (standalone, no merge target)")
    except Exception as e:
        print(f"  fred_macro: ERROR - {e}")

    if merged is None:
        merged = pd.DataFrame()

    print(f"\nFinal merged shape: {merged.shape}")
    return merged


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

def _sanity_check():
    """Quick sanity check: load each source and print shape/columns."""
    print("=" * 70)
    print("  Alt-Data Loader Sanity Check")
    print("=" * 70)

    checks = [
        ("load_insider_trading", load_insider_trading),
        ("load_ftd", load_ftd),
        ("load_short_volume", load_short_volume),
        ("load_short_interest", load_short_interest),
        ("load_options_metrics", load_options_metrics),
        ("load_reddit_sentiment", load_reddit_sentiment),
        ("load_wikipedia_pageviews", load_wikipedia_pageviews),
        ("load_fred_macro", load_fred_macro),
        ("load_congressional_trading", load_congressional_trading),
    ]

    for name, fn in checks:
        print(f"\n--- {name} ---")
        try:
            df = fn()
            if df.empty:
                print(f"  Result: EMPTY DataFrame")
            else:
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Index names: {df.index.names}")
                if hasattr(df.index, "get_level_values") and "datetime" in (df.index.names or []):
                    dates = df.index.get_level_values("datetime")
                    print(f"  Date range: {dates.min()} -> {dates.max()}")
                elif df.index.name == "datetime":
                    print(f"  Date range: {df.index.min()} -> {df.index.max()}")
                print(f"  Non-null counts:")
                for col in df.columns[:5]:
                    print(f"    {col}: {df[col].notna().sum()}")
                if len(df.columns) > 5:
                    print(f"    ... and {len(df.columns) - 5} more columns")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("  Sanity check complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    _sanity_check()
