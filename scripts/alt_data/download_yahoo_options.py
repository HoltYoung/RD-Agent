"""
Download Yahoo Finance options chains for universe tickers using yfinance.
Saves raw chains and computed aggregate metrics as parquet files.

Re-runnable: overwrites with latest data each run.
Rate-limited with 0.5s delay between tickers.
"""

import os
import time
import traceback

import pandas as pd
import yfinance as yf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TICKER_FILE = os.path.join(PROJECT_ROOT, "universe_data", "ticker_list.txt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "universe_data", "alt_data", "yahoo_options")
CHAINS_FILE = os.path.join(OUTPUT_DIR, "options_chains.parquet")
METRICS_FILE = os.path.join(OUTPUT_DIR, "options_metrics.parquet")


def load_tickers():
    with open(TICKER_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]


def compute_max_pain(chain_df):
    """Compute max pain strike for a ticker's option chain.
    Max pain = strike price where total dollar value of outstanding puts+calls
    exercised causes maximum loss for option holders (minimum loss for writers).
    """
    if chain_df.empty:
        return None

    strikes = chain_df["strike"].unique()
    if len(strikes) == 0:
        return None

    calls = chain_df[chain_df["type"] == "call"]
    puts = chain_df[chain_df["type"] == "put"]

    min_pain = float("inf")
    max_pain_strike = None

    for s in strikes:
        # For calls: loss if stock at s = sum of max(0, s - strike) * openInterest for all calls
        call_pain = 0
        for _, row in calls.iterrows():
            if s > row["strike"]:
                call_pain += (s - row["strike"]) * row.get("openInterest", 0)

        # For puts: loss if stock at s = sum of max(0, strike - s) * openInterest for all puts
        put_pain = 0
        for _, row in puts.iterrows():
            if s < row["strike"]:
                put_pain += (row["strike"] - s) * row.get("openInterest", 0)

        total_pain = call_pain + put_pain
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = s

    return max_pain_strike


def fetch_options_for_ticker(ticker_str):
    """Fetch all options chains for a single ticker. Returns (chain_df, metrics_dict) or (None, None)."""
    try:
        tk = yf.Ticker(ticker_str)
        expirations = tk.options  # tuple of expiry date strings

        if not expirations:
            return None, None

        all_chains = []

        for exp_date in expirations:
            try:
                chain = tk.option_chain(exp_date)

                if chain.calls is not None and len(chain.calls) > 0:
                    calls = chain.calls.copy()
                    calls["type"] = "call"
                    calls["expiration"] = exp_date
                    calls["ticker"] = ticker_str
                    all_chains.append(calls)

                if chain.puts is not None and len(chain.puts) > 0:
                    puts = chain.puts.copy()
                    puts["type"] = "put"
                    puts["expiration"] = exp_date
                    puts["ticker"] = ticker_str
                    all_chains.append(puts)
            except Exception:
                continue

        if not all_chains:
            return None, None

        chain_df = pd.concat(all_chains, ignore_index=True)

        # Keep relevant columns (some may vary by yfinance version)
        keep_cols = [
            "ticker", "expiration", "type", "strike", "bid", "ask",
            "volume", "openInterest", "impliedVolatility",
            "lastPrice", "contractSymbol", "inTheMoney",
        ]
        available_cols = [c for c in keep_cols if c in chain_df.columns]
        chain_df = chain_df[available_cols]

        # Fill NaN open interest and volume with 0
        for col in ["openInterest", "volume"]:
            if col in chain_df.columns:
                chain_df[col] = chain_df[col].fillna(0)

        # Compute aggregate metrics
        total_call_oi = chain_df.loc[chain_df["type"] == "call", "openInterest"].sum() if "openInterest" in chain_df.columns else 0
        total_put_oi = chain_df.loc[chain_df["type"] == "put", "openInterest"].sum() if "openInterest" in chain_df.columns else 0
        put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else None
        max_pain = compute_max_pain(chain_df)

        metrics = {
            "ticker": ticker_str,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "put_call_ratio": put_call_ratio,
            "max_pain_strike": max_pain,
            "num_expirations": len(expirations),
            "num_contracts": len(chain_df),
        }

        return chain_df, metrics

    except Exception as e:
        print(f"  Error for {ticker_str}: {e}")
        return None, None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers")

    # Process all tickers (may take a while with rate limiting)
    all_chains = []
    all_metrics = []
    success_count = 0
    fail_count = 0

    for i, ticker in enumerate(tickers):
        chain_df, metrics = fetch_options_for_ticker(ticker)

        if chain_df is not None:
            all_chains.append(chain_df)
            all_metrics.append(metrics)
            success_count += 1
        else:
            fail_count += 1

        if (i + 1) % 25 == 0 or i == len(tickers) - 1:
            print(f"  Progress: {i + 1}/{len(tickers)} tickers ({success_count} success, {fail_count} failed/no options)")

        # Rate limit: 0.5s between tickers
        time.sleep(0.5)

        # Save intermediate results every 100 tickers in case of interruption
        if (i + 1) % 100 == 0 and all_chains:
            interim_chains = pd.concat(all_chains, ignore_index=True)
            interim_chains.to_parquet(CHAINS_FILE, index=False)
            interim_metrics = pd.DataFrame(all_metrics)
            interim_metrics.to_parquet(METRICS_FILE, index=False)
            print(f"  Saved interim results ({len(interim_chains)} chain rows, {len(interim_metrics)} tickers)")

    if not all_chains:
        print("No options data downloaded.")
        return

    # Save final results
    final_chains = pd.concat(all_chains, ignore_index=True)
    final_chains.to_parquet(CHAINS_FILE, index=False)

    final_metrics = pd.DataFrame(all_metrics)
    final_metrics.to_parquet(METRICS_FILE, index=False)

    print(f"\nResults saved:")
    print(f"  Options chains: {len(final_chains)} rows -> {CHAINS_FILE}")
    print(f"  Options metrics: {len(final_metrics)} tickers -> {METRICS_FILE}")
    print(f"\nSummary:")
    print(f"  Tickers with options: {success_count}")
    print(f"  Tickers without options: {fail_count}")
    if len(final_metrics) > 0:
        print(f"  Avg put/call ratio: {final_metrics['put_call_ratio'].mean():.3f}")
        print(f"  Median total call OI: {final_metrics['total_call_oi'].median():.0f}")
        print(f"  Median total put OI: {final_metrics['total_put_oi'].median():.0f}")


if __name__ == "__main__":
    main()
