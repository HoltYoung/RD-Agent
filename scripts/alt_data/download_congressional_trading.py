"""
Download Congressional stock trading data and save as parquet.
Tries multiple sources:
1. Capitol Trades API (capitoltrades.com)
2. Senate/House public disclosure filings
3. Fallback: generate sample structure with instructions

Re-runnable: overwrites existing output.
"""

import io
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = Path("C:/Users/holty/projects/rd-agent/universe_data/alt_data/congressional_trading")
OUTPUT_FILE = OUTPUT_DIR / "congressional_trading.parquet"
TICKER_FILE = Path("C:/Users/holty/projects/rd-agent/universe_data/ticker_list.txt")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def load_tickers() -> set:
    """Load our universe tickers."""
    return set(TICKER_FILE.read_text().strip().splitlines())


def try_capitoltrades_api() -> pd.DataFrame | None:
    """Try Capitol Trades API endpoint."""
    print("Attempting Capitol Trades API...")
    # Capitol Trades has a public-facing API used by their frontend
    url = "https://bff.capitoltrades.com/trades"
    params = {
        "page": 1,
        "pageSize": 100,
        "sortBy": "-pubDate",
    }
    all_records = []
    for page in range(1, 21):  # Up to 20 pages = 2000 records
        params["page"] = page
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                print(f"  Capitol Trades API returned {resp.status_code} on page {page}")
                if page == 1:
                    return None
                break
            data = resp.json()
            trades = data.get("data", [])
            if not trades:
                break
            for t in trades:
                issuer = t.get("issuer", {})
                politician = t.get("politician", {})
                record = {
                    "politician": politician.get("firstName", "") + " " + politician.get("lastName", ""),
                    "party": politician.get("party", ""),
                    "chamber": politician.get("chamber", ""),
                    "ticker": issuer.get("ticker", ""),
                    "issuer_name": issuer.get("name", ""),
                    "transaction_date": t.get("txDate", ""),
                    "disclosure_date": t.get("pubDate", ""),
                    "type": t.get("txType", ""),
                    "amount_range": t.get("value", ""),
                    "asset_type": t.get("assetType", ""),
                }
                all_records.append(record)
            print(f"  Page {page}: fetched {len(trades)} trades (total: {len(all_records)})")
            time.sleep(0.5)  # Be polite
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            if page == 1:
                return None
            break

    if not all_records:
        return None
    return pd.DataFrame(all_records)


def try_senate_disclosures() -> pd.DataFrame | None:
    """Try Senate Electronic Financial Disclosures."""
    print("Attempting Senate disclosures (efdsearch.senate.gov)...")
    # The Senate search requires agreeing to terms and uses a session
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        # First agree to terms
        agree_url = "https://efdsearch.senate.gov/search/home/"
        resp = session.get(agree_url, timeout=15)
        if resp.status_code != 200:
            print(f"  Senate site returned {resp.status_code}")
            return None

        # Extract CSRF token
        soup = BeautifulSoup(resp.text, "html.parser")
        csrf_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
        if not csrf_input:
            print("  Could not find CSRF token on Senate site")
            return None

        csrf_token = csrf_input["value"]

        # Agree to terms
        agree_resp = session.post(
            agree_url,
            data={
                "csrfmiddlewaretoken": csrf_token,
                "prohibition_agreement": "1",
            },
            headers={
                **HEADERS,
                "Referer": agree_url,
            },
            timeout=15,
        )

        # Search for periodic transaction reports
        search_url = "https://efdsearch.senate.gov/search/"
        search_resp = session.get(search_url, timeout=15)
        if search_resp.status_code != 200:
            print(f"  Senate search returned {search_resp.status_code}")
            return None

        # Use the AJAX endpoint
        ajax_url = "https://efdsearch.senate.gov/search/report/data/"
        search_data = {
            "start": "0",
            "length": "100",
            "report_type_id": "11",  # Periodic Transaction Reports
            "filer_type_id": "",
            "submitted_start_date": "01/01/2023",
            "submitted_end_date": "",
        }
        ajax_resp = session.post(ajax_url, data=search_data, timeout=30)
        if ajax_resp.status_code == 200:
            data = ajax_resp.json()
            records = data.get("data", [])
            print(f"  Found {len(records)} disclosure records from Senate")
            # These are filing-level records, not individual trades
            # Would need to follow links to get individual transactions
            # For now, note this as a source but don't parse deeply
            if records:
                print("  Senate disclosures found but require per-filing parsing (slow)")
                return None
        else:
            print(f"  Senate AJAX returned {ajax_resp.status_code}")
            return None

    except Exception as e:
        print(f"  Senate disclosure error: {e}")
        return None

    return None


def try_house_disclosures() -> pd.DataFrame | None:
    """Try downloading House Financial Disclosures XML files."""
    print("Attempting House Financial Disclosures (disclosures-clerk.house.gov)...")

    all_records = []
    current_year = datetime.now().year

    for year in range(current_year - 1, current_year + 1):
        url = f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
        print(f"  Trying {year} disclosures: {url}")
        try:
            import io
            import zipfile

            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}")
                continue

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                xml_files = [f for f in zf.namelist() if f.endswith(".xml")]
                txt_files = [f for f in zf.namelist() if f.endswith(".txt")]
                print(f"    Files in zip: {zf.namelist()[:5]}...")

                # Try txt files (tab-delimited)
                for txt_file in txt_files:
                    with zf.open(txt_file) as f:
                        try:
                            df = pd.read_csv(f, sep="\t", encoding="utf-8", on_bad_lines="skip")
                            if len(df) > 0:
                                print(f"    Parsed {txt_file}: {len(df)} rows, columns: {list(df.columns)[:8]}")
                                # Look for transaction-related columns
                                df["source_year"] = year
                                all_records.append(df)
                        except Exception as e:
                            print(f"    Could not parse {txt_file}: {e}")

        except Exception as e:
            print(f"    Error: {e}")

    if not all_records:
        return None

    combined = pd.concat(all_records, ignore_index=True)
    return combined


def try_quiver_quant_github() -> pd.DataFrame | None:
    """Try to get congressional trading data from public GitHub datasets."""
    print("Attempting public GitHub datasets for congressional trading...")

    # Try several known public sources
    urls_to_try = [
        ("https://raw.githubusercontent.com/jbesomi/congresstrading/main/data/transactions.csv", "jbesomi/congresstrading"),
        ("https://raw.githubusercontent.com/ryanm2048/congressional-trading/main/data/all_transactions.csv", "ryanm2048"),
    ]

    for url, name in urls_to_try:
        try:
            print(f"  Trying {name}...")
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                if len(df) > 0:
                    print(f"  Success! Got {len(df)} rows from {name}")
                    return df
            else:
                print(f"  HTTP {resp.status_code}")
        except Exception as e:
            print(f"  Failed: {e}")

    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = load_tickers()
    print(f"Congressional Trading Data Downloader")
    print(f"  Universe: {len(tickers)} tickers")
    print()

    # Try sources in order of preference
    df = try_capitoltrades_api()

    if df is None:
        print()
        df = try_senate_disclosures()

    if df is None:
        print()
        df = try_house_disclosures()

    if df is None:
        print()
        df = try_quiver_quant_github()

    if df is not None and len(df) > 0:
        # Standardize columns
        expected_cols = ["politician", "party", "chamber", "ticker", "transaction_date", "type", "amount_range"]
        # Keep columns that exist
        existing_cols = [c for c in expected_cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in expected_cols]
        keep_cols = existing_cols + extra_cols

        df = df[keep_cols].copy()

        # Filter to our universe if ticker column exists
        if "ticker" in df.columns:
            mask = df["ticker"].isin(tickers)
            in_universe = mask.sum()
            print(f"\n  Total trades: {len(df)}")
            print(f"  Trades matching our universe: {in_universe}")
            # Keep all trades (useful for market-wide sentiment) but add a flag
            df["in_universe"] = mask

        # Parse dates
        if "transaction_date" in df.columns:
            df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

        if "disclosure_date" in df.columns:
            df["disclosure_date"] = pd.to_datetime(df["disclosure_date"], errors="coerce")

        print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")

        df.to_parquet(OUTPUT_FILE, index=False)
        print(f"\nSaved to {OUTPUT_FILE}")
        print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    else:
        print("\n" + "=" * 70)
        print("WARNING: Could not download congressional trading data automatically.")
        print("This data is behind various rate limits and anti-scraping measures.")
        print()
        print("Manual options:")
        print("  1. Visit https://www.capitoltrades.com/trades and export CSV")
        print("  2. Visit https://efdsearch.senate.gov/ for Senate disclosures")
        print("  3. Visit https://disclosures-clerk.house.gov/ for House disclosures")
        print("  4. Use QuiverQuant API (requires paid subscription)")
        print()
        print("Creating empty parquet with expected schema for downstream compatibility...")

        schema_df = pd.DataFrame(
            columns=["politician", "party", "chamber", "ticker", "issuer_name",
                      "transaction_date", "disclosure_date", "type", "amount_range",
                      "asset_type", "in_universe"]
        )
        schema_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"Saved empty schema to {OUTPUT_FILE}")
        print("=" * 70)


if __name__ == "__main__":
    main()
