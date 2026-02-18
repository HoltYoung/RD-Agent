"""
Download SEC Form 4 Insider Trading data for universe tickers.

Uses the SEC EDGAR API:
1. Maps tickers to CIKs via company_tickers.json
2. Fetches Form 4 filing metadata from submissions API
3. Downloads and parses Form 4 XML filings to extract transactions

Rate limit: 10 requests/second (we use ~8/sec to stay safe).
Re-runnable: skips tickers that already have data files.
"""

import os
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

# Paths
BASE_DIR = Path(r"C:\Users\holty\projects\rd-agent")
TICKER_FILE = BASE_DIR / "universe_data" / "ticker_list.txt"
OUTPUT_DIR = BASE_DIR / "universe_data" / "alt_data" / "sec_insider"

# SEC EDGAR settings
HEADERS = {"User-Agent": "RDAgent research@example.com"}
MAX_FORM4_PER_TICKER = 20  # Limit Form 4 filings to parse per ticker (most recent)
MAX_WORKERS = 4  # Concurrent threads for XML download


class RateLimiter:
    """Thread-safe rate limiter for SEC's 10 req/sec limit."""

    def __init__(self, max_per_second=8):
        self.min_interval = 1.0 / max_per_second
        self.lock = threading.Lock()
        self.last_request = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request = time.time()


rate_limiter = RateLimiter(max_per_second=8)


def rate_limited_get(url, timeout=15):
    """Make a rate-limited GET request."""
    rate_limiter.wait()
    return requests.get(url, headers=HEADERS, timeout=timeout)


def load_tickers():
    """Load ticker list from file."""
    with open(TICKER_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_ticker_to_cik_map():
    """Download SEC company tickers JSON and build ticker -> CIK mapping."""
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = rate_limited_get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    mapping = {}
    for entry in data.values():
        ticker = entry.get("ticker", "").upper()
        cik = entry.get("cik_str", "")
        if ticker and cik:
            mapping[ticker] = str(cik)
    return mapping


def get_form4_filings(cik, max_filings=MAX_FORM4_PER_TICKER):
    """Get Form 4 filing accession numbers from the submissions API."""
    cik_padded = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

    resp = rate_limited_get(url, timeout=30)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    data = resp.json()

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == "4" and i < len(accessions):
            filings.append({
                "accession": accessions[i],
                "date": dates[i] if i < len(dates) else None,
                "primary_doc": primary_docs[i] if i < len(primary_docs) else None,
            })
            if len(filings) >= max_filings:
                break

    return filings


def parse_form4_xml(xml_text, ticker):
    """Parse a Form 4 XML filing and extract transaction data."""
    transactions = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return transactions

    # Get reporting owner info
    owner_name = ""
    owner_title = ""
    for owner in root.findall(".//reportingOwner"):
        name_elem = owner.find(".//rptOwnerName")
        if name_elem is not None and name_elem.text:
            owner_name = name_elem.text.strip()
        title_elem = owner.find(".//officerTitle")
        if title_elem is not None and title_elem.text:
            owner_title = title_elem.text.strip()

    # Parse non-derivative transactions
    for txn in root.findall(".//nonDerivativeTransaction"):
        record = _parse_transaction(txn, ticker, owner_name, owner_title)
        if record:
            transactions.append(record)

    # Parse derivative transactions
    for txn in root.findall(".//derivativeTransaction"):
        record = _parse_transaction(txn, ticker, owner_name, owner_title, is_derivative=True)
        if record:
            transactions.append(record)

    return transactions


def _parse_transaction(txn, ticker, owner_name, owner_title, is_derivative=False):
    """Parse a single transaction element from Form 4 XML."""
    record = {
        "ticker": ticker,
        "insider_name": owner_name,
        "title": owner_title,
        "is_derivative": is_derivative,
    }

    date_elem = txn.find(".//transactionDate/value")
    record["transaction_date"] = date_elem.text.strip() if date_elem is not None and date_elem.text else None

    code_elem = txn.find(".//transactionCoding/transactionCode")
    record["transaction_type"] = code_elem.text.strip() if code_elem is not None and code_elem.text else None

    shares_elem = txn.find(".//transactionAmounts/transactionShares/value")
    try:
        record["shares"] = float(shares_elem.text.strip()) if shares_elem is not None and shares_elem.text else None
    except ValueError:
        record["shares"] = None

    price_elem = txn.find(".//transactionAmounts/transactionPricePerShare/value")
    try:
        record["price"] = float(price_elem.text.strip()) if price_elem is not None and price_elem.text else None
    except ValueError:
        record["price"] = None

    owned_elem = txn.find(".//postTransactionAmounts/sharesOwnedFollowingTransaction/value")
    try:
        record["shares_owned_after"] = float(owned_elem.text.strip()) if owned_elem is not None and owned_elem.text else None
    except ValueError:
        record["shares_owned_after"] = None

    ad_elem = txn.find(".//transactionAmounts/transactionAcquiredDisposedCode/value")
    record["acquired_disposed"] = ad_elem.text.strip() if ad_elem is not None and ad_elem.text else None

    return record


def download_single_xml(args):
    """Download and parse a single Form 4 XML filing. Used by thread pool."""
    ticker, cik, filing = args
    accession = filing["accession"]
    primary_doc = filing.get("primary_doc", "")
    accession_path = accession.replace("-", "")

    # Strip XSLT prefix from primaryDocument
    xml_filename = primary_doc
    if "/" in xml_filename:
        xml_filename = xml_filename.split("/")[-1]

    if not xml_filename.endswith(".xml"):
        return []

    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{xml_filename}"

    try:
        resp = rate_limited_get(url)
        if resp.status_code == 200 and "ownershipDocument" in resp.text[:500]:
            txns = parse_form4_xml(resp.text, ticker)
            for txn in txns:
                txn["filing_date"] = filing["date"]
            return txns
    except Exception:
        pass
    return []


def download_form4_for_ticker(ticker, cik):
    """Download and parse Form 4 filings for a single ticker using concurrent requests."""
    filings = get_form4_filings(cik)
    if not filings:
        return []

    all_transactions = []
    # Use thread pool for concurrent XML downloads
    tasks = [(ticker, cik, f) for f in filings]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_single_xml, task) for task in tasks]
        for future in as_completed(futures):
            txns = future.result()
            all_transactions.extend(txns)

    return all_transactions


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} tickers from universe")

    # Get ticker -> CIK mapping
    print("Fetching ticker to CIK mapping from SEC...")
    ticker_cik_map = get_ticker_to_cik_map()
    print(f"  Found {len(ticker_cik_map)} tickers in SEC database")

    # Match our tickers
    matched = {}
    unmatched = []
    for ticker in tickers:
        t = ticker.upper()
        variants = [t, t.replace("-", "."), t.replace("-", "/")]
        found = False
        for v in variants:
            if v in ticker_cik_map:
                matched[ticker] = ticker_cik_map[v]
                found = True
                break
        if not found:
            unmatched.append(ticker)

    print(f"  Matched {len(matched)} tickers to CIKs, {len(unmatched)} unmatched")
    if unmatched:
        print(f"  Unmatched: {unmatched}")

    # Process each ticker
    all_data = []
    processed = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    for i, (ticker, cik) in enumerate(matched.items()):
        ticker_file = OUTPUT_DIR / f"insider_{ticker}.parquet"

        # Skip if already downloaded (and has actual data)
        if ticker_file.exists():
            df = pd.read_parquet(ticker_file)
            if len(df) > 0:
                all_data.append(df)
                skipped += 1
                continue
            else:
                # Remove empty files from previous failed runs
                os.remove(ticker_file)

        elapsed = time.time() - start_time
        rate = (processed + 1) / max(elapsed, 1) * 60
        print(f"  [{i+1}/{len(matched)}] {ticker} (CIK: {cik})...", end=" ", flush=True)

        try:
            transactions = download_form4_for_ticker(ticker, cik)

            if transactions:
                df = pd.DataFrame(transactions)
                df.to_parquet(ticker_file, index=False)
                all_data.append(df)
                print(f"{len(transactions)} txns ({rate:.0f}/min)")
            else:
                # Save marker file to avoid re-downloading
                df = pd.DataFrame(columns=[
                    "ticker", "insider_name", "title", "transaction_date",
                    "transaction_type", "shares", "price", "shares_owned_after",
                    "acquired_disposed", "is_derivative", "filing_date"
                ])
                df.to_parquet(ticker_file, index=False)
                print(f"no txns ({rate:.0f}/min)")

            processed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    total_time = time.time() - start_time
    print(f"\nProcessed {processed} tickers, skipped {skipped} existing, {errors} errors")
    print(f"Total time: {total_time/60:.1f} minutes")

    # Combine all data
    combined_file = OUTPUT_DIR / "insider_combined.parquet"
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        if "transaction_date" in combined.columns:
            combined["transaction_date"] = pd.to_datetime(combined["transaction_date"], errors="coerce")
        if "filing_date" in combined.columns:
            combined["filing_date"] = pd.to_datetime(combined["filing_date"], errors="coerce")
        combined = combined.sort_values(["ticker", "transaction_date"]).reset_index(drop=True)
        combined.to_parquet(combined_file, index=False)
        print(f"\nCombined insider data: {len(combined)} transactions across {combined['ticker'].nunique()} tickers")
        if "transaction_type" in combined.columns:
            print(f"Transaction type breakdown:\n{combined['transaction_type'].value_counts().to_string()}")
        print(f"Saved to: {combined_file}")
    else:
        print("No insider trading data collected")


if __name__ == "__main__":
    main()
