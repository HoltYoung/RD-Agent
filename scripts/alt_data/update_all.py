"""
Master orchestrator for all alt-data download scripts.

Runs each download script in sequence with error handling.
Generates a manifest.json with last_updated timestamps and record counts.

Usage:
    python scripts/alt_data/update_all.py                           # Run all sources
    python scripts/alt_data/update_all.py --source reddit --source fred   # Run specific sources
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "alt_data"
ALT_DATA_DIR = PROJECT_ROOT / "universe_data" / "alt_data"
MANIFEST_PATH = ALT_DATA_DIR / "manifest.json"

# Maps a short source name -> (script filename, data dir, primary parquet file(s))
# The parquet paths are relative to ALT_DATA_DIR
SOURCES = {
    "sec_insider": {
        "script": "download_sec_insider.py",
        "data_dir": "sec_insider",
        "primary_file": "sec_insider/insider_combined.parquet",
    },
    "sec_ftd": {
        "script": "download_sec_ftd.py",
        "data_dir": "sec_ftd",
        "primary_file": "sec_ftd/ftd_combined.parquet",
    },
    "finra_short_volume": {
        "script": "download_finra_short_volume.py",
        "data_dir": "finra_short_volume",
        "primary_file": "finra_short_volume/finra_short_volume.parquet",
    },
    "finra_short_interest": {
        "script": "download_finra_short_interest.py",
        "data_dir": "finra_short_interest",
        "primary_file": "finra_short_interest/finra_short_interest.parquet",
    },
    "yahoo_options": {
        "script": "download_yahoo_options.py",
        "data_dir": "yahoo_options",
        "primary_file": "yahoo_options/options_metrics.parquet",
    },
    "reddit": {
        "script": "download_reddit_sentiment.py",
        "data_dir": "reddit_sentiment",
        "primary_file": "reddit_sentiment",  # glob pattern handled below
    },
    "wikipedia": {
        "script": "download_wikipedia_pageviews.py",
        "data_dir": "wikipedia_pageviews",
        "primary_file": "wikipedia_pageviews/wikipedia_pageviews_combined.parquet",
    },
    "google_trends": {
        "script": "download_google_trends.py",
        "data_dir": "google_trends",
        "primary_file": None,  # may not produce a single file
    },
    "fred": {
        "script": "download_fred_macro.py",
        "data_dir": "fred_macro",
        "primary_file": "fred_macro/fred_macro.parquet",
    },
    "congressional": {
        "script": "download_congressional_trading.py",
        "data_dir": "congressional_trading",
        "primary_file": "congressional_trading/congressional_trading.parquet",
    },
}

# Execution order (matches the user-specified order)
DEFAULT_ORDER = [
    "sec_insider",
    "sec_ftd",
    "finra_short_volume",
    "finra_short_interest",
    "yahoo_options",
    "reddit",
    "wikipedia",
    "google_trends",
    "fred",
    "congressional",
]


def _dir_total_size(path: Path) -> int:
    """Return total size in bytes of all files under *path*."""
    total = 0
    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def _count_parquet_records(path: Path) -> int | None:
    """Return the number of rows in a parquet file, or None on failure."""
    try:
        return len(pd.read_parquet(path))
    except Exception:
        return None


def _date_range_from_parquet(path: Path) -> tuple[str | None, str | None]:
    """Try to extract min/max date from a parquet file."""
    try:
        df = pd.read_parquet(path)
        # Look for date-like columns
        for col_name in ["date", "settlement_date", "transaction_date", "Date", "filing_date", "fetch_date"]:
            if col_name in df.columns:
                series = pd.to_datetime(df[col_name], errors="coerce").dropna()
                if len(series):
                    return str(series.min().date()), str(series.max().date())
        # Check index
        if df.index.name and "date" in df.index.name.lower():
            idx = pd.to_datetime(df.index, errors="coerce").dropna()
            if len(idx):
                return str(idx.min().date()), str(idx.max().date())
    except Exception:
        pass
    return None, None


def _get_primary_parquet(source_name: str) -> Path | None:
    """Resolve the primary parquet file for a source."""
    info = SOURCES[source_name]
    pf = info["primary_file"]
    if pf is None:
        return None
    full = ALT_DATA_DIR / pf
    if full.is_file():
        return full
    # Handle reddit glob: find the latest reddit_sentiment_*.parquet (not _all_)
    if source_name == "reddit":
        import glob as g
        matches = sorted(g.glob(str(ALT_DATA_DIR / "reddit_sentiment" / "reddit_sentiment_2*.parquet")))
        if matches:
            return Path(matches[-1])
    return None


def run_source(source_name: str) -> dict:
    """
    Run a single download script and return a result dict with:
      status, records, file_size_kb, date_min, date_max, duration_sec, error
    """
    info = SOURCES[source_name]
    script_path = SCRIPTS_DIR / info["script"]
    data_dir = ALT_DATA_DIR / info["data_dir"]

    result = {
        "source": source_name,
        "script": info["script"],
        "status": "unknown",
        "records": None,
        "file_size_kb": None,
        "date_min": None,
        "date_max": None,
        "duration_sec": None,
        "error": None,
    }

    if not script_path.exists():
        result["status"] = "SKIP"
        result["error"] = f"Script not found: {script_path}"
        return result

    print(f"\n{'='*70}")
    print(f"  Running: {info['script']}")
    print(f"{'='*70}")

    start = datetime.now()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            timeout=1800,  # 30 min timeout per source
        )
        duration = (datetime.now() - start).total_seconds()
        result["duration_sec"] = round(duration, 1)

        if proc.returncode == 0:
            result["status"] = "OK"
        else:
            result["status"] = "FAIL"
            result["error"] = f"Exit code {proc.returncode}"
    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start).total_seconds()
        result["duration_sec"] = round(duration, 1)
        result["status"] = "TIMEOUT"
        result["error"] = "Exceeded 30 minute timeout"
    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        result["duration_sec"] = round(duration, 1)
        result["status"] = "ERROR"
        result["error"] = str(e)

    # Gather stats from the output data regardless of status
    result["file_size_kb"] = round(_dir_total_size(data_dir) / 1024, 1)

    pf = _get_primary_parquet(source_name)
    if pf and pf.exists():
        result["records"] = _count_parquet_records(pf)
        dmin, dmax = _date_range_from_parquet(pf)
        result["date_min"] = dmin
        result["date_max"] = dmax

    return result


def update_manifest(results: list[dict]) -> None:
    """Create or update manifest.json with results from this run."""
    manifest = {}
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH) as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    now_str = datetime.now(timezone.utc).isoformat()

    for r in results:
        entry = manifest.get(r["source"], {})
        entry["last_updated"] = now_str
        entry["script"] = r["script"]
        entry["status"] = r["status"]
        entry["records"] = r["records"]
        entry["file_size_kb"] = r["file_size_kb"]
        entry["date_min"] = r["date_min"]
        entry["date_max"] = r["date_max"]
        entry["duration_sec"] = r["duration_sec"]
        if r["error"]:
            entry["last_error"] = r["error"]
        elif "last_error" in entry:
            del entry["last_error"]
        manifest[r["source"]] = entry

    manifest["_meta"] = {
        "last_run": now_str,
        "sources_run": [r["source"] for r in results],
    }

    ALT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest written to {MANIFEST_PATH}")


def print_summary(results: list[dict]) -> None:
    """Print a formatted summary table of all results."""
    print(f"\n{'='*90}")
    print(f"  UPDATE SUMMARY  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'='*90}")
    print(f"{'Source':<22} {'Status':<8} {'Records':>10} {'Size (KB)':>10} {'Time (s)':>9}  {'Date Range'}")
    print(f"{'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*9}  {'-'*25}")

    ok_count = 0
    fail_count = 0
    for r in results:
        records_str = f"{r['records']:,}" if r["records"] is not None else "-"
        size_str = f"{r['file_size_kb']:,.1f}" if r["file_size_kb"] is not None else "-"
        time_str = f"{r['duration_sec']:.1f}" if r["duration_sec"] is not None else "-"
        date_range = ""
        if r["date_min"] and r["date_max"]:
            date_range = f"{r['date_min']} -> {r['date_max']}"

        status_marker = r["status"]
        if r["status"] == "OK":
            ok_count += 1
        else:
            fail_count += 1

        print(f"{r['source']:<22} {status_marker:<8} {records_str:>10} {size_str:>10} {time_str:>9}  {date_range}")

        if r["error"]:
            print(f"  {'':22} ERROR: {r['error']}")

    print(f"\nTotal: {ok_count} succeeded, {fail_count} failed/skipped out of {len(results)} sources")


def main():
    parser = argparse.ArgumentParser(description="Run alt-data download scripts")
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        choices=list(SOURCES.keys()),
        help="Run only specific sources (can be repeated). If omitted, runs all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    args = parser.parse_args()

    sources_to_run = args.sources if args.sources else DEFAULT_ORDER

    print(f"Alt-Data Update Pipeline")
    print(f"  Sources: {', '.join(sources_to_run)}")
    print(f"  Data dir: {ALT_DATA_DIR}")

    if args.dry_run:
        print("\n[DRY RUN] Would run the following scripts:")
        for s in sources_to_run:
            info = SOURCES[s]
            script_path = SCRIPTS_DIR / info["script"]
            exists = "exists" if script_path.exists() else "MISSING"
            print(f"  {info['script']} ({exists})")
        return

    results = []
    for source_name in sources_to_run:
        result = run_source(source_name)
        results.append(result)

    print_summary(results)
    update_manifest(results)


if __name__ == "__main__":
    main()
