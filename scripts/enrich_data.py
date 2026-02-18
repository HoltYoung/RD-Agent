"""
Pre-compute useful technical indicators and add them to the HDF5 data files.
This makes factor implementation much easier for the LLM — it just needs to
combine pre-computed signals rather than computing rolling windows from scratch.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def enrich_price_data(data_folder: Path) -> None:
    """Add technical indicators to daily_pv.h5."""
    pv_path = data_folder / "daily_pv.h5"
    if not pv_path.exists():
        print(f"Skipping {data_folder}: daily_pv.h5 not found")
        return

    print(f"Enriching {pv_path}...")
    pv = pd.read_hdf(pv_path, key="data")

    # Adjusted close
    adj_close = pv["$close"] * pv["$factor"]
    pv["$adj_close"] = adj_close

    g = pv.groupby(level="instrument")

    # Returns at various horizons
    pv["$ret_1d"] = g["$adj_close"].pct_change(1)
    pv["$ret_5d"] = g["$adj_close"].pct_change(5)
    pv["$ret_10d"] = g["$adj_close"].pct_change(10)
    pv["$ret_20d"] = g["$adj_close"].pct_change(20)

    # Rolling volatility
    pv["$vol_5d"] = pv.groupby(level="instrument")["$ret_1d"].rolling(5).std().droplevel(0)
    pv["$vol_20d"] = pv.groupby(level="instrument")["$ret_1d"].rolling(20).std().droplevel(0)

    # Volume moving averages
    pv["$vol_ma5"] = g["$volume"].rolling(5).mean().droplevel(0)
    pv["$vol_ma20"] = g["$volume"].rolling(20).mean().droplevel(0)

    # Volume surge (current / 20-day average)
    pv["$vol_surge"] = pv["$volume"] / pv["$vol_ma20"]

    # Intraday range (high-low / close)
    pv["$range_pct"] = (pv["$high"] - pv["$low"]) / pv["$close"]

    # Average range over 5d
    pv["$avg_range_5d"] = pv.groupby(level="instrument")["$range_pct"].rolling(5).mean().droplevel(0)

    # VWAP proxy (mid of high-low weighted by volume)
    pv["$vwap_proxy"] = ((pv["$high"] + pv["$low"] + pv["$close"]) / 3)

    # RSI-like metric (proportion of up days in last 14 days)
    up = (pv["$ret_1d"] > 0).astype(float)
    pv["$up_ratio_14d"] = up.groupby(level="instrument").rolling(14).mean().droplevel(0)

    # Turnover (volume * close as dollar volume proxy)
    pv["$dollar_volume"] = pv["$volume"] * pv["$close"]

    print(f"  Added {len(pv.columns) - 6} technical indicators. Shape: {pv.shape}")
    pv.to_hdf(pv_path, key="data", mode="w", complevel=5)
    print(f"  Saved to {pv_path}")


def enrich_fundamentals(data_folder: Path) -> None:
    """Add useful fundamental ratios to fundamentals.h5."""
    fund_path = data_folder / "fundamentals.h5"
    pv_path = data_folder / "daily_pv.h5"
    if not fund_path.exists() or not pv_path.exists():
        print(f"Skipping {data_folder}: files not found")
        return

    print(f"Enriching {fund_path}...")
    fund = pd.read_hdf(fund_path, key="data")
    pv = pd.read_hdf(pv_path, key="data")

    # Market cap proxy
    shares = fund.get("xbrl_is_WeightedAverageNumberOfDilutedSharesOutstanding")
    adj_close = pv["$adj_close"] if "$adj_close" in pv.columns else pv["$close"] * pv["$factor"]
    if shares is not None:
        mktcap = adj_close * shares
        fund["derived_mktcap"] = mktcap

    # Profitability ratios
    revenue = fund.get("xbrl_is_Revenues")
    if revenue is None:
        revenue = fund.get("xbrl_is_RevenueFromContractWithCustomerExcludingAssessedTax")

    assets = fund.get("xbrl_bs_Assets")
    equity = fund.get("xbrl_bs_StockholdersEquity")
    net_income = fund.get("xbrl_is_NetIncomeLoss")
    op_income = fund.get("xbrl_is_OperatingIncomeLoss")
    gross_profit = fund.get("xbrl_is_GrossProfit")
    op_cf = fund.get("xbrl_cf_NetCashProvidedByUsedInOperatingActivities")
    liabilities = fund.get("xbrl_bs_Liabilities")
    capex = fund.get("xbrl_cf_PaymentsToAcquirePropertyPlantAndEquipment")
    buyback = fund.get("xbrl_cf_PaymentsForRepurchaseOfCommonStock")
    sbc = fund.get("xbrl_cf_ShareBasedCompensation")
    current_assets = fund.get("xbrl_bs_AssetsCurrent")
    current_liabs = fund.get("xbrl_bs_LiabilitiesCurrent")

    # ROE
    if net_income is not None and equity is not None:
        fund["derived_roe"] = net_income / equity.replace(0, np.nan)

    # ROA
    if net_income is not None and assets is not None:
        fund["derived_roa"] = net_income / assets.replace(0, np.nan)

    # Operating margin
    if op_income is not None and revenue is not None:
        fund["derived_op_margin"] = op_income / revenue.replace(0, np.nan)

    # Gross margin
    if gross_profit is not None and revenue is not None:
        fund["derived_gross_margin"] = gross_profit / revenue.replace(0, np.nan)

    # Gross profit to assets (Novy-Marx quality factor)
    if gross_profit is not None and assets is not None:
        fund["derived_gp_to_assets"] = gross_profit / assets.replace(0, np.nan)

    # Accruals ratio
    if net_income is not None and op_cf is not None and assets is not None:
        fund["derived_accruals"] = (net_income - op_cf) / assets.replace(0, np.nan)

    # Cash flow yield
    if op_cf is not None and "derived_mktcap" in fund.columns:
        fund["derived_cf_yield"] = op_cf / fund["derived_mktcap"].replace(0, np.nan)

    # Earnings yield
    eps = fund.get("xbrl_is_EarningsPerShareBasic")
    if eps is not None:
        fund["derived_ep"] = eps / adj_close.replace(0, np.nan)

    # Debt to equity
    if liabilities is not None and equity is not None:
        fund["derived_debt_equity"] = liabilities / equity.replace(0, np.nan)

    # Capex intensity
    if capex is not None and revenue is not None:
        fund["derived_capex_intensity"] = capex.abs() / revenue.replace(0, np.nan)

    # Buyback yield
    if buyback is not None and "derived_mktcap" in fund.columns:
        fund["derived_buyback_yield"] = buyback.abs() / fund["derived_mktcap"].replace(0, np.nan)

    # SBC ratio
    if sbc is not None and revenue is not None:
        fund["derived_sbc_ratio"] = sbc / revenue.replace(0, np.nan)

    # Current ratio
    if current_assets is not None and current_liabs is not None:
        fund["derived_current_ratio"] = current_assets / current_liabs.replace(0, np.nan)

    # Working capital to assets
    if current_assets is not None and current_liabs is not None and assets is not None:
        fund["derived_wc_to_assets"] = (current_assets - current_liabs) / assets.replace(0, np.nan)

    derived_cols = [c for c in fund.columns if c.startswith("derived_")]
    print(f"  Added {len(derived_cols)} derived fundamental ratios. Shape: {fund.shape}")
    fund.to_hdf(fund_path, key="data", mode="w", complevel=5)
    print(f"  Saved to {fund_path}")


def update_readme(data_folder: Path) -> None:
    """Update README.md with new columns."""
    readme_path = data_folder / "README.md"
    pv_path = data_folder / "daily_pv.h5"
    fund_path = data_folder / "fundamentals.h5"

    lines = ["# US Equity Factor Data\n"]

    if pv_path.exists():
        pv = pd.read_hdf(pv_path, key="data")
        n = pv.index.get_level_values("instrument").nunique()
        lines.append(f"\n## daily_pv.h5\n")
        lines.append(f"\nDaily price-volume data for {n} US equities.\n")
        lines.append(f"- Read with: `pd.read_hdf(\"daily_pv.h5\", key=\"data\")`\n")
        lines.append(f"- Index: MultiIndex [datetime, instrument]\n")
        dates = pv.index.get_level_values("datetime")
        lines.append(f"- Date range: {dates.min().date()} to {dates.max().date()}\n")
        lines.append(f"- Shape: {pv.shape[0]} rows x {pv.shape[1]} columns\n")
        lines.append(f"- Columns:\n")
        orig = [c for c in pv.columns if not c.startswith("$ret_") and not c.startswith("$vol_") and not c.startswith("$range") and not c.startswith("$avg_") and not c.startswith("$vwap") and not c.startswith("$up_") and not c.startswith("$dollar")]
        derived = [c for c in pv.columns if c not in orig]
        for c in orig:
            lines.append(f"  - `{c}`\n")
        if derived:
            lines.append(f"- Pre-computed technical indicators ({len(derived)}):\n")
            for c in derived:
                lines.append(f"  - `{c}`\n")

    if fund_path.exists():
        fund = pd.read_hdf(fund_path, key="data")
        lines.append(f"\n## fundamentals.h5\n")
        lines.append(f"\nXBRL financial statement data forward-filled to daily frequency.\n")
        lines.append(f"- Read with: `pd.read_hdf(\"fundamentals.h5\", key=\"data\")`\n")
        lines.append(f"- Index: MultiIndex [datetime, instrument]\n")
        xbrl_cols = [c for c in fund.columns if c.startswith("xbrl_")]
        derived_cols = [c for c in fund.columns if c.startswith("derived_")]
        bs = [c for c in xbrl_cols if "bs_" in c]
        is_cols = [c for c in xbrl_cols if "is_" in c]
        cf = [c for c in xbrl_cols if "cf_" in c]
        lines.append(f"- Balance sheet columns ({len(bs)}): prefixed with `xbrl_bs_`\n")
        lines.append(f"- Income statement columns ({len(is_cols)}): prefixed with `xbrl_is_`\n")
        lines.append(f"- Cash flow columns ({len(cf)}): prefixed with `xbrl_cf_`\n")
        lines.append(f"- Total XBRL columns: {len(xbrl_cols)}\n")
        if derived_cols:
            lines.append(f"- Pre-computed fundamental ratios ({len(derived_cols)}):\n")
            for c in derived_cols:
                lines.append(f"  - `{c}`\n")
        lines.append(f"- Values are forward-filled from quarterly SEC filings to daily frequency\n")

    lines.append(f"\n## Usage in factor code\n")
    lines.append(f"\n```python\nimport pandas as pd\n\n")
    lines.append(f"# Load price data\ndf = pd.read_hdf(\"daily_pv.h5\", key=\"data\")\n")
    lines.append(f"# df.index has levels: ['datetime', 'instrument']\n")
    lines.append(f"# Access: df['$close'], df['$ret_5d'], df['$vol_20d'], etc.\n\n")
    lines.append(f"# Load fundamentals (if needed)\nfund = pd.read_hdf(\"fundamentals.h5\", key=\"data\")\n")
    lines.append(f"# Access: fund['derived_ep'], fund['derived_accruals'], fund['xbrl_is_Revenues'], etc.\n```\n")

    with open(readme_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  Updated {readme_path}")


def main():
    base = Path("git_ignore_folder")
    for folder_name in ["factor_implementation_source_data", "factor_implementation_source_data_debug"]:
        folder = base / folder_name
        if folder.exists():
            enrich_price_data(folder)
            enrich_fundamentals(folder)
            update_readme(folder)
            print()

    print("Done! Data enrichment complete.")


if __name__ == "__main__":
    main()
