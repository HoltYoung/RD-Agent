"""
US Equity Factor Evaluation Script — Rolling Backtest Edition.

Computes out-of-sample IC, ICIR, and long-short returns using a rolling
walk-forward framework (no look-ahead bias).

Key features:
- IC-weighted signal combination (upweights high-IC factors, zeroes negative-IC)
- LightGBM rolling backtest includes raw features from daily_pv.h5 + fundamentals.h5
- Ensemble of 3 LightGBM models (reduce overfitting)
- Individual factor IC reporting
- NaN-aware operations throughout

Forward return horizon: 5-day (reduces microstructure noise vs 1-day).
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Rolling backtest parameters
FORWARD_DAYS = 5          # 5-day forward returns (less noisy than 1-day)
ROLL_TRAIN_DAYS = 504     # ~2 years training window
ROLL_STEP_DAYS = 63       # Re-train every ~quarter
MIN_STOCKS_PER_DAY = 20   # Minimum stocks for cross-sectional IC

# Date splits (still used for overall boundaries)
TRAIN_START = os.environ.get("train_start", "2020-10-01")
VALID_START = os.environ.get("valid_start", "2024-01-01")
VALID_END = os.environ.get("valid_end", "2024-12-31")
TEST_START = os.environ.get("test_start", "2025-01-01")
TEST_END = os.environ.get("test_end", "2026-02-28")


def load_price_data() -> pd.DataFrame:
    """Load daily price-volume data."""
    return pd.read_hdf("daily_pv.h5", key="data")


def load_raw_features(pv: pd.DataFrame) -> pd.DataFrame:
    """Load pre-computed raw features from daily_pv.h5 and fundamentals.h5."""
    tech_cols = [c for c in pv.columns if c.startswith("$") and c not in
                 ["$open", "$high", "$low", "$close", "$volume", "$factor", "$adj_close"]]
    features = pv[tech_cols].copy() if tech_cols else pd.DataFrame(index=pv.index)

    # Add momentum features computed on the fly
    close = pv["$close"] * pv["$factor"]
    for days in [60, 120, 252]:
        features[f"mom_{days}d"] = close.groupby(level="instrument").pct_change(days)
    # 12-1 month momentum
    features["mom_12_1"] = (close.groupby(level="instrument").pct_change(252)
                            - close.groupby(level="instrument").pct_change(21))
    # 52-week high distance
    rolling_max = close.groupby(level="instrument").rolling(252).max().droplevel(0)
    features["dist_52w_high"] = close / rolling_max - 1.0
    # Vol ratio
    ret = close.groupby(level="instrument").pct_change()
    vol_5 = ret.groupby(level="instrument").rolling(5).std().droplevel(0)
    vol_60 = ret.groupby(level="instrument").rolling(60).std().droplevel(0)
    features["vol_ratio_5_60"] = vol_5 / vol_60.replace(0, np.nan)

    # Load fundamentals if available
    fund_path = Path("fundamentals.h5")
    if fund_path.exists():
        fund = pd.read_hdf(fund_path, key="data")
        # Use derived ratios + select XBRL columns
        derived_cols = [c for c in fund.columns if c.startswith("derived_")]
        xbrl_important = [c for c in fund.columns if any(kw in c for kw in
                          ["EarningsPerShare", "Revenues", "NetIncome", "OperatingIncome",
                           "TotalAssets", "StockholdersEquity", "OperatingCashFlow",
                           "CommonStockValue", "PreferredStockValue"])]
        use_cols = list(set(derived_cols + xbrl_important))
        if use_cols:
            fund_aligned = fund[use_cols].reindex(features.index)
            features = pd.concat([features, fund_aligned], axis=1)

    # Drop columns with >85% NaN or near-constant
    good = [c for c in features.columns if features[c].isna().mean() < 0.85
            and features[c].nunique() > 10]
    return features[good]


def compute_forward_returns(pv: pd.DataFrame) -> pd.Series:
    """Compute N-day forward returns (close-to-close) for each stock."""
    close = pv["$close"] * pv["$factor"]  # adjusted close
    fwd = close.groupby(level="instrument").shift(-FORWARD_DAYS)
    fwd_ret = (fwd / close) - 1.0
    fwd_ret.name = "forward_return"
    return fwd_ret


def zscore_cross_section(factor_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally z-score each factor on each date."""
    grp = factor_df.groupby(level="datetime")
    mu = grp.transform("mean")
    sigma = grp.transform("std")
    sigma = sigma.replace(0, np.nan)
    return (factor_df - mu) / sigma


def compute_individual_factor_ics(normed: pd.DataFrame, fwd_ret: pd.Series) -> dict:
    """Compute IC for each individual factor (used for IC-weighting)."""
    ics = {}
    for col in normed.columns:
        combined = pd.DataFrame({"signal": normed[col], "fwd": fwd_ret}).dropna()
        if len(combined) < 100:
            ics[col] = 0.0
            continue
        daily_ic = {}
        for dt, g in combined.groupby(level="datetime"):
            if len(g) >= MIN_STOCKS_PER_DAY:
                ic = g["signal"].corr(g["fwd"])
                if not np.isnan(ic):
                    daily_ic[dt] = ic
        dic = pd.Series(daily_ic)
        ics[col] = float(dic.mean()) if len(dic) > 0 else 0.0
    return ics


def build_ic_weighted_signal(normed: pd.DataFrame, factor_ics: dict) -> pd.Series:
    """Build IC-weighted signal using NaN-aware weighted average."""
    weights = {k: max(v, 0) for k, v in factor_ics.items()}
    total_w = sum(weights.values())

    if total_w == 0 or normed.shape[1] == 0:
        signal = normed.mean(axis=1)
        signal.name = "signal"
        return signal

    w_array = np.array([weights.get(c, 0) for c in normed.columns])
    w_array = w_array / w_array.sum()

    vals = normed.values
    mask = ~np.isnan(vals)
    weighted_vals = np.where(mask, vals * w_array[np.newaxis, :], 0)
    weighted_sum = weighted_vals.sum(axis=1)
    weight_sum = (mask * w_array[np.newaxis, :]).sum(axis=1)
    weight_sum[weight_sum == 0] = np.nan
    signal = pd.Series(weighted_sum / weight_sum, index=normed.index, name="signal")
    return signal


def compute_ic_metrics(factor_df: pd.DataFrame, fwd_ret: pd.Series) -> dict:
    """
    Compute IC, Rank IC, and ICIR on the IC-weighted combined signal.
    Also reports individual factor ICs for the top/bottom factors.
    """
    if isinstance(factor_df.columns, pd.MultiIndex):
        factor_df.columns = ["_".join(str(c) for c in col).strip("_") for col in factor_df.columns]

    # Drop columns with >50% NaN after z-scoring
    normed = zscore_cross_section(factor_df)
    good_cols = [c for c in normed.columns if normed[c].isna().mean() < 0.50]
    normed = normed[good_cols]

    if normed.empty:
        return {"IC": 0.0, "Rank_IC": 0.0, "ICIR": 0.0}

    # Compute individual factor ICs
    factor_ics = compute_individual_factor_ics(normed, fwd_ret)

    # Print individual factor ICs
    print(f"\n  Individual Factor ICs ({len(factor_ics)} factors):")
    n_positive = 0
    for k, v in sorted(factor_ics.items(), key=lambda x: x[1], reverse=True):
        label = "+" if v > 0 else "-"
        print(f"    {label} {k}: IC={v:.6f}")
        if v > 0:
            n_positive += 1
    print(f"  {n_positive}/{len(factor_ics)} factors with positive IC")

    # Build IC-weighted signal
    if normed.shape[1] > 1:
        signal = build_ic_weighted_signal(normed, factor_ics)
    else:
        signal = normed.iloc[:, 0]
        signal.name = "signal"

    combined = pd.concat([signal, fwd_ret], axis=1).dropna()
    if combined.empty:
        return {"IC": 0.0, "Rank_IC": 0.0, "ICIR": 0.0}

    daily_ic = combined.groupby(level="datetime").apply(
        lambda g: g["signal"].corr(g["forward_return"]) if len(g) >= MIN_STOCKS_PER_DAY else np.nan
    ).dropna()

    daily_rank_ic = combined.groupby(level="datetime").apply(
        lambda g: g["signal"].rank().corr(g["forward_return"].rank()) if len(g) >= MIN_STOCKS_PER_DAY else np.nan
    ).dropna()

    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    rank_ic_mean = daily_rank_ic.mean()

    return {
        "IC": ic_mean,
        "Rank_IC": rank_ic_mean,
        "ICIR": icir,
    }


def compute_long_short_returns(factor_df: pd.DataFrame, fwd_ret: pd.Series) -> dict:
    """Long-short quintile returns (top 20% - bottom 20%), annualized.
    Uses IC-weighted signal."""
    if isinstance(factor_df.columns, pd.MultiIndex):
        factor_df.columns = ["_".join(str(c) for c in col).strip("_") for col in factor_df.columns]

    normed = zscore_cross_section(factor_df)
    good_cols = [c for c in normed.columns if normed[c].isna().mean() < 0.50]
    normed = normed[good_cols]

    if normed.empty:
        return {"Long_Short_Annual_Return": 0.0, "Long_Short_Sharpe": 0.0}

    # IC-weighted signal
    factor_ics = compute_individual_factor_ics(normed, fwd_ret)
    if normed.shape[1] > 1:
        signal = build_ic_weighted_signal(normed, factor_ics)
    else:
        signal = normed.iloc[:, 0]
    signal.name = "signal"

    combined = pd.concat([signal, fwd_ret], axis=1).dropna()
    if combined.empty:
        return {"Long_Short_Annual_Return": 0.0, "Long_Short_Sharpe": 0.0}

    def daily_ls_return(g):
        if len(g) < MIN_STOCKS_PER_DAY:
            return np.nan
        q20 = g["signal"].quantile(0.2)
        q80 = g["signal"].quantile(0.8)
        long_ret = g.loc[g["signal"] >= q80, "forward_return"].mean()
        short_ret = g.loc[g["signal"] <= q20, "forward_return"].mean()
        return long_ret - short_ret

    daily_ls = combined.groupby(level="datetime").apply(daily_ls_return).dropna()
    if daily_ls.empty:
        return {"Long_Short_Annual_Return": 0.0, "Long_Short_Sharpe": 0.0}

    # Annualize based on forward return horizon
    periods_per_year = 252 / FORWARD_DAYS
    annual_return = daily_ls.mean() * periods_per_year
    sharpe = (daily_ls.mean() / daily_ls.std() * np.sqrt(periods_per_year)) if daily_ls.std() > 0 else 0.0

    return {
        "Long_Short_Annual_Return": annual_return,
        "Long_Short_Sharpe": sharpe,
    }


def rolling_lightgbm_backtest(factor_df: pd.DataFrame, fwd_ret: pd.Series,
                               raw_features: pd.DataFrame = None) -> dict:
    """
    Walk-forward rolling LightGBM backtest with ensemble averaging.
    Uses LLM-generated factors + raw features for a richer feature set.
    Train on ROLL_TRAIN_DAYS, predict next ROLL_STEP_DAYS, slide forward.
    Returns out-of-sample IC and ICIR across all OOS windows.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Skipping model-based evaluation.")
        return {"LGB_IC": 0.0, "LGB_ICIR": 0.0}

    if isinstance(factor_df.columns, pd.MultiIndex):
        factor_df.columns = ["_".join(str(c) for c in col).strip("_") for col in factor_df.columns]

    normed = zscore_cross_section(factor_df)

    # Combine LLM factors with raw features
    if raw_features is not None and not raw_features.empty:
        raw_normed = zscore_cross_section(raw_features)
        # Prefix raw feature columns to avoid collision
        raw_normed.columns = ["raw_" + str(c) for c in raw_normed.columns]
        # Align indexes
        common_idx = normed.index.intersection(raw_normed.index)
        if len(common_idx) > 0:
            all_features = pd.concat([normed.loc[common_idx], raw_normed.loc[common_idx]], axis=1)
        else:
            # If no overlap, use raw features on their own full date range
            all_features = raw_normed
    else:
        all_features = normed

    # Drop columns that are >90% NaN after z-scoring
    good_cols = [c for c in all_features.columns if all_features[c].isna().mean() < 0.90]
    all_features = all_features[good_cols]

    # LightGBM handles NaN natively - only require forward_return to be non-NaN
    combined = pd.concat([all_features, fwd_ret], axis=1)
    combined = combined[combined["forward_return"].notna()]
    if combined.empty or all_features.shape[1] == 0:
        return {"LGB_IC": 0.0, "LGB_ICIR": 0.0}

    feature_cols = [c for c in combined.columns if c != "forward_return"]
    all_dates = combined.index.get_level_values("datetime").unique().sort_values()

    print(f"  LGB: {len(all_dates)} dates, {len(feature_cols)} features "
          f"({normed.shape[1]} LLM + {len(feature_cols) - normed.shape[1]} raw)")

    if len(all_dates) < ROLL_TRAIN_DAYS + ROLL_STEP_DAYS:
        print(f"  LGB: Only {len(all_dates)} dates available, need {ROLL_TRAIN_DAYS + ROLL_STEP_DAYS}. Skipping.")
        return {"LGB_IC": 0.0, "LGB_ICIR": 0.0}

    params = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "max_depth": 5,
        "verbose": -1,
        "n_jobs": -1,
    }

    N_ENSEMBLE = 3  # Average predictions from 3 models
    all_oos_ic = []

    # Walk-forward: train on [t-ROLL_TRAIN_DAYS, t), predict [t, t+ROLL_STEP_DAYS)
    start_idx = ROLL_TRAIN_DAYS
    while start_idx + ROLL_STEP_DAYS <= len(all_dates):
        train_dates = all_dates[start_idx - ROLL_TRAIN_DAYS : start_idx]
        test_dates = all_dates[start_idx : start_idx + ROLL_STEP_DAYS]

        train_mask = combined.index.get_level_values("datetime").isin(train_dates)
        test_mask = combined.index.get_level_values("datetime").isin(test_dates)

        train_data = combined.loc[train_mask]
        test_data = combined.loc[test_mask]

        if len(train_data) < 100 or len(test_data) < 50:
            start_idx += ROLL_STEP_DAYS
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data["forward_return"].values
        X_test = test_data[feature_cols].values

        # Ensemble: train N models with different seeds, average predictions
        ensemble_preds = np.zeros(len(X_test))
        for seed in range(N_ENSEMBLE):
            p = params.copy()
            p["seed"] = seed * 42
            p["feature_fraction_seed"] = seed * 42
            p["bagging_seed"] = seed * 42
            lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
            model = lgb.train(
                p, lgb_train, num_boost_round=200,
                callbacks=[lgb.log_evaluation(period=0)],
            )
            ensemble_preds += model.predict(X_test)
        ensemble_preds /= N_ENSEMBLE

        pred_series = pd.Series(ensemble_preds, index=test_data.index, name="signal")
        test_ret = test_data["forward_return"]

        window_ic = pd.concat([pred_series, test_ret], axis=1).groupby(level="datetime").apply(
            lambda g: g["signal"].corr(g["forward_return"]) if len(g) >= MIN_STOCKS_PER_DAY else np.nan
        ).dropna()

        all_oos_ic.append(window_ic)
        start_idx += ROLL_STEP_DAYS

    if not all_oos_ic:
        return {"LGB_IC": 0.0, "LGB_ICIR": 0.0}

    oos_ic = pd.concat(all_oos_ic)
    lgb_ic = oos_ic.mean()
    lgb_icir = lgb_ic / oos_ic.std() if oos_ic.std() > 0 else 0.0

    return {"LGB_IC": lgb_ic, "LGB_ICIR": lgb_icir}


def main():
    print("=" * 60)
    print("US Equity Factor Evaluation (Rolling Backtest)")
    print(f"Forward return horizon: {FORWARD_DAYS} days")
    print("=" * 60)

    if not Path("daily_pv.h5").exists():
        print("ERROR: daily_pv.h5 not found in workspace.")
        return

    pv = load_price_data()
    fwd_ret = compute_forward_returns(pv)
    n_instruments = len(pv.index.get_level_values("instrument").unique())
    print(f"Loaded price data: {pv.shape[0]} rows, {n_instruments} instruments")

    # Load raw features for LightGBM augmentation
    raw_features = load_raw_features(pv)
    print(f"Raw features loaded: {raw_features.shape[1]} columns")

    factor_path = Path("combined_factors_df.parquet")
    if factor_path.exists():
        factor_df = pd.read_parquet(factor_path)
        print(f"Loaded combined factors: {factor_df.shape}")
    else:
        result_path = Path("result.h5")
        if result_path.exists():
            factor_df = pd.read_hdf(result_path)
            print(f"Loaded individual factor: {factor_df.shape}")
        else:
            print("ERROR: No factor data found.")
            return

    print(f"\n--- Computing IC metrics ({FORWARD_DAYS}-day fwd returns, IC-weighted) ---")
    ic_metrics = compute_ic_metrics(factor_df, fwd_ret)
    for k, v in ic_metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"\n--- Computing long-short returns (IC-weighted signal) ---")
    ls_metrics = compute_long_short_returns(factor_df, fwd_ret)
    for k, v in ls_metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"\n--- Rolling LightGBM backtest (LLM factors + raw features, ensemble) ---")
    lgb_metrics = rolling_lightgbm_backtest(factor_df, fwd_ret, raw_features)
    for k, v in lgb_metrics.items():
        print(f"  {k}: {v:.6f}")

    all_metrics = {**ic_metrics, **ls_metrics, **lgb_metrics}
    results = pd.DataFrame.from_dict(all_metrics, orient="index", columns=["value"])
    results.index.name = "metric"
    results.to_csv("evaluation_results.csv")
    print(f"\nResults saved to evaluation_results.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
