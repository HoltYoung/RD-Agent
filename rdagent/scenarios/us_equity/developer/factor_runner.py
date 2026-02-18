from pathlib import Path

import pandas as pd
from pandarallel import pandarallel

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import cache_with_pickle

pandarallel.initialize(verbose=1)

from rdagent.app.us_equity_rd_loop.conf import FactorBasePropSetting
from rdagent.components.runner import CachedRunner
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.us_equity.developer.utils import process_factor_data
from rdagent.scenarios.us_equity.experiment.factor_experiment import USEquityFactorExperiment


class USEquityFactorRunner(CachedRunner[USEquityFactorExperiment]):
    """
    Runs factor evaluation locally — no Docker, no Qlib.
    Combines SOTA factors with new factors, deduplicates, and calls evaluate.py.
    """

    def calculate_information_coefficient(
        self, concat_feature: pd.DataFrame, SOTA_feature_column_size: int, new_feature_columns_size: int
    ) -> pd.Series:
        res = pd.Series(index=range(SOTA_feature_column_size * new_feature_columns_size))
        for col1 in range(SOTA_feature_column_size):
            for col2 in range(SOTA_feature_column_size, SOTA_feature_column_size + new_feature_columns_size):
                res.loc[col1 * new_feature_columns_size + col2 - SOTA_feature_column_size] = concat_feature.iloc[
                    :, col1
                ].corr(concat_feature.iloc[:, col2])
        return res

    def deduplicate_new_factors(self, SOTA_feature: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
        concat_feature = pd.concat([SOTA_feature, new_feature], axis=1)
        IC_max = (
            concat_feature.groupby("datetime")
            .parallel_apply(
                lambda x: self.calculate_information_coefficient(x, SOTA_feature.shape[1], new_feature.shape[1])
            )
            .mean()
        )
        IC_max.index = pd.MultiIndex.from_product([range(SOTA_feature.shape[1]), range(new_feature.shape[1])])
        IC_max = IC_max.unstack().max(axis=0)
        return new_feature.iloc[:, IC_max[IC_max < 0.90].index]

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: USEquityFactorExperiment) -> USEquityFactorExperiment:
        """
        Process factors and run local evaluation.
        """
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            # Only develop the baseline if it actually has factors to evaluate
            if exp.based_experiments[-1].sub_tasks:
                logger.info("Baseline experiment execution ...")
                exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])
            else:
                logger.info("Baseline experiment has no factors, skipping.")

        fbps = FactorBasePropSetting()
        env_to_use = {
            "PYTHONPATH": "./",
            "train_start": fbps.train_start,
            "train_end": fbps.train_end,
            "valid_start": fbps.valid_start,
            "valid_end": fbps.valid_end,
            "test_start": fbps.test_start,
        }
        if fbps.test_end is not None:
            env_to_use["test_end"] = fbps.test_end

        if exp.based_experiments:
            SOTA_factor = None
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, USEquityFactorExperiment)
            ]
            if len(sota_factor_experiments_list) > 1:
                logger.info("SOTA factor processing ...")
                SOTA_factor = process_factor_data(sota_factor_experiments_list)

            logger.info("New factor processing ...")
            new_factors = process_factor_data(exp)

            if new_factors.empty:
                raise FactorEmptyError("Factors failed to run on the full sample.")

            if SOTA_factor is not None and not SOTA_factor.empty:
                new_factors = self.deduplicate_new_factors(SOTA_factor, new_factors)
                if new_factors.empty:
                    raise FactorEmptyError(
                        "New factors are too similar to existing SOTA factors. Try a different direction."
                    )
                combined_factors = pd.concat([SOTA_factor, new_factors], axis=1).dropna()
            else:
                combined_factors = new_factors

            combined_factors = combined_factors.sort_index()
            combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
            new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
            combined_factors.columns = new_columns
            logger.info("Factor data processing completed.")

            target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"
            combined_factors.to_parquet(target_path, engine="pyarrow")

            result, stdout = exp.experiment_workspace.execute(run_env=env_to_use)
        else:
            logger.info("Experiment execution ...")
            result, stdout = exp.experiment_workspace.execute(run_env=env_to_use)

        if result is None:
            logger.error(f"Failed to run experiment: {stdout}")
            raise FactorEmptyError(f"Failed to run experiment: {stdout}")

        exp.result = result
        exp.stdout = stdout

        return exp
