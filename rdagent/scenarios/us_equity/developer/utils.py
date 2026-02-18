from typing import List

import pandas as pd

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import FactorEmptyError
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.us_equity.experiment.factor_experiment import USEquityFactorExperiment


def process_factor_data(exp_or_list: List[USEquityFactorExperiment] | USEquityFactorExperiment) -> pd.DataFrame:
    """
    Process and combine factor data from experiment implementations.
    Returns a DataFrame with all factor columns concatenated.
    """
    if isinstance(exp_or_list, USEquityFactorExperiment):
        exp_or_list = [exp_or_list]
    factor_dfs = []
    error_message = ""

    for exp in exp_or_list:
        if isinstance(exp, USEquityFactorExperiment):
            if len(exp.sub_tasks) > 0:
                assert isinstance(exp.prop_dev_feedback, CoSTEERMultiFeedback)
                message_and_df_list = multiprocessing_wrapper(
                    [
                        (implementation.execute, ("All",))
                        for implementation, fb in zip(exp.sub_workspace_list, exp.prop_dev_feedback)
                        if implementation and fb
                    ],
                    n=RD_AGENT_SETTINGS.multi_proc_n,
                )
                for message, df in message_and_df_list:
                    if df is not None and "datetime" in df.index.names:
                        time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                        if pd.Timedelta(minutes=1) not in time_diff:
                            factor_dfs.append(df)
                            logger.info(f"Factor data generated successfully.")
                        else:
                            logger.warning(f"Factor data appears to be intraday — skipping.")
                    else:
                        error_message += f"Factor generation failed: {message}\n"
                        logger.warning(f"Factor generation failed: {message}")

    if factor_dfs:
        return pd.concat(factor_dfs, axis=1)
    else:
        raise FactorEmptyError(
            f"No valid factor data found (in process_factor_data). {error_message}"
        )
