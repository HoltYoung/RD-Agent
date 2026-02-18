"""Lightweight factor evaluator that uses NO LLM calls.

Checks only: execution success, output file exists, valid DataFrame format.
This allows the R&D loop to run within free-tier API quotas.
"""

import re

import numpy as np
import pandas as pd

from rdagent.components.coder.CoSTEER.evaluators import CoSTEEREvaluator, CoSTEERSingleFeedbackDeprecated
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Workspace

FactorSingleFeedback = CoSTEERSingleFeedbackDeprecated


class LightweightFactorEvaluator(CoSTEEREvaluator):
    """Evaluates factor implementations without any LLM calls.

    Checks:
    1. Code executed without errors
    2. result.h5 was produced
    3. Output is a DataFrame with correct MultiIndex and a single numeric column
    4. Factor values are not all NaN/constant
    """

    def evaluate(
        self,
        target_task: FactorTask,
        implementation: Workspace,
        gt_implementation: Workspace = None,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FactorSingleFeedback:
        if implementation is None:
            return None

        factor_feedback = FactorSingleFeedback()

        # 1. Execute the factor code
        execution_feedback, gen_df = implementation.execute()
        execution_feedback = re.sub(r"(?<=\D)(,\s+-?\d+\.\d+){50,}(?=\D)", ", ", execution_feedback)
        factor_feedback.execution_feedback = "\n".join(
            [line for line in execution_feedback.split("\n") if "warning" not in line.lower()]
        )

        # 2. Check if output was produced
        if gen_df is None:
            factor_feedback.value_feedback = "No factor value generated. Code must produce result.h5."
            factor_feedback.value_generated_flag = False
            factor_feedback.code_feedback = "Factor code failed to produce output. Check the execution feedback for errors."
            factor_feedback.final_decision = False
            factor_feedback.final_feedback = "Failed: no output produced."
            factor_feedback.final_decision_based_on_gt = False
            return factor_feedback

        factor_feedback.value_generated_flag = True

        # 3. Validate DataFrame format
        issues = []

        if not isinstance(gen_df, pd.DataFrame):
            issues.append("Output is not a DataFrame.")
        elif gen_df.empty:
            issues.append("Output DataFrame is empty.")
        else:
            # Check MultiIndex
            if not isinstance(gen_df.index, pd.MultiIndex):
                issues.append("Output must have MultiIndex (datetime, instrument).")
            elif gen_df.index.nlevels != 2:
                issues.append(f"Expected 2 index levels, got {gen_df.index.nlevels}.")

            # Check single column
            if len(gen_df.columns) != 1:
                issues.append(f"Expected 1 column, got {len(gen_df.columns)}: {list(gen_df.columns)}.")

            # Check numeric values
            if len(gen_df.columns) == 1:
                col = gen_df.columns[0]
                if not np.issubdtype(gen_df[col].dtype, np.number):
                    issues.append(f"Column '{col}' is not numeric (dtype={gen_df[col].dtype}).")
                elif gen_df[col].isna().all():
                    issues.append("All factor values are NaN.")
                elif gen_df[col].nunique() <= 1:
                    issues.append("Factor has constant value — no cross-sectional variation.")
                else:
                    # Check for reasonable NaN ratio
                    nan_ratio = gen_df[col].isna().mean()
                    if nan_ratio > 0.5:
                        issues.append(f"Too many NaN values ({nan_ratio:.1%}).")

                    # Check for infinite values
                    inf_count = np.isinf(gen_df[col].dropna()).sum()
                    if inf_count > 0:
                        issues.append(f"Found {inf_count} infinite values.")

        if issues:
            factor_feedback.value_feedback = "Validation issues: " + "; ".join(issues)
            factor_feedback.code_feedback = "Fix the following issues: " + "; ".join(issues)
            factor_feedback.final_decision = False
            factor_feedback.final_feedback = "Failed validation: " + "; ".join(issues)
        else:
            col = gen_df.columns[0]
            nan_ratio = gen_df[col].isna().mean()
            factor_feedback.value_feedback = (
                f"Factor '{col}' produced successfully. "
                f"Shape: {gen_df.shape}, NaN ratio: {nan_ratio:.1%}, "
                f"unique values: {gen_df[col].nunique()}."
            )
            factor_feedback.code_feedback = "Factor code executed successfully and produced valid output."
            factor_feedback.final_decision = True
            factor_feedback.final_feedback = "Factor implementation passed all validation checks."

        factor_feedback.final_decision_based_on_gt = False
        return factor_feedback
