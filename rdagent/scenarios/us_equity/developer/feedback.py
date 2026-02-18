import json
from pathlib import Path
from typing import Dict

import pandas as pd

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T

IMPORTANT_METRICS = [
    "IC",
    "ICIR",
    "Long_Short_Annual_Return",
    "LGB_IC",
    "LGB_ICIR",
]


def process_results(current_result, sota_result):
    current_df = pd.DataFrame(current_result)
    current_df.index.name = "metric"
    current_df.rename(columns={"0": "Current Result"}, inplace=True)

    has_sota = sota_result is not None and len(sota_result) > 0

    if has_sota:
        sota_df = pd.DataFrame(sota_result)
        sota_df.index.name = "metric"
        sota_df.rename(columns={"0": "SOTA Result"}, inplace=True)
        combined_df = pd.concat([current_df, sota_df], axis=1)
    else:
        combined_df = current_df

    available_metrics = [m for m in IMPORTANT_METRICS if m in combined_df.index]
    if not available_metrics:
        return "No comparable metrics found."

    filtered_combined_df = combined_df.loc[available_metrics]

    results = []
    for metric, row in filtered_combined_df.iterrows():
        current = row.iloc[0]
        if has_sota and len(row) > 1:
            sota = row.iloc[1]
            results.append(f"{metric} of Current Result is {current:.6f}, of SOTA Result is {sota:.6f}")
        else:
            results.append(f"{metric} of Current Result is {current:.6f} (no SOTA baseline yet)")
    return "; ".join(results)


class USEquityFactorExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        hypothesis = exp.hypothesis
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
        sota_result = None
        if exp.based_experiments and exp.based_experiments[-1].result is not None:
            sota_result = exp.based_experiments[-1].result

        combined_result = process_results(current_result, sota_result)

        sys_prompt = T("scenarios.us_equity.prompts:factor_feedback_generation.system").r(
            scenario=self.scen.get_scenario_all_desc()
        )

        usr_prompt = T("scenarios.us_equity.prompts:factor_feedback_generation.user").r(
            hypothesis_text=hypothesis_text,
            task_details=tasks_factors,
            combined_result=combined_result,
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        response_json = json.loads(response)

        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        # First experiment with results should always become SOTA (nothing to compare against)
        if sota_result is None and current_result is not None:
            decision = True
            logger.info("No SOTA baseline exists -- setting current result as initial SOTA.")

        # Heuristic overrides: free-tier LLMs often fail at precise numeric comparison
        if sota_result is not None and current_result is not None:
            try:
                cur = pd.DataFrame(current_result)
                sota = pd.DataFrame(sota_result)
                cur_ic = cur.loc["IC"].iloc[0] if "IC" in cur.index else None
                sota_ic = sota.loc["IC"].iloc[0] if "IC" in sota.index else None
                cur_icir = cur.loc["ICIR"].iloc[0] if "ICIR" in cur.index else None
                sota_icir = sota.loc["ICIR"].iloc[0] if "ICIR" in sota.index else None
                cur_lgb = cur.loc["LGB_IC"].iloc[0] if "LGB_IC" in cur.index else None
                sota_lgb = sota.loc["LGB_IC"].iloc[0] if "LGB_IC" in sota.index else None

                if cur_ic is not None and sota_ic is not None and cur_icir is not None and sota_icir is not None:
                    # Primary: if LGB_IC improved, that's the strongest signal
                    if cur_lgb is not None and sota_lgb is not None and cur_lgb > sota_lgb and cur_lgb > 0:
                        if not decision:
                            decision = True
                            logger.info(f"Heuristic override (no->yes): LGB_IC improved ({cur_lgb:.6f} > {sota_lgb:.6f}). "
                                        f"Setting as new SOTA.")
                    # Override no→yes: both IC and ICIR improved
                    elif not decision and cur_ic > sota_ic and cur_icir > sota_icir:
                        decision = True
                        logger.info(f"Heuristic override (no->yes): IC improved ({cur_ic:.6f} > {sota_ic:.6f}) "
                                    f"and ICIR improved ({cur_icir:.6f} > {sota_icir:.6f}). Setting as new SOTA.")
                    # Override yes→no: both IC and ICIR got worse AND LGB_IC didn't improve
                    elif decision and cur_ic < sota_ic and cur_icir < sota_icir:
                        if cur_lgb is None or sota_lgb is None or cur_lgb <= sota_lgb:
                            decision = False
                            logger.info(f"Heuristic override (yes->no): IC worsened ({cur_ic:.6f} < {sota_ic:.6f}) "
                                        f"and ICIR worsened ({cur_icir:.6f} < {sota_icir:.6f}). Keeping current SOTA.")
            except Exception:
                pass  # If comparison fails, trust the LLM decision

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )
