from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.evolving_strategy import (
    FactorMultiProcessEvolvingStrategy,
)
from rdagent.core.experiment import Experiment
from rdagent.core.scenario import Scenario
from rdagent.scenarios.us_equity.developer.evaluators import LightweightFactorEvaluator


class USEquityFactorCoSTEER(CoSTEER):
    """FactorCoSTEER optimized for free-tier API quotas.

    - No knowledge/RAG (no embedding model needed)
    - Lightweight evaluator (no LLM calls for evaluation)
    - Single coding loop to minimize API calls
    """

    def __init__(self, scen: Scenario, *args, **kwargs) -> None:
        setting = FACTOR_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(LightweightFactorEvaluator(scen=scen), scen=scen)
        es = FactorMultiProcessEvolvingStrategy(scen=scen, settings=FACTOR_COSTEER_SETTINGS)

        super().__init__(
            *args,
            settings=setting,
            eva=eva,
            es=es,
            evolving_version=2,
            scen=scen,
            with_knowledge=False,
            knowledge_self_gen=False,
            max_loop=1,
            **kwargs,
        )

    def develop(self, exp: Experiment) -> Experiment:
        try:
            exp = super().develop(exp)
        finally:
            if hasattr(self, "evolve_agent") and self.evolve_agent.evolving_trace:
                es = self.evolve_agent.evolving_trace[-1]
                exp.prop_dev_feedback = es.feedback
        return exp
