from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.workflow.conf import BasePropSetting


class FactorBasePropSetting(BasePropSetting):
    model_config = SettingsConfigDict(env_prefix="US_EQUITY_FACTOR_", protected_namespaces=())

    scen: str = "rdagent.scenarios.us_equity.experiment.factor_experiment.USEquityFactorScenario"
    """Scenario class for US Equity Factor"""

    hypothesis_gen: str = "rdagent.scenarios.us_equity.proposal.factor_proposal.USEquityFactorHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "rdagent.scenarios.us_equity.proposal.factor_proposal.USEquityFactorHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "rdagent.scenarios.us_equity.developer.factor_coder.USEquityFactorCoSTEER"
    """Coder class"""

    runner: str = "rdagent.scenarios.us_equity.developer.factor_runner.USEquityFactorRunner"
    """Runner class"""

    summarizer: str = "rdagent.scenarios.us_equity.developer.feedback.USEquityFactorExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions per coding round"""

    train_start: str = "2020-10-01"
    """Start date of the training segment"""

    train_end: str = "2023-12-31"
    """End date of the training segment"""

    valid_start: str = "2024-01-01"
    """Start date of the validation segment"""

    valid_end: str = "2024-12-31"
    """End date of the validation segment"""

    test_start: str = "2025-01-01"
    """Start date of the test segment"""

    test_end: Optional[str] = "2026-02-28"
    """End date of the test segment"""


FACTOR_PROP_SETTING = FactorBasePropSetting()
