from copy import deepcopy
from pathlib import Path

from rdagent.app.us_equity_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.scenarios.us_equity.experiment.workspace import USEquityFBWorkspace
from rdagent.utils.agent.tpl import T


class USEquityFactorExperiment(FactorExperiment[FactorTask, USEquityFBWorkspace, FactorFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = USEquityFBWorkspace(template_folder_path=Path(__file__).parent / "factor_template")
        self.stdout = ""


class USEquityFactorScenario(Scenario):
    def __init__(self) -> None:
        super().__init__()
        self._background = deepcopy(T(".prompts:us_equity_factor_background").r())
        self._source_data = deepcopy(self._get_data_folder_intro())
        self._output_format = deepcopy(T(".prompts:us_equity_factor_output_format").r())
        self._interface = deepcopy(T(".prompts:us_equity_factor_interface").r())
        self._strategy = deepcopy(T(".prompts:us_equity_factor_strategy").r())
        self._simulator = deepcopy(T(".prompts:us_equity_factor_simulator").r())
        self._rich_style_description = deepcopy(T(".prompts:us_equity_factor_rich_style_description").r())
        self._experiment_setting = deepcopy(
            T(".prompts:us_equity_factor_experiment_setting").r(
                train_start=FACTOR_PROP_SETTING.train_start,
                train_end=FACTOR_PROP_SETTING.train_end,
                valid_start=FACTOR_PROP_SETTING.valid_start,
                valid_end=FACTOR_PROP_SETTING.valid_end,
                test_start=FACTOR_PROP_SETTING.test_start,
                test_end=FACTOR_PROP_SETTING.test_end,
            )
        )

    @staticmethod
    def _get_data_folder_intro() -> str:
        from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS

        data_folder = Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)
        if not data_folder.exists():
            return "Data folder not found. Please run `python scripts/prepare_us_equity_data.py` first."

        parts = []
        for p in sorted(data_folder.iterdir()):
            if p.name.endswith(".md"):
                parts.append(p.read_text(encoding="utf-8"))
            elif p.name.endswith(".h5"):
                import pandas as pd

                df = pd.read_hdf(p)
                info = f"## {p.name}\n"
                info += f"- Shape: {df.shape}\n"
                info += f"- Index: MultiIndex with levels {list(df.index.names)}\n"
                info += f"- Columns: {', '.join(df.columns.tolist())}\n"
                parts.append(info)
        return "\n---\n".join(parts)

    @property
    def background(self) -> str:
        return self._background

    def get_source_data_desc(self, task: Task | None = None) -> str:
        return self._source_data

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    @property
    def experiment_setting(self) -> str:
        return self._experiment_setting

    def get_runtime_environment(self) -> str:
        import sys

        return f"Python {sys.version}, local execution (no Docker)"

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        if simple_background:
            return f"""Background of the scenario:
{self.background}"""
        return f"""Background of the scenario:
{self.background}
The source data you can use:
{self.get_source_data_desc(task)}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your factor:
{self.simulator}
"""
