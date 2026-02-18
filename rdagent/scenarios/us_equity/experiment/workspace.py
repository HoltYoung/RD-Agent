import subprocess
from pathlib import Path

import pandas as pd

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger


class USEquityFBWorkspace(FBWorkspace):
    """Runs factor evaluation locally (no Docker, no Qlib)."""

    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def execute(self, run_env: dict = {}, *args, **kwargs) -> tuple:
        """
        Run evaluate.py in the workspace directory.
        Returns (pd.Series of metrics, stdout_log_str).
        """
        env = {
            "PYTHONPATH": "./",
            **run_env,
        }

        # Copy source data into workspace so evaluate.py can find it
        workspace = self.workspace_path
        source_data = Path(FACTOR_COSTEER_SETTINGS.data_folder)
        if source_data.exists():
            for f in source_data.iterdir():
                if f.suffix in (".h5", ".md"):
                    target = workspace / f.name
                    if not target.exists():
                        import shutil
                        shutil.copy2(f, target)

        # Run evaluate.py
        try:
            python_bin = FACTOR_COSTEER_SETTINGS.python_bin
            result = subprocess.run(
                [python_bin, "evaluate.py"],
                cwd=str(workspace),
                env={**__import__("os").environ, **{k: str(v) for k, v in env.items()}},
                capture_output=True,
                text=True,
                timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
            )
            stdout = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            logger.error("Evaluation timed out.")
            return None, "Evaluation timed out."
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None, str(e)

        logger.log_object(stdout, tag="evaluate_log")

        # Read results
        results_path = workspace / "evaluation_results.csv"
        if results_path.exists():
            metrics = pd.read_csv(results_path, index_col=0).iloc[:, 0]
            return metrics, stdout
        else:
            logger.error(f"No evaluation results found. Stdout:\n{stdout}")
            return None, stdout
