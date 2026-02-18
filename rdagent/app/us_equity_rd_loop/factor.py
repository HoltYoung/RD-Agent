"""
US Equity Factor workflow with session control.
"""

from dotenv import load_dotenv

load_dotenv(".env")

import asyncio
from pathlib import Path
from typing import Any, Optional

import fire
import typer
from typing_extensions import Annotated

from rdagent.app.us_equity_rd_loop.conf import FACTOR_PROP_SETTING
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import FactorEmptyError
from rdagent.log import rdagent_logger as logger


class FactorRDLoop(RDLoop):
    skip_loop_error = (FactorEmptyError,)

    def running(self, prev_out: dict[str, Any]):
        exp = self.runner.develop(prev_out["coding"])
        if exp is None:
            logger.error("Factor extraction failed.")
            raise FactorEmptyError("Factor extraction failed.")
        logger.log_object(exp, tag="runner result")
        return exp


def main(
    path: Optional[str] = None,
    step_n: Optional[int] = None,
    loop_n: Optional[int] = None,
    all_duration: str | None = None,
    checkout: Annotated[bool, typer.Option("--checkout/--no-checkout", "-c/-C")] = True,
    checkout_path: Optional[str] = None,
):
    """
    Auto R&D Evolving loop for US equity alpha factors.

    Discovers cross-sectional alpha factors for a long-short dollar-neutral strategy
    on ~600 US equities. No Docker or Qlib required.
    """
    if checkout_path is not None:
        checkout = Path(checkout_path)

    if path is None:
        model_loop = FactorRDLoop(FACTOR_PROP_SETTING)
    else:
        model_loop = FactorRDLoop.load(path, checkout=checkout)
    asyncio.run(model_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
