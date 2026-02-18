# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RD-Agent is a Microsoft Research Asia LLM-Agent framework for autonomous data science and R&D automation. It automates the research ("R" - proposing ideas) and development ("D" - implementing ideas) cycle across scenarios including quantitative trading (Qlib), Kaggle competitions, medical prediction, and general ML engineering. Python 3.10+ required; Linux is the primary supported platform.

## Common Commands

### Setup
```bash
make dev          # Install package in editable mode with all optional deps
make install      # Install in editable mode (no extras)
```

### Linting
```bash
make lint         # Run all linters in check mode (black, isort, mypy, ruff, toml-sort)
make auto-lint    # Run all linters with automatic fixes
make ruff         # Ruff only (rdagent/core/ scoped)
make mypy         # mypy only (rdagent/core/ scoped)
```

### Testing
```bash
make test-offline      # Run offline tests with coverage (no API keys needed)
make test-run-offline  # Run offline tests without coverage
make test              # Full test suite (requires API credentials)
pytest test/utils/test_conf.py                  # Run a single test file
pytest test/utils/test_conf.py -k "test_name"   # Run a single test
```

### Documentation
```bash
make docs-gen     # Build Sphinx HTML docs
```

### CLI Entry Point
The `rdagent` CLI (defined in `rdagent/app/cli.py`) provides commands like `rdagent data_science --competition <name>`, `rdagent fin_factor`, `rdagent ui`, `rdagent health_check`, etc.

## Code Style

- Line length: 120 characters (black + ruff)
- Import sorting: isort with black profile
- Ruff selects ALL rules with specific ignores (see `pyproject.toml` `[tool.ruff.lint]`)
- Conventional commit messages required for PRs (types: feat, fix, refactor, docs, test, etc.; max 100 chars)

## Architecture

### The R&D Loop

The core workflow is an iterative loop defined in `rdagent/components/workflow/rd_loop.py` (`RDLoop`) built on `rdagent/utils/workflow/loop.py` (`LoopBase`). Each iteration runs these steps in sequence:

1. **Propose** (`direct_exp_gen`): `HypothesisGen` generates a hypothesis, `Hypothesis2Experiment` converts it to an `Experiment` with sub-tasks
2. **Code** (component coders): `Developer.develop(exp)` evolves code via CoSTEER with RAG knowledge retrieval
3. **Run** (`running`): Execute the experiment in a Docker container via `DockerEnv`
4. **Feedback** (`feedback`): `Experiment2Feedback` compares results against SOTA, LLM generates structured feedback
5. **Record**: Pickle-serialize loop state for checkpoint/resume

`LoopBase` uses a metaclass (`LoopMeta`) that auto-collects public methods as ordered workflow steps. It supports async parallelism via `asyncio.Semaphore` controlled by `RD_AGENT_SETTINGS.step_semaphore`.

### Core Abstractions (`rdagent/core/`)

- **`experiment.py`**: `Task` (unit of work), `FBWorkspace` (file-based workspace with inject/execute/checkpoint), `Experiment` (collection of tasks + workspaces + hypothesis + results)
- **`proposal.py`**: `Hypothesis`, `Trace` (DAG of experiment history with parent tracking and SOTA selection), `ExpGen` (generates experiments), `Experiment2Feedback` (evaluates results)
- **`developer.py`**: `Developer` - modifies experiments **in-place** (inplace mutation model so intermediate results survive exceptions)
- **`scenario.py`**: `Scenario` - abstract base providing background description, data source info, and runtime environment
- **`conf.py`**: `ExtendedBaseSettings` - pydantic-settings subclass supporting hierarchical env var inheritance from parent classes. `.env` loaded automatically.

### Scenario System

Scenarios live in `rdagent/scenarios/` with app entry points in `rdagent/app/`. Each scenario specializes the core abstractions:

- **Data Science** (`scenarios/data_science/`): `DataScienceRDLoop` extends `RDLoop` with per-component coders (DataLoader → Feature → Model → Ensemble → Workflow) in `DSTrace.COMPLETE_ORDER`. Each component has its own `CoSTEER` coder.
- **Qlib/Finance** (`scenarios/qlib/`): Factor and model evolution for quantitative trading
- **General Model** (`scenarios/general_model/`): Extract and implement models from papers

### CoSTEER Evolution Engine (`rdagent/components/coder/CoSTEER/`)

The main code development engine. `CoSTEER(Developer)` wraps `RAGEvoAgent` which iteratively:
1. Queries RAG knowledge base for relevant context
2. Evolves code via `EvolvingStrategy` (LLM-driven code generation/modification)
3. Evaluates with `RAGEvaluator`
4. Stores evolution trace and updates knowledge base

### LLM Backend (`rdagent/oai/`)

- **`llm_conf.py`**: `LLM_SETTINGS` singleton configures model, temperature, caching, retry logic
- **`backend/base.py`**: `APIBackend` handles chat completions with auto-continuation on length limit, JSON parsing, SQLite caching, and retry with backoff
- **`backend/pydantic_ai.py`**: Pydantic-AI integration via `get_agent_model()` for tool-calling agents
- Backend is configurable via `LLM_SETTINGS.backend` (default: `LiteLLMAPIBackend`)

### Docker Execution (`rdagent/utils/env.py`)

Experiments run in Docker containers via `DockerEnv`. `FBWorkspace.execute()` injects files into the workspace, runs the entry command inside Docker with timeout/retry, and returns `EnvResult` (stdout, exit_code, running_time). Workspaces support ZIP checkpointing for state recovery.

### Plugin/Extension Pattern

Components are loaded dynamically via `import_class(dotted_path)`. Scenario configs specify class paths as strings (e.g., `"scenarios.data_science.scen:DataScienceScen"`), enabling swappable implementations without hard imports.

### Trace DAG

`Trace` maintains experiment history as a DAG (not a linear list). `dag_parent` maps each experiment index to its parent indices. `current_selection` controls branching: `(-1,)` means continue from last, `()` means start a new root. This supports parallel exploration of multiple hypothesis branches.
