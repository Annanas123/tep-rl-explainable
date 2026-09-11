# TEP-RL

This repository contains the code used in a master's thesis on reinforcement learning for transmission expansion planning. It compares PPO, preference-conditioned MO-PPO, deterministic baselines, and a continuous DC optimisation reference on a stylised Austrian PyPSA network. Policy decisions can be analysed with permutation-sampled Shapley values.

The current formulation reinforces existing transmission corridors only. It uses 60 candidate lines, a 500 MW limit per line, and a total reinforcement budget of 500 MW.

## Setup

Python 3.11 is recommended. From the repository root, create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-pypsa1.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements-pypsa1.txt
```

## Input data

The repository includes the inputs required by the thesis pipeline:

- `derived/austria_net_physical_ratings.nc`
- `data/entsoe_at_load_2015_2024_opsd.csv`
- `data/wind_at_2015_2024.csv`
- `data/solar_at_2015_2024.csv`
- `data/NUTS_RG_01M_2024_4326_LEVL_2.geojson`
- `config/official_future_scenarios.json`

## Run

Run the test suite first:

```bash
python -m unittest tests.test_framework
```

Run a short end-to-end check:

```bash
python main.py smoke-test --timesteps 256 --eval-episodes 1
```

Run the thesis pipeline:

```bash
python scripts/run_thesis_pipeline.py --output-root results/thesis_run
```

The pipeline can take several days for the full five-seed experiment. Use `--from-stage` and `--to-stage` to run or resume individual stages. Generated checkpoints, tables, figures, and logs are written below `results/` and are intentionally excluded from version control.

Use `python main.py --help` or `python scripts/run_thesis_pipeline.py --help` for all options.

## Structure

- `main.py`: command-line entry point for training, evaluation, tests, and explainability
- `tep_rl/`: environments, agents, training, evaluation, statistics, and Shapley analysis
- `scripts/`: experiment pipeline, baselines, validation, analysis, and figure generation
- `tests/`: unit and integration tests
- `config/`: future-scenario definitions
- `data/` and `derived/`: canonical input data
