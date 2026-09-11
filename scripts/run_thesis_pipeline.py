from __future__ import annotations

"""End-to-end thesis pipeline for the final TEP-RL study.

The pipeline keeps the nine thesis stages explicit so each expensive block can
be resumed independently. Result directories follow the descriptive names in
``scripts.experiment_layout``.
"""

import argparse
import gc
import pickle
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

if __name__ == "__main__":
    print("[run_thesis_pipeline] Bootstrapping imports...", flush=True)

import pandas as pd
import pypsa
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tep_rl.config import EnvironmentConfig, PPOConfig, TrainingConfig
from tep_rl.baselines import BASELINE_LABELS, build_baseline_agents
from tep_rl.data import apply_observation_scale_reference, refit_candidate_line_screening
from tep_rl.evaluation import evaluate_agent, evaluate_moppo_preference_grid
from tep_rl.future_scenarios import (
    DEFAULT_NUTS2_BOUNDARIES,
    DEFAULT_SCENARIO_CATALOG,
    apply_future_scenario,
    fit_future_scenario,
    scenario_audit,
)
from tep_rl.ppo import load_agent
from tep_rl.statistics import compare_paired_evaluations
from tep_rl.thesis_formulations import (
    DEFAULT_THESIS_FORMULATION_PRESET,
    THESIS_FORMULATION_PRESETS,
    apply_thesis_formulation_preset,
)
from tep_rl.training import train_multi_seed, train_weight_sweep
from tep_rl.visualization import plot_pareto_front

from thesis_pipeline_utils import (
    build_multi_seed_summary_payload,
    build_dataset,
    build_env,
    collect_run_dirs,
    json_default,
    list_checkpoint_candidates_from_run_dir,
    read_json,
    scalarized_mean_reward,
    select_best_checkpoint_from_run_dir,
    write_json,
    write_multi_seed_summary,
)
from validate_experiment_results import validate_results
from experiment_layout import locate_results_dir, preferred_results_dir


def resolve_device(requested: str) -> str:
    """Return a safe torch device string for the local machine."""
    try:
        import torch
        if requested in {"auto", "gpu"}:
            requested = "cuda"
        if requested == "cuda":
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                print(f"[GPU] Using CUDA device: {name}")
                return "cuda"
            else:
                print("[GPU] WARNING: --device cuda requested but CUDA not available. Falling back to cpu.")
                return "cpu"
    except ImportError:
        pass
    return "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    default_env = EnvironmentConfig()
    default_ppo = PPOConfig()
    parser = argparse.ArgumentParser(description="Run the full thesis experiment pipeline end-to-end.")
    parser.add_argument("--network", default="derived/austria_net_physical_ratings.nc")
    parser.add_argument("--load", default="data/entsoe_at_load_2015_2024_opsd.csv")
    parser.add_argument("--wind", default="data/wind_at_2015_2024.csv")
    parser.add_argument("--solar", default="data/solar_at_2015_2024.csv")
    parser.add_argument(
        "--future-scenario",
        default="apg_tyndp_nt_2040",
        help="Source-based exogenous scenario from config/official_future_scenarios.json; use 'historical' for no scaling.",
    )
    parser.add_argument("--scenario-catalog", default=str(DEFAULT_SCENARIO_CATALOG.relative_to(ROOT)))
    parser.add_argument("--nuts2-boundaries", default=str(DEFAULT_NUTS2_BOUNDARIES.relative_to(ROOT)))
    parser.add_argument("--train-start", default="2015-01-01")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--val-start", default="2021-01-01")
    parser.add_argument("--val-end", default="2022-12-31")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    parser.add_argument(
        "--formulation-preset",
        choices=sorted(THESIS_FORMULATION_PRESETS),
        default=DEFAULT_THESIS_FORMULATION_PRESET,
        help="Apply a shared thesis formulation preset unless the specific budget/candidate flags are overridden.",
    )
    parser.add_argument("--candidate-lines", type=int, default=60)
    parser.add_argument("--episode-length", type=int, default=24)
    parser.add_argument("--max-upgrade-mw", type=float, default=60.0)
    parser.add_argument("--budget-mw", type=float, default=240.0)
    parser.add_argument("--decision-interval", type=int, default=6)
    parser.add_argument("--temporal-mode", choices=["decision_block", "hourly"], default=default_env.temporal_mode)
    parser.add_argument("--budget-release", choices=["linear", "all_at_once"], default="linear")
    parser.add_argument("--action-mode", choices=["budgeted", "direct"], default="budgeted")
    parser.add_argument("--allocation-sharpness", type=float, default=default_env.allocation_sharpness)
    parser.add_argument("--allocation-sparsity-cutoff", type=float, default=default_env.allocation_sparsity_cutoff)
    parser.add_argument("--stability-margin", type=float, default=default_env.stability_margin)
    parser.add_argument(
        "--proxy-balance-mode",
        choices=["demand_proportional", "single_slack"],
        default=default_env.proxy_balance_mode,
    )
    parser.add_argument("--proxy-dispatch-limit", type=float, default=default_env.proxy_dispatch_limit)
    parser.add_argument("--load-shedding-cost", type=float, default=default_env.load_shedding_cost)
    parser.add_argument(
        "--line-investment-cost-eur-per-mw-km-year",
        type=float,
        default=default_env.line_investment_cost_eur_per_mw_km_year,
    )
    parser.add_argument("--cost-reward-scale", type=float, default=default_env.cost_reward_scale)
    parser.add_argument("--overload-reward-scale", type=float, default=default_env.overload_reward_scale)
    parser.add_argument(
        "--third-objective-mode",
        choices=["renewable_share", "curtailment", "emissions"],
        default=default_env.third_objective_mode,
    )
    parser.add_argument("--curtailment-reward-scale", type=float, default=default_env.curtailment_reward_scale)
    parser.add_argument("--emissions-reward-scale", type=float, default=default_env.emissions_reward_scale)
    parser.add_argument("--solver", default="highs")
    parser.add_argument(
        "--disable-full-env-fallback-to-proxy",
        action="store_true",
        help="In full-environment evaluation, fail on PyPSA solve errors instead of falling back to the proxy backend.",
    )
    parser.add_argument("--weights", default=",".join(str(weight) for weight in default_ppo.scalarization_weights))
    parser.add_argument("--myopic-chunk-mw", type=float, default=None)
    parser.add_argument("--benchmark-episodes", type=int, default=1)
    parser.add_argument("--weight-grid", nargs="+", default=[
        "0.70,0.20,0.10",
        "0.50,0.30,0.20",
        "0.34,0.33,0.33",
        "0.20,0.30,0.50",
        "0.20,0.20,0.60",
    ])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate-schedule", choices=["constant", "linear"], default=default_ppo.learning_rate_schedule)
    parser.add_argument("--final-learning-rate", type=float, default=default_ppo.final_learning_rate)
    parser.add_argument("--gamma", type=float, default=default_ppo.gamma)
    parser.add_argument("--gae-lambda", type=float, default=default_ppo.gae_lambda)
    parser.add_argument("--clip-epsilon", type=float, default=default_ppo.clip_epsilon)
    parser.add_argument("--entropy-coef", type=float, default=default_ppo.entropy_coef)
    parser.add_argument("--entropy-coef-schedule", choices=["constant", "linear"], default=default_ppo.entropy_coef_schedule)
    parser.add_argument("--final-entropy-coef", type=float, default=default_ppo.final_entropy_coef)
    parser.add_argument("--target-kl", type=float, default=default_ppo.target_kl)
    parser.add_argument("--normalize-rewards", action="store_true", default=default_ppo.normalize_rewards)
    parser.add_argument(
        "--moppo-training-mode",
        choices=["fixed", "conditioned"],
        default="conditioned",
        help="`conditioned` trains a universal MO-PPO policy over preference vectors; `fixed` reduces MO-PPO to one scalarisation.",
    )
    parser.add_argument("--disable-objective-advantage-normalization", action="store_true")
    parser.add_argument("--disable-moppo-preference-conditioning", action="store_true")
    parser.add_argument("--disable-moppo-preference-sampling", action="store_true")
    parser.add_argument("--moppo-dirichlet-alpha", type=float, default=default_ppo.moppo_dirichlet_alpha)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--early-stopping-patience-evals", type=int, default=8)
    parser.add_argument("--early-stopping-min-evals", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--disable-restore-best-model", action="store_true")
    parser.add_argument("--fullenv-episodes", type=int, default=3)
    parser.add_argument("--fullenv-selection-episodes", type=int, default=3)
    parser.add_argument("--checkpoint-rerank-top-k", type=int, default=5)
    parser.add_argument("--shapley-episodes", type=int, default=10)
    parser.add_argument("--shapley-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--seeds", default="7,11,19,23,31")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default=f"results/thesis_final_{timestamp}")
    parser.add_argument("--from-stage", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--to-stage", type=int, default=9, choices=range(1, 10))
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-baseline-benchmark", action="store_true")
    parser.add_argument("--skip-fullenv", action="store_true")
    args = parser.parse_args(raw_argv)
    apply_thesis_formulation_preset(args, raw_argv)
    return args


def parse_csv_tuple(text: str, cast):
    return tuple(cast(part) for part in re.split(r"[\s,]+", str(text).strip()) if part)


def make_env_kwargs(args: argparse.Namespace, seed: int) -> dict:
    return {
        "episode_length": args.episode_length,
        "max_upgrade_mw": args.max_upgrade_mw,
        "budget_mw": args.budget_mw,
        "decision_interval": args.decision_interval,
        "temporal_mode": args.temporal_mode,
        "budget_release": args.budget_release,
        "action_mode": args.action_mode,
        "allocation_sharpness": args.allocation_sharpness,
        "allocation_sparsity_cutoff": args.allocation_sparsity_cutoff,
        "stability_margin": args.stability_margin,
        "proxy_balance_mode": args.proxy_balance_mode,
        "proxy_dispatch_limit": args.proxy_dispatch_limit,
        "load_shedding_cost": args.load_shedding_cost,
        "line_investment_cost_eur_per_mw_km_year": args.line_investment_cost_eur_per_mw_km_year,
        "cost_reward_scale": args.cost_reward_scale,
        "overload_reward_scale": args.overload_reward_scale,
        "third_objective_mode": args.third_objective_mode,
        "curtailment_reward_scale": args.curtailment_reward_scale,
        "emissions_reward_scale": args.emissions_reward_scale,
        "solver": args.solver,
        "full_env_fallback_to_proxy": not args.disable_full_env_fallback_to_proxy,
        "seed": seed,
    }


def run_tests() -> None:
    subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_framework", "-v"],
        cwd=ROOT,
        check=True,
    )


def run_dataset_report(args: argparse.Namespace, output_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/report_thesis_dataset.py",
            "--network", args.network,
            "--load", args.load,
            "--wind", args.wind,
            "--solar", args.solar,
            "--start", args.train_start,
            "--end", args.test_end,
            "--candidate-lines", str(args.candidate_lines),
            "--output-dir", str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )


def run_final_analysis(output_root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/final_thesis_analysis.py",
            "--results-root",
            str(output_root),
        ],
        cwd=ROOT,
        check=True,
    )


def run_baseline_benchmark(
    args: argparse.Namespace,
    output_dir: Path,
    env_mode: str,
    episodes: int,
) -> None:
    command = [
        sys.executable,
        "scripts/benchmark_baselines.py",
        "--env", env_mode,
        "--network", args.network,
        "--load", args.load,
        "--wind", args.wind,
        "--solar", args.solar,
        "--start", args.test_start,
        "--end", args.test_end,
        "--candidate-lines", str(args.candidate_lines),
        "--episode-length", str(args.episode_length),
        "--max-upgrade-mw", str(args.max_upgrade_mw),
        "--budget-mw", str(args.budget_mw),
        "--decision-interval", str(args.decision_interval),
        "--temporal-mode", args.temporal_mode,
        "--budget-release", args.budget_release,
        "--action-mode", args.action_mode,
        "--allocation-sharpness", str(args.allocation_sharpness),
        "--allocation-sparsity-cutoff", str(args.allocation_sparsity_cutoff),
        "--stability-margin", str(args.stability_margin),
        "--proxy-balance-mode", args.proxy_balance_mode,
        "--line-investment-cost-eur-per-mw-km-year", str(args.line_investment_cost_eur_per_mw_km_year),
        "--cost-reward-scale", str(args.cost_reward_scale),
        "--overload-reward-scale", str(args.overload_reward_scale),
        "--third-objective-mode", args.third_objective_mode,
        "--curtailment-reward-scale", str(args.curtailment_reward_scale),
        "--emissions-reward-scale", str(args.emissions_reward_scale),
        "--solver", args.solver,
        "--episodes", str(episodes),
        "--weights", args.weights,
        "--baselines", "zero", "uniform", "myopic_proxy",
        "--output", str(output_dir / f"{env_mode}_baseline_runtime.json"),
    ]
    preprocessing_manifest = getattr(args, "preprocessing_manifest", None)
    if preprocessing_manifest:
        command.extend(["--preprocessing-manifest", str(preprocessing_manifest)])
    if args.disable_full_env_fallback_to_proxy:
        command.append("--disable-full-env-fallback-to-proxy")
    if args.proxy_dispatch_limit is not None:
        command.extend(["--proxy-dispatch-limit", str(args.proxy_dispatch_limit)])
    if args.myopic_chunk_mw is not None:
        command.extend(["--myopic-chunk-mw", str(args.myopic_chunk_mw)])
    subprocess.run(command, cwd=ROOT, check=True)


def evaluate_selected_runs(
    selected_runs: list[dict],
    agent_label: str,
    env_mode: str,
    dataset,
    output_dir: Path,
    env_kwargs: dict,
    episodes: int,
    device: str = "cpu",
    progress_label: str | None = None,
) -> tuple[list[dict], Path, Path]:
    agent_dir = output_dir / agent_label
    agent_dir.mkdir(parents=True, exist_ok=True)
    per_seed = []

    iterator = tqdm(
        selected_runs,
        desc=progress_label or f"Eval {agent_label} {env_mode}",
        unit="seed",
        dynamic_ncols=True,
        leave=False,
    )
    for selected in iterator:
        run_dir = Path(selected["run_dir"])
        config_snapshot = read_json(run_dir / "config_snapshot.json")
        run_id = run_dir.name
        seed = int(config_snapshot["seed"])
        seed_dir = agent_dir / run_id
        selection_path = seed_dir / "selection.json"
        evaluation_path = seed_dir / "test_evaluation.json"
        if selection_path.exists() and evaluation_path.exists():
            cached_selection = read_json(selection_path)
            cached_evaluation = read_json(evaluation_path)
            if (
                str(cached_selection.get("checkpoint")) == str(selected["checkpoint"])
                and list(cached_selection.get("eval_preference_weights") or []) == list(selected.get("eval_preference_weights") or [])
                and int(cached_evaluation.get("n_episodes", -1)) == int(episodes)
            ):
                per_seed.append(
                    {
                        "seed": seed,
                        "run_id": run_id,
                        "checkpoint": selected["checkpoint"],
                        "selection_score": selected["selection_score"],
                        "evaluation": cached_evaluation,
                    }
                )
                continue
        local_env_kwargs = dict(env_kwargs)
        local_env_kwargs["seed"] = seed
        env = build_env(dataset, env_mode=env_mode, **local_env_kwargs)
        agent = load_agent(selected["checkpoint"], device=device)
        if hasattr(agent, "set_eval_preferences") and selected.get("eval_preference_weights") is not None:
            agent.set_eval_preferences(selected["eval_preference_weights"])
        evaluation = evaluate_agent(agent, env, episodes=episodes, deterministic=True)
        iterator.set_postfix(seed=seed, episodes=episodes)

        seed_dir.mkdir(parents=True, exist_ok=True)
        write_json(selection_path, selected)
        write_json(evaluation_path, evaluation)

        per_seed.append(
            {
                "seed": seed,
                "run_id": run_id,
                "checkpoint": selected["checkpoint"],
                "selection_score": selected["selection_score"],
                "evaluation": evaluation,
            }
        )
        del env
        del agent
        gc.collect()

    prefix = f"{agent_label}_{'fullenv' if env_mode == 'full' else 'proxy_test'}_summary"
    return per_seed, *write_multi_seed_summary(output_dir, agent_label, [entry["seed"] for entry in per_seed], per_seed, filename_prefix=prefix)


def _target_env_selection_key(item: dict) -> tuple[float, float, float, float, float]:
    evaluation = item["validation_evaluation"]
    return (
        float(item["selection_score"]),
        float(evaluation.get("renewable_share_mean", float("-inf"))),
        -float(evaluation.get("load_shedding_mean", float("inf"))),
        -float(evaluation.get("total_cost_mean", float("inf"))),
        -float(evaluation.get("grid_stress_mean", float("inf"))),
    )


def _load_validation_preference_grid(run_dir: Path, fallback_weights: tuple[float, ...] | list[float]) -> list[tuple[float, ...]]:
    config_snapshot = read_json(run_dir / "config_snapshot.json")
    training_grid = config_snapshot.get("training_config", {}).get("validation_weight_grid")
    if training_grid:
        return [tuple(float(value) for value in weights) for weights in training_grid]
    return [tuple(float(value) for value in fallback_weights)]


def rerank_selected_runs_on_target_env(
    selected_runs: list[dict],
    env_mode: str,
    dataset,
    output_dir: Path,
    env_kwargs: dict,
    episodes: int,
    rerank_top_k: int,
    device: str = "cpu",
    progress_label: str | None = None,
) -> list[dict]:
    selection_version = "target_env_validation_rerank_v2"
    rerank_dir = output_dir / "selection"
    rerank_dir.mkdir(parents=True, exist_ok=True)
    reranked: list[dict] = []

    outer_iterator = tqdm(
        selected_runs,
        desc=progress_label or f"Rerank {env_mode}",
        unit="seed",
        dynamic_ncols=True,
        leave=False,
    )
    for selected in outer_iterator:
        run_dir = Path(str(selected["run_dir"]))
        cached_selection_path = rerank_dir / f"{run_dir.name}.json"
        if cached_selection_path.exists():
            cached = read_json(cached_selection_path)
            cached_details = cached.get("selection_details", {})
            if (
                cached_details.get("mode") == "target_env_validation_rerank"
                and cached_details.get("selection_version") == selection_version
                and cached_details.get("selection_env") == env_mode
                and int(cached_details.get("selection_episodes", -1)) == int(episodes)
                and int(cached_details.get("rerank_top_k", -1)) == int(max(int(rerank_top_k), 1))
            ):
                reranked.append(cached)
                continue
        config_snapshot = read_json(run_dir / "config_snapshot.json")
        seed = int(config_snapshot["seed"])
        local_env_kwargs = dict(env_kwargs)
        local_env_kwargs["seed"] = seed
        candidates = list_checkpoint_candidates_from_run_dir(run_dir)[: max(int(rerank_top_k), 1)]

        evaluated_candidates: list[dict] = []
        candidate_iterator = tqdm(
            candidates,
            desc=f"{run_dir.name}",
            unit="ckpt",
            dynamic_ncols=True,
            leave=False,
        )
        for candidate in candidate_iterator:
            env = build_env(dataset, env_mode=env_mode, **local_env_kwargs)
            agent = load_agent(candidate["checkpoint"], device=device)
            eval_preference_weights = candidate.get("eval_preference_weights")
            preference_selection_details = None
            if (
                hasattr(agent, "set_eval_preferences")
                and bool(getattr(getattr(agent, "config", None), "moppo_preference_conditioning", False))
            ):
                preference_grid = _load_validation_preference_grid(run_dir, candidate["weights"])
                preference_result = evaluate_moppo_preference_grid(
                    agent,
                    env,
                    preference_grid=preference_grid,
                    utility_weights=candidate["weights"],
                    episodes=episodes,
                    deterministic=True,
                )
                evaluation = preference_result["evaluation"]
                score = float(preference_result["selection_score"])
                eval_preference_weights = preference_result["selected_eval_preference_weights"]
                preference_selection_details = preference_result["selection_details"]
            else:
                if hasattr(agent, "set_eval_preferences") and eval_preference_weights is not None:
                    agent.set_eval_preferences(eval_preference_weights)
                evaluation = evaluate_agent(agent, env, episodes=episodes, deterministic=True)
                score = scalarized_mean_reward(evaluation, candidate["weights"])
            candidate_iterator.set_postfix(
                update=int(candidate.get("update", -1)),
                score=f"{score:.3f}",
            )
            evaluated_candidates.append(
                {
                    "checkpoint": candidate["checkpoint"],
                    "evaluation_path": candidate["evaluation_path"],
                    "proxy_selection_score": float(candidate["selection_score"]),
                    "proxy_selection_origin": candidate.get("selection_origin"),
                    "proxy_update": int(candidate.get("update", -1)),
                    "eval_preference_weights": eval_preference_weights,
                    "preference_selection_details": preference_selection_details,
                    "selection_score": score,
                    "validation_evaluation": evaluation,
                }
            )
            del env
            del agent
            gc.collect()

        best = max(evaluated_candidates, key=_target_env_selection_key)
        outer_iterator.set_postfix(seed=seed, best=f"{float(best['selection_score']):.3f}")
        run_payload = {
            "run_dir": str(run_dir),
            "weights": list(selected["weights"]),
            "checkpoint": best["checkpoint"],
            "evaluation_path": best["evaluation_path"],
            "selection_score": float(best["selection_score"]),
            "eval_preference_weights": best.get("eval_preference_weights"),
            "update": int(best["proxy_update"]),
            "selection_details": {
                "mode": "target_env_validation_rerank",
                "selection_version": selection_version,
                "selection_env": env_mode,
                "selection_episodes": int(episodes),
                "rerank_top_k": int(max(int(rerank_top_k), 1)),
                "candidates": [
                    {
                        "checkpoint": item["checkpoint"],
                        "evaluation_path": item["evaluation_path"],
                        "proxy_selection_score": item["proxy_selection_score"],
                        "proxy_selection_origin": item["proxy_selection_origin"],
                        "proxy_update": item["proxy_update"],
                        "selected_eval_preference_weights": item.get("eval_preference_weights"),
                        "target_env_selection_score": item["selection_score"],
                        "preference_selection_details": item.get("preference_selection_details"),
                        "validation_summary": _compact_evaluation_summary(
                            item["validation_evaluation"],
                            label=Path(str(item["checkpoint"])).stem,
                        ),
                    }
                    for item in evaluated_candidates
                ],
            },
        }
        write_json(cached_selection_path, run_payload)
        reranked.append(run_payload)

    return reranked


def _compact_evaluation_summary(evaluation: dict, label: str) -> dict:
    action_stats = evaluation.get("action_stats", {})
    line_upgrades = evaluation.get("line_upgrades", {})
    top_lines = sorted(line_upgrades.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "label": label,
        "total_cost_mean": evaluation.get("total_cost_mean"),
        "grid_stress_mean": evaluation.get("grid_stress_mean"),
        "renewable_share_mean": evaluation.get("renewable_share_mean"),
        "constraint_violation_mean": evaluation.get("constraint_violation_mean"),
        "slack_generation_mean": evaluation.get("slack_generation_mean"),
        "load_shedding_mean": evaluation.get("load_shedding_mean"),
        "emissions_mean": evaluation.get("emissions_mean"),
        "total_investment_mean": action_stats.get("total_investment_mean"),
        "fraction_lines_touched": action_stats.get("fraction_lines_touched"),
        "top_line_upgrades": top_lines,
    }


def evaluate_baseline_suite(
    env_mode: str,
    dataset,
    output_dir: Path,
    env_kwargs: dict,
    episodes: int,
    reference_runs: list[dict] | None = None,
    uniform_value: float = 1.0,
    heuristic_top_k: int = 3,
    myopic_weights: tuple[float, ...] = (0.34, 0.33, 0.33),
    myopic_chunk_mw: float | None = None,
    progress_label: str | None = None,
) -> dict[str, dict]:
    baseline_dir = output_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    env = build_env(dataset, env_mode=env_mode, **env_kwargs)
    baseline_names = list(
        build_baseline_agents(
            env,
            uniform_value=uniform_value,
            heuristic_top_k=heuristic_top_k,
            myopic_weights=myopic_weights,
            myopic_chunk_mw=myopic_chunk_mw,
        ).keys()
    )

    if reference_runs:
        references = []
        for item in reference_runs:
            if "seed" in item:
                seed = int(item["seed"])
            else:
                run_dir = Path(str(item["run_dir"]))
                config_snapshot = read_json(run_dir / "config_snapshot.json")
                seed = int(config_snapshot["seed"])
            run_identifier = str(item.get("run_dir", item.get("run_id", f"seed{seed:03d}")))
            references.append(
                {
                    "seed": seed,
                    "run_id": Path(run_identifier).name,
                }
            )
    else:
        references = [{"seed": int(env_kwargs["seed"]), "run_id": f"seed{int(env_kwargs['seed']):03d}"}]

    baseline_iterator = tqdm(
        baseline_names,
        desc=progress_label or f"Baselines {env_mode}",
        unit="policy",
        dynamic_ncols=True,
        leave=False,
    )
    for name in baseline_iterator:
        baseline_policy_dir = baseline_dir / name
        baseline_policy_dir.mkdir(parents=True, exist_ok=True)
        per_seed: list[dict] = []
        for reference in references:
            local_env_kwargs = dict(env_kwargs)
            local_env_kwargs["seed"] = int(reference["seed"])
            seed_dir = baseline_policy_dir / str(reference["run_id"])
            test_eval_path = seed_dir / "test_evaluation.json"
            if test_eval_path.exists():
                cached_evaluation = read_json(test_eval_path)
                if int(cached_evaluation.get("n_episodes", -1)) == int(episodes):
                    per_seed.append(
                        {
                            "seed": int(reference["seed"]),
                            "run_id": str(reference["run_id"]),
                            "evaluation": cached_evaluation,
                        }
                    )
                    continue
            env = build_env(dataset, env_mode=env_mode, **local_env_kwargs)
            agent = build_baseline_agents(
                env,
                uniform_value=uniform_value,
                heuristic_top_k=heuristic_top_k,
                myopic_weights=myopic_weights,
                myopic_chunk_mw=myopic_chunk_mw,
            )[name]
            evaluation = evaluate_agent(agent, env, episodes=episodes, deterministic=True)

            seed_dir.mkdir(parents=True, exist_ok=True)
            write_json(test_eval_path, evaluation)
            per_seed.append(
                {
                    "seed": int(reference["seed"]),
                    "run_id": str(reference["run_id"]),
                    "evaluation": evaluation,
                }
            )
            del env
            del agent
            gc.collect()

        payload = build_multi_seed_summary_payload(
            agent=name,
            seeds=[entry["seed"] for entry in per_seed],
            per_seed=per_seed,
        )
        baseline_iterator.set_postfix(policy=name, seeds=len(per_seed))
        write_json(
            baseline_dir / f"{name}_{'fullenv' if env_mode == 'full' else 'proxy_test'}_summary.json",
            payload,
        )
        summary[name] = {
            "label": BASELINE_LABELS.get(name, name),
            "n_seeds": len(per_seed),
            "summary": payload["summary"],
        }

    write_json(output_dir / f"{'fullenv' if env_mode == 'full' else 'proxy'}_baseline_suite_summary.json", summary)
    return summary


def save_comparison(results_a: list[dict], results_b: list[dict], output_csv: Path) -> None:
    comparison = compare_paired_evaluations(
        [entry["evaluation"] for entry in results_a],
        [entry["evaluation"] for entry in results_b],
        label_a="PPO",
        label_b="MO-PPO",
        alpha=0.05,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False)


def write_validation_report(results_dir: Path) -> None:
    summary = validate_results(results_dir)
    write_json(results_dir / "validation_status.json", summary)
    if int(summary["n_incomplete"]) > 0:
        raise RuntimeError(f"Incomplete runs detected in {results_dir}. See validation_status.json.")


def should_run_stage(args: argparse.Namespace, stage: int) -> bool:
    return int(args.from_stage) <= int(stage) <= int(args.to_stage)


def main() -> None:
    args = parse_args()
    if int(args.from_stage) > int(args.to_stage):
        raise ValueError("--from-stage must be less than or equal to --to-stage.")
    args.device = resolve_device(args.device)
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    weights = parse_csv_tuple(args.weights, float)
    seeds = parse_csv_tuple(args.seeds, int)
    weight_grid = [parse_csv_tuple(weight_text, float) for weight_text in args.weight_grid]
    conditioned_preferences = args.moppo_training_mode == "conditioned"

    manifest = {
        "formulation": {
            "preset": args.formulation_preset,
            "description": args.formulation_description,
        },
        "network": args.network,
        "load": args.load,
        "wind": args.wind,
        "solar": args.solar,
        "future_scenario": {
            "requested_id": args.future_scenario,
            "catalog": args.scenario_catalog,
            "nuts2_boundaries": args.nuts2_boundaries,
        },
        "splits": {
            "train": [args.train_start, args.train_end],
            "validation": [args.val_start, args.val_end],
            "test": [args.test_start, args.test_end],
        },
        "candidate_lines": args.candidate_lines,
        "environment": {
            "episode_length": args.episode_length,
            "max_upgrade_mw": args.max_upgrade_mw,
            "budget_mw": args.budget_mw,
            "decision_interval": args.decision_interval,
            "temporal_mode": args.temporal_mode,
            "budget_release": args.budget_release,
            "action_mode": args.action_mode,
            "allocation_sharpness": args.allocation_sharpness,
            "allocation_sparsity_cutoff": args.allocation_sparsity_cutoff,
            "stability_margin": args.stability_margin,
            "proxy_balance_mode": args.proxy_balance_mode,
            "proxy_dispatch_limit": args.proxy_dispatch_limit,
            "load_shedding_cost": args.load_shedding_cost,
            "line_investment_cost_eur_per_mw_km_year": args.line_investment_cost_eur_per_mw_km_year,
            "investment_cost_reference_hours": EnvironmentConfig().investment_cost_reference_hours,
            "cost_reward_scale": args.cost_reward_scale,
            "overload_reward_scale": args.overload_reward_scale,
            "third_objective_mode": args.third_objective_mode,
            "curtailment_reward_scale": args.curtailment_reward_scale,
            "emissions_reward_scale": args.emissions_reward_scale,
            "solver": args.solver,
        },
        "training": {
            "timesteps": args.timesteps,
            "rollout_steps": args.rollout_steps,
            "minibatch_size": args.minibatch_size,
            "update_epochs": args.update_epochs,
            "learning_rate": args.learning_rate,
            "learning_rate_schedule": args.learning_rate_schedule,
            "final_learning_rate": args.final_learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_epsilon": args.clip_epsilon,
            "entropy_coef": args.entropy_coef,
            "entropy_coef_schedule": args.entropy_coef_schedule,
            "final_entropy_coef": args.final_entropy_coef,
            "target_kl": args.target_kl,
            "normalize_rewards": args.normalize_rewards,
            "moppo_training_mode": args.moppo_training_mode,
            "normalize_objective_advantages": not args.disable_objective_advantage_normalization,
            "moppo_preference_conditioning": conditioned_preferences and not args.disable_moppo_preference_conditioning,
            "moppo_sample_preferences": conditioned_preferences and not args.disable_moppo_preference_sampling,
            "moppo_dirichlet_alpha": args.moppo_dirichlet_alpha,
            "hidden_size": args.hidden_size,
            "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes,
            "early_stopping_patience_evals": args.early_stopping_patience_evals,
            "early_stopping_min_evals": args.early_stopping_min_evals,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "restore_best_model_at_end": not args.disable_restore_best_model,
            "weights": list(weights),
            "weight_grid": [list(item) for item in weight_grid],
            "validation_weight_grid": [list(item) for item in weight_grid] if conditioned_preferences else None,
            "seeds": list(seeds),
            "fullenv_selection_episodes": args.fullenv_selection_episodes,
            "checkpoint_rerank_top_k": args.checkpoint_rerank_top_k,
        },
        "python": sys.version,
        "pypsa": pypsa.__version__,
    }
    write_json(output_root / "pipeline_manifest.json", manifest)

    if should_run_stage(args, 1) and not args.skip_tests:
        print("[1/9] Running unit tests")
        run_tests()
    elif should_run_stage(args, 1):
        print("[1/9] Skipping unit tests")

    if should_run_stage(args, 2):
        print("[2/9] Writing dataset sanity report")
        run_dataset_report(args, preferred_results_dir(output_root, "dataset_sanity"))

    print("[3/9] Loading train/validation/test datasets")
    train_dataset = build_dataset(
        network=ROOT / args.network,
        load=ROOT / args.load,
        wind=ROOT / args.wind,
        solar=ROOT / args.solar,
        start=args.train_start,
        end=args.train_end,
        candidate_lines=args.candidate_lines,
    )
    val_dataset = build_dataset(
        network=ROOT / args.network,
        load=ROOT / args.load,
        wind=ROOT / args.wind,
        solar=ROOT / args.solar,
        start=args.val_start,
        end=args.val_end,
        candidate_lines=args.candidate_lines,
    )
    test_dataset = build_dataset(
        network=ROOT / args.network,
        load=ROOT / args.load,
        wind=ROOT / args.wind,
        solar=ROOT / args.solar,
        start=args.test_start,
        end=args.test_end,
        candidate_lines=args.candidate_lines,
    )
    scenario_calibration = fit_future_scenario(
        train_dataset,
        args.future_scenario,
        catalog_path=ROOT / args.scenario_catalog,
        boundary_path=ROOT / args.nuts2_boundaries,
    )
    scenario_calibration["input_profiles"] = {
        "load": args.load,
        "wind_capacity_factor": args.wind,
        "solar_capacity_factor": args.solar,
    }
    train_dataset = apply_future_scenario(train_dataset, scenario_calibration)
    val_dataset = apply_future_scenario(val_dataset, scenario_calibration)
    test_dataset = apply_future_scenario(test_dataset, scenario_calibration)
    # The exogenous scenario changes the pressure pattern. Candidate screening
    # is therefore refitted on the transformed training split and then frozen.
    refit_candidate_line_screening(train_dataset, args.candidate_lines)
    # Candidate screening is a training-data operation.  Freeze the selected
    # action set before validation/test to avoid temporal leakage and changing
    # action semantics across splits.
    val_dataset.candidate_lines = list(train_dataset.candidate_lines)
    test_dataset.candidate_lines = list(train_dataset.candidate_lines)
    val_dataset.candidate_line_scores = train_dataset.candidate_line_scores
    test_dataset.candidate_line_scores = train_dataset.candidate_line_scores
    apply_observation_scale_reference(val_dataset, train_dataset)
    apply_observation_scale_reference(test_dataset, train_dataset)
    preprocessing_manifest = output_root / "preprocessing_manifest.json"
    scenario_audits = {
        "train": scenario_audit(train_dataset),
        "validation": scenario_audit(val_dataset),
        "test": scenario_audit(test_dataset),
    }
    write_json(
        preprocessing_manifest,
        {
            "fit_split": [args.train_start, args.train_end],
            "formulation": {
                "preset": args.formulation_preset,
                "description": args.formulation_description,
            },
            "environment": {
                "episode_length": args.episode_length,
                "max_upgrade_mw": args.max_upgrade_mw,
                "budget_mw": args.budget_mw,
                "load_shedding_cost": args.load_shedding_cost,
                "line_investment_cost_eur_per_mw_km_year": args.line_investment_cost_eur_per_mw_km_year,
                "investment_cost_reference_hours": EnvironmentConfig().investment_cost_reference_hours,
            },
            "future_scenario": scenario_calibration,
            "future_scenario_audit": scenario_audits,
            "candidate_lines": list(train_dataset.candidate_lines),
            "demand_scale": train_dataset.demand_scale.to_dict(),
            "renewable_scale": train_dataset.renewable_scale.to_dict(),
            "total_demand_scale": float(train_dataset.total_demand_scale),
        },
    )
    manifest["future_scenario"]["calibration"] = scenario_calibration
    manifest["future_scenario"]["audit"] = scenario_audits
    write_json(output_root / "pipeline_manifest.json", manifest)
    write_json(output_root / "future_scenario_audit.json", scenario_audits)
    args.preprocessing_manifest = preprocessing_manifest
    if args.to_stage <= 3:
        print(f"Pipeline finished up to stage {args.to_stage}. Results are in {output_root}")
        return

    ppo_config = PPOConfig(
        hidden_sizes=(args.hidden_size, args.hidden_size),
        learning_rate=args.learning_rate,
        learning_rate_schedule=args.learning_rate_schedule,
        final_learning_rate=args.final_learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        entropy_coef_schedule=args.entropy_coef_schedule,
        final_entropy_coef=args.final_entropy_coef,
        target_kl=args.target_kl,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        normalize_rewards=args.normalize_rewards,
        normalize_objective_advantages=not args.disable_objective_advantage_normalization,
        device=args.device,
        seed=seeds[0],
        scalarization_weights=weights,
        moppo_preference_conditioning=conditioned_preferences and not args.disable_moppo_preference_conditioning,
        moppo_sample_preferences=conditioned_preferences and not args.disable_moppo_preference_sampling,
        moppo_dirichlet_alpha=args.moppo_dirichlet_alpha,
    )
    training_config = TrainingConfig(
        total_timesteps=args.timesteps,
        eval_every_updates=args.eval_every,
        eval_episodes=args.eval_episodes,
        show_progress=True,
        early_stopping_patience_evals=args.early_stopping_patience_evals,
        early_stopping_min_evals=args.early_stopping_min_evals,
        early_stopping_min_delta=args.early_stopping_min_delta,
        restore_best_model_at_end=not args.disable_restore_best_model,
        validation_weight_grid=tuple(weight_grid) if conditioned_preferences else None,
    )

    proxy_env_kwargs = make_env_kwargs(args, seed=seeds[0])

    def train_env_factory():
        return build_env(train_dataset, env_mode="proxy", **proxy_env_kwargs)

    def val_env_factory():
        return build_env(val_dataset, env_mode="proxy", **proxy_env_kwargs)

    ppo_validation_dir = preferred_results_dir(output_root, "ppo_validation")
    if should_run_stage(args, 4):
        print("[4/9] Training PPO multi-seed validation experiment")
        ppo_results = train_multi_seed(
            env_factory=train_env_factory,
            base_ppo_config=ppo_config,
            training_config=training_config,
            seeds=seeds,
            mode="ppo",
            eval_env_factory=val_env_factory,
            output_dir=ppo_validation_dir,
            experiment_name="ppo_proxy_val",
        )
        write_multi_seed_summary(
            ppo_validation_dir,
            "ppo",
            seeds,
            [
                {
                    "seed": entry["seed"],
                    "run_id": entry["run_id"],
                    "evaluation": entry["evaluation"],
                }
                for entry in ppo_results
            ],
        )
        write_validation_report(ppo_validation_dir)
    else:
        print("[4/9] Reusing existing PPO validation experiment")
        if not ppo_validation_dir.exists():
            ppo_validation_dir = locate_results_dir(output_root, "ppo_validation")
        ppo_results = None

    moppo_validation_dir = preferred_results_dir(output_root, "moppo_validation")
    if should_run_stage(args, 5):
        print("[5/9] Training MO-PPO multi-seed validation experiment")
        moppo_results = train_multi_seed(
            env_factory=train_env_factory,
            base_ppo_config=ppo_config,
            training_config=training_config,
            seeds=seeds,
            mode="moppo",
            eval_env_factory=val_env_factory,
            output_dir=moppo_validation_dir,
            experiment_name="moppo_proxy_val",
        )
        write_multi_seed_summary(
            moppo_validation_dir,
            "moppo",
            seeds,
            [
                {
                    "seed": entry["seed"],
                    "run_id": entry["run_id"],
                    "evaluation": entry["evaluation"],
                }
                for entry in moppo_results
            ],
        )
        write_validation_report(moppo_validation_dir)
    else:
        print("[5/9] Reusing existing MO-PPO validation experiment")
        if not moppo_validation_dir.exists():
            moppo_validation_dir = locate_results_dir(output_root, "moppo_validation")
        moppo_results = None
    if args.to_stage <= 5:
        print(f"Pipeline finished up to stage {args.to_stage}. Results are in {output_root}")
        return

    if should_run_stage(args, 6):
        print("[6/9] Running scalarisation sweep")
        scalarization_sweep_dir = preferred_results_dir(output_root, "scalarization_sweep")
        sweep_results = train_weight_sweep(
            env_factory=train_env_factory,
            eval_env_factory=val_env_factory,
            training_config=training_config,
            ppo_config=ppo_config,
            weight_grid=weight_grid,
            mode="ppo",
            output_dir=scalarization_sweep_dir,
        )
        with (scalarization_sweep_dir / "sweep_results.pkl").open("wb") as handle:
            pickle.dump(sweep_results, handle)
        sweep_summary = {
            "/".join(f"{value:.2f}" for value in key): payload["evaluation"]
            for key, payload in sweep_results.items()
        }
        write_json(scalarization_sweep_dir / "sweep_summary.json", sweep_summary)
        plot_pareto_front(sweep_results, scalarization_sweep_dir / "pareto_front.png")

    if should_run_stage(args, 7) and not args.skip_baseline_benchmark:
        print("[7/9] Benchmarking baseline runtime")
        benchmark_dir = preferred_results_dir(output_root, "baseline_benchmark")
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        run_baseline_benchmark(args, benchmark_dir, env_mode="proxy", episodes=args.benchmark_episodes)
        if not args.skip_fullenv:
            run_baseline_benchmark(args, benchmark_dir, env_mode="full", episodes=args.benchmark_episodes)
    elif should_run_stage(args, 7):
        print("[7/9] Skipping baseline runtime benchmark")
    if args.to_stage <= 7:
        print(f"Pipeline finished up to stage {args.to_stage}. Results are in {output_root}")
        return

    selected_ppo = [select_best_checkpoint_from_run_dir(run_dir) for run_dir in collect_run_dirs(ppo_validation_dir)]
    selected_moppo = [
        select_best_checkpoint_from_run_dir(run_dir)
        for run_dir in collect_run_dirs(moppo_validation_dir)
    ]
    if should_run_stage(args, 8):
        print("[8/9] Selecting validation checkpoints and evaluating on the held-out proxy test split")
        proxy_evaluation_dir = preferred_results_dir(output_root, "proxy_evaluation") / "proxy"
        proxy_evaluation_dir.mkdir(parents=True, exist_ok=True)
        write_json(proxy_evaluation_dir / "best_checkpoints_ppo.json", selected_ppo)
        write_json(proxy_evaluation_dir / "best_checkpoints_moppo.json", selected_moppo)

        ppo_test_results, _, _ = evaluate_selected_runs(
            selected_ppo,
            "ppo",
            "proxy",
            test_dataset,
            proxy_evaluation_dir,
            proxy_env_kwargs,
            args.eval_episodes,
            progress_label="Stage 8 PPO proxy test",
        )
        moppo_test_results, _, _ = evaluate_selected_runs(
            selected_moppo,
            "moppo",
            "proxy",
            test_dataset,
            proxy_evaluation_dir,
            proxy_env_kwargs,
            args.eval_episodes,
            progress_label="Stage 8 MO-PPO proxy test",
        )
        if ppo_results is None:
            ppo_results = [
                {
                    "seed": item["seed"],
                    "run_id": item["run_id"],
                    "evaluation": item["evaluation"],
                }
                for item in read_json(ppo_validation_dir / "multi_seed_summary.json")["per_seed"]
            ]
        if moppo_results is None:
            moppo_results = [
                {
                    "seed": item["seed"],
                    "run_id": item["run_id"],
                    "evaluation": item["evaluation"],
                }
                for item in read_json(moppo_validation_dir / "multi_seed_summary.json")["per_seed"]
            ]
        comparison_dir = preferred_results_dir(output_root, "agent_comparison")
        save_comparison(ppo_results, moppo_results, comparison_dir / "validation_ppo_vs_moppo.csv")
        save_comparison(ppo_test_results, moppo_test_results, comparison_dir / "test_proxy_ppo_vs_moppo.csv")
        evaluate_baseline_suite(
            env_mode="proxy",
            dataset=test_dataset,
            output_dir=proxy_evaluation_dir,
            env_kwargs=proxy_env_kwargs,
            episodes=args.eval_episodes,
            reference_runs=selected_ppo,
            myopic_weights=weights,
            myopic_chunk_mw=args.myopic_chunk_mw,
            progress_label="Stage 8 proxy baselines",
        )

    if should_run_stage(args, 9):
        if not args.skip_fullenv:
            print("[9/9] Evaluating selected checkpoints in the full PyPSA environment")
            fullenv_evaluation_dir = preferred_results_dir(output_root, "fullenv_evaluation")
            selected_ppo_for_fullenv = rerank_selected_runs_on_target_env(
                selected_ppo,
                "full",
                val_dataset,
                fullenv_evaluation_dir / "ppo_validation_rerank",
                proxy_env_kwargs,
                args.fullenv_selection_episodes,
                args.checkpoint_rerank_top_k,
                device=args.device,
                progress_label="Stage 9 PPO full-env rerank",
            )
            selected_moppo_for_fullenv = rerank_selected_runs_on_target_env(
                selected_moppo,
                "full",
                val_dataset,
                fullenv_evaluation_dir / "moppo_validation_rerank",
                proxy_env_kwargs,
                args.fullenv_selection_episodes,
                args.checkpoint_rerank_top_k,
                device=args.device,
                progress_label="Stage 9 MO-PPO full-env rerank",
            )
            write_json(
                fullenv_evaluation_dir / "selected_checkpoints_ppo_fullenv_validation.json",
                selected_ppo_for_fullenv,
            )
            write_json(
                fullenv_evaluation_dir / "selected_checkpoints_moppo_fullenv_validation.json",
                selected_moppo_for_fullenv,
            )
            ppo_full_results, _, _ = evaluate_selected_runs(
                selected_ppo_for_fullenv,
                "ppo",
                "full",
                test_dataset,
                fullenv_evaluation_dir,
                proxy_env_kwargs,
                args.fullenv_episodes,
                device=args.device,
                progress_label="Stage 9 PPO full-env test",
            )
            moppo_full_results, _, _ = evaluate_selected_runs(
                selected_moppo_for_fullenv,
                "moppo",
                "full",
                test_dataset,
                fullenv_evaluation_dir,
                proxy_env_kwargs,
                args.fullenv_episodes,
                device=args.device,
                progress_label="Stage 9 MO-PPO full-env test",
            )
            comparison_dir = preferred_results_dir(output_root, "agent_comparison")
            save_comparison(ppo_full_results, moppo_full_results, comparison_dir / "test_fullenv_ppo_vs_moppo.csv")
            evaluate_baseline_suite(
                env_mode="full",
                dataset=test_dataset,
                output_dir=fullenv_evaluation_dir,
                env_kwargs=proxy_env_kwargs,
                episodes=args.fullenv_episodes,
                reference_runs=selected_ppo,
                myopic_weights=weights,
                myopic_chunk_mw=args.myopic_chunk_mw,
                progress_label="Stage 9 full-env baselines",
            )
        else:
            print("[9/9] Skipping full-environment evaluation")

        if not args.skip_fullenv:
            best_moppo = max(selected_moppo_for_fullenv, key=lambda item: float(item["selection_score"]))
        else:
            best_moppo = max(selected_moppo, key=lambda item: float(item["selection_score"]))
        shapley_output_dir = preferred_results_dir(output_root, "explainability")
        shapley_command = [
            sys.executable,
            "main.py",
            "shapley",
            "--checkpoint", best_moppo["checkpoint"],
            "--env", "proxy",
            "--network", args.network,
            "--load", args.load,
            "--wind", args.wind,
            "--solar", args.solar,
            "--start", args.test_start,
            "--end", args.test_end,
            "--candidate-lines", str(args.candidate_lines),
            "--preprocessing-manifest", str(preprocessing_manifest),
            "--episode-length", str(args.episode_length),
            "--max-upgrade-mw", str(args.max_upgrade_mw),
            "--budget-mw", str(args.budget_mw),
            "--decision-interval", str(args.decision_interval),
            "--temporal-mode", args.temporal_mode,
            "--budget-release", args.budget_release,
            "--action-mode", args.action_mode,
            "--allocation-sharpness", str(args.allocation_sharpness),
            "--allocation-sparsity-cutoff", str(args.allocation_sparsity_cutoff),
            "--stability-margin", str(args.stability_margin),
            "--proxy-balance-mode", args.proxy_balance_mode,
            "--line-investment-cost-eur-per-mw-km-year", str(args.line_investment_cost_eur_per_mw_km_year),
            "--cost-reward-scale", str(args.cost_reward_scale),
            "--overload-reward-scale", str(args.overload_reward_scale),
            "--third-objective-mode", args.third_objective_mode,
            "--curtailment-reward-scale", str(args.curtailment_reward_scale),
            "--emissions-reward-scale", str(args.emissions_reward_scale),
            "--solver", args.solver,
            "--eval-episodes", str(args.shapley_episodes),
            "--shapley-samples", str(args.shapley_samples),
            "--top-k", str(args.top_k),
            "--output-dir", str(shapley_output_dir),
        ]
        if args.proxy_dispatch_limit is not None:
            shapley_command.extend(["--proxy-dispatch-limit", str(args.proxy_dispatch_limit)])
        if best_moppo.get("eval_preference_weights") is not None:
            shapley_command.extend(
                [
                    "--weights",
                    ",".join(f"{float(value):.6f}" for value in best_moppo["eval_preference_weights"]),
                ]
            )
        subprocess.run(shapley_command, cwd=ROOT, check=True)

        run_final_analysis(output_root)
    print(f"Pipeline finished. Results are in {output_root}")


if __name__ == "__main__":
    main()
