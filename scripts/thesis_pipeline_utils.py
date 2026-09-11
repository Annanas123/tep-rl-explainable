from __future__ import annotations

"""Shared helpers for the thesis pipeline and post-processing scripts.

This module keeps the high-level experiment scripts thin by centralising
dataset loading, environment reconstruction, checkpoint discovery, and JSON
serialisation. The helpers deliberately operate on finished run directories so
analysis can be reproduced without retraining.
"""

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tep_rl.config import EnvironmentConfig, NetworkConfig
from tep_rl.data import TEPDataset, load_austria_case
from tep_rl.envs import ProxyTEPEnv, PyPSATEPEnv
from tep_rl.statistics import summarise_experiment


def json_default(value: Any):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_history(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_dataset(
    network: Path,
    load: Path,
    wind: Path,
    solar: Path,
    start: str | None,
    end: str | None,
    candidate_lines: int | None,
    year: int = 2020,
) -> TEPDataset:
    """Rebuild one chronological slice of the Austrian case-study dataset."""
    return load_austria_case(
        NetworkConfig(
            network_path=network,
            load_path=load,
            wind_path=wind,
            solar_path=solar,
            year=year,
            start=start,
            end=end,
            candidate_line_limit=candidate_lines,
        )
    )


def build_env(
    dataset: TEPDataset,
    env_mode: str,
    episode_length: int,
    max_upgrade_mw: float,
    budget_mw: float,
    decision_interval: int,
    temporal_mode: str,
    budget_release: str,
    action_mode: str,
    allocation_sharpness: float,
    allocation_sparsity_cutoff: float,
    stability_margin: float,
    proxy_balance_mode: str,
    proxy_dispatch_limit: float | None,
    load_shedding_cost: float,
    cost_reward_scale: float,
    overload_reward_scale: float,
    third_objective_mode: str,
    curtailment_reward_scale: float,
    emissions_reward_scale: float,
    solver: str,
    full_env_fallback_to_proxy: bool,
    seed: int,
    line_investment_cost_eur_per_mw_km_year: float | None = None,
):
    """Reconstruct a proxy or full environment from a saved thesis config."""
    config = EnvironmentConfig(
        episode_length=episode_length,
        max_line_upgrade_mw=max_upgrade_mw,
        total_upgrade_budget_mw=budget_mw,
        decision_interval=decision_interval,
        temporal_mode=temporal_mode,
        budget_release=budget_release,
        action_mode=action_mode,
        allocation_sharpness=allocation_sharpness,
        allocation_sparsity_cutoff=allocation_sparsity_cutoff,
        stability_margin=stability_margin,
        proxy_balance_mode=proxy_balance_mode,
        proxy_dispatch_limit=proxy_dispatch_limit,
        load_shedding_cost=load_shedding_cost,
        line_investment_cost_eur_per_mw_km_year=(
            EnvironmentConfig().line_investment_cost_eur_per_mw_km_year
            if line_investment_cost_eur_per_mw_km_year is None
            else line_investment_cost_eur_per_mw_km_year
        ),
        cost_reward_scale=cost_reward_scale,
        overload_reward_scale=overload_reward_scale,
        third_objective_mode=third_objective_mode,
        curtailment_reward_scale=curtailment_reward_scale,
        emissions_reward_scale=emissions_reward_scale,
        seed=seed,
        solver_name=solver,
        full_env_fallback_to_proxy=full_env_fallback_to_proxy,
    )
    if env_mode == "proxy":
        return ProxyTEPEnv(dataset, config)
    return PyPSATEPEnv(dataset, config)


def scalarized_mean_reward(evaluation: dict[str, Any], weights: Sequence[float]) -> float:
    reward_vector = evaluation.get("reward_stats", {}).get("mean_vector_reward")
    if reward_vector is None:
        raise ValueError("Evaluation is missing reward_stats.mean_vector_reward.")
    vector = np.asarray(reward_vector, dtype=float)
    weights_arr = np.asarray(list(weights)[: len(vector)], dtype=float)
    total = float(weights_arr.sum())
    if total <= 0.0:
        weights_arr = np.full(len(vector), 1.0 / len(vector))
    else:
        weights_arr = weights_arr / total
    return float(np.dot(weights_arr, vector))


def _selection_score_from_details(
    selection_details: dict[str, Any] | None,
    evaluation: dict[str, Any],
    weights: Sequence[float],
) -> float:
    if isinstance(selection_details, dict):
        if "best_selection_score" in selection_details:
            return float(selection_details["best_selection_score"])
        if "selection_score" in selection_details:
            return float(selection_details["selection_score"])
        if "mean_selection_score" in selection_details:
            return float(selection_details["mean_selection_score"])
    return scalarized_mean_reward(evaluation, weights)


def _checkpoint_candidate_sort_key(item: dict[str, Any]) -> tuple[float, ...]:
    evaluation = item["evaluation"]
    total_investment = float(evaluation.get("action_stats", {}).get("total_investment_mean", float("inf")))
    # Ranking first follows the explicit validation selection score, then
    # resolves near-ties with the thesis metrics used in the final analysis.
    return (
        float(item["selection_score"]),
        float(evaluation.get("renewable_share_mean", float("-inf"))),
        -float(evaluation.get("renewable_curtailment_mean", float("inf"))),
        -float(evaluation.get("load_shedding_mean", float("inf"))),
        -float(evaluation.get("total_cost_mean", float("inf"))),
        -float(evaluation.get("grid_stress_mean", float("inf"))),
        -total_investment,
    )


def _selected_eval_preference_weights(selection_details: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(selection_details, dict):
        return None
    selected = selection_details.get("selected_eval_preference_weights")
    if selected is None:
        return None
    return [float(value) for value in selected]


def list_checkpoint_candidates_from_run_dir(run_dir: Path) -> list[dict[str, Any]]:
    """Collect ranked checkpoint candidates from one finished training run.

    The function merges three possible sources:
    - explicit checkpoint evaluations produced during training
    - the final restored model
    - the persisted best-validation record

    Returning one normalised candidate list keeps reranking and supplementary
    experiments agnostic to the exact training-time output layout.
    """
    config_snapshot = read_json(run_dir / "config_snapshot.json")
    weights = config_snapshot["ppo_config"]["scalarization_weights"]
    checkpoint_dir = run_dir / "checkpoints"
    candidates: list[dict[str, Any]] = []

    if checkpoint_dir.exists():
        for eval_path in sorted(checkpoint_dir.glob("update_*.evaluation.json")):
            checkpoint_path = eval_path.with_suffix("").with_suffix(".pt")
            evaluation = read_json(eval_path)
            selection_path = eval_path.with_suffix("").with_suffix(".selection.json")
            selection_details = read_json(selection_path) if selection_path.exists() else None
            score = _selection_score_from_details(selection_details, evaluation, weights)
            candidates.append(
                {
                    "run_dir": str(run_dir),
                    "weights": list(weights),
                    "checkpoint": str(checkpoint_path),
                    "evaluation_path": str(eval_path),
                    "evaluation": evaluation,
                    "selection_score": score,
                    "update": int(eval_path.stem.split("_")[1].split(".")[0]),
                    "selection_details": selection_details,
                    "eval_preference_weights": _selected_eval_preference_weights(selection_details),
                    "selection_origin": "checkpoint",
                }
            )

    final_eval_path = run_dir / "evaluation.json"
    final_agent_path = run_dir / "agent.pt"
    if final_eval_path.exists():
        final_eval = read_json(final_eval_path)
        final_eval = read_json(run_dir / "evaluation.json")
        candidates.append(
            {
                "run_dir": str(run_dir),
                "weights": list(weights),
                "checkpoint": str(run_dir / "agent.pt"),
                "evaluation_path": str(run_dir / "evaluation.json"),
                "evaluation": final_eval,
                "selection_score": scalarized_mean_reward(final_eval, weights),
                "update": -1,
                "selection_details": None,
                "selection_origin": "final",
            }
        )

    best_validation_path = run_dir / "best_validation.json"
    if best_validation_path.exists():
        best_validation = read_json(best_validation_path)
        checkpoint_info = best_validation.get("checkpoint") or {}
        evaluation = best_validation.get("evaluation")
        checkpoint_path = Path(checkpoint_info.get("checkpoint", final_agent_path))
        evaluation_path = Path(checkpoint_info.get("evaluation_path", final_eval_path))
        if evaluation is None and evaluation_path.exists():
            evaluation = read_json(evaluation_path)
        if evaluation is None and final_eval_path.exists():
            evaluation = read_json(final_eval_path)
        if evaluation is not None:
            best_candidate = {
                "run_dir": str(run_dir),
                "weights": list(weights),
                "checkpoint": str(checkpoint_path),
                "evaluation_path": str(evaluation_path),
                "evaluation": evaluation,
                "selection_score": float(
                    best_validation.get(
                        "selection_score",
                        _selection_score_from_details(best_validation.get("selection_details"), evaluation, weights),
                    )
                ),
                "update": int(best_validation.get("update", -1)),
                "selection_details": best_validation.get("selection_details"),
                "eval_preference_weights": _selected_eval_preference_weights(best_validation.get("selection_details")),
                "selection_origin": "best_validation",
            }
            existing_index = next(
                (
                    index
                    for index, candidate in enumerate(candidates)
                    if Path(str(candidate["checkpoint"])) == checkpoint_path
                ),
                None,
            )
            if existing_index is None:
                candidates.append(best_candidate)
            else:
                candidates[existing_index] = best_candidate

    if not candidates:
        raise FileNotFoundError(f"No checkpoint evaluations found in {run_dir}.")

    candidates.sort(key=_checkpoint_candidate_sort_key, reverse=True)
    return candidates


def select_best_checkpoint_from_run_dir(run_dir: Path) -> dict[str, Any]:
    return list_checkpoint_candidates_from_run_dir(run_dir)[0]


def collect_run_dirs(experiment_dir: Path) -> list[Path]:
    """Return finished run directories inside an experiment-level folder."""
    return sorted(path for path in experiment_dir.iterdir() if path.is_dir() and (path / "config_snapshot.json").exists())


def load_seed_histories(experiment_dir: Path, deduplicate_by_seed: bool = True) -> list[dict[str, Any]]:
    histories = []
    run_dirs = collect_run_dirs(experiment_dir)
    if deduplicate_by_seed:
        seed_pattern = re.compile(r"seed(\d+)")
        grouped: dict[str, Path] = {}
        for run_dir in run_dirs:
            match = seed_pattern.search(run_dir.name)
            key = match.group(1) if match else run_dir.name
            current = grouped.get(key)
            # The final pipeline can contain restarted runs for the same seed.
            # For thesis figures we keep the most recent run per seed.
            if current is None or (run_dir.name, run_dir.stat().st_mtime) > (current.name, current.stat().st_mtime):
                grouped[key] = run_dir
        run_dirs = [grouped[key] for key in sorted(grouped)]
    for run_dir in run_dirs:
        history_path = run_dir / "history.pkl"
        if history_path.exists():
            histories.append(load_history(history_path))
    return histories


def load_seed_evaluations(experiment_dir: Path) -> list[dict[str, Any]]:
    evaluations = []
    for run_dir in collect_run_dirs(experiment_dir):
        evaluation_path = run_dir / "evaluation.json"
        if evaluation_path.exists():
            evaluations.append(read_json(evaluation_path))
    return evaluations


def build_multi_seed_summary_payload(
    agent: str,
    seeds: Sequence[int],
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluations = [entry["evaluation"] for entry in per_seed]
    summary_df = summarise_experiment(evaluations)
    return {
        "seeds": list(seeds),
        "agent": agent,
        "summary": summary_df.to_dict(orient="records"),
        "per_seed": per_seed,
    }


def write_multi_seed_summary(
    output_dir: Path,
    agent: str,
    seeds: Sequence[int],
    per_seed: list[dict[str, Any]],
    filename_prefix: str = "multi_seed_summary",
) -> tuple[Path, Path]:
    payload = build_multi_seed_summary_payload(agent, seeds, per_seed)
    summary_df = pd.DataFrame(payload["summary"])
    json_path = output_dir / f"{filename_prefix}.json"
    csv_path = output_dir / f"{filename_prefix}.csv"
    write_json(json_path, payload)
    summary_df.to_csv(csv_path, index=False)
    return json_path, csv_path


def combine_episode_records(evaluations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    combined_episodes: list[dict[str, Any]] = []
    line_means: dict[str, list[float]] = {}
    for evaluation in evaluations:
        combined_episodes.extend(evaluation.get("episodes", []))
        for line, value in evaluation.get("line_upgrades", {}).items():
            line_means.setdefault(line, []).append(float(value))

    mean_mw_per_line = {
        line: float(np.mean(values))
        for line, values in line_means.items()
    }
    return {
        "episodes": combined_episodes,
        "action_stats": {
            "mean_mw_per_line": mean_mw_per_line,
        },
    }


def network_summary_frame(network: pypsa.Network) -> pd.DataFrame:
    rows = [
        ("buses", len(network.buses)),
        ("lines", len(network.lines)),
        ("transformers", len(network.transformers)),
        ("generators", len(network.generators)),
        ("loads", len(network.loads)),
        ("storage_units", len(network.storage_units)),
    ]
    return pd.DataFrame(rows, columns=["component", "count"])
