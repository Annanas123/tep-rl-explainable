from __future__ import annotations

"""Measure how strongly conditioned MO-PPO responds to deployment preferences."""

import argparse
import gc
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pypsa
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_layout import preferred_results_dir
from run_pareto_archive_sweep import (
    DEFAULT_WEIGHT_GRID,
    _plot_front,
    collapse_outcome_points,
    make_env_kwargs,
    metric_row,
    nondominated_minimize,
    normalize_weights,
    parse_csv_tuple,
    resolve_device,
    weight_label,
)
from thesis_pipeline_utils import (
    build_dataset,
    build_env,
    collect_run_dirs,
    list_checkpoint_candidates_from_run_dir,
    read_json,
    write_json,
)
from tep_rl.config import EnvironmentConfig, PPOConfig, TrainingConfig
from tep_rl.evaluation import evaluate_agent, evaluate_moppo_preference_grid
from tep_rl.line_metadata import endpoint_label, osm_way_url
from tep_rl.ppo import load_agent
from tep_rl.reproducibility import seed_everything
from tep_rl.thesis_formulations import (
    DEFAULT_THESIS_FORMULATION_PRESET,
    THESIS_FORMULATION_PRESETS,
    apply_thesis_formulation_preset,
)
from tep_rl.training import train_multi_seed


PROXY_SELECTION_OBJECTIVES: tuple[tuple[str, str], ...] = (
    ("total_cost_mean", "min"),
    ("grid_stress_mean", "min"),
    ("renewable_share_mean", "max"),
)


def generate_simplex_weight_grid(step: float, dimension: int = 3) -> list[tuple[float, ...]]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    units = int(round(1.0 / float(step)))
    if units <= 0 or not np.isclose(units * float(step), 1.0, atol=1e-8):
        raise ValueError("step must evenly divide 1.0, for example 0.50, 0.25, 0.20, 0.10")
    if dimension != 3:
        raise ValueError("This helper currently supports the 3-objective thesis setup only.")
    grid: list[tuple[float, ...]] = []
    for first in range(units, -1, -1):
        for second in range(units - first, -1, -1):
            third = units - first - second
            grid.append(
                (
                    float(first / units),
                    float(second / units),
                    float(third / units),
                )
            )
    return [normalize_weights(weights) for weights in grid]


def _action_signature(evaluation: dict[str, Any], threshold_mw: float = 0.1) -> str:
    mean_per_line = evaluation.get("action_stats", {}).get("mean_mw_per_line", {})
    active = []
    for line, value in sorted(mean_per_line.items()):
        mw = float(value)
        if abs(mw) <= float(threshold_mw):
            continue
        active.append(f"{line}:{mw:.3f}")
    return " | ".join(active) if active else "no_upgrade"


def _outcome_signature(row: dict[str, Any] | pd.Series) -> str:
    return "|".join(
        [
            f"{float(row['total_cost_mean']):.6f}",
            f"{float(row.get('grid_stress_mean', np.nan)):.6f}",
            f"{float(row.get('load_shedding_mean', np.nan)):.6f}",
            f"{float(row['renewable_share_mean']):.6f}",
            f"{float(row['renewable_curtailment_mean']):.6f}",
            f"{float(row['total_investment_mean']):.6f}",
            f"{float(row['active_lines_mean']):.6f}",
        ]
    )


def _nondominated_mask(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(values), dtype=bool)
    for idx, candidate in enumerate(values):
        for challenger_idx, challenger in enumerate(values):
            if idx == challenger_idx:
                continue
            no_worse = np.all(challenger <= candidate + 1e-12)
            strictly_better = np.any(challenger < candidate - 1e-12)
            if no_worse and strictly_better:
                keep[idx] = False
                break
    return keep


def approximate_hypervolume(
    points: np.ndarray,
    ref_point: np.ndarray | None = None,
    samples: int = 20_000,
    seed: int = 7,
) -> float:
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return 0.0
    if array.ndim != 2:
        raise ValueError("points must be a 2D array")
    reference = np.asarray(ref_point if ref_point is not None else np.full(array.shape[1], 1.02), dtype=float)
    if reference.shape != (array.shape[1],):
        raise ValueError("ref_point must match the point dimension")
    filtered = array[_nondominated_mask(array)]
    filtered = np.clip(filtered, 0.0, reference)
    rng = np.random.default_rng(seed)
    draws = rng.random((int(samples), filtered.shape[1])) * reference
    dominated = np.any(np.all(filtered[:, None, :] <= draws[None, :, :], axis=2), axis=0)
    return float(dominated.mean() * np.prod(reference))


def _objective_bounds(frame: pd.DataFrame, specs: Sequence[tuple[str, str]]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for column, sense in specs:
        values = frame[column].to_numpy(dtype=float)
        if sense == "max":
            values = -values
        bounds[column] = (float(np.nanmin(values)), float(np.nanmax(values)))
    return bounds


def _normalize_objectives(
    frame: pd.DataFrame,
    specs: Sequence[tuple[str, str]],
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for column, sense in specs:
        values = frame[column].to_numpy(dtype=float)
        if sense == "max":
            values = -values
        lower, upper = bounds[column]
        scale = max(float(upper - lower), 1e-9)
        columns.append((values - lower) / scale)
    return np.column_stack(columns) if columns else np.zeros((len(frame), 0), dtype=float)


def _candidate_response_sort_key(row: pd.Series | dict[str, Any]) -> tuple[float, int, int, int, float, float]:
    return (
        float(row["validation_hypervolume_estimate"]),
        int(row["validation_unique_outcomes"]),
        int(row["validation_unique_action_signatures"]),
        int(row["validation_unique_eval_preferences"]),
        float(row["validation_mean_selection_score"]),
        float(row["validation_min_selection_score"]),
    )


def _weight_grid_from_args(raw_grid: Sequence[str] | None, simplex_step: float | None) -> list[tuple[float, ...]]:
    if raw_grid:
        return [normalize_weights(parse_csv_tuple(text, float)) for text in raw_grid]
    if simplex_step is None:
        raise ValueError("Either an explicit weight grid or a simplex step must be provided.")
    return generate_simplex_weight_grid(simplex_step, dimension=3)


def _evaluation_row(
    evaluation: dict[str, Any],
    utility_weights: Sequence[float],
    selected_eval_preference_weights: Sequence[float],
    selection_score: float,
) -> dict[str, Any]:
    weights = normalize_weights(utility_weights)
    selected_weights = normalize_weights(selected_eval_preference_weights)
    row = {
        "weights": weight_label(weights),
        "weight_cost": weights[0],
        "weight_stress": weights[1],
        "weight_sustainability": weights[2],
        "selected_eval_preference_weights": ",".join(f"{value:.6f}" for value in selected_weights),
        "selected_eval_preference_label": weight_label(selected_weights),
        "selection_score": float(selection_score),
    }
    row.update(metric_row(evaluation, prefix="", weights=weights))
    row["action_signature"] = _action_signature(evaluation)
    row["outcome_signature"] = _outcome_signature(row)
    return row


def _cache_signature(
    args: argparse.Namespace,
    utility_grid: Sequence[Sequence[float]],
    conditioning_grid: Sequence[Sequence[float]],
    checkpoint: str,
) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "selection_episodes": int(args.selection_episodes),
        "conditioning_grid": [list(weights) for weights in conditioning_grid],
        "utility_grid": [list(weights) for weights in utility_grid],
        "episode_length": int(args.episode_length),
        "decision_interval": int(args.decision_interval),
        "third_objective_mode": str(args.third_objective_mode),
        "allocation_sparsity_cutoff": float(args.allocation_sparsity_cutoff),
        "budget_mw": float(args.budget_mw),
        "max_upgrade_mw": float(args.max_upgrade_mw),
    }


def evaluate_checkpoint_response_on_validation(
    run_dir: Path,
    candidate: dict[str, Any],
    dataset,
    args: argparse.Namespace,
    utility_grid: Sequence[Sequence[float]],
    conditioning_grid: Sequence[Sequence[float]],
    cache_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    seed = int(read_json(run_dir / "config_snapshot.json")["seed"])
    checkpoint_path = str(candidate["checkpoint"])
    cache_path = cache_dir / run_dir.name / f"{Path(checkpoint_path).stem}.json"
    signature = _cache_signature(args, utility_grid, conditioning_grid, checkpoint_path)

    if cache_path.exists():
        cached = read_json(cache_path)
        if cached.get("cache_signature") == signature:
            return pd.DataFrame(cached.get("rows", [])), cache_path

    env = build_env(dataset, env_mode="proxy", **make_env_kwargs(args, seed=seed))
    agent = load_agent(checkpoint_path, device=args.device)
    rows: list[dict[str, Any]] = []
    iterator = tqdm(utility_grid, desc=f"{run_dir.name}:{Path(checkpoint_path).stem}", unit="pref", leave=False, dynamic_ncols=True)
    for utility_weights in iterator:
        response = evaluate_moppo_preference_grid(
            agent,
            env,
            preference_grid=conditioning_grid,
            utility_weights=utility_weights,
            episodes=args.selection_episodes,
            deterministic=True,
        )
        row = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "checkpoint": checkpoint_path,
            "update": int(candidate.get("update", -1)),
            "selection_origin": str(candidate.get("selection_origin", "")),
            "seed": seed,
        }
        row.update(
            _evaluation_row(
                response["evaluation"],
                utility_weights=utility_weights,
                selected_eval_preference_weights=response["selected_eval_preference_weights"],
                selection_score=float(response["selection_score"]),
            )
        )
        rows.append(row)

    payload = {
        "cache_signature": signature,
        "rows": rows,
    }
    write_json(cache_path, payload)
    del env
    del agent
    gc.collect()
    return pd.DataFrame(rows), cache_path


def summarize_validation_candidate(
    frame: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    cache_path: Path,
) -> dict[str, Any]:
    normalized = _normalize_objectives(frame, PROXY_SELECTION_OBJECTIVES, bounds)
    nondominated_mask = _nondominated_mask(normalized)
    hypervolume = approximate_hypervolume(normalized[nondominated_mask], ref_point=np.full(normalized.shape[1], 1.02), samples=12_000)
    first = frame.iloc[0]
    return {
        "run_id": first["run_id"],
        "run_dir": first["run_dir"],
        "checkpoint": first["checkpoint"],
        "update": int(first["update"]),
        "selection_origin": first["selection_origin"],
        "seed": int(first["seed"]),
        "validation_response_path": str(cache_path),
        "validation_hypervolume_estimate": float(hypervolume),
        "validation_unique_outcomes": int(frame["outcome_signature"].nunique()),
        "validation_unique_action_signatures": int(frame["action_signature"].nunique()),
        "validation_unique_eval_preferences": int(frame["selected_eval_preference_label"].nunique()),
        "validation_nondominated_outcomes": int(pd.Series(nondominated_mask).sum()),
        "validation_mean_selection_score": float(frame["selection_score"].mean()),
        "validation_min_selection_score": float(frame["selection_score"].min()),
        "validation_max_selection_score": float(frame["selection_score"].max()),
        "validation_score_spread": float(frame["selection_score"].max() - frame["selection_score"].min()),
        "validation_mean_cost": float(frame["total_cost_mean"].mean()),
        "validation_mean_grid_stress": float(frame["grid_stress_mean"].mean()),
        "validation_mean_renewable_share": float(frame["renewable_share_mean"].mean()),
    }


def build_validation_response_archive(
    response_dir: Path,
    dataset,
    args: argparse.Namespace,
    utility_grid: Sequence[Sequence[float]],
    conditioning_grid: Sequence[Sequence[float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = response_dir / "validation_response_cache"
    all_candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for run_dir in tqdm(collect_run_dirs(response_dir), desc="Validation response selection", unit="run"):
        candidate_pool = list_checkpoint_candidates_from_run_dir(run_dir)[: max(int(args.selection_top_k_per_seed), 1)]
        evaluated: list[tuple[pd.DataFrame, Path]] = []
        for candidate in candidate_pool:
            frame, cache_path = evaluate_checkpoint_response_on_validation(
                run_dir=run_dir,
                candidate=candidate,
                dataset=dataset,
                args=args,
                utility_grid=utility_grid,
                conditioning_grid=conditioning_grid,
                cache_dir=cache_dir,
            )
            if not frame.empty:
                evaluated.append((frame, cache_path))

        if not evaluated:
            continue

        pooled = pd.concat([frame for frame, _ in evaluated], ignore_index=True)
        bounds = _objective_bounds(pooled, PROXY_SELECTION_OBJECTIVES)
        per_run_summaries = [
            summarize_validation_candidate(frame, bounds=bounds, cache_path=cache_path)
            for frame, cache_path in evaluated
        ]
        all_candidate_rows.extend(per_run_summaries)
        best = max(per_run_summaries, key=_candidate_response_sort_key)
        selected_rows.append(best)

    candidates_frame = pd.DataFrame(all_candidate_rows)
    selected_frame = pd.DataFrame(selected_rows)
    if not candidates_frame.empty:
        candidates_frame = candidates_frame.sort_values(
            [
                "validation_hypervolume_estimate",
                "validation_unique_outcomes",
                "validation_unique_action_signatures",
                "validation_mean_selection_score",
            ],
            ascending=[False, False, False, False],
        )
    if not selected_frame.empty:
        selected_frame = selected_frame.sort_values(["seed", "validation_hypervolume_estimate"], ascending=[True, False])
    return candidates_frame.reset_index(drop=True), selected_frame.reset_index(drop=True)


def _load_validation_response_rows(cache_path: Path) -> pd.DataFrame:
    payload = read_json(cache_path)
    return pd.DataFrame(payload.get("rows", []))


def evaluate_selected_runs_on_test(
    selected_runs: pd.DataFrame,
    dataset,
    args: argparse.Namespace,
    utility_grid: Sequence[Sequence[float]],
    env_mode: str,
    episodes: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    network = pypsa.Network(str(ROOT / args.network))

    for _, selected in tqdm(selected_runs.iterrows(), total=len(selected_runs), desc=f"Test {env_mode}", unit="seed"):
        cache_rows = _load_validation_response_rows(Path(selected["validation_response_path"]))
        if cache_rows.empty:
            continue
        preference_map = {
            str(row["weights"]): parse_csv_tuple(str(row["selected_eval_preference_weights"]), float)
            for _, row in cache_rows.iterrows()
        }
        score_map = {str(row["weights"]): float(row["selection_score"]) for _, row in cache_rows.iterrows()}
        agent = load_agent(str(selected["checkpoint"]), device=args.device)
        env = build_env(dataset, env_mode=env_mode, **make_env_kwargs(args, seed=int(selected["seed"])))

        for utility_weights in utility_grid:
            utility_weights = normalize_weights(utility_weights)
            utility_label = weight_label(utility_weights)
            eval_weights = preference_map.get(utility_label, utility_weights)
            agent.set_eval_preferences(eval_weights)
            evaluation = evaluate_agent(agent, env, episodes=episodes, deterministic=True)

            row = {
                "candidate_id": f"{selected['run_id']}__{env_mode}__{utility_label}",
                "run_id": selected["run_id"],
                "run_dir": selected["run_dir"],
                "checkpoint": selected["checkpoint"],
                "update": int(selected["update"]),
                "selection_origin": selected["selection_origin"],
                "seed": int(selected["seed"]),
                "weights": utility_label,
                "weight_cost": utility_weights[0],
                "weight_stress": utility_weights[1],
                "weight_sustainability": utility_weights[2],
                "selected_eval_preference_weights": ",".join(f"{value:.6f}" for value in normalize_weights(eval_weights)),
                "selected_eval_preference_label": weight_label(eval_weights),
                "validation_selection_score": float(score_map.get(utility_label, np.nan)),
            }
            row.update(metric_row(evaluation, prefix="", weights=utility_weights))
            rows.append(row)

            for line, value in evaluation.get("action_stats", {}).get("mean_mw_per_line", {}).items():
                mw = float(value)
                if abs(mw) <= 1e-9:
                    continue
                line_rows.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "env": env_mode,
                        "seed": int(selected["seed"]),
                        "weights": utility_label,
                        "selected_eval_preference_label": row["selected_eval_preference_label"],
                        "line": line,
                        "line_label": endpoint_label(network, line),
                        "osm_way_url": osm_way_url(line),
                        "mean_upgrade_mw": mw,
                    }
                )

        del env
        del agent
        gc.collect()

    result_frame = pd.DataFrame(rows)
    line_frame = pd.DataFrame(line_rows)
    if not result_frame.empty:
        objective_cols = ["total_cost_mean", "load_shedding_mean", "renewable_curtailment_mean", "total_investment_mean"]
        if env_mode == "proxy":
            objective_cols = ["total_cost_mean", "grid_stress_mean", "renewable_curtailment_mean", "total_investment_mean"]
        result_frame["test_nondominated"] = nondominated_minimize(result_frame, objective_cols)
        unique_frame = collapse_outcome_points(result_frame, env_mode=env_mode)
        result_frame.to_csv(output_dir / f"preference_response_test_results_{env_mode}.csv", index=False)
        unique_frame.to_csv(output_dir / f"preference_response_test_results_{env_mode}_unique.csv", index=False)
    if not line_frame.empty:
        line_frame.to_csv(output_dir / f"preference_response_test_line_upgrades_{env_mode}.csv", index=False)
    return result_frame, line_frame


def write_story(output_dir: Path, proxy_frame: pd.DataFrame, full_frame: pd.DataFrame) -> None:
    proxy_unique = collapse_outcome_points(proxy_frame, env_mode="proxy") if not proxy_frame.empty else proxy_frame.copy()
    full_unique = collapse_outcome_points(full_frame, env_mode="full") if not full_frame.empty else full_frame.copy()
    lines = [
        "# Preference-Responsive MO-PPO",
        "",
        "This experiment trains a preference-conditioned MO-PPO agent over a fixed preference grid and selects checkpoints by validation preference responsiveness.",
        "Responsiveness is summarised by validation hypervolume across deployment preferences, the number of unique outcomes, and the number of distinct action signatures.",
        "",
    ]
    if not proxy_frame.empty:
        lines.extend(
            [
                f"- Proxy deployment preference points evaluated: {len(proxy_frame)}.",
                f"- Proxy unique evaluated outcomes: {len(proxy_unique)}.",
                f"- Proxy non-dominated unique outcomes: {int(proxy_unique['test_nondominated'].sum())}.",
            ]
        )
    if not full_frame.empty:
        lines.extend(
            [
                f"- Full-environment deployment preference points evaluated: {len(full_frame)}.",
                f"- Full-environment unique evaluated outcomes: {len(full_unique)}.",
                f"- Full-environment non-dominated unique outcomes: {int(full_unique['test_nondominated'].sum())}.",
            ]
        )
        if len(full_unique) > 1:
            lines.append("- The conditioned MO-PPO generated multiple distinct evaluated outcomes across deployment preferences, indicating a meaningful preference response.")
        elif len(full_unique) == 1:
            lines.append("- The conditioned MO-PPO still collapsed to a single full-environment outcome across deployment preferences.")
    (output_dir / "preference_response_story.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    default_env = EnvironmentConfig()
    default_ppo = PPOConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Train preference-conditioned MO-PPO on a fixed conditioning grid, "
            "select checkpoints by validation preference responsiveness, and evaluate the resulting policy front."
        )
    )
    parser.add_argument(
        "--formulation-preset",
        choices=sorted(THESIS_FORMULATION_PRESETS),
        default=DEFAULT_THESIS_FORMULATION_PRESET,
        help="Apply a shared thesis formulation preset unless the specific budget/candidate flags are overridden.",
    )
    parser.add_argument("--results-root", default="results/thesis_final")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--network", default="derived/austria_net_physical_ratings.nc")
    parser.add_argument("--load", default="data/entsoe_at_load_2015_2024_opsd.csv")
    parser.add_argument("--wind", default="data/wind_at_2015_2024.csv")
    parser.add_argument("--solar", default="data/solar_at_2015_2024.csv")
    parser.add_argument("--train-start", default="2015-01-01")
    parser.add_argument("--train-end", default="2020-12-31")
    parser.add_argument("--val-start", default="2021-01-01")
    parser.add_argument("--val-end", default="2022-12-31")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
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
    parser.add_argument("--proxy-balance-mode", choices=["demand_proportional", "single_slack"], default=default_env.proxy_balance_mode)
    parser.add_argument("--proxy-dispatch-limit", type=float, default=default_env.proxy_dispatch_limit)
    parser.add_argument("--load-shedding-cost", type=float, default=default_env.load_shedding_cost)
    parser.add_argument("--cost-reward-scale", type=float, default=default_env.cost_reward_scale)
    parser.add_argument("--overload-reward-scale", type=float, default=default_env.overload_reward_scale)
    parser.add_argument("--third-objective-mode", choices=["renewable_share", "curtailment", "emissions"], default=default_env.third_objective_mode)
    parser.add_argument("--curtailment-reward-scale", type=float, default=default_env.curtailment_reward_scale)
    parser.add_argument("--emissions-reward-scale", type=float, default=default_env.emissions_reward_scale)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--disable-full-env-fallback-to-proxy", action="store_true")
    parser.add_argument("--conditioning-grid", nargs="+", default=list(DEFAULT_WEIGHT_GRID))
    parser.add_argument("--utility-grid", nargs="*", default=None)
    parser.add_argument("--utility-grid-step", type=float, default=0.20)
    parser.add_argument("--seeds", default="7,11,19")
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=default_ppo.learning_rate)
    parser.add_argument("--learning-rate-schedule", choices=["constant", "linear"], default=default_ppo.learning_rate_schedule)
    parser.add_argument("--final-learning-rate", type=float, default=default_ppo.final_learning_rate)
    parser.add_argument("--gamma", type=float, default=default_ppo.gamma)
    parser.add_argument("--gae-lambda", type=float, default=default_ppo.gae_lambda)
    parser.add_argument("--clip-epsilon", type=float, default=default_ppo.clip_epsilon)
    parser.add_argument("--entropy-coef", type=float, default=default_ppo.entropy_coef)
    parser.add_argument("--entropy-coef-schedule", choices=["constant", "linear"], default=default_ppo.entropy_coef_schedule)
    parser.add_argument("--final-entropy-coef", type=float, default=default_ppo.final_entropy_coef)
    parser.add_argument("--target-kl", type=float, default=default_ppo.target_kl)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--selection-episodes", type=int, default=3)
    parser.add_argument("--proxy-test-episodes", type=int, default=10)
    parser.add_argument("--fullenv-episodes", type=int, default=5)
    parser.add_argument("--selection-top-k-per-seed", type=int, default=6)
    parser.add_argument("--early-stopping-patience-evals", type=int, default=8)
    parser.add_argument("--early-stopping-min-evals", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-fullenv", action="store_true")
    parser.add_argument("--reference-fullenv-csv", default=None)
    return parser


def main() -> None:
    raw_argv = sys.argv[1:]
    args = build_parser().parse_args(raw_argv)
    apply_thesis_formulation_preset(args, raw_argv)
    args.device = resolve_device(args.device)

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else preferred_results_dir(results_root, "preference_response")
    output_dir.mkdir(parents=True, exist_ok=True)

    conditioning_grid = _weight_grid_from_args(args.conditioning_grid, simplex_step=None)
    utility_grid = _weight_grid_from_args(args.utility_grid, simplex_step=args.utility_grid_step)
    seeds = tuple(int(seed) for seed in parse_csv_tuple(args.seeds, int))
    reference_fullenv_csv = (
        Path(args.reference_fullenv_csv)
        if args.reference_fullenv_csv
        else results_root / "analysis" / "fullenv_policy_comparison.csv"
    )

    manifest = {
        "experiment": "supplement_moppo_preference_response",
        "description": "Preference-conditioned MO-PPO with validation-time preference-response selection.",
        "network": args.network,
        "load": args.load,
        "wind": args.wind,
        "solar": args.solar,
        "splits": {
            "train": [args.train_start, args.train_end],
            "validation": [args.val_start, args.val_end],
            "test": [args.test_start, args.test_end],
        },
        "conditioning_grid": [list(weights) for weights in conditioning_grid],
        "utility_grid": [list(weights) for weights in utility_grid],
        "seeds": list(seeds),
        "timesteps": args.timesteps,
        "selection_top_k_per_seed": args.selection_top_k_per_seed,
        "selection_episodes": args.selection_episodes,
        "proxy_test_episodes": args.proxy_test_episodes,
        "fullenv_episodes": args.fullenv_episodes,
    }
    write_json(output_dir / "moppo_preference_response_manifest.json", manifest)

    print("[moppo-preference-response] Loading train/validation/test datasets")
    train_dataset = build_dataset(ROOT / args.network, ROOT / args.load, ROOT / args.wind, ROOT / args.solar, args.train_start, args.train_end, args.candidate_lines)
    val_dataset = build_dataset(ROOT / args.network, ROOT / args.load, ROOT / args.wind, ROOT / args.solar, args.val_start, args.val_end, args.candidate_lines)
    test_dataset = build_dataset(ROOT / args.network, ROOT / args.load, ROOT / args.wind, ROOT / args.solar, args.test_start, args.test_end, args.candidate_lines)

    if not args.skip_training:
        print("[moppo-preference-response] Training preference-conditioned MO-PPO")
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
            device=args.device,
            seed=seeds[0],
            scalarization_weights=(0.34, 0.33, 0.33),
            moppo_preference_conditioning=True,
            moppo_sample_preferences=True,
            moppo_preference_sampling_mode="grid",
            moppo_preference_grid=tuple(tuple(weights) for weights in conditioning_grid),
        )
        training_config = TrainingConfig(
            total_timesteps=args.timesteps,
            eval_every_updates=args.eval_every,
            eval_episodes=args.eval_episodes,
            show_progress=True,
            early_stopping_patience_evals=args.early_stopping_patience_evals,
            early_stopping_min_evals=args.early_stopping_min_evals,
            early_stopping_min_delta=args.early_stopping_min_delta,
            restore_best_model_at_end=True,
            validation_weight_grid=tuple(tuple(weights) for weights in conditioning_grid),
        )

        def train_env_factory():
            return build_env(train_dataset, env_mode="proxy", **make_env_kwargs(args, seed=seeds[0]))

        def val_env_factory():
            return build_env(val_dataset, env_mode="proxy", **make_env_kwargs(args, seed=seeds[0]))

        seed_everything(seeds[0])
        train_multi_seed(
            env_factory=train_env_factory,
            base_ppo_config=ppo_config,
            training_config=training_config,
            seeds=seeds,
            mode="moppo",
            eval_env_factory=val_env_factory,
            output_dir=output_dir,
            experiment_name="moppo_preference_response",
        )
        gc.collect()

    print("[moppo-preference-response] Selecting checkpoints by validation preference response")
    candidate_frame, selected_frame = build_validation_response_archive(
        response_dir=output_dir,
        dataset=val_dataset,
        args=args,
        utility_grid=utility_grid,
        conditioning_grid=conditioning_grid,
    )
    if candidate_frame.empty or selected_frame.empty:
        raise RuntimeError(f"No validation response candidates found under {output_dir}.")
    candidate_frame.to_csv(output_dir / "validation_response_candidates.csv", index=False)
    selected_frame.to_csv(output_dir / "validation_response_selected.csv", index=False)
    write_json(output_dir / "validation_response_selected.json", selected_frame.to_dict(orient="records"))

    print("[moppo-preference-response] Evaluating selected MO-PPO checkpoints on proxy test split")
    proxy_frame, proxy_lines = evaluate_selected_runs_on_test(
        selected_runs=selected_frame,
        dataset=test_dataset,
        args=args,
        utility_grid=utility_grid,
        env_mode="proxy",
        episodes=args.proxy_test_episodes,
        output_dir=output_dir,
    )
    full_frame = pd.DataFrame()
    full_lines = pd.DataFrame()
    if not args.skip_fullenv:
        print("[moppo-preference-response] Evaluating selected MO-PPO checkpoints on full PyPSA test split")
        full_frame, full_lines = evaluate_selected_runs_on_test(
            selected_runs=selected_frame,
            dataset=test_dataset,
            args=args,
            utility_grid=utility_grid,
            env_mode="full",
            episodes=args.fullenv_episodes,
            output_dir=output_dir,
        )

    collapsed_proxy = collapse_outcome_points(proxy_frame, env_mode="proxy") if not proxy_frame.empty else proxy_frame.copy()
    collapsed_full = collapse_outcome_points(full_frame, env_mode="full") if not full_frame.empty else full_frame.copy()
    reference = None
    if reference_fullenv_csv is not None and reference_fullenv_csv.exists():
        reference = pd.read_csv(reference_fullenv_csv)
        reference = reference[reference["policy"].isin(["zero", "ppo", "moppo", "uniform", "myopic_proxy"])].copy()

    _plot_front(
        collapsed_proxy,
        output_dir / "preference_response_proxy.png",
        "MO-PPO proxy preference response",
        y_col="grid_stress_mean",
        y_label="Proxy grid stress (lower better)",
        reference_frame=None,
    )
    if not collapsed_full.empty:
        _plot_front(
            collapsed_full,
            output_dir / "preference_response_full_load_shedding.png",
            "MO-PPO full-environment preference response",
            y_col="load_shedding_mean",
            y_label="Load shedding (MWh, lower better)",
            reference_frame=reference,
        )
        _plot_front(
            collapsed_full,
            output_dir / "preference_response_full_curtailment.png",
            "MO-PPO full-environment renewable trade-off",
            y_col="renewable_curtailment_mean",
            y_label="Renewable curtailment (MWh, lower better)",
            reference_frame=reference,
        )

    write_story(output_dir, proxy_frame, full_frame)
    outputs = {
        "validation_candidates": str(output_dir / "validation_response_candidates.csv"),
        "selected_runs": str(output_dir / "validation_response_selected.csv"),
        "proxy_results": str(output_dir / "preference_response_test_results_proxy.csv"),
        "proxy_results_unique": str(output_dir / "preference_response_test_results_proxy_unique.csv"),
    }
    if not full_frame.empty:
        outputs["full_results"] = str(output_dir / "preference_response_test_results_full.csv")
        outputs["full_results_unique"] = str(output_dir / "preference_response_test_results_full_unique.csv")

    summary = {
        "formulation_preset": args.formulation_preset,
        "formulation_description": args.formulation_description,
        "candidate_lines": int(args.candidate_lines),
        "max_upgrade_mw": float(args.max_upgrade_mw),
        "budget_mw": float(args.budget_mw),
        "n_training_runs": int(len(collect_run_dirs(output_dir))),
        "n_validation_candidates": int(len(candidate_frame)),
        "n_selected_runs": int(len(selected_frame)),
        "proxy_test_points": int(len(proxy_frame)),
        "proxy_test_unique_outcomes": int(len(collapsed_proxy)),
        "proxy_test_nondominated_unique_outcomes": int(collapsed_proxy["test_nondominated"].sum()) if not collapsed_proxy.empty else 0,
        "full_test_points": int(len(full_frame)),
        "full_test_unique_outcomes": int(len(collapsed_full)),
        "full_test_nondominated_unique_outcomes": int(collapsed_full["test_nondominated"].sum()) if not collapsed_full.empty else 0,
        "outputs": outputs,
    }
    write_json(output_dir / "preference_response_summary.json", summary)
    print(f"[moppo-preference-response] Finished. Results are in {output_dir}")


if __name__ == "__main__":
    main()
