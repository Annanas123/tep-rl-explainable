from __future__ import annotations

"""Train and evaluate a scalarised PPO candidate set for empirical trade-off analysis."""

import argparse
import copy
import gc
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from tep_rl.config import EnvironmentConfig, PPOConfig, TrainingConfig
from tep_rl.evaluation import evaluate_agent, scalarized_mean_reward
from tep_rl.line_metadata import endpoint_label, osm_way_url
from tep_rl.ppo import load_agent
from tep_rl.reproducibility import seed_everything
from tep_rl.thesis_formulations import (
    DEFAULT_THESIS_FORMULATION_PRESET,
    THESIS_FORMULATION_PRESETS,
    apply_thesis_formulation_preset,
)
from tep_rl.training import train_multi_seed

from thesis_pipeline_utils import (
    build_dataset,
    build_env,
    collect_run_dirs,
    list_checkpoint_candidates_from_run_dir,
    read_json,
    write_json,
    write_multi_seed_summary,
)


DEFAULT_WEIGHT_GRID = (
    "0.95,0.03,0.02",
    "0.80,0.15,0.05",
    "0.60,0.35,0.05",
    "0.45,0.45,0.10",
    "0.34,0.33,0.33",
    "0.20,0.55,0.25",
    "0.10,0.80,0.10",
    "0.10,0.35,0.55",
    "0.05,0.15,0.80",
    "0.03,0.02,0.95",
)

METRIC_COLUMNS = (
    "total_cost_mean",
    "grid_stress_mean",
    "load_shedding_mean",
    "renewable_curtailment_mean",
    "renewable_share_mean",
    "emissions_mean",
)


def parse_csv_tuple(text: str, cast=float) -> tuple:
    return tuple(cast(part) for part in re.split(r"[\s,]+", str(text).strip()) if part)


def normalize_weights(weights: Sequence[float]) -> tuple[float, ...]:
    array = np.asarray(list(weights), dtype=float)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0.0:
        array = np.full_like(array, 1.0 / max(len(array), 1))
    else:
        array = array / total
    return tuple(float(value) for value in array)


def weight_label(weights: Sequence[float]) -> str:
    return "_".join(f"{float(value):.2f}" for value in weights)


def candidate_id(row: pd.Series | dict[str, Any]) -> str:
    weights = row["weights_tuple"] if "weights_tuple" in row else row["weights"]
    if isinstance(weights, str):
        weights_label = weights
    else:
        weights_label = weight_label(weights)
    seed = int(row.get("seed", -1))
    update = int(row.get("update", -1))
    origin = str(row.get("selection_origin", "candidate"))
    return f"w_{weights_label}_seed{seed:03d}_u{update:04d}_{origin}"


def nondominated_minimize(frame: pd.DataFrame, objective_cols: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    values = frame[objective_cols].to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=np.inf, posinf=np.inf, neginf=-np.inf)
    keep = np.ones(len(frame), dtype=bool)
    for idx, candidate in enumerate(values):
        for challenger_idx, challenger in enumerate(values):
            if idx == challenger_idx:
                continue
            no_worse = np.all(challenger <= candidate + 1e-12)
            strictly_better = np.any(challenger < candidate - 1e-12)
            if no_worse and strictly_better:
                keep[idx] = False
                break
    return pd.Series(keep, index=frame.index)


def select_diverse_archive(frame: pd.DataFrame, max_size: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_size:
        return frame.copy()

    selected: list[int] = []
    sorted_frame = frame.sort_values(["validation_nondominated", "validation_selection_score"], ascending=[False, False])
    for _, group in sorted_frame.groupby("weights", sort=False):
        selected.append(int(group.index[0]))
        if len(selected) >= max_size:
            break

    remaining = sorted_frame.drop(index=selected, errors="ignore")
    objective_cols = ["val_total_cost_mean", "val_grid_stress_mean", "val_renewable_curtailment_mean", "val_total_investment_mean"]
    values = remaining[objective_cols].to_numpy(dtype=float)
    if len(selected) < max_size and len(values):
        selected_values = sorted_frame.loc[selected, objective_cols].to_numpy(dtype=float)
        mins = np.nanmin(sorted_frame[objective_cols].to_numpy(dtype=float), axis=0)
        maxs = np.nanmax(sorted_frame[objective_cols].to_numpy(dtype=float), axis=0)
        scale = np.maximum(maxs - mins, 1e-9)
        selected_norm = (selected_values - mins) / scale
        for idx, row_values in zip(remaining.index, values):
            row_norm = (row_values - mins) / scale
            distances = np.linalg.norm(selected_norm - row_norm, axis=1)
            remaining.loc[idx, "_diversity_score"] = float(np.min(distances))
        remaining = remaining.sort_values(["validation_nondominated", "_diversity_score"], ascending=[False, False])
        for idx in remaining.index:
            selected.append(int(idx))
            if len(selected) >= max_size:
                break

    return sorted_frame.loc[selected].drop(columns=["_diversity_score"], errors="ignore").reset_index(drop=True)


def collapse_outcome_points(frame: pd.DataFrame, env_mode: str) -> pd.DataFrame:
    """Collapse candidate checkpoints that induce the same evaluated outcome."""
    if frame.empty:
        return frame.copy()
    if env_mode == "proxy":
        key_cols = [
            "total_cost_mean",
            "grid_stress_mean",
            "renewable_curtailment_mean",
            "renewable_share_mean",
            "total_investment_mean",
            "active_lines_mean",
        ]
    else:
        key_cols = [
            "total_cost_mean",
            "load_shedding_mean",
            "renewable_curtailment_mean",
            "renewable_share_mean",
            "total_investment_mean",
            "active_lines_mean",
        ]

    rounded = frame.copy()
    for column in key_cols:
        rounded[f"_sig_{column}"] = rounded[column].round(6)
    signature_cols = [f"_sig_{column}" for column in key_cols]

    groups: list[dict[str, Any]] = []
    for _, group in rounded.groupby(signature_cols, sort=False, dropna=False):
        first = group.iloc[0].to_dict()
        payload = {column: first[column] for column in frame.columns if not column.startswith("_sig_")}
        payload["support_size"] = int(len(group))
        payload["support_weights"] = " | ".join(sorted(set(group["weights"].astype(str))))
        payload["support_weight_count"] = int(group["weights"].nunique())
        payload["support_seeds"] = " | ".join(str(int(seed)) for seed in sorted(set(group["seed"].astype(int))))
        payload["support_seed_count"] = int(group["seed"].nunique())
        payload["support_candidate_ids"] = " | ".join(group["candidate_id"].astype(str).tolist())
        payload["candidate_id"] = f"{first['candidate_id']}__x{len(group)}"
        groups.append(payload)

    collapsed = pd.DataFrame(groups)
    objective_cols = ["total_cost_mean", "load_shedding_mean", "renewable_curtailment_mean", "total_investment_mean"]
    if env_mode == "proxy":
        objective_cols = ["total_cost_mean", "grid_stress_mean", "renewable_curtailment_mean", "total_investment_mean"]
    collapsed["test_nondominated"] = nondominated_minimize(collapsed, objective_cols)
    return collapsed.sort_values(["test_nondominated", "support_size", "total_cost_mean"], ascending=[False, False, True]).reset_index(drop=True)


def make_env_kwargs(args: argparse.Namespace, seed: int, budget_mw: float | None = None) -> dict[str, Any]:
    return {
        "episode_length": args.episode_length,
        "max_upgrade_mw": args.max_upgrade_mw,
        "budget_mw": float(args.budget_mw if budget_mw is None else budget_mw),
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
        "cost_reward_scale": args.cost_reward_scale,
        "overload_reward_scale": args.overload_reward_scale,
        "third_objective_mode": args.third_objective_mode,
        "curtailment_reward_scale": args.curtailment_reward_scale,
        "emissions_reward_scale": args.emissions_reward_scale,
        "solver": args.solver,
        "full_env_fallback_to_proxy": not args.disable_full_env_fallback_to_proxy,
        "seed": seed,
    }


def resolve_device(requested: str) -> str:
    try:
        import torch

        if requested in {"auto", "gpu"}:
            requested = "cuda"
        if requested == "cuda":
            if torch.cuda.is_available():
                print(f"[GPU] Using CUDA device: {torch.cuda.get_device_name(0)}")
                return "cuda"
            print("[GPU] CUDA requested but unavailable; using CPU.")
            return "cpu"
    except ImportError:
        pass
    return requested if requested not in {"auto", "gpu"} else "cpu"


def build_parser() -> argparse.ArgumentParser:
    default_env = EnvironmentConfig()
    default_ppo = PPOConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Train a scalarised PPO candidate set over a broad preference grid "
            "and evaluate the validation-selected non-dominated candidates as an empirical Pareto front."
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
    parser.add_argument("--weight-grid", nargs="+", default=list(DEFAULT_WEIGHT_GRID))
    parser.add_argument("--seeds", default="7,11,19")
    parser.add_argument("--timesteps", type=int, default=60_000)
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
    parser.add_argument("--proxy-test-episodes", type=int, default=10)
    parser.add_argument("--fullenv-episodes", type=int, default=5)
    parser.add_argument("--early-stopping-patience-evals", type=int, default=8)
    parser.add_argument("--early-stopping-min-evals", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--archive-size", type=int, default=15)
    parser.add_argument("--validation-top-k-per-weight", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-fullenv", action="store_true")
    parser.add_argument("--reference-fullenv-csv", default=None)
    return parser


def metric_row(
    evaluation: dict[str, Any],
    prefix: str = "",
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    row: dict[str, float] = {}
    for metric in METRIC_COLUMNS:
        row[f"{prefix}{metric}"] = float(evaluation.get(metric, np.nan))
    action_stats = evaluation.get("action_stats", {})
    mean_per_line = action_stats.get("mean_mw_per_line", {})
    total_investment = float(action_stats.get("total_investment_mean", np.nan))
    if not np.isfinite(total_investment) and mean_per_line:
        total_investment = float(np.sum(list(mean_per_line.values())))
    active_lines = float(sum(1 for value in mean_per_line.values() if float(value) > 0.1))
    row[f"{prefix}total_investment_mean"] = total_investment
    row[f"{prefix}active_lines_mean"] = active_lines
    row[f"{prefix}invested_line_fraction_mean"] = active_lines / max(len(mean_per_line), 1) if mean_per_line else np.nan
    if weights is not None:
        row[f"{prefix}scalarized_mean_reward"] = scalarized_mean_reward(evaluation, weights)
    return row


def build_validation_archive(archive_dir: Path, max_per_weight: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for weight_dir in sorted(path for path in archive_dir.glob("weights_*") if path.is_dir()):
        for run_dir in collect_run_dirs(weight_dir):
            try:
                candidates = list_checkpoint_candidates_from_run_dir(run_dir)
            except FileNotFoundError:
                continue
            config = read_json(run_dir / "config_snapshot.json")
            weights = normalize_weights(config["ppo_config"]["scalarization_weights"])
            seed = int(config["ppo_config"].get("seed", -1))
            for candidate in candidates:
                evaluation = candidate["evaluation"]
                row = {
                    "candidate_id": "",
                    "weights": weight_label(weights),
                    "weights_tuple": weights,
                    "weight_cost": weights[0],
                    "weight_stress": weights[1],
                    "weight_sustainability": weights[2],
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "checkpoint": str(candidate["checkpoint"]),
                    "update": int(candidate.get("update", -1)),
                    "selection_origin": str(candidate.get("selection_origin", "")),
                    "validation_selection_score": float(candidate.get("selection_score", np.nan)),
                }
                row.update(metric_row(evaluation, prefix="val_", weights=weights))
                rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["candidate_id"] = frame.apply(candidate_id, axis=1)
    objective_cols = ["val_total_cost_mean", "val_grid_stress_mean", "val_renewable_curtailment_mean", "val_total_investment_mean"]
    frame["validation_nondominated"] = nondominated_minimize(frame, objective_cols)
    selected_rows: list[pd.DataFrame] = []
    for _, group in frame.sort_values("validation_selection_score", ascending=False).groupby("weights", sort=False):
        nd = group[group["validation_nondominated"]].head(max_per_weight)
        if len(nd) < max_per_weight:
            nd = pd.concat([nd, group.drop(index=nd.index).head(max_per_weight - len(nd))], ignore_index=False)
        selected_rows.append(nd)
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else frame.head(0)
    return selected.sort_values(["validation_nondominated", "validation_selection_score"], ascending=[False, False]).reset_index(drop=True)


def evaluate_archive(
    archive: pd.DataFrame,
    dataset,
    args: argparse.Namespace,
    env_mode: str,
    episodes: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    network = pypsa.Network(str(ROOT / args.network))
    for _, archive_row in tqdm(archive.iterrows(), total=len(archive), desc=f"Evaluate {env_mode}", unit="policy"):
        seed = int(archive_row["seed"])
        weights = normalize_weights((archive_row["weight_cost"], archive_row["weight_stress"], archive_row["weight_sustainability"]))
        env = build_env(dataset, env_mode=env_mode, **make_env_kwargs(args, seed=seed))
        agent = load_agent(archive_row["checkpoint"], device=args.device)
        evaluation = evaluate_agent(agent, env, episodes=episodes, deterministic=True)

        row = {
            "candidate_id": archive_row["candidate_id"],
            "weights": archive_row["weights"],
            "weight_cost": weights[0],
            "weight_stress": weights[1],
            "weight_sustainability": weights[2],
            "seed": seed,
            "checkpoint": archive_row["checkpoint"],
            "update": int(archive_row["update"]),
            "selection_origin": archive_row["selection_origin"],
        }
        row.update(metric_row(evaluation, prefix="", weights=weights))
        rows.append(row)

        for line, value in evaluation.get("action_stats", {}).get("mean_mw_per_line", {}).items():
            if abs(float(value)) <= 1e-9:
                continue
            line_rows.append(
                {
                    "candidate_id": archive_row["candidate_id"],
                    "env": env_mode,
                    "weights": archive_row["weights"],
                    "seed": seed,
                    "line": line,
                    "line_label": endpoint_label(network, line),
                    "osm_way_url": osm_way_url(line),
                    "mean_upgrade_mw": float(value),
                }
            )
        del env, agent
        gc.collect()

    result_frame = pd.DataFrame(rows)
    line_frame = pd.DataFrame(line_rows)
    if not result_frame.empty:
        objective_cols = ["total_cost_mean", "load_shedding_mean", "renewable_curtailment_mean", "total_investment_mean"]
        if env_mode == "proxy":
            objective_cols = ["total_cost_mean", "grid_stress_mean", "renewable_curtailment_mean", "total_investment_mean"]
        result_frame["test_nondominated"] = nondominated_minimize(result_frame, objective_cols)
        result_frame.to_csv(output_dir / f"pareto_test_results_{env_mode}.csv", index=False)
    if not line_frame.empty:
        line_frame.to_csv(output_dir / f"pareto_test_line_upgrades_{env_mode}.csv", index=False)
    return result_frame, line_frame


def _plot_front(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    y_col: str,
    y_label: str,
    reference_frame: pd.DataFrame | None = None,
) -> None:
    if frame.empty:
        return
    plot_frame = frame.copy()
    plot_frame["total_cost_million"] = plot_frame["total_cost_mean"] / 1e6
    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    unique_outcomes = plot_frame[
        [
            "total_cost_mean",
            y_col,
            "renewable_curtailment_mean",
            "total_investment_mean",
            "active_lines_mean",
        ]
    ].drop_duplicates().shape[0]
    dominated = plot_frame[~plot_frame["test_nondominated"]]
    nondom = plot_frame[plot_frame["test_nondominated"]]
    if not dominated.empty:
        ax.scatter(
            dominated["total_cost_million"],
            dominated[y_col],
            s=45 + 0.15 * dominated["total_investment_mean"].fillna(0),
            color="#b7c4d6",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.65,
            label="Dominated candidate policy",
        )
    if not nondom.empty:
        ax.scatter(
            nondom["total_cost_million"],
            nondom[y_col],
            s=85 + 6.0 * nondom.get("support_size", pd.Series(1, index=nondom.index)).fillna(1) + 0.18 * nondom["total_investment_mean"].fillna(0),
            color="#d97706",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.92,
            label="Non-dominated unique outcome",
        )
        ordered = nondom.sort_values("total_cost_million")
        ax.plot(ordered["total_cost_million"], ordered[y_col], color="#d97706", linewidth=1.2, alpha=0.75)

    if reference_frame is not None and not reference_frame.empty:
        ref = reference_frame.copy()
        if "total_cost_mean" in ref:
            ref["total_cost_million"] = ref["total_cost_mean"] / 1e6
            for _, row in ref.iterrows():
                label = row.get("Policy", row.get("policy", "reference"))
                if y_col not in row:
                    continue
                ax.scatter(row["total_cost_million"], row[y_col], marker="*", s=150, color="#2563eb", edgecolor="black", linewidth=0.6)
                ax.annotate(str(label), (row["total_cost_million"], row[y_col]), xytext=(6, 5), textcoords="offset points", fontsize=8)

    for _, row in nondom.iterrows():
        support_size = int(row.get("support_size", 1))
        support_weights = int(row.get("support_weight_count", 1))
        label = f"{row['total_investment_mean']:.0f} MW\n{support_size} ckpts / {support_weights} weights"
        ax.annotate(label, (row["total_cost_million"], row[y_col]), xytext=(6, 5), textcoords="offset points", fontsize=7)

    ax.set_xlabel("Total cost (M, lower better)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    ax.text(
        0.02,
        0.98,
        f"{len(plot_frame)} selected checkpoints\n{unique_outcomes} unique evaluated outcome(s)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_pareto_outputs(
    output_dir: Path,
    proxy_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    reference_fullenv_csv: Path | None,
) -> None:
    collapsed_proxy = collapse_outcome_points(proxy_frame, env_mode="proxy") if not proxy_frame.empty else proxy_frame.copy()
    collapsed_full = collapse_outcome_points(full_frame, env_mode="full") if not full_frame.empty else full_frame.copy()
    if not collapsed_proxy.empty:
        collapsed_proxy.to_csv(output_dir / "pareto_test_results_proxy_unique.csv", index=False)
    if not collapsed_full.empty:
        collapsed_full.to_csv(output_dir / "pareto_test_results_full_unique.csv", index=False)

    reference = None
    if reference_fullenv_csv is not None and reference_fullenv_csv.exists():
        reference = pd.read_csv(reference_fullenv_csv)
        reference = reference[reference["policy"].isin(["zero", "ppo", "moppo", "uniform", "myopic_proxy"])].copy()

    _plot_front(
        collapsed_proxy,
        output_dir / "pareto_front_proxy.png",
        "Proxy Candidate Set",
        y_col="grid_stress_mean",
        y_label="Proxy grid stress (lower better)",
        reference_frame=None,
    )
    if not collapsed_full.empty:
        _plot_front(
            collapsed_full,
            output_dir / "pareto_front_full_load_shedding.png",
            "Full-Environment Candidate Set",
            y_col="load_shedding_mean",
            y_label="Load shedding (MWh, lower better)",
            reference_frame=reference,
        )
        _plot_front(
            collapsed_full,
            output_dir / "pareto_front_full_curtailment.png",
            "Full-environment renewable trade-off",
            y_col="renewable_curtailment_mean",
            y_label="Renewable curtailment (MWh, lower better)",
            reference_frame=reference,
        )
        table = collapsed_full[collapsed_full["test_nondominated"]].copy()
        if table.empty:
            table = collapsed_full.sort_values("scalarized_mean_reward", ascending=False).head(8).copy()
        display = table[
            [
                "support_weights",
                "support_seeds",
                "support_size",
                "total_cost_mean",
                "load_shedding_mean",
                "renewable_curtailment_mean",
                "renewable_share_mean",
                "total_investment_mean",
                "active_lines_mean",
            ]
        ].sort_values("total_cost_mean")
        display.to_latex(
            output_dir / "pareto_front_full_table.tex",
            index=False,
            float_format=lambda value: f"{value:.3f}",
            caption="Unique non-dominated scalarised PPO candidate-set outcomes evaluated in the full PyPSA environment.",
            label="tab:pareto_archive_front",
        )


def write_story(output_dir: Path, proxy_frame: pd.DataFrame, full_frame: pd.DataFrame) -> None:
    collapsed_proxy = collapse_outcome_points(proxy_frame, env_mode="proxy") if not proxy_frame.empty else proxy_frame.copy()
    collapsed_full = collapse_outcome_points(full_frame, env_mode="full") if not full_frame.empty else full_frame.copy()
    lines = [
        "# Scalarised PPO Candidate-Set Sweep",
        "",
        "This experiment trains scalarised PPO policies over a broad objective-weight grid and treats validation checkpoints as a candidate set.",
        "The candidate set is selected on the validation split before any test evaluation, then the selected policies are evaluated on the held-out test split.",
        "",
    ]
    if not proxy_frame.empty:
        lines.extend(
            [
                f"- Proxy candidate policies evaluated: {len(proxy_frame)}.",
                f"- Proxy unique evaluated outcomes: {len(collapsed_proxy)}.",
                f"- Proxy non-dominated unique outcomes: {int(collapsed_proxy['test_nondominated'].sum())}.",
            ]
        )
    if not full_frame.empty:
        nd = collapsed_full[collapsed_full["test_nondominated"]].copy()
        lines.extend(
            [
                f"- Full-environment candidate policies evaluated: {len(full_frame)}.",
                f"- Full-environment unique evaluated outcomes: {len(collapsed_full)}.",
                f"- Full-environment non-dominated unique outcomes: {int(collapsed_full['test_nondominated'].sum())}.",
            ]
        )
        if not nd.empty:
            best_cost = nd.sort_values("total_cost_mean").iloc[0]
            best_curtail = nd.sort_values("renewable_curtailment_mean").iloc[0]
            lines.extend(
                [
                    f"- Representative full-environment outcome: cost {best_cost['total_cost_mean'] / 1e6:.2f}M, load shedding {best_cost['load_shedding_mean']:.2f} MWh, curtailment {best_cost['renewable_curtailment_mean']:.2f} MWh, investment {best_cost['total_investment_mean']:.1f} MW.",
                    f"- This single outcome is supported by {int(best_cost['support_size'])} selected checkpoints spanning {int(best_cost['support_weight_count'])} distinct weight vectors and {int(best_cost['support_seed_count'])} seeds.",
                ]
            )
        else:
            lines.append("- The selected candidate set did not produce distinct full-environment trade-off points.")
    (output_dir / "pareto_story.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    raw_argv = sys.argv[1:]
    args = build_parser().parse_args(raw_argv)
    apply_thesis_formulation_preset(args, raw_argv)
    args.device = resolve_device(args.device)
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else preferred_results_dir(results_root, "pareto_archive")
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_grid = [normalize_weights(parse_csv_tuple(text, float)) for text in args.weight_grid]
    seeds = tuple(int(seed) for seed in parse_csv_tuple(args.seeds, int))
    reference_fullenv_csv = (
        Path(args.reference_fullenv_csv)
        if args.reference_fullenv_csv
        else results_root / "analysis" / "fullenv_policy_comparison.csv"
    )

    manifest = {
        "experiment": "supplement_pareto_archive",
        "description": "Scalarised PPO candidate set for empirical Pareto-front analysis.",
        "network": args.network,
        "load": args.load,
        "wind": args.wind,
        "solar": args.solar,
        "splits": {
            "train": [args.train_start, args.train_end],
            "validation": [args.val_start, args.val_end],
            "test": [args.test_start, args.test_end],
        },
        "candidate_lines": args.candidate_lines,
        "weight_grid": [list(weights) for weights in weight_grid],
        "seeds": list(seeds),
        "timesteps": args.timesteps,
        "archive_size": args.archive_size,
        "validation_top_k_per_weight": args.validation_top_k_per_weight,
        "proxy_test_episodes": args.proxy_test_episodes,
        "fullenv_episodes": args.fullenv_episodes,
        "environment": {
            "episode_length": args.episode_length,
            "decision_interval": args.decision_interval,
            "max_upgrade_mw": args.max_upgrade_mw,
            "budget_mw": args.budget_mw,
            "allocation_sparsity_cutoff": args.allocation_sparsity_cutoff,
            "third_objective_mode": args.third_objective_mode,
        },
    }
    write_json(output_dir / "pareto_archive_manifest.json", manifest)

    print("[candidate-set] Loading train/validation/test datasets")
    train_dataset = build_dataset(ROOT / args.network, ROOT / args.load, ROOT / args.wind, ROOT / args.solar, args.train_start, args.train_end, args.candidate_lines)
    val_dataset = build_dataset(ROOT / args.network, ROOT / args.load, ROOT / args.wind, ROOT / args.solar, args.val_start, args.val_end, args.candidate_lines)
    test_dataset = build_dataset(ROOT / args.network, ROOT / args.load, ROOT / args.wind, ROOT / args.solar, args.test_start, args.test_end, args.candidate_lines)

    if not args.skip_training:
        for weights in tqdm(weight_grid, desc="Train scalarised PPO weights", unit="weight"):
            label = weight_label(weights)
            weight_dir = output_dir / f"weights_{label}"
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
                scalarization_weights=weights,
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
            )

            def train_env_factory():
                return build_env(train_dataset, env_mode="proxy", **make_env_kwargs(args, seed=seeds[0]))

            def val_env_factory():
                return build_env(val_dataset, env_mode="proxy", **make_env_kwargs(args, seed=seeds[0]))

            seed_everything(seeds[0])
            results = train_multi_seed(
                env_factory=train_env_factory,
                base_ppo_config=ppo_config,
                training_config=training_config,
                seeds=seeds,
                mode="ppo",
                eval_env_factory=val_env_factory,
                output_dir=weight_dir,
                experiment_name=f"pareto_ppo_w_{label}",
            )
            write_multi_seed_summary(
                weight_dir,
                "ppo",
                seeds,
                [{"seed": item["seed"], "run_id": item["run_id"], "evaluation": item["evaluation"]} for item in results],
            )
            gc.collect()

    print("[candidate-set] Building validation-selected policy candidate set")
    validation_candidates = build_validation_archive(output_dir, max_per_weight=args.validation_top_k_per_weight)
    if validation_candidates.empty:
        raise RuntimeError(f"No validation checkpoint candidates found under {output_dir}.")
    validation_candidates.to_csv(output_dir / "pareto_archive_validation_candidates.csv", index=False)
    validation_unique = collapse_outcome_points(
        validation_candidates.rename(
            columns={
                "val_total_cost_mean": "total_cost_mean",
                "val_grid_stress_mean": "grid_stress_mean",
                "val_renewable_curtailment_mean": "renewable_curtailment_mean",
                "val_renewable_share_mean": "renewable_share_mean",
                "val_total_investment_mean": "total_investment_mean",
                "val_active_lines_mean": "active_lines_mean",
            }
        )[
            [
                "candidate_id",
                "weights",
                "seed",
                "checkpoint",
                "update",
                "selection_origin",
                "validation_selection_score",
                "weight_cost",
                "weight_stress",
                "weight_sustainability",
                "total_cost_mean",
                "grid_stress_mean",
                "renewable_curtailment_mean",
                "renewable_share_mean",
                "total_investment_mean",
                "active_lines_mean",
            ]
        ].assign(scalarized_mean_reward=validation_candidates["val_scalarized_mean_reward"]),
        env_mode="proxy",
    )
    validation_unique.to_csv(output_dir / "pareto_archive_validation_candidates_unique.csv", index=False)
    archive = select_diverse_archive(validation_candidates, max_size=args.archive_size)
    archive.to_csv(output_dir / "pareto_archive_selected.csv", index=False)
    write_json(output_dir / "pareto_archive_selected.json", archive.to_dict(orient="records"))

    print("[candidate-set] Evaluating selected candidates on proxy test split")
    proxy_frame, proxy_lines = evaluate_archive(archive, test_dataset, args, env_mode="proxy", episodes=args.proxy_test_episodes, output_dir=output_dir)
    full_frame = pd.DataFrame()
    if not args.skip_fullenv:
        print("[candidate-set] Evaluating selected candidates on full PyPSA test split")
        full_frame, full_lines = evaluate_archive(archive, test_dataset, args, env_mode="full", episodes=args.fullenv_episodes, output_dir=output_dir)

    write_pareto_outputs(output_dir, proxy_frame, full_frame, reference_fullenv_csv)
    write_story(output_dir, proxy_frame, full_frame)
    collapsed_proxy = collapse_outcome_points(proxy_frame, env_mode="proxy") if not proxy_frame.empty else proxy_frame.copy()
    collapsed_full = collapse_outcome_points(full_frame, env_mode="full") if not full_frame.empty else full_frame.copy()

    summary = {
        "formulation_preset": args.formulation_preset,
        "formulation_description": args.formulation_description,
        "candidate_lines": int(args.candidate_lines),
        "max_upgrade_mw": float(args.max_upgrade_mw),
        "budget_mw": float(args.budget_mw),
        "n_validation_candidates": int(len(validation_candidates)),
        "n_validation_unique_outcomes": int(len(validation_unique)),
        "n_archive_selected": int(len(archive)),
        "proxy_test_policies": int(len(proxy_frame)),
        "proxy_test_unique_outcomes": int(len(collapsed_proxy)),
        "proxy_test_nondominated": int(proxy_frame["test_nondominated"].sum()) if not proxy_frame.empty else 0,
        "proxy_test_nondominated_unique_outcomes": int(collapsed_proxy["test_nondominated"].sum()) if not collapsed_proxy.empty else 0,
        "full_test_policies": int(len(full_frame)),
        "full_test_unique_outcomes": int(len(collapsed_full)),
        "full_test_nondominated": int(full_frame["test_nondominated"].sum()) if not full_frame.empty else 0,
        "full_test_nondominated_unique_outcomes": int(collapsed_full["test_nondominated"].sum()) if not collapsed_full.empty else 0,
        "outputs": {
            "validation_candidates": str(output_dir / "pareto_archive_validation_candidates.csv"),
            "validation_candidates_unique": str(output_dir / "pareto_archive_validation_candidates_unique.csv"),
            "selected_archive": str(output_dir / "pareto_archive_selected.csv"),
            "proxy_results": str(output_dir / "pareto_test_results_proxy.csv"),
            "proxy_results_unique": str(output_dir / "pareto_test_results_proxy_unique.csv"),
            "full_results": str(output_dir / "pareto_test_results_full.csv"),
            "full_results_unique": str(output_dir / "pareto_test_results_full_unique.csv"),
        },
    }
    write_json(output_dir / "pareto_summary.json", summary)
    print(f"[candidate-set] Finished. Results are in {output_dir}")


if __name__ == "__main__":
    main()
