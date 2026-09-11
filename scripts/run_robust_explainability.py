"""Robust Shapley audit and strict corridor counterfactual for the thesis."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import _build_dataset_for_window, _build_env
from tep_rl.evaluation import evaluate_agent, stratified_episode_start_indices
from tep_rl.line_metadata import aggregate_line_importance, endpoint_anchor_label
from tep_rl.ppo import load_agent
from tep_rl.shapley import (
    PolicyShapleyExplainer,
    capacity_preserving_corridor_ablation,
    collect_policy_states,
    episode_grouped_train_test_indices,
    shapley_guided_ridge_surrogate,
)
from tep_rl.statistics import compare_paired_evaluations


DEFAULT_RESULTS_ROOT = ROOT / "results" / "thesis_final_v12_apg_nt2040"
DEFAULT_TARGET_LINES = (
    "way/134219945-220",
    "way/82662462-220",
    "way/134219946-220",
)
CENTRAL_WEIGHTS = (0.34, 0.33, 0.33)
FH_BLUE = "#00649C"
FH_GREEN = "#8BB31D"
FH_GREY = "#72777A"
FH_AMBER = "#FFBF00"
FH_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "fh_sequential", ["#F4F6F7", "#C9DD8D", FH_GREEN, "#2A829F", FH_BLUE]
)
FH_DIVERGING = LinearSegmentedColormap.from_list(
    "fh_diverging", [FH_BLUE, "#F4F6F7", FH_GREEN]
)

logging.getLogger("fontTools").setLevel(logging.WARNING)
for _logger_name in (
    "pypsa.network.io",
    "pypsa.consistency",
    "pypsa.optimization.optimize",
    "linopy.model",
    "linopy.io",
    "linopy.constants",
):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value)!r}.")


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _seed_from_selection(selection: dict[str, Any]) -> int:
    match = re.search(r"seed(\d+)", str(selection.get("run_dir", "")))
    if not match:
        raise ValueError(f"Cannot parse seed from selection: {selection.get('run_dir')!r}")
    return int(match.group(1))


def _selected_moppo_checkpoints(results_root: Path) -> list[dict[str, Any]]:
    path = results_root / "stage9_full_environment" / "selected_checkpoints_moppo_fullenv_validation.json"
    selections = _read_json(path)
    if not isinstance(selections, list) or not selections:
        raise ValueError(f"No MO-PPO checkpoint selections found in {path}.")
    for item in selections:
        item["seed"] = _seed_from_selection(item)
        if not Path(item["checkpoint"]).exists():
            raise FileNotFoundError(item["checkpoint"])
    return sorted(selections, key=lambda item: int(item["seed"]))


def _environment_args(
    manifest: dict[str, Any],
    results_root: Path,
    env_mode: str,
    seed: int,
) -> SimpleNamespace:
    environment = manifest["environment"]
    test_start, test_end = manifest["splits"]["test"]
    return SimpleNamespace(
        toy=False,
        toy_steps=96,
        seed=int(seed),
        network=str(_resolve_repo_path(manifest["network"])),
        load=str(_resolve_repo_path(manifest["load"])),
        wind=str(_resolve_repo_path(manifest["wind"])),
        solar=str(_resolve_repo_path(manifest["solar"])),
        year=2020,
        candidate_lines=int(manifest["candidate_lines"]),
        preprocessing_manifest=str(results_root / "preprocessing_manifest.json"),
        start=test_start,
        end=test_end,
        env=env_mode,
        episode_length=int(environment["episode_length"]),
        max_upgrade_mw=float(environment["max_upgrade_mw"]),
        budget_mw=float(environment["budget_mw"]),
        decision_interval=int(environment["decision_interval"]),
        temporal_mode=str(environment["temporal_mode"]),
        budget_release=str(environment["budget_release"]),
        action_mode=str(environment["action_mode"]),
        allocation_sharpness=float(environment["allocation_sharpness"]),
        allocation_sparsity_cutoff=float(environment["allocation_sparsity_cutoff"]),
        stability_margin=float(environment["stability_margin"]),
        proxy_balance_mode=str(environment["proxy_balance_mode"]),
        proxy_dispatch_limit=environment.get("proxy_dispatch_limit"),
        load_shedding_cost=float(environment["load_shedding_cost"]),
        line_investment_cost_eur_per_mw_km_year=float(
            environment["line_investment_cost_eur_per_mw_km_year"]
        ),
        cost_reward_scale=float(environment["cost_reward_scale"]),
        overload_reward_scale=float(environment["overload_reward_scale"]),
        third_objective_mode=str(environment["third_objective_mode"]),
        curtailment_reward_scale=float(environment["curtailment_reward_scale"]),
        emissions_reward_scale=float(environment["emissions_reward_scale"]),
        solver=str(environment["solver"]),
        disable_full_env_fallback_to_proxy=True,
    )


def _build_test_environment(
    manifest: dict[str, Any],
    results_root: Path,
    env_mode: str,
    seed: int,
):
    args = _environment_args(manifest, results_root, env_mode=env_mode, seed=seed)
    dataset = _build_dataset_for_window(args, start=args.start, end=args.end)
    return _build_env(dataset, args)


def _target_actions(env, target_lines: Sequence[str]) -> tuple[list[int], list[str]]:
    action_names = env.get_action_names()
    requested = ["spend_fraction", *[f"allocation::{line}" for line in target_lines]]
    missing = [name for name in requested if name not in action_names]
    if missing:
        raise ValueError(f"Requested explainability outputs are absent from the policy: {missing}")
    return [action_names.index(name) for name in requested], requested


def _top_k_overlap(left: pd.Series, right: pd.Series, top_k: int) -> float:
    left_top = set(left.nlargest(min(top_k, len(left))).index)
    right_top = set(right.nlargest(min(top_k, len(right))).index)
    denominator = max(min(top_k, len(left_top), len(right_top)), 1)
    return float(len(left_top.intersection(right_top)) / denominator)


def _column_mean_or_nan(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    return float(frame[column].mean())


def _rank_stability(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    comparison_column: str,
    top_k: int,
    comparison_kind: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_values, group in frame.groupby(list(group_columns), sort=True):
        group_values = group_values if isinstance(group_values, tuple) else (group_values,)
        series_by_comparison = {
            comparison: sub.set_index("feature")["global_importance"].astype(float)
            for comparison, sub in group.groupby(comparison_column)
        }
        for left_key, right_key in itertools.combinations(sorted(series_by_comparison), 2):
            left = series_by_comparison[left_key]
            right = series_by_comparison[right_key]
            common = left.index.intersection(right.index)
            left = left.reindex(common)
            right = right.reindex(common)
            spearman = stats.spearmanr(left.to_numpy(), right.to_numpy())
            kendall = stats.kendalltau(left.to_numpy(), right.to_numpy())
            row = dict(zip(group_columns, group_values))
            row.update(
                {
                    "comparison_kind": comparison_kind,
                    "left": left_key,
                    "right": right_key,
                    "n_common_features": int(len(common)),
                    "spearman_rho": float(spearman.statistic),
                    "kendall_tau": float(kendall.statistic),
                    "top_k_overlap": _top_k_overlap(left, right, top_k),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _summarise_attributions(long_frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranked = long_frame.copy()
    ranked["rank"] = ranked.groupby(["seed", "reference_run", "output"])[
        "global_importance"
    ].rank(method="average", ascending=False)
    ranked["top_k"] = ranked["rank"] <= int(top_k)
    summary = (
        ranked.groupby(["output", "feature"], as_index=False)
        .agg(
            importance_mean=("global_importance", "mean"),
            importance_median=("global_importance", "median"),
            importance_std=("global_importance", "std"),
            importance_q25=("global_importance", lambda values: float(np.quantile(values, 0.25))),
            importance_q75=("global_importance", lambda values: float(np.quantile(values, 0.75))),
            median_rank=("rank", "median"),
            top_k_frequency=("top_k", "mean"),
            n_seeds=("seed", "nunique"),
            n_seed_estimator_runs=("reference_run", "count"),
        )
        .sort_values(["output", "importance_median"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return ranked, summary


def _corridor_attributions(
    feature_frame: pd.DataFrame,
    network,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    for (seed, reference_run, output), group in feature_frame.groupby(
        ["seed", "reference_run", "output"], sort=True
    ):
        corridor = aggregate_line_importance(group, network, value_col="global_importance")
        if corridor.empty:
            continue
        corridor["seed"] = int(seed)
        corridor["reference_run"] = int(reference_run)
        corridor["output"] = output
        corridor["rank"] = corridor["line_importance"].rank(method="average", ascending=False)
        corridor["top_k"] = corridor["rank"] <= int(top_k)
        records.append(corridor)
    long = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if long.empty:
        return long, long
    summary = (
        long.groupby(["output", "line", "line_anchor_label"], as_index=False)
        .agg(
            importance_mean=("line_importance", "mean"),
            importance_median=("line_importance", "median"),
            importance_std=("line_importance", "std"),
            median_rank=("rank", "median"),
            top_k_frequency=("top_k", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["output", "importance_median"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return long, summary


def _short_corridor_label(label: str) -> str:
    label = str(label).replace("220 kV ", "").replace("380 kV ", "")
    coordinate_pairs = re.findall(r"([\d.]+E),[\d.]+N", label)
    cities = [part.strip().split(" (")[0] for part in label.split("->")]
    if len(cities) == 2 and len(coordinate_pairs) >= 2:
        if cities[0] == cities[1]:
            return f"{cities[0]} ({coordinate_pairs[0]}-{coordinate_pairs[1]})"
        return f"{cities[0]}-{cities[1]} ({coordinate_pairs[0]}-{coordinate_pairs[1]})"
    return label if len(label) <= 44 else label[:41] + "..."


def _plot_seed_corridor_stability(
    corridor_long: pd.DataFrame,
    output: str,
    output_dir: Path,
    top_k: int,
) -> None:
    selected = corridor_long[corridor_long["output"] == output].copy()
    per_seed = (
        selected.groupby(["seed", "line", "line_anchor_label"], as_index=False)["line_importance"]
        .mean()
    )
    top_lines = (
        per_seed.groupby(["line", "line_anchor_label"], as_index=False)["line_importance"]
        .mean()
        .nlargest(10, "line_importance")
    )
    line_order = list(top_lines["line"])
    seeds = sorted(per_seed["seed"].unique())
    matrix = np.zeros((len(line_order), len(seeds)), dtype=float)
    labels: list[str] = []
    for row_index, line in enumerate(line_order):
        row = per_seed[per_seed["line"] == line]
        label = row["line_anchor_label"].iloc[0] if not row.empty else line
        labels.append(_short_corridor_label(label))
        for column_index, seed in enumerate(seeds):
            values = row.loc[row["seed"] == seed, "line_importance"]
            matrix[row_index, column_index] = float(values.iloc[0]) if len(values) else 0.0
    maxima = np.maximum(matrix.max(axis=0, keepdims=True), 1e-12)
    normalised = matrix / maxima

    run_ranks = selected.copy()
    run_ranks["rank"] = run_ranks.groupby(["seed", "reference_run"])["line_importance"].rank(
        method="average", ascending=False
    )
    frequencies = np.asarray(
        [float((run_ranks.loc[run_ranks["line"] == line, "rank"] <= top_k).mean()) for line in line_order]
    )

    fig, (ax_heat, ax_freq) = plt.subplots(
        1, 2, figsize=(11.6, 5.9), gridspec_kw={"width_ratios": [4.7, 1.3]}
    )
    image = ax_heat.imshow(normalised, cmap=FH_SEQUENTIAL, vmin=0.0, vmax=1.0, aspect="auto")
    ax_heat.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
    ax_heat.set_yticks(range(len(labels)), labels)
    ax_heat.set_xlabel("MO-PPO training seed")
    ax_heat.set_ylabel("Input corridor")
    ax_heat.set_title("Normalised Shapley importance by seed", fontweight="normal")
    for row_index in range(normalised.shape[0]):
        for column_index in range(normalised.shape[1]):
            value = normalised[row_index, column_index]
            ax_heat.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.52 else "black",
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.02)
    colorbar.set_label("Importance / within-seed maximum")

    positions = np.arange(len(line_order))
    ax_freq.barh(positions, frequencies, color=FH_BLUE)
    ax_freq.set_ylim(len(line_order) - 0.5, -0.5)
    ax_freq.set_xlim(0.0, 1.0)
    ax_freq.set_yticks([])
    ax_freq.set_xlabel(f"Top-{top_k}\nfrequency")
    ax_freq.set_title("Seed-estimator-run\nconsensus", fontweight="normal")
    ax_freq.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
    for position, value in zip(positions, frequencies):
        ax_freq.text(min(value + 0.025, 0.96), position, f"{value:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "xai_seed_corridor_stability.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "xai_seed_corridor_stability.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_surrogate_fidelity(metrics: pd.DataFrame, requested_outputs: Sequence[str], output_dir: Path) -> None:
    selected = metrics[metrics["action"].isin(requested_outputs)].copy()
    seeds = sorted(selected["seed"].unique())
    matrix = np.full((len(requested_outputs), len(seeds)), np.nan, dtype=float)
    for row_index, action in enumerate(requested_outputs):
        for column_index, seed in enumerate(seeds):
            values = selected.loc[(selected["action"] == action) & (selected["seed"] == seed), "r2"]
            if len(values):
                matrix[row_index, column_index] = float(values.iloc[0])

    display = [
        "Spend fraction",
        "Salzburg dominant allocation score",
        "Salzburg parallel A allocation score",
        "Salzburg parallel B allocation score",
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    image = ax.imshow(matrix, cmap=FH_DIVERGING, vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
    ax.set_yticks(range(len(display)), display)
    ax.set_xlabel("MO-PPO training seed")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = "n/a" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(column_index, row_index, label, ha="center", va="center", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label(r"Holdout $R^2$")
    fig.tight_layout()
    fig.savefig(output_dir / "xai_output_surrogate_fidelity.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "xai_output_surrogate_fidelity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


class CapacityPreservingAblationAgent:
    """Online intervention on decoded actions for a multi-line corridor."""

    def __init__(self, base_agent, env, target_lines: Sequence[str]):
        self.base_agent = base_agent
        self.env = env
        self.target_lines = tuple(target_lines)
        self.env_reward_dim = int(base_agent.env_reward_dim)

    def start_rollout(self, training: bool = False) -> None:
        if hasattr(self.base_agent, "start_rollout"):
            self.base_agent.start_rollout(training=training)

    def start_episode(self, training: bool = False) -> None:
        if hasattr(self.base_agent, "start_episode"):
            self.base_agent.start_episode(training=training)

    def act(self, observation, deterministic: bool = True):
        action, log_prob, value = self.base_agent.act(observation, deterministic=deterministic)
        available_budget = float(self.env._available_budget())
        remaining_cap = self.env._remaining_line_upgrade_cap().reindex(self.env.candidate_lines).fillna(0.0)
        increments = self.env._decode_action(
            action,
            available_budget=available_budget,
            remaining_line_cap=remaining_cap,
        )
        released = float(increments.reindex(self.target_lines).fillna(0.0).clip(lower=0.0).sum())
        if released <= 1e-12:
            # A true no-op must use the original budgeted action. Re-encoding an
            # unchanged action through the direct-action interface can
            # otherwise introduce a small float32 round-trip difference.
            return action, log_prob, value
        modified = capacity_preserving_corridor_ablation(
            increments,
            remaining_cap,
            self.target_lines,
        )
        fractions = np.divide(
            modified.to_numpy(dtype=float),
            remaining_cap.to_numpy(dtype=float),
            out=np.zeros(len(modified), dtype=float),
            where=remaining_cap.to_numpy(dtype=float) > 1e-10,
        )
        return np.clip(fractions, 0.0, 1.0).astype(np.float32), log_prob, value


def _run_shapley_audit(args, manifest, selections, output_dir: Path) -> dict[str, Any]:
    all_frames: list[pd.DataFrame] = []
    surrogate_frames: list[pd.DataFrame] = []
    state_cache: dict[int, tuple[Any, np.ndarray, np.ndarray, list[str]]] = {}
    network = None

    for selection_index, selection in enumerate(selections, start=1):
        seed = int(selection["seed"])
        print(f"[shapley] seed {seed} ({selection_index}/{len(selections)})")
        seed_path = output_dir / f"seed_{seed:03d}_output_shapley.csv"
        selection_path = output_dir / f"seed_{seed:03d}_surrogate_selection_shapley.csv"
        surrogate_path = output_dir / f"seed_{seed:03d}_surrogate_metrics.csv"
        env = _build_test_environment(manifest, args.results_root, "proxy", seed)
        network = env.dataset.network
        agent = load_agent(selection["checkpoint"], device=args.device)
        if hasattr(agent, "set_eval_preferences"):
            agent.set_eval_preferences(np.asarray(args.weights, dtype=np.float32))
        states, episode_ids = collect_policy_states(
            env,
            agent,
            episodes=args.episodes,
            deterministic=True,
            return_episode_ids=True,
        )
        output_indices, output_names = _target_actions(env, args.target_lines)
        state_cache[seed] = (agent, states, episode_ids, env.get_action_names())

        if seed_path.exists() and not args.force:
            seed_frame = pd.read_csv(seed_path)
        else:
            run_frames: list[pd.DataFrame] = []
            for reference_run in range(args.reference_runs):
                explainer = PolicyShapleyExplainer(
                    agent=agent,
                    feature_names=env.get_feature_names(),
                    n_samples=args.shapley_samples,
                    seed=args.base_seed + 10000 * seed + reference_run,
                )
                frame = explainer.global_output_shapley(
                    states,
                    output_indices=output_indices,
                    output_names=output_names,
                    baseline_states=states,
                )
                frame["seed"] = seed
                frame["reference_run"] = reference_run
                run_frames.append(frame)
            seed_frame = pd.concat(run_frames, ignore_index=True)
            seed_frame.to_csv(seed_path, index=False)
        all_frames.append(seed_frame)

        selection_existed = selection_path.exists()
        if selection_existed and not args.force:
            selection_frame = pd.read_csv(selection_path)
        else:
            train_idx, _ = episode_grouped_train_test_indices(
                episode_ids,
                test_fraction=0.25,
                seed=args.base_seed + seed,
            )
            selection_states = states[train_idx]
            selection_runs: list[pd.DataFrame] = []
            for reference_run in range(args.reference_runs):
                explainer = PolicyShapleyExplainer(
                    agent=agent,
                    feature_names=env.get_feature_names(),
                    n_samples=args.shapley_samples,
                    seed=args.base_seed + 20000 * seed + reference_run,
                )
                frame = explainer.global_output_shapley(
                    selection_states,
                    output_indices=output_indices,
                    output_names=output_names,
                    baseline_states=selection_states,
                )
                frame["seed"] = seed
                frame["reference_run"] = reference_run
                frame["selection_states"] = len(selection_states)
                selection_runs.append(frame)
            selection_frame = pd.concat(selection_runs, ignore_index=True)
            selection_frame.to_csv(selection_path, index=False)

        if surrogate_path.exists() and selection_existed and not args.force:
            surrogate = pd.read_csv(surrogate_path)
        else:
            ranking = (
                selection_frame.groupby("feature", as_index=False)["global_importance"]
                .mean()
                .sort_values("global_importance", ascending=False)
            )
            surrogate, _, _ = shapley_guided_ridge_surrogate(
                agent=agent,
                states=states,
                feature_names=env.get_feature_names(),
                global_importance=ranking,
                action_names=env.get_action_names(),
                top_k=args.surrogate_top_k,
                ridge_alpha=args.ridge_alpha,
                seed=args.base_seed + seed,
                group_ids=episode_ids,
            )
            surrogate["seed"] = seed
            surrogate.to_csv(surrogate_path, index=False)
        surrogate_frames.append(surrogate)

    long_frame = pd.concat(all_frames, ignore_index=True)
    ranked_frame, summary_frame = _summarise_attributions(long_frame, args.top_k)
    long_frame.to_csv(output_dir / "output_shapley_long.csv", index=False)
    ranked_frame.to_csv(output_dir / "output_shapley_ranked.csv", index=False)
    summary_frame.to_csv(output_dir / "output_shapley_summary.csv", index=False)

    reference_mean = (
        long_frame.groupby(["seed", "output", "feature"], as_index=False)["global_importance"].mean()
    )
    seed_stability = _rank_stability(
        reference_mean,
        group_columns=["output"],
        comparison_column="seed",
        top_k=args.top_k,
        comparison_kind="training_seed",
    )
    estimator_stability = _rank_stability(
        long_frame,
        group_columns=["seed", "output"],
        comparison_column="reference_run",
        top_k=args.top_k,
        comparison_kind="stochastic_estimator_run",
    )
    stability = pd.concat([seed_stability, estimator_stability], ignore_index=True)
    stability.to_csv(output_dir / "output_shapley_stability.csv", index=False)

    corridor_long, corridor_summary = _corridor_attributions(ranked_frame, network, args.top_k)
    corridor_long.to_csv(output_dir / "corridor_shapley_long.csv", index=False)
    corridor_summary.to_csv(output_dir / "corridor_shapley_summary.csv", index=False)

    surrogate_metrics = pd.concat(surrogate_frames, ignore_index=True)
    surrogate_metrics.to_csv(output_dir / "surrogate_metrics_all_seeds.csv", index=False)
    requested_outputs = ["spend_fraction", *[f"allocation::{line}" for line in args.target_lines]]
    dominant_output = f"allocation::{args.target_lines[0]}"
    _plot_seed_corridor_stability(
        corridor_long,
        output=dominant_output,
        output_dir=output_dir,
        top_k=args.top_k,
    )
    _plot_surrogate_fidelity(surrogate_metrics, requested_outputs, output_dir)

    return {
        "n_checkpoints": len(selections),
        "training_seeds": [int(item["seed"]) for item in selections],
        "fixed_preference_weights": list(args.weights),
        "episodes_per_seed": args.episodes,
        "states_per_seed": {str(seed): int(len(cache[1])) for seed, cache in state_cache.items()},
        "permutation_samples": args.shapley_samples,
        "estimator_runs": args.reference_runs,
        "observed_reference_runs": args.reference_runs,
        "estimator_run_randomness": "observed-reference and permutation sampling jointly varied",
        "outputs": requested_outputs,
        "top_k": args.top_k,
        "mean_seed_spearman": _column_mean_or_nan(seed_stability, "spearman_rho"),
        "mean_seed_kendall": _column_mean_or_nan(seed_stability, "kendall_tau"),
        "mean_seed_top_k_overlap": _column_mean_or_nan(seed_stability, "top_k_overlap"),
        "mean_estimator_run_spearman": _column_mean_or_nan(estimator_stability, "spearman_rho"),
        "mean_estimator_run_kendall": _column_mean_or_nan(estimator_stability, "kendall_tau"),
        "mean_estimator_run_top_k_overlap": _column_mean_or_nan(estimator_stability, "top_k_overlap"),
    }


def _plot_counterfactual_effects(inference: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("total_cost_mean", "Total cost", "EUR / 24 h", 1.0),
        ("operating_cost_mean", "Operating cost", "EUR / 24 h", 1.0),
        ("investment_cost_mean", "Investment charge", "EUR / 24 h", 1.0),
        ("renewable_curtailment_mean", "Renewable curtailment", "MWh / 24 h", 1.0),
    ]
    metric_colours = [FH_BLUE] * len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.5))
    limits: list[tuple[float, float]] = []
    for ax, (metric, title, unit, scale), colour in zip(axes.flat, metrics, metric_colours):
        row = inference.loc[inference["metric"] == metric]
        if row.empty:
            ax.set_visible(False)
            continue
        row = row.iloc[0]
        # compare_paired_evaluations reports factual minus counterfactual.
        estimate = -float(row["mean_paired_difference"]) / scale
        lower = -float(row["crossed_bootstrap_ci_upper"]) / scale
        upper = -float(row["crossed_bootstrap_ci_lower"]) / scale
        ax.errorbar(
            [0],
            [estimate],
            yerr=[[estimate - lower], [upper - estimate]],
            fmt="o",
            color=colour,
            ecolor=colour,
            capsize=5,
            elinewidth=1.6,
            markersize=7.5,
        )
        ax.axhline(0.0, color=FH_GREY, linewidth=0.8, linestyle="--")
        ax.set_xlim(-0.7, 0.7)
        ax.set_xticks([0], ["Reallocated minus factual"])
        ax.set_ylabel(f"Difference ({unit})")
        ax.set_title(title, fontweight="normal")
        ax.grid(axis="y", alpha=0.25)
        limits.append((lower, upper))
    for row in range(2):
        row_limits = limits[row * 2 : row * 2 + 2]
        lower_extent = max(abs(min(lower, 0.0)) for lower, _ in row_limits)
        upper_extent = max(max(upper, 0.0) for _, upper in row_limits)
        extent = max(lower_extent, upper_extent, 1.0)
        for ax in axes[row, :]:
            ax.set_ylim(-1.12 * extent, 1.12 * extent)
    fig.tight_layout()
    fig.savefig(output_dir / "xai_strict_corridor_counterfactual.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "xai_strict_corridor_counterfactual.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _run_strict_counterfactual(args, manifest, selections, output_dir: Path) -> dict[str, Any]:
    factual_results: list[dict[str, Any]] = []
    reallocated_results: list[dict[str, Any]] = []
    factual_sources: list[str] = []
    zero_target_seeds: list[int] = []
    for selection_index, selection in enumerate(selections, start=1):
        seed = int(selection["seed"])
        force_reallocated = bool(args.force or seed in set(args.force_strict_seed))
        print(f"[strict] seed {seed} ({selection_index}/{len(selections)})")
        factual_path = output_dir / f"strict_seed_{seed:03d}_factual.json"
        reallocated_path = output_dir / f"strict_seed_{seed:03d}_reallocated.json"

        if factual_path.exists() and not args.force:
            factual = _read_json(factual_path)
            factual_sources.append(str(factual_path))
        else:
            existing_factual_path = (
                args.results_root
                / "stage9_full_environment"
                / "moppo"
                / Path(selection["run_dir"]).name
                / "test_evaluation.json"
            )
            can_reuse = (
                existing_factual_path.exists()
                and np.allclose(
                    np.asarray(selection.get("weights", []), dtype=float),
                    np.asarray(args.weights, dtype=float),
                )
            )
            if can_reuse:
                candidate_factual = _read_json(existing_factual_path)
                can_reuse = int(candidate_factual.get("n_episodes", 0)) == int(args.episodes)
            if can_reuse:
                factual = candidate_factual
                factual_sources.append(str(existing_factual_path))
            else:
                env = _build_test_environment(manifest, args.results_root, "full", seed)
                starts = stratified_episode_start_indices(
                    env.dataset.snapshots, env.config.episode_length, args.episodes
                )
                agent = load_agent(selection["checkpoint"], device=args.device)
                if hasattr(agent, "set_eval_preferences"):
                    agent.set_eval_preferences(np.asarray(args.weights, dtype=np.float32))
                factual = evaluate_agent(
                    agent,
                    env,
                    episodes=args.episodes,
                    deterministic=True,
                    episode_start_indices=starts,
                )
                factual_sources.append("new strict full-environment evaluation")
            _write_json(factual_path, factual)

        factual_target_mean = float(
            sum(
                factual.get("line_upgrades", {}).get(line, 0.0)
                for line in args.target_lines
            )
        )
        if factual_target_mean <= 1e-12:
            # The intervention is mathematically identical to the factual policy.
            # Re-solving the same OPF can nevertheless produce solver-tolerance
            # noise, so retain exact identity for this true no-op case.
            reallocated = factual
            zero_target_seeds.append(seed)
            _write_json(reallocated_path, reallocated)
        elif reallocated_path.exists() and not force_reallocated:
            reallocated = _read_json(reallocated_path)
        else:
            env = _build_test_environment(manifest, args.results_root, "full", seed)
            starts = stratified_episode_start_indices(
                env.dataset.snapshots, env.config.episode_length, args.episodes
            )
            base_agent = load_agent(selection["checkpoint"], device=args.device)
            if hasattr(base_agent, "set_eval_preferences"):
                base_agent.set_eval_preferences(np.asarray(args.weights, dtype=np.float32))
            intervention = CapacityPreservingAblationAgent(
                base_agent,
                env,
                target_lines=args.target_lines,
            )
            reallocated = evaluate_agent(
                intervention,
                env,
                episodes=args.episodes,
                deterministic=True,
                episode_start_indices=starts,
            )
            _write_json(reallocated_path, reallocated)

        factual_results.append(factual)
        reallocated_results.append(reallocated)
        print(f"[strict] seed {seed} complete")

    metrics = (
        "total_cost_mean",
        "investment_cost_mean",
        "annualized_investment_cost_mean",
        "operating_cost_mean",
        "renewable_share_mean",
        "renewable_curtailment_mean",
        "slack_generation_mean",
        "load_shedding_mean",
        "emissions_mean",
        "total_investment_mean",
    )
    inference = compare_paired_evaluations(
        factual_results,
        reallocated_results,
        label_a="factual",
        label_b="reallocated",
        metrics=metrics,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.base_seed,
    )
    inference.to_csv(output_dir / "strict_counterfactual_inference.csv", index=False)
    _plot_counterfactual_effects(inference, output_dir)

    targets = list(args.target_lines)
    factual_target = [
        float(sum(item.get("line_upgrades", {}).get(line, 0.0) for line in targets))
        for item in factual_results
    ]
    reallocated_target = [
        float(sum(item.get("line_upgrades", {}).get(line, 0.0) for line in targets))
        for item in reallocated_results
    ]
    factual_total = [
        float(sum(item.get("line_upgrades", {}).values()))
        for item in factual_results
    ]
    reallocated_total = [
        float(sum(item.get("line_upgrades", {}).values()))
        for item in reallocated_results
    ]
    total_differences = [
        reallocated - factual
        for factual, reallocated in zip(factual_total, reallocated_total)
    ]
    total_relative_differences = [
        difference / max(abs(factual), 1e-12)
        for difference, factual in zip(total_differences, factual_total)
    ]
    return {
        "target_lines": targets,
        "intervention": (
            "At each decision, remove all decoded increments on the three DC-selected Salzburg-area "
            "lines and redistribute the same MW over non-target corridors, first in proportion to "
            "the policy's other positive increments."
        ),
        "training_seeds": [int(item["seed"]) for item in selections],
        "fixed_preference_weights": list(args.weights),
        "episodes_per_seed": args.episodes,
        "factual_sources": factual_sources,
        "factual_target_mean_mw_by_seed": factual_target,
        "reallocated_target_mean_mw_by_seed": reallocated_target,
        "factual_total_mean_mw_by_seed": factual_total,
        "reallocated_total_mean_mw_by_seed": reallocated_total,
        "reallocated_minus_factual_total_mw_by_seed": total_differences,
        "maximum_absolute_total_mw_difference": float(max(map(abs, total_differences), default=0.0)),
        "maximum_absolute_relative_total_mw_difference": float(
            max(map(abs, total_relative_differences), default=0.0)
        ),
        "zero_target_seeds_copied_from_factual": zero_target_seeds,
        "inference_file": str(output_dir / "strict_counterfactual_inference.csv"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run robust multi-seed output-specific Shapley and strict corridor counterfactual analyses."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--shapley-samples", type=int, default=20)
    parser.add_argument(
        "--estimator-runs",
        "--reference-runs",
        dest="reference_runs",
        type=int,
        default=5,
        help="Repeated runs jointly varying observed-reference and permutation sampling.",
    )
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--surrogate-top-k", type=int, default=12)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--weights", type=float, nargs=3, default=CENTRAL_WEIGHTS)
    parser.add_argument("--target-lines", nargs=3, default=DEFAULT_TARGET_LINES)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of selected MO-PPO training seeds.",
    )
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--skip-shapley", action="store_true")
    parser.add_argument("--run-strict-counterfactual", action="store_true")
    parser.add_argument(
        "--force-strict-seed",
        type=int,
        action="append",
        default=[],
        help="Recompute only the reallocated strict evaluation for this seed; may be repeated.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.results_root = args.results_root.resolve()
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.results_root / "stage10_robust_explainability"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(args.results_root / "pipeline_manifest.json")
    selections = _selected_moppo_checkpoints(args.results_root)
    if args.seeds:
        requested_seeds = set(args.seeds)
        selections = [item for item in selections if int(item["seed"]) in requested_seeds]
        found_seeds = {int(item["seed"]) for item in selections}
        missing_seeds = requested_seeds - found_seeds
        if missing_seeds:
            raise ValueError(f"No selected MO-PPO checkpoint for seeds {sorted(missing_seeds)}.")

    robust_manifest_path = args.output_dir / "robust_explainability_manifest.json"
    manifest_payload: dict[str, Any] = (
        _read_json(robust_manifest_path)
        if robust_manifest_path.exists() and not args.force
        else {}
    )
    manifest_payload.update({
        "method": "multi-seed output-specific interventional permutation Shapley",
        "results_root": str(args.results_root),
        "output_dir": str(args.output_dir),
    })
    if not args.skip_shapley:
        manifest_payload["shapley"] = _run_shapley_audit(
            args, manifest, selections, args.output_dir
        )
    if args.run_strict_counterfactual:
        manifest_payload["strict_counterfactual"] = _run_strict_counterfactual(
            args, manifest, selections, args.output_dir
        )
    _write_json(robust_manifest_path, manifest_payload)
    print(json.dumps(manifest_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
