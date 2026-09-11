from __future__ import annotations

"""Dense deployment-preference diagnostic for conditioned MO-PPO checkpoints."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from thesis_pipeline_utils import build_dataset, build_env
from tep_rl.evaluation import evaluate_agent
from tep_rl.line_metadata import endpoint_label, line_metadata_frame
from tep_rl.ppo import load_agent
from tep_rl.thesis_formulations import (
    DEFAULT_THESIS_FORMULATION_PRESET,
    THESIS_FORMULATION_PRESETS,
    apply_thesis_formulation_preset,
)


def simplex_grid(resolution: int, objective_dim: int = 3) -> list[tuple[float, ...]]:
    if objective_dim != 3:
        raise ValueError("This diagnostic currently expects three reward objectives.")
    if resolution < 1:
        raise ValueError("--grid-resolution must be >= 1")
    weights: list[tuple[float, ...]] = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            weights.append((i / resolution, j / resolution, k / resolution))
    # Put thesis-like balanced weights first visually, then extremes remain present.
    return sorted(weights, key=lambda w: (abs(w[0] - w[1]) + abs(w[1] - w[2]), w))


def parse_weights(text: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in re.split(r"[\s,]+", text.strip()) if part)
    total = sum(values)
    if total <= 0:
        raise ValueError("Weights must have positive sum.")
    return tuple(value / total for value in values)


def load_selected_checkpoints(path: Path, best_only: bool = True, max_policies: int | None = None) -> list[Path]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for item in payload:
        checkpoint = item.get("checkpoint")
        if checkpoint:
            rows.append((float(item.get("selection_score", float("-inf"))), Path(checkpoint)))
    rows.sort(key=lambda item: item[0], reverse=True)
    unique: list[Path] = []
    seen: set[str] = set()
    for _, checkpoint in rows:
        key = str(checkpoint)
        if key in seen:
            continue
        seen.add(key)
        unique.append(checkpoint)
        if best_only:
            break
        if max_policies is not None and len(unique) >= max_policies:
            break
    return unique


def metric_value(evaluation: dict[str, Any], key: str) -> float:
    if key in evaluation:
        return float(evaluation[key])
    action_stats = evaluation.get("action_stats", {})
    if isinstance(action_stats, dict) and key == "total_investment_mean":
        return float(action_stats.get("total_investment_mean", np.nan))
    if key == "active_lines_mean":
        upgrades = evaluation.get("line_upgrades", {})
        if isinstance(upgrades, dict):
            return float(sum(float(value) > 1e-6 for value in upgrades.values()))
    return float("nan")


def nondominated_minimize(frame: pd.DataFrame, objective_cols: list[str]) -> pd.Series:
    values = frame[objective_cols].to_numpy(dtype=float)
    keep = np.ones(len(frame), dtype=bool)
    for i, candidate in enumerate(values):
        for j, challenger in enumerate(values):
            if i == j:
                continue
            no_worse = np.all(challenger <= candidate + 1e-12)
            strictly_better = np.any(challenger < candidate - 1e-12)
            if no_worse and strictly_better:
                keep[i] = False
                break
    return pd.Series(keep, index=frame.index)


def action_signature(line_upgrades: dict[str, float], precision_mw: float = 1.0) -> str:
    active = [
        f"{line}:{round(float(value) / precision_mw) * precision_mw:.0f}"
        for line, value in sorted(line_upgrades.items())
        if float(value) > 1e-6
    ]
    return "|".join(active) if active else "zero"


def plot_preference_scatter(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    marker_sizes = 70 + 1.4 * frame["total_investment_mean"].fillna(0.0)
    for ax, y_col, y_label in [
        (axes[0], "grid_stress_mean", "Grid stress (lower better)"),
        (axes[1], "renewable_curtailment_mean", "Curtailment (MWh, lower better)"),
    ]:
        scatter = ax.scatter(
            frame["total_cost_mean"] / 1e6,
            frame[y_col],
            c=frame["renewable_share_mean"],
            s=marker_sizes,
            cmap="viridis",
            edgecolor=np.where(frame["nondominated"], "black", "0.65"),
            linewidth=np.where(frame["nondominated"], 1.0, 0.4),
            alpha=0.88,
        )
        for _, row in frame.iterrows():
            if bool(row["nondominated"]):
                ax.annotate(row["weight_label"], (row["total_cost_mean"] / 1e6, row[y_col]), xytext=(5, 5), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Total cost (M)")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.82)
    cbar.set_label("Renewable share")
    fig.suptitle("Dense MO-PPO preference diagnostic", y=1.02)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_action_diversity(upgrade_frame: pd.DataFrame, output_path: Path, top_k: int = 8) -> None:
    if upgrade_frame.empty:
        return
    top_lines = (
        upgrade_frame.groupby("line", as_index=False)["upgrade_mw"]
        .mean()
        .sort_values("upgrade_mw", ascending=False)
        .head(top_k)["line"]
        .tolist()
    )
    plot_frame = upgrade_frame[upgrade_frame["line"].isin(top_lines)]
    pivot = plot_frame.pivot_table(index="weight_label", columns="line_label", values="upgrade_mw", aggfunc="mean").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.32 * len(pivot))))
    left = np.zeros(len(pivot))
    for column in pivot.columns:
        ax.barh(pivot.index, pivot[column], left=left, label=column[:46])
        left += pivot[column].to_numpy()
    ax.set_xlabel("Mean upgrade (MW)")
    ax.set_ylabel("Preference weights")
    ax.set_title("Action diversity across preference weights")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an existing MO-PPO checkpoint over a dense preference grid.")
    parser.add_argument(
        "--formulation-preset",
        choices=sorted(THESIS_FORMULATION_PRESETS),
        default=DEFAULT_THESIS_FORMULATION_PRESET,
        help="Apply a shared thesis formulation preset unless the specific budget/candidate flags are overridden.",
    )
    parser.add_argument("--network", default="derived/austria_net_physical_ratings.nc")
    parser.add_argument("--load", default="data/entsoe_at_load_2015_2024_opsd.csv")
    parser.add_argument("--wind", default="data/wind_at_2015_2024.csv")
    parser.add_argument("--solar", default="data/solar_at_2015_2024.csv")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--candidate-lines", type=int, default=60)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--selected-checkpoints", type=Path, default=None)
    parser.add_argument("--all-selected", action="store_true", help="Evaluate all unique selected checkpoints instead of only the best one.")
    parser.add_argument("--max-policies", type=int, default=5)
    parser.add_argument("--grid-resolution", type=int, default=5, help="Simplex denominator; 5 gives 21 three-objective weights.")
    parser.add_argument("--extra-weight", action="append", default=[], help="Additional weight tuple, e.g. 0.34,0.33,0.33")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--env-mode", choices=["proxy", "full"], default="proxy")
    parser.add_argument("--output-dir", default="results/thesis_final/analysis/preference_diagnostic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episode-length", type=int, default=24)
    parser.add_argument("--decision-interval", type=int, default=6)
    parser.add_argument("--temporal-mode", choices=["decision_block", "hourly"], default="decision_block")
    parser.add_argument("--max-upgrade-mw", type=float, default=60.0)
    parser.add_argument("--budget-mw", type=float, default=240.0)
    parser.add_argument("--budget-release", choices=["linear", "all_at_once"], default="linear")
    parser.add_argument("--action-mode", choices=["budgeted", "direct"], default="budgeted")
    parser.add_argument("--allocation-sharpness", type=float, default=12.0)
    parser.add_argument("--allocation-sparsity-cutoff", type=float, default=0.70)
    parser.add_argument("--stability-margin", type=float, default=0.70)
    parser.add_argument("--proxy-balance-mode", choices=["demand_proportional", "single_slack"], default="demand_proportional")
    parser.add_argument("--proxy-dispatch-limit", type=float, default=None)
    parser.add_argument("--load-shedding-cost", type=float, default=1e5)
    parser.add_argument("--cost-reward-scale", type=float, default=1e7)
    parser.add_argument("--overload-reward-scale", type=float, default=175.0)
    parser.add_argument("--third-objective-mode", choices=["renewable_share", "curtailment", "emissions"], default="renewable_share")
    parser.add_argument("--curtailment-reward-scale", type=float, default=1000.0)
    parser.add_argument("--emissions-reward-scale", type=float, default=1000.0)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--disable-full-env-fallback-to-proxy", action="store_true")
    raw_argv = sys.argv[1:]
    args = parser.parse_args(raw_argv)
    apply_thesis_formulation_preset(args, raw_argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [Path(path) for path in args.checkpoint]
    if args.selected_checkpoints:
        checkpoints.extend(
            load_selected_checkpoints(
                args.selected_checkpoints,
                best_only=not args.all_selected,
                max_policies=args.max_policies,
            )
        )
    if not checkpoints:
        raise SystemExit("Provide --checkpoint or --selected-checkpoints.")
    checkpoints = list(dict.fromkeys(checkpoints))

    weights = simplex_grid(args.grid_resolution)
    for text in args.extra_weight:
        weight = parse_weights(text)
        if weight not in weights:
            weights.append(weight)
    weights = sorted(weights, key=lambda w: (w[0], w[1], w[2]))

    dataset = build_dataset(
        network=Path(args.network),
        load=Path(args.load),
        wind=Path(args.wind),
        solar=Path(args.solar),
        start=args.start,
        end=args.end,
        candidate_lines=args.candidate_lines,
    )

    rows: list[dict[str, Any]] = []
    upgrade_rows: list[dict[str, Any]] = []
    for policy_idx, checkpoint in enumerate(checkpoints):
        agent = load_agent(checkpoint, device=args.device)
        checkpoint_label = checkpoint.parent.parent.name
        for weight_idx, weight in enumerate(weights):
            if hasattr(agent, "set_eval_preferences"):
                agent.set_eval_preferences(weight)
            env = build_env(
                dataset,
                env_mode=args.env_mode,
                episode_length=args.episode_length,
                max_upgrade_mw=args.max_upgrade_mw,
                budget_mw=args.budget_mw,
                decision_interval=args.decision_interval,
                temporal_mode=args.temporal_mode,
                budget_release=args.budget_release,
                action_mode=args.action_mode,
                allocation_sharpness=args.allocation_sharpness,
                allocation_sparsity_cutoff=args.allocation_sparsity_cutoff,
                stability_margin=args.stability_margin,
                proxy_balance_mode=args.proxy_balance_mode,
                proxy_dispatch_limit=args.proxy_dispatch_limit,
                load_shedding_cost=args.load_shedding_cost,
                cost_reward_scale=args.cost_reward_scale,
                overload_reward_scale=args.overload_reward_scale,
                third_objective_mode=args.third_objective_mode,
                curtailment_reward_scale=args.curtailment_reward_scale,
                emissions_reward_scale=args.emissions_reward_scale,
                solver=args.solver,
                full_env_fallback_to_proxy=not args.disable_full_env_fallback_to_proxy,
                seed=7000 + policy_idx * 1000 + weight_idx,
            )
            print(f"Evaluating {checkpoint_label} at weights {weight}")
            evaluation = evaluate_agent(agent, env, episodes=args.episodes, deterministic=True)
            line_upgrades = {str(line): float(value) for line, value in evaluation.get("line_upgrades", {}).items()}
            row = {
                "checkpoint": str(checkpoint),
                "checkpoint_label": checkpoint_label,
                "weight_cost": weight[0],
                "weight_stress": weight[1],
                "weight_renewable": weight[2],
                "weight_label": "/".join(f"{part:.2f}" for part in weight),
                "action_signature": action_signature(line_upgrades),
                "unique_active_lines": sum(value > 1e-6 for value in line_upgrades.values()),
                "episodes": args.episodes,
                "env_mode": args.env_mode,
            }
            for key in (
                "total_cost_mean",
                "grid_stress_mean",
                "renewable_share_mean",
                "renewable_curtailment_mean",
                "load_shedding_mean",
                "emissions_mean",
                "total_investment_mean",
                "active_lines_mean",
            ):
                row[key] = metric_value(evaluation, key)
            rows.append(row)
            metadata = line_metadata_frame(dataset.network, line_upgrades.keys())
            label_by_line = dict(zip(metadata.get("line", []), metadata.get("line_label", [])))
            for line, value in line_upgrades.items():
                upgrade_rows.append(
                    {
                        "checkpoint_label": checkpoint_label,
                        "weight_label": row["weight_label"],
                        "weight_cost": weight[0],
                        "weight_stress": weight[1],
                        "weight_renewable": weight[2],
                        "line": line,
                        "line_label": label_by_line.get(line, endpoint_label(dataset.network, line) if line in dataset.network.lines.index else line),
                        "upgrade_mw": value,
                    }
                )

    frame = pd.DataFrame(rows)
    frame["negative_renewable_share_mean"] = -frame["renewable_share_mean"]
    frame["nondominated"] = nondominated_minimize(
        frame,
        [
            "total_cost_mean",
            "grid_stress_mean",
            "renewable_curtailment_mean",
            "total_investment_mean",
            "negative_renewable_share_mean",
        ],
    )
    upgrade_frame = pd.DataFrame(upgrade_rows)

    frame.to_csv(output_dir / "dense_preference_results.csv", index=False)
    upgrade_frame.to_csv(output_dir / "dense_preference_line_upgrades.csv", index=False)

    summary = {
        "n_checkpoints": len(checkpoints),
        "n_weights": len(weights),
        "n_evaluations": len(frame),
        "episodes": args.episodes,
        "env_mode": args.env_mode,
        "unique_action_signatures": int(frame["action_signature"].nunique()),
        "nondominated_points": int(frame["nondominated"].sum()),
        "metric_ranges": {
            col: {"min": float(frame[col].min()), "max": float(frame[col].max())}
            for col in [
                "total_cost_mean",
                "grid_stress_mean",
                "renewable_share_mean",
                "renewable_curtailment_mean",
                "total_investment_mean",
                "active_lines_mean",
            ]
        },
    }
    (output_dir / "dense_preference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_preference_scatter(frame, output_dir / "dense_preference_scatter.png")
    plot_action_diversity(upgrade_frame, output_dir / "dense_preference_action_diversity.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
