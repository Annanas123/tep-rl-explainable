from __future__ import annotations

"""Generate editable thesis figure variants from completed experiment outputs.

All figure assembly is deterministic and file-based so the thesis visuals can
be refreshed after wording, styling, or result-layout changes without touching
the underlying experiments.
"""

import argparse
import json
import logging
import re
import sys
from textwrap import fill
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import pypsa

logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 14,
        "axes.labelsize": 15,
        "axes.titleweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.titlesize": 14,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_layout import locate_results_dir
from thesis_pipeline_utils import load_seed_histories
from tep_rl.line_metadata import (
    anchor_points_for_lines,
    annotate_anchor_points,
    apply_geo_aspect,
    draw_country_outline,
    endpoint_label,
    endpoint_anchor_label,
    feature_display_name,
    line_metadata_frame,
    parse_linestring,
    plot_line_importance_map,
)
from tep_rl.visualization import plot_multi_seed_curves


POLICY_ORDER = ["zero", "ppo", "moppo", "uniform", "myopic_proxy"]
POLICY_LABELS = {
    "zero": "Zero",
    "ppo": "PPO",
    "moppo": "MO-PPO",
    "uniform": "Uniform",
    "myopic_proxy": "Myopic proxy",
    "top_1_excess": "Top-1 excess",
    "top_3_excess": "Top-3 excess",
    "proportional_excess": "Proportional excess",
}
COLORS = {
    "zero": "#7f858c",
    "ppo": "#6f9fcf",
    "moppo": "#e5a35c",
    "uniform": "#79b96f",
    "myopic_proxy": "#3f8f86",
    "neutral": "#b9c1ca",
    "stress": "#e5a35c",
    "cost": "#6f9fcf",
    "renewable": "#79b96f",
    "curtailment": "#3f8f86",
}
ANNOTATION_FONT_SIZE = 12
COMPACT_FONT_SIZE = 11


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    policy_order = {policy: idx for idx, policy in enumerate(POLICY_ORDER)}
    return (
        frame[frame["policy"].isin(POLICY_ORDER)]
        .assign(_order=lambda df: df["policy"].map(policy_order))
        .sort_values("_order")
        .drop(columns="_order")
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"}
    if path.suffix.lower() == ".pdf":
        fig.savefig(path, **save_kwargs)
    else:
        fig.savefig(path, dpi=300, **save_kwargs)
        fig.savefig(path.with_suffix(".pdf"), **save_kwargs)
    plt.close(fig)


def _display_labels(frame: pd.DataFrame) -> pd.Series:
    existing = frame["Policy"] if "Policy" in frame.columns else frame["policy"]
    return frame["policy"].map(POLICY_LABELS).fillna(existing)


def _annotate_vertical_bars(
    ax: plt.Axes,
    container,
    fmt: str = "{:.1f}",
    small_value_threshold: float = 0.02,
    skip_small: bool = False,
    rotation: float = 0,
) -> None:
    ymax = ax.get_ylim()[1]
    ymin = ax.get_ylim()[0]
    span = ymax - ymin if ymax > ymin else 1.0
    for patch in container:
        height = float(patch.get_height())
        x = patch.get_x() + patch.get_width() / 2.0
        if np.isnan(height):
            continue
        if abs(height) <= small_value_threshold:
            if skip_small:
                continue
            baseline = 0.0 if ymin <= 0.0 <= ymax else ymin
            y = baseline + 0.015 * span
            va = "bottom"
        elif height >= 0:
            y = height + 0.015 * span
            va = "bottom"
        else:
            y = height - 0.015 * span
            va = "top"
        ax.text(x, y, fmt.format(height), ha="center", va=va, fontsize=COMPACT_FONT_SIZE, rotation=rotation)


def _pad_axis_y(ax: plt.Axes, top_fraction: float = 0.16, bottom_fraction: float = 0.04) -> None:
    ymin, ymax = ax.get_ylim()
    if not np.isfinite(ymin) or not np.isfinite(ymax) or np.isclose(ymin, ymax):
        return
    span = ymax - ymin
    ax.set_ylim(ymin - bottom_fraction * span, ymax + top_fraction * span)


def _pad_axis_x(ax: plt.Axes, left_fraction: float = 0.04, right_fraction: float = 0.08) -> None:
    xmin, xmax = ax.get_xlim()
    if not np.isfinite(xmin) or not np.isfinite(xmax) or np.isclose(xmin, xmax):
        return
    span = xmax - xmin
    ax.set_xlim(xmin - left_fraction * span, xmax + right_fraction * span)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {word}"


def _jitter_duplicate_xy(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    radius_x_fraction: float = 0.018,
    radius_y_fraction: float = 0.028,
) -> pd.DataFrame:
    """Return display coordinates with deterministic small offsets for duplicate outcomes."""
    plot_frame = frame.copy()
    x_range = max(float(plot_frame[x_col].max() - plot_frame[x_col].min()), 1e-6)
    y_range = max(float(plot_frame[y_col].max() - plot_frame[y_col].min()), 1e-6)
    plot_frame["_plot_x"] = plot_frame[x_col].astype(float)
    plot_frame["_plot_y"] = plot_frame[y_col].astype(float)
    for _, group in plot_frame.groupby([x_col, y_col], sort=False):
        if len(group) <= 1:
            continue
        angles = np.linspace(0.0, 2.0 * np.pi, len(group), endpoint=False)
        radius_x = radius_x_fraction * x_range
        radius_y = radius_y_fraction * y_range
        for idx, angle in zip(group.index, angles):
            plot_frame.loc[idx, "_plot_x"] += radius_x * np.cos(angle)
            plot_frame.loc[idx, "_plot_y"] += radius_y * np.sin(angle)
    return plot_frame


def _format_feature_label(label: str) -> str:
    if ": " in label:
        label = label.replace(": ", ":\n", 1)
    if " -> " in label and len(label) > 36:
        label = label.replace(" -> ", "\n-> ", 1)
    return fill(label, width=30, break_long_words=False, break_on_hyphens=False)


def _format_small_number(value: float) -> str:
    if value == 0:
        return "0"
    text = f"{value:.2e}"
    return text.replace("e-0", "e-").replace("e+0", "e+")


def _ci_error(frame: pd.DataFrame, metric: str) -> np.ndarray:
    lower = frame[f"{metric}_mean"].to_numpy() - frame[f"{metric}_ci_lower"].to_numpy()
    upper = frame[f"{metric}_ci_upper"].to_numpy() - frame[f"{metric}_mean"].to_numpy()
    return np.vstack([np.maximum(lower, 0.0), np.maximum(upper, 0.0)])


def regenerate_learning_curves(results_root: Path, output_dir: Path) -> None:
    ppo_histories = load_seed_histories(locate_results_dir(results_root, "ppo_validation"), deduplicate_by_seed=True)
    moppo_histories = load_seed_histories(locate_results_dir(results_root, "moppo_validation"), deduplicate_by_seed=True)
    plot_multi_seed_curves(
        ppo_histories,
        output_dir / "ppo_learning_curves_clean.png",
        label="PPO",
        smoothing_window=151,
        min_valid_seeds=len(ppo_histories),
    )
    plot_multi_seed_curves(
        moppo_histories,
        output_dir / "moppo_learning_curves_clean.png",
        label="MO-PPO",
        smoothing_window=151,
        min_valid_seeds=len(moppo_histories),
    )
    pd.DataFrame(
        [
            {"agent": "PPO", "deduplicated_seed_runs": len(ppo_histories)},
            {"agent": "MO-PPO", "deduplicated_seed_runs": len(moppo_histories)},
        ]
    ).to_csv(output_dir / "learning_curve_metadata.csv", index=False)


def plot_fullenv_delta_vs_zero(analysis_dir: Path, output_dir: Path) -> None:
    frame = _ordered(pd.read_csv(analysis_dir / "fullenv_policy_comparison.csv"))
    zero = frame.loc[frame["policy"] == "zero"].iloc[0]
    metrics = [
        ("total_cost", "Total cost", "lower"),
        ("load_shedding", "Load shedding", "lower"),
        ("renewable_curtailment", "Curtailment", "lower"),
        ("emissions", "Emissions", "lower"),
        ("renewable_share", "Renewable share", "higher"),
    ]
    rows = []
    for _, row in frame.iterrows():
        display_label = POLICY_LABELS.get(str(row["policy"]), str(row.get("Policy", row["policy"])))
        for metric, label, direction in metrics:
            baseline = float(zero[f"{metric}_mean"])
            value = float(row[f"{metric}_mean"])
            if abs(baseline) < 1e-12:
                delta = np.nan
            elif direction == "higher":
                delta = 100.0 * (value - baseline) / abs(baseline)
            else:
                delta = 100.0 * (baseline - value) / abs(baseline)
            rows.append({"policy": row["policy"], "Policy": display_label, "metric": label, "delta_pct": delta})
    delta_frame = pd.DataFrame(rows)
    delta_frame.to_csv(output_dir / "fullenv_delta_vs_zero.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    selected = delta_frame[delta_frame["policy"].isin(["ppo", "moppo", "uniform", "myopic_proxy"])]
    pivot = selected.pivot(index="metric", columns="Policy", values="delta_pct").loc[[m[1] for m in metrics]]
    x = np.arange(len(pivot.index))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(pivot.columns))
    color_map = {
        "PPO": COLORS["ppo"],
        "MO-PPO": COLORS["moppo"],
        "Uniform": COLORS["uniform"],
        "Myopic proxy": COLORS["myopic_proxy"],
    }
    for offset, column in zip(offsets, pivot.columns):
        bars = ax.bar(x + offset, pivot[column], width=width, label=column, color=color_map.get(column, None), alpha=0.88)
        _annotate_vertical_bars(ax, bars, fmt="{:.2f}", small_value_threshold=0.05, skip_small=True, rotation=90)
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_ylabel("Improvement Relative to Zero (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    _pad_axis_y(ax, top_fraction=0.20, bottom_fraction=0.08)
    fig.tight_layout()
    _save(fig, output_dir / "fullenv_delta_vs_zero.png")


def plot_proxy_and_full_context(analysis_dir: Path, output_dir: Path) -> None:
    proxy = _ordered(pd.read_csv(analysis_dir / "proxy_policy_comparison.csv"))
    full = _ordered(pd.read_csv(analysis_dir / "fullenv_policy_comparison.csv"))
    zero_proxy = float(proxy.loc[proxy["policy"] == "zero", "grid_stress_mean"].iloc[0])
    proxy = proxy.assign(stress_reduction_pct=100.0 * (zero_proxy - proxy["grid_stress_mean"]) / zero_proxy)
    proxy = proxy.assign(display_label=_display_labels(proxy))
    full = full.assign(display_label=_display_labels(full))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    plot_proxy = proxy[proxy["policy"].isin(POLICY_ORDER)]
    proxy_bars = axes[0].bar(
        plot_proxy["display_label"],
        plot_proxy["stress_reduction_pct"],
        color=[COLORS.get(policy, "#999999") for policy in plot_proxy["policy"]],
        alpha=0.85,
    )
    axes[0].set_ylabel("Proxy Stress Reduction Relative to Zero (%)")
    axes[0].set_title("Proxy Learning Signal")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    _annotate_vertical_bars(axes[0], proxy_bars, fmt="{:.1f}")
    _pad_axis_y(axes[0], top_fraction=0.18, bottom_fraction=0.04)

    plot_full = full[full["policy"].isin(["ppo", "moppo", "zero", "uniform", "myopic_proxy"])]
    load_err = _ci_error(plot_full, "load_shedding")
    curt_err = _ci_error(plot_full, "renewable_curtailment")
    for pos, (_, row) in enumerate(plot_full.iterrows()):
        axes[1].errorbar(
            row["load_shedding_mean"],
            row["renewable_curtailment_mean"],
            xerr=load_err[:, [pos]],
            yerr=curt_err[:, [pos]],
            fmt="o",
            markersize=7,
            color=COLORS.get(row["policy"], "0.25"),
            ecolor="0.70",
            markeredgecolor="black",
            markeredgewidth=0.5,
            capsize=2,
        )
    label_offsets = {
        "Zero": (8, 8),
        "PPO": (8, -14),
        "MO-PPO": (8, 12),
        "Uniform": (8, -2),
        "Myopic proxy": (8, 22),
        "Zero / MO-PPO /\nMyopic proxy": (8, 12),
    }
    label_groups = (
        plot_full.assign(
            _label_x=plot_full["load_shedding_mean"].round(2),
            _label_y=plot_full["renewable_curtailment_mean"].round(2),
        )
        .groupby(["_label_x", "_label_y"], sort=False)
        .agg(
            load_shedding_mean=("load_shedding_mean", "first"),
            renewable_curtailment_mean=("renewable_curtailment_mean", "first"),
            labels=("display_label", list),
        )
        .reset_index(drop=True)
    )
    label_groups["display_label"] = label_groups["labels"].map(
        lambda labels: " / ".join(labels[:2]) + (" /\n" + " / ".join(labels[2:]) if len(labels) > 2 else "")
    )
    for _, row in label_groups.iterrows():
        dx, dy = label_offsets.get(str(row["display_label"]), (6, 5))
        axes[1].annotate(
            row["display_label"],
            (row["load_shedding_mean"], row["renewable_curtailment_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=ANNOTATION_FONT_SIZE,
        )
    axes[1].set_xlabel("Load Shedding (MWh, Lower Is Better)")
    axes[1].set_ylabel("Renewable Curtailment (MWh, Lower Is Better)")
    axes[1].set_title("Full-Environment Operational Trade-Off")
    axes[1].grid(alpha=0.25)

    x_min = float(plot_full["load_shedding_mean"].min()) - 18.0
    x_max = float(plot_full["load_shedding_mean"].max()) + 12.0
    y_min = float(plot_full["renewable_curtailment_mean"].min()) - 1.8
    y_max = float(plot_full["renewable_curtailment_mean"].max()) + 2.4
    fig.tight_layout(w_pad=3.0)
    _save(fig, output_dir / "proxy_full_context_clean.png")

    fig2, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(12.6, 5.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
    for ax in [ax_main, ax_zoom]:
        for pos, (_, row) in enumerate(plot_full.iterrows()):
            ax.errorbar(
                row["load_shedding_mean"],
                row["renewable_curtailment_mean"],
                xerr=load_err[:, [pos]],
                yerr=curt_err[:, [pos]],
                fmt="o",
                markersize=7 if ax is ax_main else 5.0,
                color=COLORS.get(row["policy"], "0.25"),
                ecolor="0.70",
                markeredgecolor="black",
                markeredgewidth=0.5,
                capsize=3 if ax is ax_main else 2,
            )
        ax.set_xlabel("Load Shedding (MWh, Lower Is Better)")
        ax.set_ylabel("Renewable Curtailment (MWh, Lower Is Better)")
        ax.grid(alpha=0.25)

    load_lower = (plot_full["load_shedding_mean"] - (plot_full["load_shedding_mean"] - plot_full["load_shedding_ci_lower"])).to_numpy()
    load_upper = (plot_full["load_shedding_mean"] + (plot_full["load_shedding_ci_upper"] - plot_full["load_shedding_mean"])).to_numpy()
    curt_lower = (plot_full["renewable_curtailment_mean"] - (plot_full["renewable_curtailment_mean"] - plot_full["renewable_curtailment_ci_lower"])).to_numpy()
    curt_upper = (plot_full["renewable_curtailment_mean"] + (plot_full["renewable_curtailment_ci_upper"] - plot_full["renewable_curtailment_mean"])).to_numpy()
    full_x_min = float(np.nanmin(load_lower)) - 30.0
    full_x_max = float(np.nanmax(load_upper)) + 30.0
    full_y_min = float(np.nanmin(curt_lower)) - 10.0
    full_y_max = float(np.nanmax(curt_upper)) + 14.0
    ax_main.set_xlim(full_x_min, full_x_max)
    ax_main.set_ylim(full_y_min, full_y_max)
    ax_main.set_title("Full View")
    for _, row in label_groups.iterrows():
        dx, dy = label_offsets.get(str(row["display_label"]), (6, 5))
        ax_main.annotate(
            row["display_label"],
            (row["load_shedding_mean"], row["renewable_curtailment_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=ANNOTATION_FONT_SIZE,
        )

    ax_zoom.set_xlim(x_min, x_max)
    ax_zoom.set_ylim(y_min, y_max)
    ax_zoom.set_title("Zoomed Comparison")
    for _, row in label_groups.iterrows():
        dx, dy = label_offsets.get(str(row["display_label"]), (4, 3))
        ax_zoom.annotate(
            row["display_label"],
            (row["load_shedding_mean"], row["renewable_curtailment_mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=COMPACT_FONT_SIZE,
        )

    fig2.tight_layout()
    _save(fig2, output_dir / "fullenv_operational_tradeoff_zoom.png")


def plot_additional_metrics(analysis_dir: Path, output_dir: Path) -> None:
    frame = _ordered(pd.read_csv(analysis_dir / "fullenv_policy_comparison.csv"))
    zero = frame.loc[frame["policy"] == "zero"].iloc[0]
    summary = frame[
        [
            "policy",
            "Policy",
            "total_cost_mean",
            "total_investment_mean",
            "active_lines_mean",
            "invested_line_fraction_mean",
            "load_shedding_mean",
            "renewable_curtailment_mean",
            "renewable_share_mean",
            "emissions_mean",
            "slack_generation_mean",
        ]
    ].copy()
    summary["total_cost_million"] = summary["total_cost_mean"] / 1e6
    summary["invested_line_fraction_pct"] = 100.0 * summary["invested_line_fraction_mean"]
    summary["cost_improvement_vs_zero_pct"] = 100.0 * (float(zero["total_cost_mean"]) - summary["total_cost_mean"]) / float(zero["total_cost_mean"])
    summary["curtailment_reduction_vs_zero_pct"] = 100.0 * (float(zero["renewable_curtailment_mean"]) - summary["renewable_curtailment_mean"]) / float(zero["renewable_curtailment_mean"])
    summary.to_csv(output_dir / "additional_metrics_summary.csv", index=False)
    latex_cols = [
        "Policy",
        "total_cost_million",
        "total_investment_mean",
        "active_lines_mean",
        "invested_line_fraction_pct",
        "renewable_share_mean",
        "emissions_mean",
    ]
    summary[latex_cols].to_latex(
        output_dir / "additional_metrics_summary.tex",
        index=False,
        float_format=lambda value: f"{value:.2f}",
        caption="Additional full-environment planning metrics.",
        label="tab:additional_planning_metrics",
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.25))
    plot_frame = summary[summary["policy"].isin(["zero", "ppo", "moppo", "uniform", "myopic_proxy"])]
    display = plot_frame["policy"].map(POLICY_LABELS).fillna(plot_frame["Policy"])
    bars0 = axes[0].bar(display, plot_frame["total_investment_mean"], color=[COLORS.get(p, "#999") for p in plot_frame["policy"]])
    axes[0].set_ylabel("Investment (MW)")
    axes[0].set_title("Budget Use")
    bars1 = axes[1].bar(display, plot_frame["active_lines_mean"], color=[COLORS.get(p, "#999") for p in plot_frame["policy"]])
    axes[1].set_ylabel("Active Lines")
    axes[1].set_title("Active Reinforced Lines")
    bars2 = axes[2].bar(display, plot_frame["curtailment_reduction_vs_zero_pct"], color=[COLORS.get(p, "#999") for p in plot_frame["policy"]])
    axes[2].set_ylabel("Curtailment Reduction Relative to Zero (%)")
    axes[2].set_title("Renewable Integration")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    _annotate_vertical_bars(axes[0], bars0, fmt="{:.2f}")
    _annotate_vertical_bars(axes[1], bars1, fmt="{:.2f}")
    _annotate_vertical_bars(axes[2], bars2, fmt="{:.2f}", small_value_threshold=0.005)
    for ax in axes:
        _pad_axis_y(ax, top_fraction=0.20, bottom_fraction=0.04)
    fig.tight_layout()
    _save(fig, output_dir / "additional_metrics_dashboard.png")


def plot_pareto_tradeoff(results_root: Path, output_dir: Path) -> None:
    with (locate_results_dir(results_root, "scalarization_sweep") / "sweep_summary.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = []
    for weights, evaluation in payload.items():
        rows.append(
            {
                "weights": weights,
                "cost_million": float(evaluation["total_cost_mean"]) / 1e6,
                "grid_stress": float(evaluation["grid_stress_mean"]),
                "renewable_share": float(evaluation["renewable_share_mean"]),
                "curtailment_mwh": float(evaluation["renewable_curtailment_mean"]),
                "investment_mw": float(evaluation.get("action_stats", {}).get("total_investment_mean", np.nan)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "pareto_tradeoff_points.csv", index=False)

    objective_cols = ["cost_million", "grid_stress", "renewable_share", "curtailment_mwh"]
    if frame[objective_cols].drop_duplicates().shape[0] == 1:
        row = frame.iloc[0]
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.scatter([row["cost_million"]], [row["grid_stress"]], s=220, color=COLORS["moppo"], edgecolor="black", linewidth=0.8)
        ax.annotate(
            "All tested scalarisation weights\nselect the same evaluated policy",
            (row["cost_million"], row["grid_stress"]),
            xytext=(25, 25),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "0.25"},
            fontsize=ANNOTATION_FONT_SIZE,
        )
        weights_text = "\n".join(frame["weights"].tolist())
        ax.text(
            0.02,
            0.98,
            f"Finite sweep weights:\n{weights_text}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=COMPACT_FONT_SIZE,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
        ax.set_xlabel("Total Cost (M, Lower Is Better)")
        ax.set_ylabel("Proxy Grid Stress (Lower Is Better)")
        ax.grid(alpha=0.25)
        ax.set_xlim(row["cost_million"] - 0.5, row["cost_million"] + 0.5)
        ax.set_ylim(max(0.0, row["grid_stress"] - 0.5), row["grid_stress"] + 0.5)
        _save(fig, output_dir / "pareto_tradeoff_clean.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    def annotate_weight_groups(ax: plt.Axes, plot_frame: pd.DataFrame, y_col: str) -> None:
        cluster_lines: list[str] = []
        clustered_indices: set[int] = set()
        for _, group in plot_frame.groupby(plot_frame["investment_mw"].round(), sort=False):
            if len(group) <= 1:
                continue
            clustered_indices.update(int(idx) for idx in group.index)
            cluster_lines.extend(group["weights"].astype(str).tolist())
            ax.annotate(
                f"{_plural(len(group), 'weight')}\n{group['investment_mw'].mean():.0f} MW",
                (float(group["_plot_x"].mean()), float(group["_plot_y"].mean())),
                xytext=(30, -38),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.8},
                fontsize=COMPACT_FONT_SIZE,
                bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "0.80", "alpha": 0.92},
            )
        for idx, row in plot_frame.iterrows():
            if int(idx) in clustered_indices:
                continue
            ax.annotate(
                row["weights"],
                (row["_plot_x"], row["_plot_y"]),
                xytext=(10, -18),
                textcoords="offset points",
                fontsize=COMPACT_FONT_SIZE,
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )
        if cluster_lines:
            ax.text(
                0.03,
                0.06,
                "Collapsed 60 MW solution:\n" + "\n".join(cluster_lines),
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=COMPACT_FONT_SIZE,
                bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
            )

    stress_frame = _jitter_duplicate_xy(frame, "cost_million", "grid_stress", radius_x_fraction=0.032, radius_y_fraction=0.040)
    sc0 = axes[0].scatter(
        stress_frame["_plot_x"],
        stress_frame["_plot_y"],
        c=stress_frame["renewable_share"],
        s=95 + stress_frame["investment_mw"].fillna(0),
        cmap="YlGnBu",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.92,
    )
    annotate_weight_groups(axes[0], stress_frame, "grid_stress")
    axes[0].set_xlabel("Total Cost (M, Lower Is Better)")
    axes[0].set_ylabel("Proxy Grid Stress (Lower Is Better)")
    axes[0].set_title("Cost and Grid Stress")
    axes[0].grid(alpha=0.25)

    curtailment_frame = _jitter_duplicate_xy(frame, "cost_million", "curtailment_mwh", radius_x_fraction=0.032, radius_y_fraction=0.040)
    axes[1].scatter(
        curtailment_frame["_plot_x"],
        curtailment_frame["_plot_y"],
        c=curtailment_frame["renewable_share"],
        s=95 + curtailment_frame["investment_mw"].fillna(0),
        cmap="YlGnBu",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.92,
    )
    annotate_weight_groups(axes[1], curtailment_frame, "curtailment_mwh")
    axes[1].set_xlabel("Total Cost (M, Lower Is Better)")
    axes[1].set_ylabel("Curtailment (MWh, Lower Is Better)")
    axes[1].set_title("Cost and Curtailment")
    axes[1].grid(alpha=0.25)
    for ax in axes:
        _pad_axis_y(ax, top_fraction=0.16, bottom_fraction=0.08)
        _pad_axis_x(ax, left_fraction=0.03, right_fraction=0.08)
    cbar = fig.colorbar(sc0, ax=axes, shrink=0.82)
    cbar.set_label("Renewable Share")
    fig.subplots_adjust(wspace=0.34)
    _save(fig, output_dir / "pareto_tradeoff_clean.png")


def _nondominated_minimize(frame: pd.DataFrame, objective_cols: list[str]) -> pd.Series:
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


def plot_empirical_policy_front(analysis_dir: Path, output_dir: Path) -> None:
    """Plot an empirical non-dominated set across final evaluated policies."""
    rows = []
    for env_name, filename in [
        ("Proxy", "proxy_policy_comparison.csv"),
        ("Full PyPSA", "fullenv_policy_comparison.csv"),
    ]:
        frame = _ordered(pd.read_csv(analysis_dir / filename))
        frame = frame[frame["policy"].isin(POLICY_ORDER)].copy()
        frame["env"] = env_name
        frame["total_cost_million"] = frame["total_cost_mean"] / 1e6
        frame["nondominated"] = _nondominated_minimize(
            frame,
            ["total_cost_mean", "load_shedding_mean", "renewable_curtailment_mean", "total_investment_mean"],
        )
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    combined[
        [
            "env",
            "policy",
            "Policy",
            "total_cost_million",
            "load_shedding_mean",
            "renewable_curtailment_mean",
            "total_investment_mean",
            "nondominated",
        ]
    ].to_csv(output_dir / "empirical_policy_front.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=False)
    for ax, (env_name, env_frame) in zip(axes, combined.groupby("env", sort=False)):
        y_col = "renewable_curtailment_mean"
        plot_frame = _jitter_duplicate_xy(env_frame, "total_cost_million", y_col)
        label_offsets = [(7, 7), (7, -14), (-72, 7), (-72, -14), (8, 20)]
        for offset, (_, row) in zip(label_offsets, plot_frame.iterrows()):
            policy = row["policy"]
            marker = "*" if bool(row["nondominated"]) else "o"
            size = 120 if bool(row["nondominated"]) else 70
            ax.scatter(
                row["_plot_x"],
                row["_plot_y"],
                s=size + 0.15 * row["total_investment_mean"],
                marker=marker,
                color=COLORS.get(policy, "#999999"),
                edgecolor="black",
                linewidth=0.6,
                alpha=0.88,
                label=POLICY_LABELS.get(policy, row["Policy"]) if env_name == "Proxy" else None,
            )
            suffix = " (ND)" if bool(row["nondominated"]) else ""
            ax.annotate(
                f"{POLICY_LABELS.get(policy, row['Policy'])}{suffix}",
                (row["_plot_x"], row["_plot_y"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=COMPACT_FONT_SIZE,
            )
        ax.set_title(f"{env_name}: Empirical Non-Dominated Set")
        ax.set_xlabel("Total Cost (M, Lower Is Better)")
        ax.set_ylabel("Renewable Curtailment (MWh, Lower Is Better)")
        ax.grid(alpha=0.25)
        _pad_axis_y(ax, top_fraction=0.14, bottom_fraction=0.08)
    axes[0].legend(loc="best")
    fig.subplots_adjust(wspace=0.34)
    _save(fig, output_dir / "empirical_policy_front.png")


def _nondominated_minimize_cols(frame: pd.DataFrame, objective_cols: list[str]) -> pd.Series:
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


def _collapse_unique_outcome_points(frame: pd.DataFrame, env_name: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    key_cols = [
        "total_cost_mean",
        "renewable_curtailment_mean",
        "renewable_share_mean",
        "total_investment_mean",
        "active_lines_mean",
    ]
    if env_name == "Proxy":
        key_cols.insert(1, "grid_stress_mean")
    else:
        key_cols.insert(1, "load_shedding_mean")
    rounded = frame.copy()
    for column in key_cols:
        rounded[f"_sig_{column}"] = rounded[column].round(6)
    signature_cols = [f"_sig_{column}" for column in key_cols]
    rows = []
    for _, group in rounded.groupby(signature_cols, sort=False, dropna=False):
        first = group.iloc[0].to_dict()
        payload = {column: first[column] for column in frame.columns if not column.startswith("_sig_")}
        payload["support_size"] = int(len(group))
        payload["support_weight_count"] = int(group["weights"].nunique()) if "weights" in group else int(len(group))
        payload["support_seed_count"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        payload["support_weights"] = " | ".join(sorted(set(group["weights"].astype(str)))) if "weights" in group else ""
        payload["support_seeds"] = " | ".join(str(int(seed)) for seed in sorted(set(group["seed"].astype(int)))) if "seed" in group else ""
        rows.append(payload)
    collapsed = pd.DataFrame(rows)
    objective_cols = ["total_cost_mean", "load_shedding_mean", "renewable_curtailment_mean", "total_investment_mean"]
    if env_name == "Proxy":
        objective_cols = ["total_cost_mean", "grid_stress_mean", "renewable_curtailment_mean", "total_investment_mean"]
    collapsed["test_nondominated"] = _nondominated_minimize_cols(collapsed, objective_cols)
    return collapsed


def plot_scalarized_candidate_set_diagnostic(results_root: Path, output_dir: Path) -> None:
    """Regenerate thesis-ready scalarised candidate-set figures if available."""
    candidate_dir = locate_results_dir(results_root, "pareto_archive")
    full_path = candidate_dir / "pareto_test_results_full.csv"
    proxy_path = candidate_dir / "pareto_test_results_proxy.csv"
    if not full_path.exists() and not proxy_path.exists():
        return

    frames: list[tuple[str, pd.DataFrame, str, str, list[str]]] = []
    if proxy_path.exists():
        proxy = pd.read_csv(proxy_path)
        if "test_nondominated" not in proxy.columns:
            proxy["test_nondominated"] = _nondominated_minimize_cols(
                proxy,
                ["total_cost_mean", "grid_stress_mean", "renewable_curtailment_mean", "total_investment_mean"],
            )
        proxy = _collapse_unique_outcome_points(proxy, "Proxy")
        frames.append(
            (
                "Proxy",
                proxy,
                "grid_stress_mean",
                "Proxy Grid Stress (Lower Is Better)",
                ["total_cost_mean", "grid_stress_mean", "renewable_curtailment_mean", "total_investment_mean"],
            )
        )
    if full_path.exists():
        full = pd.read_csv(full_path)
        if "test_nondominated" not in full.columns:
            full["test_nondominated"] = _nondominated_minimize_cols(
                full,
                ["total_cost_mean", "load_shedding_mean", "renewable_curtailment_mean", "total_investment_mean"],
            )
        full = _collapse_unique_outcome_points(full, "Full PyPSA")
        frames.append(
            (
                "Full PyPSA",
                full,
                "renewable_curtailment_mean",
                "Renewable Curtailment (MWh, Lower Is Better)",
                ["total_cost_mean", "load_shedding_mean", "renewable_curtailment_mean", "total_investment_mean"],
            )
        )

    fig, axes = plt.subplots(1, len(frames), figsize=(6.0 * len(frames), 5.0), squeeze=False)
    export_rows = []
    for ax, (env_label, frame, y_col, y_label, objective_cols) in zip(axes.ravel(), frames):
        plot_frame = frame.copy()
        plot_frame["total_cost_million"] = plot_frame["total_cost_mean"] / 1e6
        dominated = plot_frame[~plot_frame["test_nondominated"]]
        nondominated = plot_frame[plot_frame["test_nondominated"]]
        if not dominated.empty:
            ax.scatter(
                dominated["total_cost_million"],
                dominated[y_col],
                s=45 + 0.15 * dominated["total_investment_mean"].fillna(0),
                color=COLORS["neutral"],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.65,
                label="Dominated",
            )
        if not nondominated.empty:
            ordered = nondominated.sort_values("total_cost_million")
            ax.plot(ordered["total_cost_million"], ordered[y_col], color=COLORS["moppo"], linewidth=1.3, alpha=0.75)
            ax.scatter(
                nondominated["total_cost_million"],
                nondominated[y_col],
                s=85 + 6.0 * nondominated["support_size"].fillna(1) + 0.18 * nondominated["total_investment_mean"].fillna(0),
                color=COLORS["moppo"],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.92,
                label="Non-dominated unique outcome",
            )
            label_offsets = [(7, 7), (7, -18), (-76, 8), (-76, -18)]
            for offset, (_, row) in zip(label_offsets, nondominated.iterrows()):
                support_size = int(row["support_size"])
                support_weight_count = int(row["support_weight_count"])
                ax.annotate(
                    f"{row['total_investment_mean']:.0f} MW\n{_plural(support_size, 'checkpoint')}\n{_plural(support_weight_count, 'weight')}",
                    (row["total_cost_million"], row[y_col]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=COMPACT_FONT_SIZE,
                )
        ax.set_title(f"{env_label} Candidate Set")
        ax.set_xlabel("Total Cost (M, Lower Is Better)")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        ax.legend()
        _pad_axis_y(ax, top_fraction=0.16, bottom_fraction=0.08)
        _pad_axis_x(ax, left_fraction=0.04, right_fraction=0.10)
        ax.text(
            0.98,
            0.02,
            f"{_plural(int(plot_frame['support_size'].sum()), 'selected checkpoint')}\n{_plural(len(plot_frame), 'unique outcome')}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=COMPACT_FONT_SIZE,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
        export = plot_frame[
            [
                "candidate_id",
                "weights",
                "seed",
                "total_cost_mean",
                "grid_stress_mean",
                "load_shedding_mean",
                "renewable_curtailment_mean",
                "renewable_share_mean",
                "total_investment_mean",
                "active_lines_mean",
                "support_size",
                "support_weight_count",
                "support_seed_count",
                "support_weights",
                "support_seeds",
                "test_nondominated",
            ]
        ].copy()
        export["env"] = env_label
        export_rows.append(export)

    _save(fig, output_dir / "scalarized_candidate_set_diagnostic.png")
    if export_rows:
        pd.concat(export_rows, ignore_index=True).to_csv(output_dir / "scalarized_candidate_set_diagnostic_points.csv", index=False)


def plot_preference_response_diagnostic(results_root: Path, output_dir: Path) -> None:
    """Regenerate thesis-ready MO-PPO preference-response figures if available."""
    response_dir = locate_results_dir(results_root, "preference_response")
    full_path = response_dir / "preference_response_test_results_full.csv"
    proxy_path = response_dir / "preference_response_test_results_proxy.csv"
    if not full_path.exists() and not proxy_path.exists():
        return

    frames: list[tuple[str, pd.DataFrame, str, str]] = []
    if proxy_path.exists():
        proxy = pd.read_csv(proxy_path)
        if "test_nondominated" not in proxy.columns:
            proxy["test_nondominated"] = _nondominated_minimize_cols(
                proxy,
                ["total_cost_mean", "grid_stress_mean", "renewable_curtailment_mean", "total_investment_mean"],
            )
        frames.append(
            (
                "Proxy",
                _collapse_unique_outcome_points(proxy, "Proxy"),
                "grid_stress_mean",
                "Proxy Grid Stress (Lower Is Better)",
            )
        )
    if full_path.exists():
        full = pd.read_csv(full_path)
        if "test_nondominated" not in full.columns:
            full["test_nondominated"] = _nondominated_minimize_cols(
                full,
                ["total_cost_mean", "load_shedding_mean", "renewable_curtailment_mean", "total_investment_mean"],
            )
        frames.append(
            (
                "Full PyPSA",
                _collapse_unique_outcome_points(full, "Full PyPSA"),
                "renewable_curtailment_mean",
                "Renewable Curtailment (MWh, Lower Is Better)",
            )
        )

    fig, axes = plt.subplots(1, len(frames), figsize=(6.0 * len(frames), 5.0), squeeze=False)
    export_rows = []
    for ax, (env_label, frame, y_col, y_label) in zip(axes.ravel(), frames):
        plot_frame = frame.copy()
        plot_frame["total_cost_million"] = plot_frame["total_cost_mean"] / 1e6
        dominated = plot_frame[~plot_frame["test_nondominated"]]
        nondominated = plot_frame[plot_frame["test_nondominated"]]
        if not dominated.empty:
            ax.scatter(
                dominated["total_cost_million"],
                dominated[y_col],
                s=45 + 0.15 * dominated["total_investment_mean"].fillna(0),
                color=COLORS["neutral"],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.65,
                label="Dominated",
            )
        if not nondominated.empty:
            ordered = nondominated.sort_values("total_cost_million")
            ax.plot(ordered["total_cost_million"], ordered[y_col], color=COLORS["moppo"], linewidth=1.3, alpha=0.75)
            ax.scatter(
                nondominated["total_cost_million"],
                nondominated[y_col],
                s=85 + 6.0 * nondominated["support_size"].fillna(1) + 0.18 * nondominated["total_investment_mean"].fillna(0),
                color=COLORS["moppo"],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.92,
                label="Non-dominated unique outcome",
            )
            label_offsets = [(7, 7), (7, -18), (-72, 8), (-72, -18)]
            for offset, (_, row) in zip(label_offsets, nondominated.iterrows()):
                support_size = int(row["support_size"])
                support_seed_count = int(row["support_seed_count"])
                ax.annotate(
                    f"{row['total_investment_mean']:.0f} MW\n{_plural(support_size, 'query', 'queries')} / {_plural(support_seed_count, 'seed')}",
                    (row["total_cost_million"], row[y_col]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=COMPACT_FONT_SIZE,
                )
        ax.set_title(f"{env_label} Response")
        ax.set_xlabel("Total Cost (M, Lower Is Better)")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        ax.legend()
        _pad_axis_y(ax, top_fraction=0.16, bottom_fraction=0.08)
        _pad_axis_x(ax, left_fraction=0.04, right_fraction=0.10)
        ax.text(
            0.02,
            0.02,
            f"{_plural(int(plot_frame['support_size'].sum()), 'preference query', 'preference queries')}\n{_plural(len(plot_frame), 'unique outcome')}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=COMPACT_FONT_SIZE,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
        export = plot_frame[
            [
                "candidate_id",
                "weights",
                "seed",
                "total_cost_mean",
                "grid_stress_mean",
                "load_shedding_mean",
                "renewable_curtailment_mean",
                "renewable_share_mean",
                "total_investment_mean",
                "active_lines_mean",
                "support_size",
                "support_weight_count",
                "support_seed_count",
                "support_weights",
                "support_seeds",
                "test_nondominated",
            ]
        ].copy()
        export["env"] = env_label
        export_rows.append(export)

    _save(fig, output_dir / "moppo_preference_response_diagnostic.png")
    if export_rows:
        pd.concat(export_rows, ignore_index=True).to_csv(output_dir / "moppo_preference_response_diagnostic_points.csv", index=False)


def write_architecture_tikz(output_dir: Path) -> None:
    source = r"""\begin{tikzpicture}[
    node distance=7mm and 11mm,
    block/.style={draw, rounded corners, align=center, minimum width=31mm, minimum height=10mm, fill=blue!6},
    process/.style={draw, rounded corners, align=center, minimum width=34mm, minimum height=10mm, fill=orange!10},
    result/.style={draw, rounded corners, align=center, minimum width=34mm, minimum height=10mm, fill=green!10},
    arrow/.style={-{Latex[length=2mm]}, thick}
]
\node[block] (data) {Input data\\PyPSA-Eur network\\load, wind, solar};
\node[process, right=of data] (case) {Austria case builder\\candidate corridors\\chronological splits};
\node[process, right=of case] (envs) {TEP environments\\proxy training\\full PyPSA evaluation};

\node[process, below left=of envs] (agents) {RL agents\\PPO\\preference-conditioned MO-PPO};
\node[process, below=of envs] (baselines) {Baselines\\zero, uniform\\congestion heuristics\\myopic proxy};
\node[result, below right=of envs] (selection) {Validation and selection\\best checkpoint\\target-env reranking};

\node[result, below=of agents] (eval) {Held-out evaluation\\proxy test\\full PyPSA test\\future scenarios};
\node[result, below=of baselines] (stats) {Statistical analysis\\confidence intervals\\non-parametric tests\\empirical front};
\node[result, below=of selection] (xai) {Explainability\\output-specific Shapley\\seed/estimator stability\\strict intervention};

\node[result, below=of stats, minimum width=50mm] (report) {Thesis reporting\\tables, figures\\method discussion\\limitations};

\draw[arrow] (data) -- (case);
\draw[arrow] (case) -- (envs);
\draw[arrow] (envs) -- (agents);
\draw[arrow] (envs) -- (baselines);
\draw[arrow] (agents) -- (selection);
\draw[arrow] (selection) -- (eval);
\draw[arrow] (baselines) -- (eval);
\draw[arrow] (eval) -- (stats);
\draw[arrow] (eval) -- (xai);
\draw[arrow] (stats) -- (report);
\draw[arrow] (xai) -- (report);
\end{tikzpicture}
"""
    (output_dir / "architecture_diagram_tikz.tex").write_text(source, encoding="utf-8")


def plot_architecture_diagram(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    ax.axis("off")
    ax.set_xlim(-0.05, 0.94)
    ax.set_ylim(-0.05, 0.86)
    boxes = {
        "data": (0.06, 0.72, "Input data\nPyPSA-Eur network\nload, wind, solar", "#e8f1ff"),
        "case": (0.32, 0.72, "Austria case builder\ncandidate corridors\nchronological splits", "#fff3df"),
        "envs": (0.58, 0.72, "TEP environments\nproxy training\nfull PyPSA evaluation", "#fff3df"),
        "agents": (0.20, 0.45, "RL agents\nPPO\nMO-PPO", "#fff3df"),
        "baselines": (0.45, 0.45, "Baselines\nzero, uniform\nheuristics, myopic", "#fff3df"),
        "selection": (0.70, 0.45, "Validation and selection\nbest checkpoints\ntarget-env reranking", "#e9f8ed"),
        "eval": (0.20, 0.19, "Held-out evaluation\nproxy test\nfull PyPSA test\nfuture scenarios", "#e9f8ed"),
        "stats": (0.45, 0.19, "Statistical analysis\nconfidence intervals\ntests, empirical front", "#e9f8ed"),
        "xai": (0.70, 0.19, "Explainability\noutput-specific Shapley\nstability, strict intervention", "#e9f8ed"),
        "report": (0.45, 0.02, "Thesis reporting\ntables, figures, interpretation", "#f2f2f2"),
    }

    def draw_box(key: str) -> None:
        x, y, text, color = boxes[key]
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=ANNOTATION_FONT_SIZE,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "#34495e", "linewidth": 1.1},
        )

    def arrow(a: str, b: str, curve: float = 0.0) -> None:
        x1, y1, *_ = boxes[a]
        x2, y2, *_ = boxes[b]
        arrowprops = {"arrowstyle": "->", "lw": 1.5, "color": "#34495e", "shrinkA": 26, "shrinkB": 26}
        if curve:
            arrowprops["connectionstyle"] = f"arc3,rad={curve}"
        ax.annotate(
            "",
            xy=(x2, y2 + (0.055 if y2 < y1 else 0.0)),
            xytext=(x1, y1 - (0.055 if y1 > y2 else 0.0)),
            arrowprops=arrowprops,
        )

    for a, b, curve in [
        ("data", "case", 0.0),
        ("case", "envs", 0.0),
        ("envs", "agents", 0.0),
        ("envs", "baselines", 0.0),
        ("agents", "selection", -0.25),
        ("selection", "eval", 0.0),
        ("baselines", "eval", 0.0),
        ("eval", "stats", 0.0),
        ("eval", "xai", 0.0),
        ("stats", "report", 0.0),
        ("xai", "report", 0.0),
    ]:
        arrow(a, b, curve)
    for key in boxes:
        draw_box(key)
    _save(fig, output_dir / "architecture_diagram.png")
    write_architecture_tikz(output_dir)


def plot_future_upgrade_trajectory(analysis_dir: Path, output_dir: Path) -> None:
    path = analysis_dir / "future_projection" / "future_projection_top_lines.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    nonzero = frame[frame["nonzero_policy_count"] > 0].copy()
    if nonzero.empty:
        return
    top_lines = (
        nonzero.groupby(["line", "line_label"], as_index=False)["mean_upgrade_mw"]
        .mean()
        .sort_values("mean_upgrade_mw", ascending=False)
        .head(5)
    )
    plot_frame = nonzero[nonzero["line"].isin(top_lines["line"])]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for _, top_row in top_lines.iterrows():
        line_data = plot_frame[plot_frame["line"] == top_row["line"]].sort_values("scenario_year")
        label = str(top_row["line_label"])
        ax.plot(line_data["scenario_year"], line_data["mean_upgrade_mw"], marker="o", label=label)
    invariant = all(
        plot_frame[plot_frame["line"] == line_name]["mean_upgrade_mw"].nunique(dropna=False) == 1
        for line_name in top_lines["line"]
    )
    ax.set_xlabel("Scenario Year")
    ax.set_ylabel("Mean Recommended Upgrade (MW)")
    if invariant:
        ax.text(
            0.02,
            0.96,
            "Same corridor recommendations in all scenarios",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    _save(fig, output_dir / "future_upgrade_trajectory_clean.png")


def plot_future_projection_outcomes(analysis_dir: Path, output_dir: Path) -> None:
    path = analysis_dir / "future_projection" / "future_projection_summary.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path).sort_values("scenario_year")
    if frame.empty:
        return
    years = frame["scenario_year"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    specs = [
        ("grid_stress_mean", "grid_stress_std_across_policies", "Proxy Grid Stress", "Grid Stress", COLORS["stress"]),
        ("renewable_curtailment_mean", "renewable_curtailment_std_across_policies", "Renewable Curtailment (MWh)", "Curtailment", COLORS["curtailment"]),
        ("renewable_share_mean", "renewable_share_std_across_policies", "Renewable Share", "Renewable Share", COLORS["renewable"]),
    ]
    for ax, (mean_col, std_col, ylabel, title, colour) in zip(axes, specs):
        ax.errorbar(
            years,
            frame[mean_col],
            yerr=frame[std_col],
            fmt="-o",
            color=colour,
            ecolor="0.65",
            linewidth=2.0,
            capsize=3,
        )
        ax.set_title(title)
        ax.set_xlabel("Scenario Year")
        ax.set_ylabel(ylabel)
        ax.set_xticks(years)
        ax.grid(alpha=0.25)
        _pad_axis_y(ax, top_fraction=0.16, bottom_fraction=0.06)
    fig.tight_layout()
    _save(fig, output_dir / "future_projection_outcomes.png")


def plot_future_maps_with_austria(network: pypsa.Network, analysis_dir: Path) -> None:
    projection_dir = analysis_dir / "future_projection"
    top_path = projection_dir / "future_projection_top_lines.csv"
    if not top_path.exists():
        return
    frame = pd.read_csv(top_path)
    for year, year_data in frame.groupby("scenario_year"):
        top = year_data.sort_values("mean_upgrade_mw", ascending=False).head(8)
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_country_outline(ax, network)
        for _, line in network.lines.iterrows():
            coords = parse_linestring(line.get("geometry", ""))
            if len(coords) >= 2:
                xs, ys = zip(*coords)
                ax.plot(xs, ys, color="black", linewidth=0.45, alpha=0.38, zorder=1)
        max_upgrade = max(float(top["mean_upgrade_mw"].max()), 1.0)
        highlight_colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(top), 1)))
        for color, (_, row) in zip(highlight_colors, top.iterrows()):
            line_name = row["line"]
            if line_name not in network.lines.index or float(row["mean_upgrade_mw"]) <= 0:
                continue
            coords = parse_linestring(network.lines.loc[line_name].get("geometry", ""))
            if len(coords) < 2:
                continue
            xs, ys = zip(*coords)
            label = endpoint_anchor_label(network, line_name)
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=1.5 + 4.0 * float(row["mean_upgrade_mw"]) / max_upgrade,
                label=f"{label} ({row['mean_upgrade_mw']:.0f} MW)",
                zorder=3,
            )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        apply_geo_aspect(ax, network)
        ax.grid(alpha=0.25)
        annotate_anchor_points(ax, anchor_points_for_lines(network, top["line"].tolist(), max_unique_anchors=6), fontsize=COMPACT_FONT_SIZE)
        ax.legend(fontsize=COMPACT_FONT_SIZE, loc="best")
        _save(fig, projection_dir / f"future_network_map_{int(year)}.png")


def plot_network_context_and_future_map(
    network: pypsa.Network,
    analysis_dir: Path,
    output_dir: Path,
    scenario_year: int = 2040,
) -> None:
    projection_dir = analysis_dir / "future_projection"
    top_path = projection_dir / "future_projection_top_lines.csv"
    if not top_path.exists():
        return
    frame = pd.read_csv(top_path)
    if frame.empty:
        return
    year_frame = frame.loc[frame["scenario_year"] == scenario_year].copy()
    if year_frame.empty:
        scenario_year = int(frame["scenario_year"].max())
        year_frame = frame.loc[frame["scenario_year"] == scenario_year].copy()
    top = year_frame.sort_values("mean_upgrade_mw", ascending=False).head(8)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.1))
    for idx, ax in enumerate(axes):
        draw_country_outline(ax, network)
        for _, line in network.lines.iterrows():
            coords = parse_linestring(line.get("geometry", ""))
            if len(coords) < 2:
                continue
            xs, ys = zip(*coords)
            if idx == 0:
                ax.plot(xs, ys, color="#2f5d7e", linewidth=0.62, alpha=0.72, zorder=1)
            else:
                ax.plot(xs, ys, color="black", linewidth=0.45, alpha=0.40, zorder=1)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        apply_geo_aspect(ax, network)
        ax.grid(alpha=0.20)

    axes[0].set_title("Austrian Transmission Network Context")
    anchor_points = anchor_points_for_lines(network, top["line"].tolist(), max_unique_anchors=6)
    annotate_anchor_points(axes[0], anchor_points, fontsize=COMPACT_FONT_SIZE)

    max_upgrade = max(float(top["mean_upgrade_mw"].max()), 1.0)
    highlight_colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(top), 1)))
    handles = []
    labels = []
    for color, (_, row) in zip(highlight_colors, top.iterrows()):
        line_name = row["line"]
        if line_name not in network.lines.index or float(row["mean_upgrade_mw"]) <= 0:
            continue
        coords = parse_linestring(network.lines.loc[line_name].get("geometry", ""))
        if len(coords) < 2:
            continue
        xs, ys = zip(*coords)
        handle = axes[1].plot(
            xs,
            ys,
            color=color,
            linewidth=1.6 + 4.6 * float(row["mean_upgrade_mw"]) / max_upgrade,
            zorder=3,
        )[0]
        handles.append(handle)
        labels.append(f"{endpoint_anchor_label(network, line_name)} ({row['mean_upgrade_mw']:.0f} MW)")

    axes[1].set_title(f"Projected Reinforcements in {int(scenario_year)}")
    annotate_anchor_points(axes[1], anchor_points, fontsize=COMPACT_FONT_SIZE)
    if handles:
        axes[1].legend(
            handles,
            labels,
            fontsize=COMPACT_FONT_SIZE,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            title="Highlighted corridors",
        )
    fig.tight_layout()
    _save(fig, output_dir / f"future_network_context_{int(scenario_year)}.png")


def plot_explainability_feature_bars(results_root: Path, output_dir: Path) -> None:
    shapley_dir = locate_results_dir(results_root, "explainability")
    global_path = shapley_dir / "global_shapley_labeled.csv"
    perm_path = shapley_dir / "permutation_importance_labeled.csv"
    if not global_path.exists() or not perm_path.exists():
        return

    network = pypsa.Network("derived/austria_net_physical_ratings.nc")
    global_frame = pd.read_csv(global_path).head(10).iloc[::-1].copy()
    perm_frame = pd.read_csv(perm_path).head(10).iloc[::-1].copy()
    if "feature_raw" in global_frame.columns:
        global_frame["feature_display"] = [feature_display_name(raw, network) for raw in global_frame["feature_raw"]]
    else:
        global_frame["feature_display"] = global_frame["feature"]
    if "feature_raw" in perm_frame.columns:
        perm_frame["feature_display"] = [feature_display_name(raw, network) for raw in perm_frame["feature_raw"]]
    else:
        perm_frame["feature_display"] = perm_frame["feature"]
    global_frame["feature_wrapped"] = global_frame["feature_display"].map(_format_feature_label)
    perm_frame["feature_wrapped"] = perm_frame["feature_display"].map(_format_feature_label)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.2), sharey=False)
    importance_scale = 1e-4
    importance_formatter = FuncFormatter(lambda value, _position: f"{value / importance_scale:.1f}")

    bars0 = axes[0].barh(
        global_frame["feature_wrapped"],
        global_frame["global_importance"],
        xerr=global_frame["global_std"],
        color=COLORS["ppo"],
        alpha=0.92,
        capsize=3,
        error_kw={"elinewidth": 0.9, "ecolor": "0.55"},
    )
    axes[0].set_title("Global Permutation-Shapley Attribution")
    axes[0].set_xlabel(r"Mean Importance ($\times 10^{-4}$)")
    axes[0].set_ylabel("Feature")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].xaxis.set_major_formatter(importance_formatter)

    bars1 = axes[1].barh(
        perm_frame["feature_wrapped"],
        perm_frame["importance"],
        color=COLORS["uniform"],
        alpha=0.92,
    )
    axes[1].set_title("Permutation Importance")
    axes[1].set_xlabel(r"Mean Importance ($\times 10^{-4}$)")
    axes[1].set_ylabel("")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].xaxis.set_major_formatter(importance_formatter)
    right_max = float(perm_frame["importance"].max()) if not perm_frame.empty else 1.0
    axes[1].set_xlim(0.0, right_max * 1.16)
    x_span1 = axes[1].get_xlim()[1] - axes[1].get_xlim()[0]
    for patch, value in zip(bars1, perm_frame["importance"]):
        axes[1].text(
            patch.get_width() + 0.015 * x_span1,
            patch.get_y() + patch.get_height() / 2.0,
            _format_small_number(float(value)),
            va="center",
            ha="left",
            fontsize=COMPACT_FONT_SIZE,
        )

    axes[0].set_xlim(0.0, float(global_frame["global_importance"].max()) * 1.22)
    axes[1].set_xlim(0.0, right_max * 1.22)
    fig.tight_layout()
    _save(fig, output_dir / "explainability_feature_bars_combined.png")


def export_geographic_anchor_mapping(results_root: Path, output_dir: Path) -> None:
    network = pypsa.Network("derived/austria_net_physical_ratings.nc")
    shapley_dir = locate_results_dir(results_root, "explainability")
    feature_sources = [
        shapley_dir / "global_shapley_labeled.csv",
        shapley_dir / "permutation_importance_labeled.csv",
    ]
    line_ids: set[str] = set()
    bus_ids: set[str] = set()
    for source in feature_sources:
        if not source.exists():
            continue
        frame = pd.read_csv(source).head(15)
        if "feature_raw" not in frame.columns:
            continue
        for raw in frame["feature_raw"]:
            raw = str(raw)
            if "::" not in raw:
                continue
            prefix, value = raw.split("::", 1)
            if value in network.lines.index:
                line_ids.add(value)
            elif value in network.buses.index:
                bus_ids.add(value)
    future_path = results_root / "analysis" / "future_projection" / "future_projection_top_lines.csv"
    if future_path.exists():
        future = pd.read_csv(future_path)
        for line_name in future.loc[future["nonzero_policy_count"] > 0, "line"]:
            if line_name in network.lines.index:
                line_ids.add(str(line_name))

    records: list[dict[str, object]] = []
    for line_name in sorted(line_ids):
        line = network.lines.loc[line_name]
        bus0 = network.buses.loc[line.bus0]
        bus1 = network.buses.loc[line.bus1]
        records.append(
            {
                "type": "line",
                "raw_id": line_name,
                "display_label": endpoint_anchor_label(network, line_name),
                "start_bus": line.bus0,
                "start_anchor": feature_display_name(f"demand::{line.bus0}", network).replace("Demand at bus: ", ""),
                "start_coords": f"{float(bus0.x):.2f}E,{float(bus0.y):.2f}N",
                "end_bus": line.bus1,
                "end_anchor": feature_display_name(f"demand::{line.bus1}", network).replace("Demand at bus: ", ""),
                "end_coords": f"{float(bus1.x):.2f}E,{float(bus1.y):.2f}N",
            }
        )
    for bus_name in sorted(bus_ids):
        bus = network.buses.loc[bus_name]
        records.append(
            {
                "type": "bus",
                "raw_id": bus_name,
                "display_label": feature_display_name(f"demand::{bus_name}", network).replace("Demand at bus: ", ""),
                "start_bus": "",
                "start_anchor": "",
                "start_coords": "",
                "end_bus": "",
                "end_anchor": "",
                "end_coords": f"{float(bus.x):.2f}E,{float(bus.y):.2f}N",
            }
        )
    if not records:
        return
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "geographic_anchor_mapping.csv", index=False)
    frame.to_latex(
        output_dir / "geographic_anchor_mapping.tex",
        index=False,
        escape=True,
        longtable=True,
        caption="Mapping of raw network identifiers to readable geographic anchor labels used in figures.",
        label="tab:geographic_anchor_mapping",
    )


def refresh_line_importance_map(network: pypsa.Network, results_root: Path, output_dir: Path) -> None:
    shapley_dir = locate_results_dir(results_root, "explainability")
    path = shapley_dir / "line_importance_shapley.csv"
    if path.exists():
        frame = pd.read_csv(path)
        plot_line_importance_map(network, frame, shapley_dir / "line_importance_shapley_map.png", top_k=8)
        plot_line_importance_map(network, frame, output_dir / "line_importance_shapley_map.png", top_k=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate editable thesis figure variants from a finished thesis run.")
    parser.add_argument("--results-root", default="results/thesis_final")
    parser.add_argument("--network", default="derived/austria_net_physical_ratings.nc")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    analysis_dir = results_root / "analysis"
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / "figure_workbench"
    output_dir.mkdir(parents=True, exist_ok=True)

    network = pypsa.Network(args.network)
    regenerate_learning_curves(results_root, output_dir)
    plot_fullenv_delta_vs_zero(analysis_dir, output_dir)
    plot_proxy_and_full_context(analysis_dir, output_dir)
    plot_additional_metrics(analysis_dir, output_dir)
    plot_pareto_tradeoff(results_root, output_dir)
    plot_empirical_policy_front(analysis_dir, output_dir)
    plot_scalarized_candidate_set_diagnostic(results_root, output_dir)
    plot_preference_response_diagnostic(results_root, output_dir)
    plot_architecture_diagram(output_dir)
    plot_future_upgrade_trajectory(analysis_dir, output_dir)
    plot_future_projection_outcomes(analysis_dir, output_dir)
    plot_future_maps_with_austria(network, analysis_dir)
    plot_network_context_and_future_map(network, analysis_dir, output_dir)
    plot_explainability_feature_bars(results_root, output_dir)
    export_geographic_anchor_mapping(results_root, output_dir)
    refresh_line_importance_map(network, results_root, output_dir)
    print(f"Wrote figure workbench outputs to {output_dir}")


if __name__ == "__main__":
    main()
