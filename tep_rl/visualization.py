"""
visualization.py - Publication-quality plots for the TEP-RL thesis.

All figures that are listed in Section 9 ("Minimum Thesis Figure Set") of
THESIS_RUNBOOK_AND_EXPERIMENT_PLAN.md are implemented here.

Figure inventory
----------------
Training / learning behaviour
  * plot_training_curves          - single-run cost / stress / renewable
  * plot_multi_seed_curves        - mean and uncertainty band across seeds

Multi-objective analysis
  * plot_pareto_front             - scatter with stress colour coding

Model comparison
  * plot_metric_boxplots          - box plots for all primary metrics
  * plot_action_distribution      - bar chart for per-line upgrades

Grid impact
  * plot_line_loading_comparison  - before vs. after bar chart
  * plot_network_upgrades         - spatial network upgrade map

Explainability
  * plot_feature_importance       - horizontal bar chart
  * plot_temporal_attribution     - heatmap over decision steps

Utilities
  * _save                         - shared figure export helper
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 14,
        "axes.labelsize": 15,
        "axes.titleweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)



# Internal helpers


_PALETTE = {
    "cost":       "#6f9fcf",
    "stress":     "#e5a35c",
    "renewable":  "#79b96f",
    "emissions":  "#3f8f86",
    "ppo":        "#6f9fcf",
    "moppo":      "#e5a35c",
    "uniform":    "#79b96f",
    "zero":       "#7f858c",
    "neutral":    "#b9c1ca",
}
ANNOTATION_FONT_SIZE = 12
COMPACT_FONT_SIZE = 11


def _save(fig: plt.Figure, output_path: Optional[Path | str], dpi: int = 200) -> None:
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight"}
        if p.suffix.lower() == ".pdf":
            fig.savefig(p, **save_kwargs)
        else:
            fig.savefig(p, dpi=max(dpi, 300), **save_kwargs)
            fig.savefig(p.with_suffix(".pdf"), **save_kwargs)
    plt.close(fig)


def _pad_axis_y(ax: plt.Axes, top_fraction: float = 0.14, bottom_fraction: float = 0.03) -> None:
    ymin, ymax = ax.get_ylim()
    if not np.isfinite(ymin) or not np.isfinite(ymax) or np.isclose(ymin, ymax):
        return
    span = ymax - ymin
    ax.set_ylim(ymin - bottom_fraction * span, ymax + top_fraction * span)



# Training curves: single run


def plot_training_curves(
        history: dict[str, Any],
        output_path: Optional[Path | str] = None,
) -> None:
    """
    Plot per-episode cost, grid stress, and renewable share for one run.
    """
    episodes = history.get("episodes", [])
    if not episodes:
        return

    costs = [episode["total_cost"] for episode in episodes]
    stress = [episode["grid_stress"] for episode in episodes]
    renewable = [episode["renewable_share"] for episode in episodes]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(costs, color=_PALETTE["cost"])
    axes[0].set_ylabel("Total Cost")
    axes[0].grid(alpha=0.3)

    axes[1].plot(stress, color=_PALETTE["stress"])
    axes[1].set_ylabel("Grid Stress")
    axes[1].grid(alpha=0.3)

    axes[2].plot(renewable, color=_PALETTE["renewable"])
    axes[2].set_ylabel("Renewable Share")
    axes[2].set_xlabel("Episode")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, output_path)



# Multi-seed learning curves with standard-deviation shading


def plot_multi_seed_curves(
        seed_histories: Sequence[dict[str, Any]],
        output_path: Optional[Path | str] = None,
        label: str = "Agent",
        smoothing_window: int = 101,
        min_valid_seeds: Optional[int] = None,
) -> None:
    """
    Plot smoothed mean learning curves across multiple seeds.

        This is the thesis learning-curve diagnostic. It is intentionally
        smoothed for readability and uses a confidence band for the mean,
        because raw episode-level rewards are weather/time-slice noisy.

    Parameters
    ----------
    seed_histories:
        List of ``history`` dicts, one per seed (from ``train_agent``).
    label:
        Legend label for the agent.
    smoothing_window:
        Training episodes to smooth over.  Set to 1 to disable.
    """
    def _extract(histories, key):
        """Return a 2-D array (seeds x episodes), zero-padded to equal length."""
        series = []
        for h in histories:
            vals = [ep[key] for ep in h.get("episodes", [])]
            series.append(vals)
        max_len = max(len(s) for s in series) if series else 0
        if max_len == 0:
            return np.zeros((len(histories), 0))
        padded = np.full((len(series), max_len), fill_value=np.nan)
        for i, s in enumerate(series):
            padded[i, : len(s)] = s
        return padded

    metrics = [
        ("total_cost", "Total Cost\n(Million Model-Cost Units)", _PALETTE["cost"], 1e6),
        ("grid_stress", "Grid Stress", _PALETTE["stress"], 1.0),
        ("renewable_share", "Renewable Share", _PALETTE["renewable"], 1.0),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 9), sharex=True)

    for ax, (key, ylabel, colour, scale) in zip(axes, metrics):
        matrix = _extract(seed_histories, key) / scale  # (seeds, episodes)
        if matrix.shape[1] == 0:
            continue

        # Optional smoothing per seed.  Pandas rolling avoids the artificial
        # start/end drops caused by zero-padded convolution.
        if smoothing_window > 1:
            smoothed = np.full_like(matrix, fill_value=np.nan)
            for i in range(matrix.shape[0]):
                row = matrix[i]
                valid = ~np.isnan(row)
                if valid.sum() > 0:
                    smoothed_values = (
                        pd.Series(row[valid])
                        .rolling(window=smoothing_window, center=True, min_periods=1)
                        .mean()
                        .to_numpy()
                    )
                    smoothed[i, valid] = smoothed_values
            matrix = smoothed

        mean = np.nanmean(matrix, axis=0)
        valid_count = np.sum(~np.isnan(matrix), axis=0)
        sem = np.divide(
            np.nanstd(matrix, axis=0),
            np.sqrt(valid_count),
            out=np.zeros_like(mean),
            where=valid_count > 0,
        )
        required = len(seed_histories) if min_valid_seeds is None else max(int(min_valid_seeds), 1)
        mask = valid_count >= required
        if not np.any(mask):
            continue
        cutoff = int(np.max(np.flatnonzero(mask))) + 1
        mean = mean[:cutoff]
        sem = sem[:cutoff]
        xs = np.arange(len(mean))

        ax.plot(xs, mean, color=colour, label=label, linewidth=1.8)
        ax.fill_between(xs, mean - 1.96 * sem, mean + 1.96 * sem, color=colour, alpha=0.20,
                        label=f"95% CI of mean ({len(seed_histories)} seeds)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0.0)
        _pad_axis_y(ax, top_fraction=0.18, bottom_fraction=0.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

    axes[-1].set_xlabel("Training Episode (Not Test Episode)")
    fig.tight_layout()
    _save(fig, output_path)



# Box plots for primary metrics  (Experiments 2 / 3 comparison)


def plot_metric_boxplots(
        evaluations: dict[str, dict[str, Any]],
        metrics: Optional[Sequence[str]] = None,
        output_path: Optional[Path | str] = None,
) -> None:
    """
    Side-by-side box plots comparing agents across primary metrics.

    Each agent is one entry in ``evaluations``; each box shows the
    distribution over evaluation episodes.

    Parameters
    ----------
    evaluations:
        Mapping from agent label to evaluation dict (from ``evaluate_agent``).
        Requires the ``"episodes"`` list to be present.

    Example
    -------
    >>> plot_metric_boxplots(
    ...     {"PPO": eval_ppo, "MO-PPO": eval_moppo},
    ...     output_path="results/boxplots.png",
    ... )
    """
    if metrics is None:
        metrics = ["total_cost", "grid_stress", "renewable_share", "emissions"]

    metric_labels = {
        "total_cost": "Total Cost",
        "grid_stress": "Grid Stress",
        "renewable_share": "Renewable Share",
        "emissions": "Emissions",
        "constraint_violation": "Constraint Violations",
        "renewable_curtailment": "Curtailment",
    }

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    agent_labels = list(evaluations.keys())
    colours = [_PALETTE.get(label.lower(), f"C{i}") for i, label in enumerate(agent_labels)]

    for ax, metric in zip(axes, metrics):
        data = []
        for label in agent_labels:
            ev = evaluations[label]
            eps = ev.get("episodes", [])
            vals = [float(ep.get(metric, float("nan"))) for ep in eps]
            data.append([v for v in vals if not np.isnan(v)])

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 1.5})
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)

        ax.set_xticklabels(agent_labels, rotation=15, ha="right")
        ax.set_title(metric_labels.get(metric, metric))
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Action / upgrade distribution  (Experiments 2 / 3)
# ---------------------------------------------------------------------------

def plot_action_distribution(
        evaluations: dict[str, dict[str, Any]],
        top_k: int = 10,
        output_path: Optional[Path | str] = None,
) -> None:
    """
    Bar chart of mean MW invested per candidate line, comparing agents.

    Shows which lines the agents prioritise, a key result for the
    Grid Impact section of the thesis.

    Parameters
    ----------
    evaluations:
        Mapping from agent label to evaluation dict. Requires
        ``action_stats.mean_mw_per_line`` to be present.
    top_k:
        Show only the top-k lines by maximum investment across agents.
    """
    agent_labels = list(evaluations.keys())
    # Collect all line names
    all_lines: list[str] = []
    for ev in evaluations.values():
        lines = list(ev.get("action_stats", {}).get("mean_mw_per_line", {}).keys())
        all_lines = lines
        break
    if not all_lines:
        return

    # Build data matrix: (n_agents, n_lines)
    data = np.zeros((len(agent_labels), len(all_lines)))
    for i, label in enumerate(agent_labels):
        per_line = evaluations[label].get("action_stats", {}).get("mean_mw_per_line", {})
        for j, line in enumerate(all_lines):
            data[i, j] = per_line.get(line, 0.0)

    # Keep only top-k lines by max investment
    max_per_line = data.max(axis=0)
    top_indices = np.argsort(max_per_line)[::-1][:top_k]
    data = data[:, top_indices]
    top_lines = [all_lines[idx] for idx in top_indices]

    x = np.arange(len(top_lines))
    width = 0.8 / len(agent_labels)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(agent_labels))

    fig, ax = plt.subplots(figsize=(max(8, top_k), 5))
    for i, (label, offset) in enumerate(zip(agent_labels, offsets)):
        colour = _PALETTE.get(label.lower(), f"C{i}")
        ax.bar(x + offset, data[i], width=width * 0.9,
               label=label, color=colour, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(top_lines, rotation=40, ha="right")
    ax.set_ylabel("Mean Upgrade (MW)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Line loading before and after reinforcement
# ---------------------------------------------------------------------------

def plot_line_loading_comparison(
        loading_before: pd.Series,
        loading_after: pd.Series,
        output_path: Optional[Path | str] = None,
        top_k: int = 15,
        capacity_threshold: float = 0.85,
) -> None:
    """
    Horizontal bar chart comparing line loading before and after policy.

    Lines are sorted by their pre-policy loading (descending) to
    highlight the most stressed connections.

    Parameters
    ----------
    loading_before:
        Series indexed by line name, values in [0, infinity) fraction of capacity.
    loading_after:
        Same format; represents post-upgrade loading.
    capacity_threshold:
        Reference line drawn at this fraction (default = stability margin).
    top_k:
        Show only the top-k most stressed lines (by pre-policy loading).
    """
    df = pd.DataFrame({"before": loading_before, "after": loading_after}).dropna()
    df = df.nlargest(top_k, "before").iloc[::-1]  # ascending for horizontal bars

    fig, ax = plt.subplots(figsize=(9, max(5, top_k * 0.45)))
    y = np.arange(len(df))
    height = 0.35

    ax.barh(y + height / 2, df["before"], height=height,
            color=_PALETTE["stress"], alpha=0.8, label="Before policy")
    ax.barh(y - height / 2, df["after"], height=height,
            color=_PALETTE["renewable"], alpha=0.8, label="After policy")

    ax.axvline(capacity_threshold, color="black", linestyle="--",
               linewidth=1.0, label=f"Stability margin ({capacity_threshold:.0%})")
    ax.axvline(1.0, color="red", linestyle=":", linewidth=1.0, label="Thermal limit (100%)")

    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Line Loading (Fraction of Capacity)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Network upgrade map
# ---------------------------------------------------------------------------

def plot_network_upgrades(
        network,  # pypsa.Network
        line_upgrades: dict[str, float],
        output_path: Optional[Path | str] = None,
        title: str = "Transmission Expansion Decisions",
        min_upgrade_mw: float = 1.0,
) -> None:
    """
    Spatial map of the network with line widths proportional to MW upgraded.

    Requires the PyPSA network to have valid bus coordinates (x, y).

    Parameters
    ----------
    network:
        ``pypsa.Network`` with ``buses.x``, ``buses.y`` coordinates.
    line_upgrades:
        Dict mapping line name to mean MW upgraded (from ``evaluate_agent``
        ``line_upgrades`` field).
    min_upgrade_mw:
        Lines with less than this upgrade are shown as thin grey lines.
    """
    buses = network.buses
    lines = network.lines

    if "x" not in buses.columns or "y" not in buses.columns:
        return  # no coordinates available

    fig, ax = plt.subplots(figsize=(9, 7))
    try:
        from tep_rl.line_metadata import draw_country_outline

        draw_country_outline(ax, network)
    except Exception:
        pass

    # Draw all lines in grey first
    for _, row in lines.iterrows():
        b0 = buses.loc[row.bus0] if row.bus0 in buses.index else None
        b1 = buses.loc[row.bus1] if row.bus1 in buses.index else None
        if b0 is None or b1 is None:
            continue
        ax.plot(
            [b0.x, b1.x], [b0.y, b1.y],
            color="lightgrey", linewidth=0.8, zorder=1,
        )

    # Draw upgraded lines with width proportional to MW.
    max_upgrade = max((v for v in line_upgrades.values()), default=1.0)
    max_upgrade = max(max_upgrade, 1.0)

    for line_name, upgrade_mw in line_upgrades.items():
        if upgrade_mw < min_upgrade_mw:
            continue
        if line_name not in lines.index:
            continue
        row = lines.loc[line_name]
        b0 = buses.loc[row.bus0] if row.bus0 in buses.index else None
        b1 = buses.loc[row.bus1] if row.bus1 in buses.index else None
        if b0 is None or b1 is None:
            continue
        lw = 1.0 + 6.0 * (upgrade_mw / max_upgrade)
        ax.plot(
            [b0.x, b1.x], [b0.y, b1.y],
            color=_PALETTE["cost"], linewidth=lw, alpha=0.85, zorder=2,
        )
        mid_x = (b0.x + b1.x) / 2
        mid_y = (b0.y + b1.y) / 2
        ax.annotate(
            f"{upgrade_mw:.0f} MW",
            (mid_x, mid_y),
            fontsize=COMPACT_FONT_SIZE,
            ha="center",
            color="darkred",
            zorder=3,
        )

    # Draw buses
    ax.scatter(buses.x, buses.y, s=40, color="steelblue", zorder=4, label="Buses")
    for bus_name, bus_row in buses.iterrows():
        ax.annotate(str(bus_name), (bus_row.x, bus_row.y),
                    fontsize=COMPACT_FONT_SIZE, xytext=(3, 3), textcoords="offset points",
                    color="steelblue", zorder=5)

    # Legend proxy
    legend_patches = [
        mpatches.Patch(color=_PALETTE["cost"], label="Upgraded line (width proportional to MW)"),
        mpatches.Patch(color="lightgrey", label="Existing line (no upgrade)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, output_path)



# Pareto front


def plot_pareto_front(
    sweep_results: dict[tuple[float, ...], dict[str, Any]],
    output_path: Optional[Path | str] = None,
    annotate: bool = True,
) -> None:
    """
    Scatter plot of mean cost vs. renewable share, colour-coded by grid stress.
    """
    if not sweep_results:
        return

    costs, renewable, stress, labels = [], [], [], []
    for weights, payload in sweep_results.items():
        evaluation = payload["evaluation"]
        costs.append(evaluation["total_cost_mean"])
        renewable.append(evaluation["renewable_share_mean"])
        stress.append(evaluation["grid_stress_mean"])
        labels.append("/".join(f"{weight:.2f}" for weight in weights))

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(costs, renewable, c=stress, s=90, cmap="viridis", zorder=3)
    if annotate:
        for x, y, lbl in zip(costs, renewable, labels):
            ax.annotate(lbl, (x, y), xytext=(6, 6),
                        textcoords="offset points", fontsize=ANNOTATION_FONT_SIZE)

    ax.set_xlabel("Mean Total Cost")
    ax.set_ylabel("Mean Renewable Share")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Mean Grid Stress")
    fig.tight_layout()
    _save(fig, output_path)



# Feature importance bar chart


def plot_feature_importance(
        importance_frame: pd.DataFrame,
        output_path: Optional[Path | str] = None,
        top_k: int = 15,
        colour: str = _PALETTE["neutral"],
        error_col: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart of feature importance.

    Parameters
    ----------
    importance_frame:
        DataFrame with at least ``feature`` and one numeric column.
        If ``global_std`` is present (permutation-Shapley output) it is used as
        error bars unless ``error_col`` overrides.
    error_col:
        Column name to use as error bars (half-width).  Auto-detected
        from ``global_std`` or ``*_std`` columns if present.
    """
    frame = importance_frame.head(top_k).iloc[::-1].copy()
    value_col = [c for c in frame.columns if c != "feature"][0]

    # Auto-detect error column
    if error_col is None:
        candidates = [c for c in frame.columns
                      if "std" in c.lower() and c != value_col]
        error_col = candidates[0] if candidates else None

    fig, ax = plt.subplots(figsize=(9, max(5, top_k * 0.45)))
    xerr = frame[error_col].to_numpy() if error_col and error_col in frame.columns else None

    ax.barh(
        frame["feature"],
        frame[value_col],
        xerr=xerr,
        color=colour,
        alpha=0.85,
        capsize=3,
        error_kw={"elinewidth": 0.8, "ecolor": "grey"},
    )
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)



# Temporal permutation-Shapley attribution heatmap


def plot_temporal_attribution(
        temporal_df: pd.DataFrame,
        output_path: Optional[Path | str] = None,
        top_k: int = 12,
) -> None:
    """
    Heatmap of feature Shapley values across decision steps.

    Parameters
    ----------
    temporal_df:
        Long-format DataFrame with columns ``step``, ``feature``,
        ``shapley_mean``.  Returned by
        ``PolicyShapleyExplainer.temporal_attribution``.
    top_k:
        Show only the top-k most important features.
    """
    if temporal_df.empty:
        return

    pivot = temporal_df.pivot(index="feature", columns="step", values="shapley_mean")

    # Keep top-k features by mean importance across steps
    top_features = pivot.mean(axis=1).nlargest(top_k).index
    pivot = pivot.loc[top_features]

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.6), max(5, top_k * 0.45)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"Step {c}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Decision Step")
    ax.set_ylabel("Feature")
    fig.colorbar(im, ax=ax, label="|Shapley value|")
    fig.tight_layout()
    _save(fig, output_path)
