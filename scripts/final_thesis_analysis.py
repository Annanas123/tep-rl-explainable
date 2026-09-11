from __future__ import annotations

"""Aggregate thesis-ready tables, figures, and summaries from finished runs.

The analysis stage is intentionally separate from training so the final report
can be regenerated after text or plotting changes without rerunning the
expensive experiments. Directory lookup uses the shared experiment-layout
registry.
"""

import argparse
import re
import sys
from pathlib import Path

# When this file is executed directly (``python scripts/final_thesis_analysis.py``),
# Python adds ``scripts`` rather than the repository root to ``sys.path``.
# Make the local package importable before importing ``tep_rl`` below.
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
import pypsa
from scipy import stats

from tep_rl.baselines import BASELINE_LABELS
from tep_rl.statistics import compare_paired_evaluations
from tep_rl.visualization import (
    plot_action_distribution,
    plot_metric_boxplots,
    plot_multi_seed_curves,
    plot_network_upgrades,
)

from experiment_layout import locate_results_dir
from thesis_pipeline_utils import (
    combine_episode_records,
    load_seed_histories,
    read_json,
)


POLICY_LABELS = {
    "ppo": "PPO",
    "moppo": "MO-PPO",
    **BASELINE_LABELS,
}

POLICY_ORDER = [
    "ppo",
    "moppo",
    "zero",
    "uniform",
    "myopic_proxy",
    "top_1_excess",
    "top_3_excess",
    "proportional_excess",
]

METRIC_SPECS = [
    ("grid_stress", "Grid Stress"),
    ("total_cost", "Total Cost"),
    ("renewable_share", "Renewable Share"),
    ("total_investment", "Total Investment (MW)"),
    ("active_lines", "Active Lines"),
    ("invested_line_fraction", "Lines Touched"),
    ("renewable_curtailment", "Renewable Curtailment"),
    ("load_shedding", "Load Shedding"),
    ("constraint_violation", "Constraint Violations"),
    ("slack_generation", "Slack Generation"),
    ("emissions", "Emissions"),
]

PRIMARY_TABLE_METRICS = [
    ("grid_stress", "Grid Stress"),
    ("total_cost", "Total Cost"),
    ("renewable_share", "Renewable Share"),
    ("total_investment", "Investment (MW)"),
    ("active_lines", "Active Lines"),
    ("renewable_curtailment", "Curtailment"),
    ("load_shedding", "Load Shedding"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate thesis-ready plots, tables, and narrative summaries from finished experiment runs.")
    parser.add_argument("--results-root", default="results/thesis_final")
    parser.add_argument("--output-dir", default=None)
    return parser


def _policy_sort_key(policy_name: str) -> tuple[int, str]:
    try:
        return POLICY_ORDER.index(policy_name), policy_name
    except ValueError:
        return len(POLICY_ORDER), policy_name


def _safe_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size == 1:
        return mean, float("nan"), float("nan")
    stderr = float(stats.sem(values))
    half_width = stderr * stats.t.ppf((1.0 + confidence) / 2.0, df=values.size - 1)
    return mean, mean - half_width, mean + half_width


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Return Holm--Bonferroni adjusted p-values, preserving missing entries."""
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        return pd.Series(adjusted, index=p_values.index, dtype=float)
    order = valid[np.argsort(values[valid])]
    factors = valid.size - np.arange(valid.size)
    ordered_adjusted = np.maximum.accumulate(values[order] * factors)
    adjusted[order] = np.minimum(ordered_adjusted, 1.0)
    return pd.Series(adjusted, index=p_values.index, dtype=float)


def _format_ci(mean: float, lower: float, upper: float, precision: int = 3) -> str:
    if pd.isna(mean):
        return "-"
    if pd.isna(lower) or pd.isna(upper):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} [{lower:.{precision}f}, {upper:.{precision}f}]"


def _write_table_variants(numeric_table: pd.DataFrame, display_table: pd.DataFrame, base_path: Path, caption: str, label: str) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_table.to_csv(base_path.with_suffix(".csv"), index=False)

    markdown_lines = ["| " + " | ".join(display_table.columns) + " |"]
    markdown_lines.append("| " + " | ".join(["---"] * len(display_table.columns)) + " |")
    for _, row in display_table.iterrows():
        markdown_lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    base_path.with_suffix(".md").write_text("\n".join(markdown_lines), encoding="utf-8")

    latex = display_table.to_latex(index=False, escape=False, caption=caption, label=label)
    base_path.with_suffix(".tex").write_text(latex, encoding="utf-8")


def _load_test_evaluations(agent_dir: Path) -> list[dict]:
    evaluations_by_key: dict[str, dict] = {}
    if not agent_dir.exists():
        return []
    seed_pattern = re.compile(r"seed(\d+)")
    for run_dir in sorted(path for path in agent_dir.iterdir() if path.is_dir()):
        test_eval_path = run_dir / "test_evaluation.json"
        if test_eval_path.exists():
            evaluation = read_json(test_eval_path)
            match = seed_pattern.search(run_dir.name)
            dedup_key = f"seed:{match.group(1)}" if match else run_dir.name
            evaluations_by_key[dedup_key] = evaluation
    return list(evaluations_by_key.values())


def _load_baseline_evaluations(baseline_dir: Path) -> dict[str, list[dict]]:
    evaluations: dict[str, list[dict]] = {}
    if not baseline_dir.exists():
        return evaluations

    nested_dirs = [path for path in baseline_dir.iterdir() if path.is_dir()]
    if nested_dirs:
        for policy_dir in sorted(nested_dirs):
            evaluations[policy_dir.name] = _load_test_evaluations(policy_dir)
        return {name: values for name, values in evaluations.items() if values}

    return evaluations


def _episode_investment_stats(episode: dict, candidate_count: int) -> tuple[float, float, float]:
    actions = episode.get("actions", [])
    if not actions:
        return 0.0, 0.0, 0.0
    action_matrix = [np.asarray(step, dtype=float) for step in actions if step]
    if not action_matrix:
        return 0.0, 0.0, 0.0
    totals = np.sum(np.vstack(action_matrix), axis=0)
    active_lines = int(np.sum(totals > 0.1))
    fraction = active_lines / max(candidate_count, 1)
    return float(totals.sum()), float(active_lines), float(fraction)


def _episode_rows_from_evaluations(policy_name: str, evaluations: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for evaluation in evaluations:
        candidate_count = max(
            len(evaluation.get("line_upgrades", {})),
            len(evaluation.get("action_stats", {}).get("mean_mw_per_line", {})),
            1,
        )
        for episode in evaluation.get("episodes", []):
            total_investment, active_lines, touched_fraction = _episode_investment_stats(episode, candidate_count)
            rows.append(
                {
                    "policy": policy_name,
                    "Policy": POLICY_LABELS.get(policy_name, policy_name),
                    "total_cost": float(episode.get("total_cost", float("nan"))),
                    "grid_stress": float(episode.get("grid_stress", float("nan"))),
                    "renewable_share": float(episode.get("renewable_share", float("nan"))),
                    "constraint_violation": float(episode.get("constraint_violation", float("nan"))),
                    "renewable_curtailment": float(episode.get("renewable_curtailment", float("nan"))),
                    "slack_generation": float(episode.get("slack_generation", float("nan"))),
                    "load_shedding": float(episode.get("load_shedding", float("nan"))),
                    "emissions": float(episode.get("emissions", float("nan"))),
                    "total_investment": total_investment,
                    "active_lines": active_lines,
                    "invested_line_fraction": touched_fraction,
                }
            )
    return rows


def _build_policy_episode_frame(policy_evaluations: dict[str, list[dict]]) -> pd.DataFrame:
    rows: list[dict] = []
    for policy_name, evaluations in policy_evaluations.items():
        rows.extend(_episode_rows_from_evaluations(policy_name, evaluations))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["policy_order"] = frame["policy"].map(lambda value: _policy_sort_key(value)[0])
    return frame.sort_values(["policy_order", "policy"]).drop(columns=["policy_order"])


def _summarise_policy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for policy_name, group in frame.groupby("policy", sort=False):
        row = {
            "policy": policy_name,
            "Policy": POLICY_LABELS.get(policy_name, policy_name),
            "n_episodes": int(len(group)),
        }
        for metric_key, _ in METRIC_SPECS:
            values = group[metric_key].to_numpy(dtype=float)
            mean, ci_lower, ci_upper = _safe_ci(values)
            std = float(np.std(values, ddof=1)) if values.size >= 2 else float("nan")
            row[f"{metric_key}_mean"] = mean
            row[f"{metric_key}_std"] = std
            row[f"{metric_key}_ci_lower"] = ci_lower
            row[f"{metric_key}_ci_upper"] = ci_upper
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["policy_order"] = summary["policy"].map(lambda value: _policy_sort_key(value)[0])
    return summary.sort_values(["policy_order", "policy"]).drop(columns=["policy_order"]).reset_index(drop=True)


def _build_policy_display_table(summary: pd.DataFrame) -> pd.DataFrame:
    display = pd.DataFrame({"Policy": summary["Policy"], "Episodes": summary["n_episodes"]})
    for metric_key, title in PRIMARY_TABLE_METRICS:
        precision = 4 if metric_key == "renewable_share" else 2
        display[title] = summary.apply(
            lambda row: _format_ci(
                row[f"{metric_key}_mean"],
                row[f"{metric_key}_ci_lower"],
                row[f"{metric_key}_ci_upper"],
                precision=precision,
            ),
            axis=1,
        )
    return display


def _build_budget_efficiency_table(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    lookup = {row["policy"]: row for _, row in summary.iterrows()}
    zero_stress = lookup.get("zero", {}).get("grid_stress_mean", float("nan"))
    uniform_stress = lookup.get("uniform", {}).get("grid_stress_mean", float("nan"))
    uniform_cost = lookup.get("uniform", {}).get("total_cost_mean", float("nan"))
    uniform_investment = lookup.get("uniform", {}).get("total_investment_mean", float("nan"))

    rows = []
    for _, row in summary.iterrows():
        investment = float(row.get("total_investment_mean", float("nan")))
        stress = float(row.get("grid_stress_mean", float("nan")))
        total_cost = float(row.get("total_cost_mean", float("nan")))

        stress_reduction_zero = float("nan")
        if not pd.isna(zero_stress) and zero_stress != 0:
            stress_reduction_zero = 100.0 * (zero_stress - stress) / zero_stress

        stress_delta_uniform = float("nan")
        if not pd.isna(uniform_stress) and uniform_stress != 0:
            stress_delta_uniform = 100.0 * (uniform_stress - stress) / uniform_stress

        cost_delta_uniform = float("nan")
        if not pd.isna(uniform_cost) and uniform_cost != 0:
            cost_delta_uniform = 100.0 * (uniform_cost - total_cost) / uniform_cost

        investment_delta_uniform = float("nan")
        if not pd.isna(uniform_investment) and uniform_investment != 0:
            investment_delta_uniform = 100.0 * (uniform_investment - investment) / uniform_investment

        stress_reduction_per_100_mw = float("nan")
        if not pd.isna(zero_stress) and investment > 0:
            stress_reduction_per_100_mw = 100.0 * (zero_stress - stress) / investment

        rows.append(
            {
                "policy": row["policy"],
                "Policy": row["Policy"],
                "stress_reduction_vs_zero_pct": stress_reduction_zero,
                "stress_delta_vs_uniform_pct": stress_delta_uniform,
                "cost_delta_vs_uniform_pct": cost_delta_uniform,
                "investment_delta_vs_uniform_pct": investment_delta_uniform,
                "stress_reduction_per_100_mw": stress_reduction_per_100_mw,
                "total_investment_mean": investment,
            }
        )

    numeric = pd.DataFrame(rows)
    numeric["policy_order"] = numeric["policy"].map(lambda value: _policy_sort_key(value)[0])
    numeric = numeric.sort_values(["policy_order", "policy"]).drop(columns=["policy_order"]).reset_index(drop=True)

    display = numeric.copy()
    for column in (
        "stress_reduction_vs_zero_pct",
        "stress_delta_vs_uniform_pct",
        "cost_delta_vs_uniform_pct",
        "investment_delta_vs_uniform_pct",
        "stress_reduction_per_100_mw",
    ):
        display[column] = display[column].map(lambda value: "-" if pd.isna(value) else f"{value:.2f}")
    display["total_investment_mean"] = display["total_investment_mean"].map(lambda value: "-" if pd.isna(value) else f"{value:.2f}")
    display = display.rename(
        columns={
            "stress_reduction_vs_zero_pct": "Stress Reduction vs Zero (%)",
            "stress_delta_vs_uniform_pct": "Stress Delta vs Uniform (%)",
            "cost_delta_vs_uniform_pct": "Cost Delta vs Uniform (%)",
            "investment_delta_vs_uniform_pct": "Investment Delta vs Uniform (%)",
            "stress_reduction_per_100_mw": "Stress Reduction per 100 MW",
            "total_investment_mean": "Investment (MW)",
        }
    )
    return numeric, display[
        [
            "Policy",
            "Stress Reduction vs Zero (%)",
            "Stress Delta vs Uniform (%)",
            "Cost Delta vs Uniform (%)",
            "Investment Delta vs Uniform (%)",
            "Stress Reduction per 100 MW",
            "Investment (MW)",
        ]
    ]


def _build_policy_seed_frame(policy_evaluations: dict[str, list[dict]]) -> pd.DataFrame:
    rows: list[dict] = []
    for policy_name, evaluations in policy_evaluations.items():
        for seed_index, evaluation in enumerate(evaluations):
            action_stats = evaluation.get("action_stats", {})
            rows.append(
                {
                    "policy": policy_name,
                    "Policy": POLICY_LABELS.get(policy_name, policy_name),
                    "seed_index": seed_index,
                    "total_cost_mean": float(evaluation.get("total_cost_mean", float("nan"))),
                    "grid_stress_mean": float(evaluation.get("grid_stress_mean", float("nan"))),
                    "renewable_share_mean": float(evaluation.get("renewable_share_mean", float("nan"))),
                    "renewable_curtailment_mean": float(evaluation.get("renewable_curtailment_mean", float("nan"))),
                    "load_shedding_mean": float(evaluation.get("load_shedding_mean", float("nan"))),
                    "emissions_mean": float(evaluation.get("emissions_mean", float("nan"))),
                    "constraint_violation_mean": float(evaluation.get("constraint_violation_mean", float("nan"))),
                    "total_investment_mean": float(action_stats.get("total_investment_mean", float("nan"))),
                    "fraction_lines_touched": float(action_stats.get("fraction_lines_touched", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def _build_significance_table(policy_evaluations: dict[str, list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = (
        "grid_stress_mean",
        "total_cost_mean",
        "renewable_share_mean",
        "renewable_curtailment_mean",
        "load_shedding_mean",
        "emissions_mean",
        "total_investment_mean",
    )
    metric_labels = {
        "grid_stress_mean": "Grid Stress",
        "total_cost_mean": "Total Cost",
        "renewable_share_mean": "Renewable Share",
        "renewable_curtailment_mean": "Curtailment",
        "load_shedding_mean": "Load Shedding",
        "emissions_mean": "Emissions",
        "total_investment_mean": "Investment (MW)",
    }
    numeric_rows: list[pd.DataFrame] = []
    learned_policies = [name for name in ("ppo", "moppo") if policy_evaluations.get(name)]
    baseline_policies = [name for name in POLICY_ORDER if name not in {"ppo", "moppo"} and policy_evaluations.get(name)]

    for learned_name in learned_policies:
        learned_evaluations = policy_evaluations.get(learned_name, [])
        for baseline_name in baseline_policies:
            baseline_evaluations = policy_evaluations.get(baseline_name, [])
            comparison = compare_paired_evaluations(
                learned_evaluations,
                baseline_evaluations,
                label_a=POLICY_LABELS.get(learned_name, learned_name),
                label_b=POLICY_LABELS.get(baseline_name, baseline_name),
                alpha=0.05,
                metrics=metrics,
            )
            if comparison.empty:
                continue
            comparison.insert(0, "baseline", baseline_name)
            comparison.insert(0, "policy", learned_name)
            numeric_rows.append(comparison)

    if not numeric_rows:
        return pd.DataFrame(), pd.DataFrame()

    numeric = pd.concat(numeric_rows, ignore_index=True)
    numeric["Policy"] = numeric["policy"].map(lambda value: POLICY_LABELS.get(value, value))
    numeric["Baseline"] = numeric["baseline"].map(lambda value: POLICY_LABELS.get(value, value))
    numeric["Metric"] = numeric["metric"].map(lambda value: metric_labels.get(value, value))
    numeric["Significant (paired p<0.05)"] = numeric["significant"].map(lambda value: "yes" if value else "no")
    # Treat every learned-policy/baseline/metric contrast in one evaluation
    # setting as one family. Keep paired-t and exact Wilcoxon families separate.
    numeric["paired_p_holm"] = _holm_adjust(numeric["paired_p"])
    numeric["wilcoxon_p_holm"] = _holm_adjust(numeric["wilcoxon_p"])
    numeric["Holm significant (paired t)"] = numeric["paired_p_holm"].map(
        lambda value: "yes" if pd.notna(value) and value < 0.05 else "no"
    )
    numeric["Holm significant (Wilcoxon)"] = numeric["wilcoxon_p_holm"].map(
        lambda value: "yes" if pd.notna(value) and value < 0.05 else "no"
    )
    numeric["Better"] = numeric["better"]

    display = numeric[
        [
            "Policy",
            "Baseline",
            "Metric",
            "paired_p",
            "paired_p_holm",
            "wilcoxon_p",
            "wilcoxon_p_holm",
            "mean_paired_difference",
            "crossed_bootstrap_ci_lower",
            "crossed_bootstrap_ci_upper",
            "Holm significant (paired t)",
            "Holm significant (Wilcoxon)",
            "Better",
        ]
    ].copy()
    for column in ("paired_p", "paired_p_holm", "wilcoxon_p", "wilcoxon_p_holm"):
        display[column] = display[column].map(lambda value: "-" if pd.isna(value) else f"{value:.4f}")
    display = display.rename(
        columns={
            "paired_p": "Paired t p",
            "paired_p_holm": "Paired t Holm p",
            "wilcoxon_p": "Wilcoxon p",
            "wilcoxon_p_holm": "Wilcoxon Holm p",
        }
    )
    return numeric, display


def _write_rl_comparison(policy_evaluations: dict[str, list[dict]], output_path: Path) -> None:
    ppo_evaluations = policy_evaluations.get("ppo", [])
    moppo_evaluations = policy_evaluations.get("moppo", [])
    if not ppo_evaluations or not moppo_evaluations:
        return

    metrics = (
        "total_cost_mean",
        "renewable_share_mean",
        "grid_stress_mean",
        "load_shedding_mean",
        "renewable_curtailment_mean",
        "constraint_violation_mean",
        "emissions_mean",
    )
    comparison = compare_paired_evaluations(
        ppo_evaluations,
        moppo_evaluations,
        label_a="PPO",
        label_b="MO-PPO",
        alpha=0.05,
        metrics=metrics,
    )
    if not comparison.empty:
        comparison["paired_p_holm"] = _holm_adjust(comparison["paired_p"])
        comparison["wilcoxon_p_holm"] = _holm_adjust(comparison["wilcoxon_p"])
        comparison.to_csv(output_path, index=False)


def _write_upgrade_maps(
    output_dir: Path,
    results_root: Path,
    proxy_policy_evaluations: dict[str, list[dict]],
    proxy_summary: pd.DataFrame,
) -> list[Path]:
    manifest_path = results_root / "pipeline_manifest.json"
    if not manifest_path.exists():
        return []

    manifest = read_json(manifest_path)
    network_path = REPO_ROOT / str(manifest.get("network", ""))
    if not network_path.exists():
        return []

    network = pypsa.Network(network_path)
    generated_paths: list[Path] = []

    for policy_name in ("ppo", "moppo"):
        evaluations = proxy_policy_evaluations.get(policy_name, [])
        if not evaluations:
            continue
        payload = combine_episode_records(evaluations)
        line_upgrades = payload.get("action_stats", {}).get("mean_mw_per_line", {})
        output_path = output_dir / f"{policy_name}_upgrade_map.png"
        plot_network_upgrades(
            network,
            line_upgrades=line_upgrades,
            output_path=output_path,
            title=f"{POLICY_LABELS.get(policy_name, policy_name)} Upgrade Map",
        )
        if output_path.exists():
            generated_paths.append(output_path)

    baseline_candidates = proxy_summary[~proxy_summary["policy"].isin(["ppo", "moppo"])].copy()
    if not baseline_candidates.empty:
        best_baseline = str(baseline_candidates.sort_values("grid_stress_mean").iloc[0]["policy"])
        evaluations = proxy_policy_evaluations.get(best_baseline, [])
        if evaluations:
            payload = combine_episode_records(evaluations)
            line_upgrades = payload.get("action_stats", {}).get("mean_mw_per_line", {})
            output_path = output_dir / "best_baseline_upgrade_map.png"
            plot_network_upgrades(
                network,
                line_upgrades=line_upgrades,
                output_path=output_path,
                title=f"{POLICY_LABELS.get(best_baseline, best_baseline)} Upgrade Map",
            )
            if output_path.exists():
                generated_paths.append(output_path)
    return generated_paths


def _load_runtime_benchmark(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {}
    return {str(name): stats for name, stats in payload.items() if isinstance(stats, dict)}


def _write_runtime_table(runtime_payload: dict[str, dict], output_path: Path, caption: str, label: str) -> None:
    if not runtime_payload:
        return
    rows: list[dict] = []
    for policy_name, stats in runtime_payload.items():
        rows.append(
            {
                "policy": policy_name,
                "Policy": POLICY_LABELS.get(policy_name, policy_name),
                "seconds_per_episode": float(stats.get("seconds_per_episode", float("nan"))),
                "elapsed_seconds": float(stats.get("elapsed_seconds", float("nan"))),
                "episodes": int(stats.get("episodes", 0)),
                "grid_stress_mean": float(stats.get("grid_stress_mean", float("nan"))),
                "total_cost_mean": float(stats.get("total_cost_mean", float("nan"))),
            }
        )
    numeric = pd.DataFrame(rows)
    numeric["policy_order"] = numeric["policy"].map(lambda value: _policy_sort_key(value)[0])
    numeric = numeric.sort_values(["policy_order", "policy"]).drop(columns=["policy_order"]).reset_index(drop=True)
    display = numeric.copy()
    for column in ("seconds_per_episode", "elapsed_seconds", "grid_stress_mean", "total_cost_mean"):
        display[column] = display[column].map(lambda value: "-" if pd.isna(value) else f"{value:.2f}")
    display = display.rename(
        columns={
            "seconds_per_episode": "Seconds per Episode",
            "elapsed_seconds": "Elapsed Seconds",
            "grid_stress_mean": "Grid Stress",
            "total_cost_mean": "Total Cost",
            "episodes": "Episodes",
        }
    )
    _write_table_variants(
        numeric,
        display[["Policy", "Episodes", "Seconds per Episode", "Elapsed Seconds", "Grid Stress", "Total Cost"]],
        output_path,
        caption=caption,
        label=label,
    )


def _visualization_payload_from_evaluations(policy_evaluations: dict[str, list[dict]]) -> dict[str, dict]:
    payload: dict[str, dict] = {}
    for policy_name, evaluations in policy_evaluations.items():
        if not evaluations:
            continue
        label = POLICY_LABELS.get(policy_name, policy_name)
        if len(evaluations) == 1:
            payload[label] = evaluations[0]
        else:
            payload[label] = combine_episode_records(evaluations)
    return payload


def _write_thesis_story(
    output_path: Path,
    summary: pd.DataFrame,
    efficiency: pd.DataFrame,
    results_root: Path,
    evaluation_label: str,
    runtime_benchmark: dict[str, dict] | None = None,
) -> None:
    if summary.empty:
        output_path.write_text("# Thesis Story\n\nNo policy evaluations were found.\n", encoding="utf-8")
        return

    learned = summary[summary["policy"].isin(["ppo", "moppo"])].copy()
    baselines = summary[~summary["policy"].isin(["ppo", "moppo"])].copy()
    stress_degenerate = (
        evaluation_label == "fullenv"
        and "grid_stress_mean" in summary.columns
        and np.nanmax(np.abs(summary["grid_stress_mean"].to_numpy(dtype=float))) < 1e-9
    )
    if stress_degenerate:
        best_learned = learned.sort_values(
            ["load_shedding_mean", "total_cost_mean", "renewable_share_mean"],
            ascending=[True, True, False],
        ).head(1)
        best_baseline = (
            baselines.sort_values(
                ["load_shedding_mean", "total_cost_mean", "renewable_share_mean"],
                ascending=[True, True, False],
            ).head(1)
            if not baselines.empty
            else pd.DataFrame()
        )
    else:
        best_learned = learned.sort_values("grid_stress_mean").head(1)
        best_baseline = baselines.sort_values("grid_stress_mean").head(1) if not baselines.empty else pd.DataFrame()
    most_efficient = efficiency.sort_values("stress_reduction_per_100_mw", ascending=False).head(1) if not efficiency.empty else pd.DataFrame()
    manifest_path = results_root / "pipeline_manifest.json"
    third_objective_mode = "renewable_share"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        third_objective_mode = str(manifest.get("environment", {}).get("third_objective_mode", "renewable_share"))

    lines = [
        "# Thesis-Oriented Result Framing",
        "",
        "## Research Question",
        "",
        "Can a learned policy allocate a fixed transmission-upgrade budget more selectively than simple heuristics, and thereby reduce congestion on unseen years under time-varying load and renewable availability?",
        "",
        "## Recommended Framing",
        "",
        "Budget-efficient congestion relief with learned selective transmission upgrades under time-varying load and renewables.",
        "",
        f"Headline evaluation source: `{evaluation_label}`.",
        "",
        f"Configured third objective mode: `{third_objective_mode}`.",
        "",
        "## Headline Findings",
        "",
    ]

    if stress_degenerate:
        lines.extend(
            [
                "- In strict full-environment evaluation, PyPSA enforces feasibility, so grid stress is zero for every policy. The meaningful outcomes are load shedding, total cost, renewable share, and investment cost.",
            ]
        )
        if not best_learned.empty:
            row = best_learned.iloc[0]
            lines.extend(
                [
                    f"- Best learned policy on strict full-environment adequacy/cost: {row['Policy']} with mean load shedding {row['load_shedding_mean']:.2f}, mean cost {row['total_cost_mean']:.2f}, renewable share {row['renewable_share_mean']:.4f}, and mean investment {row['total_investment_mean']:.2f} MW.",
                ]
            )
        if not best_baseline.empty:
            row = best_baseline.iloc[0]
            lines.extend(
                [
                    f"- Strongest baseline on the same criteria: {row['Policy']} with mean load shedding {row['load_shedding_mean']:.2f}, mean cost {row['total_cost_mean']:.2f}, renewable share {row['renewable_share_mean']:.4f}, and mean investment {row['total_investment_mean']:.2f} MW.",
                ]
            )
    else:
        if not best_learned.empty:
            row = best_learned.iloc[0]
            lines.extend(
                [
                    f"- Best learned policy on grid stress: {row['Policy']} with mean stress {row['grid_stress_mean']:.2f}, mean cost {row['total_cost_mean']:.2f}, and mean investment {row['total_investment_mean']:.2f} MW.",
                ]
            )
        if not best_baseline.empty:
            row = best_baseline.iloc[0]
            lines.extend(
                [
                    f"- Strongest heuristic baseline on grid stress: {row['Policy']} with mean stress {row['grid_stress_mean']:.2f} and mean investment {row['total_investment_mean']:.2f} MW.",
                ]
            )
        if not most_efficient.empty and np.isfinite(float(most_efficient.iloc[0]["stress_reduction_per_100_mw"])):
            row = most_efficient.iloc[0]
            lines.extend(
                [
                    f"- Highest stress reduction per invested 100 MW: {row['Policy']} at {row['stress_reduction_per_100_mw']:.2f}.",
                ]
            )
        elif baselines.empty:
            lines.extend(
                [
                    "- Heuristic baseline outputs were not found in this results folder, so the budget-efficiency ranking is omitted.",
                ]
            )
    if runtime_benchmark:
        zero_runtime = runtime_benchmark.get("zero", {})
        myopic_runtime = runtime_benchmark.get("myopic_proxy", {})
        zero_s = float(zero_runtime.get("seconds_per_episode", float("nan")))
        myopic_s = float(myopic_runtime.get("seconds_per_episode", float("nan")))
        if np.isfinite(zero_s) and np.isfinite(myopic_s) and zero_s > 0:
            lines.extend(
                [
                    f"- Runtime benchmark: `myopic_proxy` required {myopic_s:.2f} s/episode versus {zero_s:.2f} s/episode for `zero`, a {myopic_s / zero_s:.2f}x overhead for a stronger optimisation-style baseline.",
                ]
            )

    lines.extend(
        [
            "",
            "## What To Show In The Thesis",
            "",
            "- Use the policy comparison table as the main quantitative result, not the raw learning curves.",
        ]
    )
    if stress_degenerate:
        lines.extend(
            [
                "- For strict full-environment results, emphasise load shedding, total cost, renewable share, and investment cost instead of grid stress.",
                "- Use the proxy section for congestion-learning evidence and the full-environment section for transfer and operational relevance.",
                "- Show the action-distribution figure to demonstrate selective corridor reinforcement rather than diffuse spending.",
                "- Treat PPO vs MO-PPO as a methodological comparison inside a broader planning study with heuristic baselines.",
            ]
        )
    else:
        lines.extend(
            [
                "- Emphasise stress reduction per MW invested and investment savings relative to `uniform`.",
                "- Show the action-distribution figure to demonstrate selective corridor reinforcement rather than diffuse spending.",
                "- Treat PPO vs MO-PPO as a methodological comparison inside a broader planning study with heuristic baselines.",
            ]
        )
    lines.extend(
        [
            "",
            "## Output Locations",
            "",
            f"- Proxy comparison tables and figures live under `{output_path.parent}`.",
            f"- Source experiment root: `{results_root}`.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    ppo_validation_dir = locate_results_dir(results_root, "ppo_validation")
    moppo_validation_dir = locate_results_dir(results_root, "moppo_validation")
    proxy_evaluation_dir = locate_results_dir(results_root, "proxy_evaluation") / "proxy"
    fullenv_evaluation_dir = locate_results_dir(results_root, "fullenv_evaluation")
    benchmark_dir = locate_results_dir(results_root, "baseline_benchmark")
    proxy_runtime_benchmark = _load_runtime_benchmark(benchmark_dir / "proxy_baseline_runtime.json")
    full_runtime_benchmark = _load_runtime_benchmark(benchmark_dir / "full_baseline_runtime.json")

    ppo_histories = load_seed_histories(ppo_validation_dir)
    moppo_histories = load_seed_histories(moppo_validation_dir)
    if ppo_histories:
        plot_multi_seed_curves(ppo_histories, output_dir / "ppo_learning_curves.png", label="PPO")
    if moppo_histories:
        plot_multi_seed_curves(moppo_histories, output_dir / "moppo_learning_curves.png", label="MO-PPO")

    proxy_policy_evaluations: dict[str, list[dict]] = {
        "ppo": _load_test_evaluations(proxy_evaluation_dir / "ppo"),
        "moppo": _load_test_evaluations(proxy_evaluation_dir / "moppo"),
    }
    for baseline_name, evaluations in _load_baseline_evaluations(proxy_evaluation_dir / "baselines").items():
        proxy_policy_evaluations[baseline_name] = evaluations

    proxy_episode_frame = _build_policy_episode_frame(proxy_policy_evaluations)
    proxy_summary = pd.DataFrame()
    proxy_efficiency_numeric = pd.DataFrame()
    if not proxy_episode_frame.empty:
        proxy_episode_frame.to_csv(output_dir / "proxy_policy_episode_records.csv", index=False)
        proxy_summary = _summarise_policy_frame(proxy_episode_frame)
        proxy_summary.to_csv(output_dir / "proxy_policy_summary_numeric.csv", index=False)
        proxy_seed_frame = _build_policy_seed_frame(proxy_policy_evaluations)
        proxy_seed_frame.to_csv(output_dir / "proxy_policy_seed_records.csv", index=False)

        proxy_display = _build_policy_display_table(proxy_summary)
        _write_table_variants(
            proxy_summary,
            proxy_display,
            output_dir / "proxy_policy_comparison",
            caption="Held-out proxy test results with RL agents and heuristic baselines.",
            label="tab:proxy_policy_comparison",
        )

        efficiency_numeric, efficiency_display = _build_budget_efficiency_table(proxy_summary)
        proxy_efficiency_numeric = efficiency_numeric
        _write_table_variants(
            efficiency_numeric,
            efficiency_display,
            output_dir / "proxy_budget_efficiency",
            caption="Budget-efficiency comparison on the held-out proxy test window.",
            label="tab:proxy_budget_efficiency",
        )

        significance_numeric, significance_display = _build_significance_table(proxy_policy_evaluations)
        if not significance_numeric.empty:
            _write_table_variants(
                significance_numeric,
                significance_display,
                output_dir / "proxy_significance_vs_baselines",
                caption="Seed-level statistical comparison of learned policies against heuristic baselines on the proxy test split.",
                label="tab:proxy_significance_vs_baselines",
            )

        proxy_visual_payload = _visualization_payload_from_evaluations(proxy_policy_evaluations)
        if proxy_visual_payload:
            plot_metric_boxplots(proxy_visual_payload, output_path=output_dir / "proxy_policy_boxplots.png")
            plot_action_distribution(proxy_visual_payload, output_path=output_dir / "proxy_action_distribution.png")
        _write_upgrade_maps(output_dir, results_root, proxy_policy_evaluations, proxy_summary)
        if proxy_runtime_benchmark:
            _write_runtime_table(
                proxy_runtime_benchmark,
                output_dir / "proxy_baseline_runtime",
                caption="Runtime benchmark for heuristic baselines on the proxy environment.",
                label="tab:proxy_baseline_runtime",
            )
        _write_thesis_story(
            output_dir / "proxy_thesis_story.md",
            proxy_summary,
            efficiency_numeric,
            results_root,
            evaluation_label="proxy",
            runtime_benchmark=proxy_runtime_benchmark,
        )
        _write_rl_comparison(proxy_policy_evaluations, output_dir / "test_proxy_ppo_vs_moppo_statistics.csv")

    comparison_dir = locate_results_dir(results_root, "agent_comparison")
    if (comparison_dir / "validation_ppo_vs_moppo.csv").exists():
        pd.read_csv(comparison_dir / "validation_ppo_vs_moppo.csv").to_csv(
            output_dir / "validation_statistics.csv",
            index=False,
        )
    full_policy_evaluations: dict[str, list[dict]] = {
        "ppo": _load_test_evaluations(fullenv_evaluation_dir / "ppo"),
        "moppo": _load_test_evaluations(fullenv_evaluation_dir / "moppo"),
    }
    for baseline_name, evaluations in _load_baseline_evaluations(fullenv_evaluation_dir / "baselines").items():
        full_policy_evaluations[baseline_name] = evaluations

    full_episode_frame = _build_policy_episode_frame(full_policy_evaluations)
    full_summary = pd.DataFrame()
    full_efficiency_numeric = pd.DataFrame()
    if not full_episode_frame.empty:
        full_episode_frame.to_csv(output_dir / "fullenv_policy_episode_records.csv", index=False)
        full_summary = _summarise_policy_frame(full_episode_frame)
        full_display = _build_policy_display_table(full_summary)
        _write_table_variants(
            full_summary,
            full_display,
            output_dir / "fullenv_policy_comparison",
            caption="Full-environment test results with RL agents and heuristic baselines.",
            label="tab:fullenv_policy_comparison",
        )
        full_efficiency_numeric, full_efficiency_display = _build_budget_efficiency_table(full_summary)
        _write_table_variants(
            full_efficiency_numeric,
            full_efficiency_display,
            output_dir / "fullenv_budget_efficiency",
            caption="Budget-efficiency comparison on the full-environment test window.",
            label="tab:fullenv_budget_efficiency",
        )
        full_significance_numeric, full_significance_display = _build_significance_table(full_policy_evaluations)
        if not full_significance_numeric.empty:
            _write_table_variants(
                full_significance_numeric,
                full_significance_display,
                output_dir / "fullenv_significance_vs_baselines",
                caption="Seed-level statistical comparison of learned policies against heuristic baselines on the full-environment test split.",
                label="tab:fullenv_significance_vs_baselines",
            )
        if full_runtime_benchmark:
            _write_runtime_table(
                full_runtime_benchmark,
                output_dir / "fullenv_baseline_runtime",
                caption="Runtime benchmark for heuristic baselines on the full environment.",
                label="tab:fullenv_baseline_runtime",
            )
        _write_rl_comparison(full_policy_evaluations, output_dir / "test_fullenv_ppo_vs_moppo_statistics.csv")

    if not full_summary.empty:
        main_display = _build_policy_display_table(full_summary)
        _write_table_variants(
            full_summary,
            main_display,
            output_dir / "main_policy_comparison",
            caption="Main thesis result table on the full environment.",
            label="tab:main_policy_comparison",
        )
        if not full_efficiency_numeric.empty:
            _, main_efficiency_display = _build_budget_efficiency_table(full_summary)
            _write_table_variants(
                full_efficiency_numeric,
                main_efficiency_display,
                output_dir / "main_budget_efficiency",
                caption="Main thesis budget-efficiency table on the full environment.",
                label="tab:main_budget_efficiency",
            )
        _write_thesis_story(
            output_dir / "thesis_story.md",
            full_summary,
            full_efficiency_numeric,
            results_root,
            evaluation_label="fullenv",
            runtime_benchmark=full_runtime_benchmark,
        )
    elif not proxy_summary.empty:
        main_display = _build_policy_display_table(proxy_summary)
        _write_table_variants(
            proxy_summary,
            main_display,
            output_dir / "main_policy_comparison",
            caption="Main thesis result table on the proxy environment.",
            label="tab:main_policy_comparison",
        )
        if not proxy_efficiency_numeric.empty:
            _, main_efficiency_display = _build_budget_efficiency_table(proxy_summary)
            _write_table_variants(
                proxy_efficiency_numeric,
                main_efficiency_display,
                output_dir / "main_budget_efficiency",
                caption="Main thesis budget-efficiency table on the proxy environment.",
                label="tab:main_budget_efficiency",
            )
        _write_thesis_story(
            output_dir / "thesis_story.md",
            proxy_summary,
            proxy_efficiency_numeric,
            results_root,
            evaluation_label="proxy",
            runtime_benchmark=proxy_runtime_benchmark,
        )

    readme_lines = [
        "# Thesis Analysis Outputs",
        "",
        f"- Main policy comparison table: `{output_dir / 'main_policy_comparison.md'}`",
        f"- Main budget-efficiency table: `{output_dir / 'main_budget_efficiency.md'}`",
        f"- Learning curves: `{output_dir / 'ppo_learning_curves.png'}` and `{output_dir / 'moppo_learning_curves.png'}`",
        f"- Proxy policy comparison table: `{output_dir / 'proxy_policy_comparison.md'}`",
        f"- Proxy budget-efficiency table: `{output_dir / 'proxy_budget_efficiency.md'}`",
        f"- Proxy distribution figures: `{output_dir / 'proxy_policy_boxplots.png'}` and `{output_dir / 'proxy_action_distribution.png'}`",
        f"- Thesis framing summary: `{output_dir / 'thesis_story.md'}`",
    ]
    if (output_dir / "proxy_significance_vs_baselines.md").exists():
        readme_lines.append(f"- Proxy significance table: `{output_dir / 'proxy_significance_vs_baselines.md'}`")
    for map_name in ("ppo_upgrade_map.png", "moppo_upgrade_map.png", "best_baseline_upgrade_map.png"):
        if (output_dir / map_name).exists():
            readme_lines.append(f"- Upgrade map: `{output_dir / map_name}`")
    if (output_dir / "fullenv_policy_comparison.md").exists():
        readme_lines.append(f"- Full-environment comparison table: `{output_dir / 'fullenv_policy_comparison.md'}`")
    if (output_dir / "fullenv_significance_vs_baselines.md").exists():
        readme_lines.append(f"- Full-environment significance table: `{output_dir / 'fullenv_significance_vs_baselines.md'}`")
    if (output_dir / "proxy_baseline_runtime.md").exists():
        readme_lines.append(f"- Proxy baseline runtime table: `{output_dir / 'proxy_baseline_runtime.md'}`")
    if (output_dir / "fullenv_baseline_runtime.md").exists():
        readme_lines.append(f"- Full-environment baseline runtime table: `{output_dir / 'fullenv_baseline_runtime.md'}`")
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Wrote thesis analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
