"""
statistics.py - Statistical analysis utilities for multi-seed RL experiments.

Implements the significance-testing protocol specified in the thesis runbook
and the ML checklist requirements:

  * ``aggregate_seed_results`` - collect per-seed metrics into arrays
  * ``compute_confidence_interval`` - bootstrap or t-based CI
  * ``compare_agents`` - paired t/Wilcoxon tests for shared seeds/windows
  * ``summarise_experiment`` - full summary table ready for the thesis

All functions return plain Python dicts / DataFrames so results can be
serialised to JSON or CSV without extra dependencies beyond scipy and pandas.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_METRICS: tuple[str, ...] = (
    "total_cost_mean",
    "renewable_share_mean",
    "grid_stress_mean",
    "load_shedding_mean",
    "renewable_curtailment_mean",
    "constraint_violation_mean",
    "emissions_mean",
)


def aggregate_seed_results(
    seed_results: Sequence[dict[str, Any]],
    metrics: Sequence[str] = PRIMARY_METRICS,
) -> pd.DataFrame:
    """
    Collect evaluation dicts from multiple seeds into a tidy DataFrame.
    """

    def _extract_metric(result: dict[str, Any], metric: str) -> float:
        if metric in result:
            return result.get(metric, float("nan"))
        action_stats = result.get("action_stats", {})
        if metric in action_stats:
            return action_stats.get(metric, float("nan"))
        return float("nan")

    rows = []
    for result in seed_results:
        rows.append({metric: _extract_metric(result, metric) for metric in metrics})
    return pd.DataFrame(rows, columns=list(metrics))


def compute_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Compute mean and symmetric confidence interval via Student's t-distribution.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2:
        mean = float(arr.mean()) if n == 1 else float("nan")
        return mean, float("nan"), float("nan")
    mean = float(arr.mean())
    se = float(stats.sem(arr))
    interval = se * stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    return mean, mean - interval, mean + interval


def compare_agents(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    label_a: str = "A",
    label_b: str = "B",
    alpha: float = 0.05,
    metrics: Sequence[str] = PRIMARY_METRICS,
) -> pd.DataFrame:
    """
    Compare two agents using paired observations (normally identical training
    seeds evaluated on identical operating windows). Independent-sample tests
    are inappropriate for this common-random-number design.
    """

    rows = []
    for metric in metrics:
        if metric not in results_a.columns or metric not in results_b.columns:
            continue

        a = results_a[metric].dropna().to_numpy(dtype=float)
        b = results_b[metric].dropna().to_numpy(dtype=float)

        if len(a) != len(b):
            raise ValueError(
                f"Paired comparison for {metric!r} requires equal sample counts; got {len(a)} and {len(b)}."
            )

        if len(a) < 2:
            rows.append(
                {
                    "metric": metric,
                    f"mean_{label_a}": float(a.mean()) if len(a) else float("nan"),
                    f"std_{label_a}": float(a.std()) if len(a) else float("nan"),
                    f"mean_{label_b}": float(b.mean()) if len(b) else float("nan"),
                    f"std_{label_b}": float(b.std()) if len(b) else float("nan"),
                    "n_pairs": len(a),
                    "mean_paired_difference": float("nan"),
                    "difference_ci_lower": float("nan"),
                    "difference_ci_upper": float("nan"),
                    "paired_t": float("nan"),
                    "paired_p": float("nan"),
                    "wilcoxon_w": float("nan"),
                    "wilcoxon_p": float("nan"),
                    "cohens_dz": float("nan"),
                    "significant": False,
                    "better": "-",
                }
            )
            continue

        differences = a - b
        if np.allclose(differences, differences[0]):
            if np.isclose(differences[0], 0.0):
                paired_t, paired_p = 0.0, 1.0
            else:
                paired_t = float("inf") * np.sign(differences[0])
                paired_p = 0.0
        else:
            paired_t, paired_p = stats.ttest_rel(a, b)
        if np.allclose(differences, 0.0):
            wilcoxon_w, wilcoxon_p = 0.0, 1.0
        else:
            wilcoxon = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
            wilcoxon_w, wilcoxon_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
        diff_mean, diff_ci_lower, diff_ci_upper = compute_confidence_interval(differences)
        diff_std = float(np.std(differences, ddof=1))
        if diff_std > 1e-12:
            cohens_dz = diff_mean / diff_std
        elif np.isclose(diff_mean, 0.0):
            cohens_dz = 0.0
        else:
            cohens_dz = float("inf") * np.sign(diff_mean)
        significant = bool(np.isfinite(paired_p) and paired_p < alpha)

        lower_is_better = (
            "cost" in metric
            or "stress" in metric
            or "emission" in metric
            or "violation" in metric
            or "curtail" in metric
            or "shedding" in metric
        )
        if not significant:
            better = "-"
        elif lower_is_better:
            better = label_a if a.mean() < b.mean() else label_b
        else:
            better = label_a if a.mean() > b.mean() else label_b

        rows.append(
            {
                "metric": metric,
                f"mean_{label_a}": float(a.mean()),
                f"std_{label_a}": float(a.std()),
                f"mean_{label_b}": float(b.mean()),
                f"std_{label_b}": float(b.std()),
                "n_pairs": len(a),
                "mean_paired_difference": float(diff_mean),
                "difference_ci_lower": float(diff_ci_lower),
                "difference_ci_upper": float(diff_ci_upper),
                "paired_t": float(paired_t),
                "paired_p": float(paired_p),
                "wilcoxon_w": float(wilcoxon_w),
                "wilcoxon_p": float(wilcoxon_p),
                "cohens_dz": float(cohens_dz),
                "significant": significant,
                "better": better,
            }
        )

    return pd.DataFrame(rows)


def compare_paired_evaluations(
    evaluations_a: Sequence[dict[str, Any]],
    evaluations_b: Sequence[dict[str, Any]],
    label_a: str = "A",
    label_b: str = "B",
    alpha: float = 0.05,
    metrics: Sequence[str] = PRIMARY_METRICS,
    bootstrap_samples: int = 5000,
    seed: int = 7,
) -> pd.DataFrame:
    """Paired seed inference plus a crossed seed/window bootstrap interval.

    The paired t and Wilcoxon tests use training-seed means as independent
    experimental units.  The additional bootstrap samples training seeds and
    shared chronology windows independently with replacement, avoiding the
    former treatment of every repeated episode row as independent evidence.
    """
    if len(evaluations_a) != len(evaluations_b):
        raise ValueError("Paired evaluation comparison requires equal training-seed counts.")
    summary = compare_agents(
        aggregate_seed_results(evaluations_a, metrics=metrics),
        aggregate_seed_results(evaluations_b, metrics=metrics),
        label_a=label_a,
        label_b=label_b,
        alpha=alpha,
        metrics=metrics,
    )
    rng = np.random.default_rng(seed)

    def _episode_metric_value(item: dict[str, Any], episode_metric: str) -> float:
        if episode_metric in item:
            return float(item[episode_metric])
        raise KeyError(
            f"Episode metric {episode_metric!r} is unavailable; present keys: {sorted(item)}"
        )

    for row_index, metric in enumerate(summary["metric"]):
        episode_metric = metric[:-5] if metric.endswith("_mean") else metric
        seed_differences: list[list[float]] = []
        reference_windows: list[tuple[int, str]] | None = None
        for result_a, result_b in zip(evaluations_a, evaluations_b):
            episodes_a = result_a.get("episodes", [])
            episodes_b = result_b.get("episodes", [])
            keyed_a = {
                (int(item.get("start_index", item.get("episode", -1))), str(item.get("start_timestamp", ""))): item
                for item in episodes_a
            }
            keyed_b = {
                (int(item.get("start_index", item.get("episode", -1))), str(item.get("start_timestamp", ""))): item
                for item in episodes_b
            }
            windows = sorted(set(keyed_a).intersection(keyed_b))
            if not windows:
                raise ValueError(f"No shared episode windows found for metric {metric!r}.")
            if reference_windows is None:
                reference_windows = windows
            elif windows != reference_windows:
                raise ValueError("Every seed pair must use the same chronology windows.")
            seed_differences.append(
                [
                    _episode_metric_value(keyed_a[key], episode_metric)
                    - _episode_metric_value(keyed_b[key], episode_metric)
                    for key in windows
                ]
            )

        difference_matrix = np.asarray(seed_differences, dtype=float)
        n_seeds, n_windows = difference_matrix.shape
        boot = np.empty(max(int(bootstrap_samples), 1), dtype=float)
        for draw in range(len(boot)):
            seed_indices = rng.integers(0, n_seeds, size=n_seeds)
            window_indices = rng.integers(0, n_windows, size=n_windows)
            boot[draw] = float(difference_matrix[np.ix_(seed_indices, window_indices)].mean())
        lower, upper = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        summary.loc[row_index, "n_training_seeds"] = n_seeds
        summary.loc[row_index, "n_shared_windows"] = n_windows
        summary.loc[row_index, "crossed_bootstrap_ci_lower"] = float(lower)
        summary.loc[row_index, "crossed_bootstrap_ci_upper"] = float(upper)
        summary.loc[row_index, "crossed_bootstrap_samples"] = len(boot)

    return summary


def summarise_experiment(
    seed_results: Sequence[dict[str, Any]],
    metrics: Sequence[str] = PRIMARY_METRICS,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Build a thesis-ready summary table: mean +/- std and CI per metric.
    """
    df = aggregate_seed_results(seed_results, metrics=metrics)
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        values = df[metric].dropna().tolist()
        mean, ci_lo, ci_hi = compute_confidence_interval(values, confidence=confidence)
        std = float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")
        cv = abs(std / mean) if mean != 0.0 else float("nan")
        rows.append(
            {
                "metric": metric,
                "n_seeds": len(values),
                "mean": mean,
                "std": std,
                f"ci{int(confidence * 100)}_lower": ci_lo,
                f"ci{int(confidence * 100)}_upper": ci_hi,
                "cv": cv,
            }
        )
    return pd.DataFrame(rows)


def is_pareto_efficient(
    metric_vectors: Sequence[Sequence[float]],
    maximize: Sequence[bool],
) -> list[bool]:
    """
    Return a boolean mask indicating non-dominated (Pareto-efficient) points.
    """
    points = np.asarray(metric_vectors, dtype=float)
    maximize_arr = np.asarray(maximize, dtype=bool)
    transformed = points.copy()
    transformed[:, maximize_arr] *= -1.0

    efficient = np.ones(len(points), dtype=bool)
    for i, point in enumerate(transformed):
        if not efficient[i]:
            continue
        dominates = (
            np.all(transformed <= point, axis=1)
            & np.any(transformed < point, axis=1)
        )
        dominates[i] = False
        efficient[dominates] = False
    return efficient.tolist()
