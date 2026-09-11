"""
evaluation.py - Episode-level evaluation with full metric tracking.

Improvements over the original version
---------------------------------------
* Per-episode confidence intervals (t-based, 95 % by default).
* Separate tracking of investment vs. operating cost.
* Renewable curtailment and slack generation in every summary.
* Per-line upgrade tracking for grid-impact analysis.
* Action distribution statistics for repeated-seed training.
* Backend usage rate (proxy vs. full PyPSA fallback).
* Fixed Pareto-dominance bug (a point could dominate itself).
* ``select_best_checkpoint`` for model selection on the validation split
  (choose hyperparameters on validation data, then evaluate once on test data).
* ``format_thesis_table`` renders a LaTeX-ready summary table.

All functions are self-contained and depend only on NumPy/SciPy/pandas
so they can be called from the CLI, notebooks, or test suites.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats


def _normalize_weights(weights: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float32)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0.0:
        return np.full_like(array, 1.0 / max(len(array), 1))
    return array / total


def scalarized_mean_reward(evaluation: dict[str, Any], weights: Sequence[float]) -> float:
    reward_vector = evaluation.get("reward_stats", {}).get("mean_vector_reward")
    if reward_vector is None:
        raise ValueError("Evaluation is missing reward_stats.mean_vector_reward.")
    vector = np.asarray(reward_vector, dtype=np.float32)
    normalized = _normalize_weights(np.asarray(list(weights)[: len(vector)], dtype=np.float32))
    return float(np.dot(normalized, vector))


def stratified_episode_start_indices(
    snapshots: pd.DatetimeIndex,
    episode_length: int,
    episodes: int,
) -> list[int]:
    """Return deterministic starts spread over the complete evaluation split."""
    if episodes <= 0:
        return []
    max_start = max(len(snapshots) - max(int(episode_length), 1), 0)
    if max_start == 0:
        return [0] * episodes
    return np.rint(np.linspace(0, max_start, num=episodes)).astype(int).tolist()


class _ZeroActionAgent:
    """Minimal evaluation-only baseline used to anchor MO-PPO checkpoint selection."""

    def __init__(self, action_dim: int, env_reward_dim: int):
        self.action_dim = int(action_dim)
        self.env_reward_dim = int(env_reward_dim)

    def start_rollout(self, training: bool = False) -> None:
        del training

    def start_episode(self, training: bool = False) -> None:
        del training

    def act(self, observation, deterministic: bool = True):
        del observation, deterministic
        return (
            np.zeros(self.action_dim, dtype=np.float32),
            0.0,
            np.zeros(self.env_reward_dim, dtype=np.float32),
        )


def _selection_reference_cache_key(
    episodes: int,
    deterministic: bool,
    confidence: float,
    track_line_upgrades: bool,
) -> tuple[int, bool, float, bool]:
    return (
        int(episodes),
        bool(deterministic),
        float(confidence),
        bool(track_line_upgrades),
    )


def _selection_reference_evaluation(
    env,
    episodes: int,
    deterministic: bool,
    confidence: float,
    track_line_upgrades: bool,
) -> dict[str, Any]:
    cache = getattr(env, "_moppo_selection_reference_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_moppo_selection_reference_cache", cache)

    key = _selection_reference_cache_key(
        episodes=episodes,
        deterministic=deterministic,
        confidence=confidence,
        track_line_upgrades=track_line_upgrades,
    )
    if key not in cache:
        zero_agent = _ZeroActionAgent(
            action_dim=env.action_space.shape[0],
            env_reward_dim=len(env.config.objective_names),
        )
        cache[key] = evaluate_agent(
            zero_agent,
            env,
            episodes=episodes,
            deterministic=True,
            confidence=confidence,
            track_line_upgrades=track_line_upgrades,
        )
    return cache[key]


def _relative_improvement(current: float, reference: float, *, floor: float = 1.0) -> float:
    denominator = max(abs(float(reference)), float(floor))
    return (float(reference) - float(current)) / denominator


def _relative_share_gain(current: float, reference: float) -> float:
    denominator = max(abs(float(reference)), 0.05)
    return (float(current) - float(reference)) / denominator


def metric_selection_components(
    evaluation: dict[str, Any],
    reference_evaluation: dict[str, Any],
    third_objective_mode: str = "renewable_share",
) -> np.ndarray:
    """
    Convert deployment metrics into a comparable three-objective utility vector.

    The vector is anchored to the zero-upgrade reference policy on the same split,
    which makes checkpoint selection much less sensitive to reward-shaping scales.
    Larger values are always better.
    """
    cost_component = _relative_improvement(
        evaluation.get("total_cost_mean", float("inf")),
        reference_evaluation.get("total_cost_mean", float("inf")),
        floor=1.0,
    )
    stress_component = _relative_improvement(
        evaluation.get("grid_stress_mean", float("inf")),
        reference_evaluation.get("grid_stress_mean", float("inf")),
        floor=1.0,
    )

    mode = str(third_objective_mode).lower()
    if mode == "renewable_share":
        sustainability_component = _relative_share_gain(
            evaluation.get("renewable_share_mean", 0.0),
            reference_evaluation.get("renewable_share_mean", 0.0),
        )
    elif mode == "curtailment":
        sustainability_component = _relative_improvement(
            evaluation.get("renewable_curtailment_mean", float("inf")),
            reference_evaluation.get("renewable_curtailment_mean", float("inf")),
            floor=1.0,
        )
    elif mode == "emissions":
        sustainability_component = _relative_improvement(
            evaluation.get("emissions_mean", float("inf")),
            reference_evaluation.get("emissions_mean", float("inf")),
            floor=1.0,
        )
    else:
        raise ValueError(f"Unsupported third objective mode {third_objective_mode!r}.")

    return np.asarray(
        [cost_component, stress_component, sustainability_component],
        dtype=np.float32,
    )


def metric_selection_score(
    evaluation: dict[str, Any],
    utility_weights: Sequence[float],
    reference_evaluation: dict[str, Any],
    third_objective_mode: str = "renewable_share",
) -> float:
    components = metric_selection_components(
        evaluation=evaluation,
        reference_evaluation=reference_evaluation,
        third_objective_mode=third_objective_mode,
    )
    normalized = _normalize_weights(np.asarray(list(utility_weights)[: len(components)], dtype=np.float32))
    return float(np.dot(normalized, components))


def _selection_key_for_evaluation(item: dict[str, Any]) -> tuple[float, ...]:
    evaluation = item["evaluation"]
    components = np.asarray(item.get("selection_components", []), dtype=np.float32)
    sustainability_component = float(components[2]) if len(components) >= 3 else float("-inf")
    cost_component = float(components[0]) if len(components) >= 1 else float("-inf")
    stress_component = float(components[1]) if len(components) >= 2 else float("-inf")
    total_investment = float(evaluation.get("action_stats", {}).get("total_investment_mean", float("inf")))
    return (
        float(item["selection_score"]),
        sustainability_component,
        cost_component,
        stress_component,
        -float(evaluation.get("load_shedding_mean", float("inf"))),
        -float(evaluation.get("renewable_curtailment_mean", float("inf"))),
        -float(evaluation.get("total_cost_mean", float("inf"))),
        -float(evaluation.get("grid_stress_mean", float("inf"))),
        -total_investment,
    )


def evaluate_moppo_preference_grid(
    agent,
    env,
    preference_grid: Sequence[Sequence[float]],
    utility_weights: Sequence[float],
    episodes: int = 5,
    deterministic: bool = True,
    confidence: float = 0.95,
    track_line_upgrades: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a conditioned MO-PPO policy over a grid of deployment preferences
    and select the preference that maximizes the decision-maker utility.

    This is the evaluation-time analogue of a preference-conditioned policy:
    the actor is trained across the preference simplex, but deployment picks
    the conditioning vector on a validation split before the final test.
    """
    if not hasattr(agent, "set_eval_preferences"):
        raise AttributeError("Preference-grid evaluation requires an agent with set_eval_preferences().")

    normalized_utility = _normalize_weights(utility_weights)
    raw_grid = list(preference_grid) if preference_grid else [normalized_utility.tolist()]
    if not raw_grid:
        raw_grid = [normalized_utility.tolist()]

    reference_evaluation = _selection_reference_evaluation(
        env,
        episodes=episodes,
        deterministic=deterministic,
        confidence=confidence,
        track_line_upgrades=track_line_upgrades,
    )
    third_objective_mode = str(getattr(env.config, "third_objective_mode", "renewable_share"))

    unique_grid: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for weights in raw_grid:
        normalized = _normalize_weights(weights)
        key = tuple(np.round(normalized, 6).tolist())
        if key in seen:
            continue
        seen.add(key)
        unique_grid.append(normalized)

    grid_results: list[dict[str, Any]] = []
    for weights in unique_grid:
        agent.set_eval_preferences(weights)
        agent.start_rollout(training=False)
        evaluation = evaluate_agent(
            agent,
            env,
            episodes=episodes,
            deterministic=deterministic,
            confidence=confidence,
            track_line_upgrades=track_line_upgrades,
        )
        selection_components = metric_selection_components(
            evaluation=evaluation,
            reference_evaluation=reference_evaluation,
            third_objective_mode=third_objective_mode,
        )
        grid_results.append(
            {
                "eval_preference_weights": weights.tolist(),
                "selection_score": metric_selection_score(
                    evaluation=evaluation,
                    utility_weights=normalized_utility,
                    reference_evaluation=reference_evaluation,
                    third_objective_mode=third_objective_mode,
                ),
                "selection_components": selection_components.tolist(),
                "evaluation": evaluation,
            }
        )

    best = max(grid_results, key=_selection_key_for_evaluation)
    best_weights = np.asarray(best["eval_preference_weights"], dtype=np.float32)
    agent.set_eval_preferences(best_weights)
    agent.start_rollout(training=False)

    selection_scores = np.asarray([float(item["selection_score"]) for item in grid_results], dtype=np.float32)
    return {
        "evaluation": best["evaluation"],
        "selection_score": float(best["selection_score"]),
        "selected_eval_preference_weights": best_weights.tolist(),
        "selection_details": {
            "mode": "weight_grid_best_for_reference_utility",
            "selection_score_mode": "metric_relative_to_zero_upgrade_reference",
            "reference_weights": normalized_utility.tolist(),
            "reference_evaluation": {
                "total_cost_mean": float(reference_evaluation.get("total_cost_mean", float("nan"))),
                "grid_stress_mean": float(reference_evaluation.get("grid_stress_mean", float("nan"))),
                "renewable_share_mean": float(reference_evaluation.get("renewable_share_mean", float("nan"))),
                "renewable_curtailment_mean": float(reference_evaluation.get("renewable_curtailment_mean", float("nan"))),
                "emissions_mean": float(reference_evaluation.get("emissions_mean", float("nan"))),
                "load_shedding_mean": float(reference_evaluation.get("load_shedding_mean", float("nan"))),
            },
            "third_objective_mode": third_objective_mode,
            "selected_eval_preference_weights": best_weights.tolist(),
            "best_selection_score": float(best["selection_score"]),
            "mean_selection_score": float(selection_scores.mean()),
            "min_selection_score": float(selection_scores.min()),
            "max_selection_score": float(selection_scores.max()),
            "grid": [
                {
                    "eval_preference_weights": item["eval_preference_weights"],
                    "selection_score": float(item["selection_score"]),
                    "selection_components": item["selection_components"],
                    "evaluation": item["evaluation"],
                }
                for item in grid_results
            ],
        },
    }



# Internal helpers


def _t_ci(
        values: np.ndarray,
        confidence: float = 0.95,
) -> tuple[float, float, float]:
    """
    Mean and symmetric confidence interval via Student's t-distribution.

    Returns (mean, lower, upper).  Falls back gracefully for n < 2.
    """
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if n == 1:
        return mean, float("nan"), float("nan")
    se = float(stats.sem(values))
    half_width = se * stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    return mean, mean - half_width, mean + half_width


def _scalar_summary(
        values: np.ndarray,
        confidence: float = 0.95,
        label: str = "",
) -> dict[str, float]:
    """Build a standard {mean, std, ci_lower, ci_upper} dict for one metric."""
    mean, ci_lo, ci_hi = _t_ci(values, confidence)
    std = float(values.std(ddof=1)) if len(values) >= 2 else float("nan")
    key = (label + "_") if label else ""
    return {
        f"{key}mean": mean,
        f"{key}std": std,
        f"{key}ci_lower": ci_lo,
        f"{key}ci_upper": ci_hi,
    }



# Main evaluation function


def evaluate_agent(
    agent,
    env,
    episodes: int = 5,
    deterministic: bool = True,
    confidence: float = 0.95,
    track_line_upgrades: bool = True,
    episode_start_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a trained agent over multiple episodes and return a full
    metric summary including confidence intervals.

    Parameters
    ----------
    agent:
        Any agent with an ``act(obs, deterministic) -> (action, log_prob, value)``
        interface.
    env:
        Gym-compatible TEP environment.  Must return the standard ``info``
        dict from ``step()``.
    episodes:
        Number of shared operating windows. They are not treated as additional
        independent training seeds in cross-policy inference.
    deterministic:
        Whether to use the deterministic (mean) policy.
    confidence:
        Coverage for the t-based confidence intervals (default 0.95).
    track_line_upgrades:
        If True, record per-line cumulative upgrade magnitudes.  Adds a
        small overhead but is required for the Grid Impact chapter.

    Returns
    -------
    dict
        Contains:
        * ``episodes``       - list of per-episode summary dicts
        * Aggregate scalars  - mean / std / CI for every metric
        * ``action_stats``   - mean, std, max action per candidate line
        * ``line_upgrades``  - dict {line_name: mean_mw} across episodes
        * ``backend_stats``  - fraction of steps using the full PyPSA backend
        * ``reward_stats``   - per-objective reward summary

    Notes
    -----
    Evaluation windows are deterministic and spread across the complete split
    by default. Explicit start indices support a shared-window paired design.
    """
    episode_summaries: list[dict[str, Any]] = [] # episode_summaries = []

    # Per-episode accumulators
    ep_total_costs: list[float] = []
    ep_investment_costs: list[float] = []
    ep_annualized_investment_costs: list[float] = []
    ep_operating_costs: list[float] = []
    ep_grid_stresses: list[float] = []
    ep_constraint_viols: list[float] = []
    ep_renewable_shares: list[float] = []
    ep_renewable_curtails: list[float] = []
    ep_slack_gens: list[float] = []
    ep_load_shedding: list[float] = []
    ep_emissions: list[float] = []
    ep_reward_vecs: list[np.ndarray] = []

    # Action tracking
    n_candidates = len(env.candidate_lines)
    ep_action_sums: list[np.ndarray] = []  # total MW invested per line per episode
    ep_steps_count: list[int] = []

    # Backend tracking
    all_backend_flags: list[float] = []  # 1.0 = full PyPSA, 0.0 = proxy

    if episode_start_indices is None:
        episode_start_indices = stratified_episode_start_indices(
            pd.DatetimeIndex(env.dataset.snapshots),
            int(env.config.episode_length),
            episodes,
        )
    if len(episode_start_indices) != episodes:
        raise ValueError("episode_start_indices must contain one index per evaluation episode.")

    for episode, start_index in enumerate(episode_start_indices):
        if hasattr(agent, "start_rollout"):
            agent.start_rollout(training=False)
        if hasattr(agent, "start_episode"):
            agent.start_episode(training=False)
        observation, reset_info = env.reset(
            seed=episode,
            options={"start_index": int(start_index)},
        )
        terminated = False

        reward_sum = np.zeros(agent.env_reward_dim, dtype=np.float32)
        total_cost = 0.0
        investment_cost = 0.0
        annualized_investment_cost = 0.0
        operating_cost = 0.0
        total_stress = 0.0
        total_constraint = 0.0
        total_curtail = 0.0
        total_slack = 0.0
        total_load_shedding = 0.0
        total_emissions = 0.0
        renewable_share_weighted_sum = 0.0
        renewable_share_hours = 0
        action_sum = np.zeros(n_candidates, dtype=np.float64)
        decision_count = 0
        physical_hour_count = 0
        action_trace: list[list[float]] = []

        while not terminated:
            action, _, _ = agent.act(observation, deterministic=deterministic)
            observation, reward_vector, terminated, _, info = env.step(action)

            reward_vector = np.asarray(reward_vector, dtype=np.float32)
            reward_sum[: len(reward_vector)] += reward_vector

            total_cost += float(info["total_cost"])
            investment_cost += float(info.get("investment_cost", 0.0))
            annualized_investment_cost += float(info.get("annualized_investment_cost", 0.0))
            operating_cost += float(info.get("operating_cost", 0.0))
            total_stress += float(info["grid_stress"])
            total_constraint += float(info["constraint_violation"])
            total_curtail += float(info.get("renewable_curtailment", 0.0))
            total_slack += float(info.get("slack_generation", 0.0))
            total_load_shedding += float(info.get("load_shedding", 0.0))
            total_emissions += float(info["emissions"])
            hours = int(info.get("n_hours", 1))
            renewable_share_weighted_sum += float(info["renewable_share"]) * hours
            renewable_share_hours += hours

            action_mw = info.get("action_mw", [])
            if len(action_mw) == n_candidates:
                action_sum += np.asarray(action_mw, dtype=np.float64)
            action_trace.append(action_mw)

            # Backend: "pypsa-optimize" or "pypsa-lopf" -> full, "proxy-dc" -> proxy
            backend = str(info.get("backend", "proxy-dc"))
            all_backend_flags.extend([0.0 if backend == "proxy-dc" else 1.0] * max(hours, 1))

            decision_count += 1
            physical_hour_count += hours

        mean_renewable = renewable_share_weighted_sum / max(renewable_share_hours, 1)

        episode_summaries.append(
            {
                "episode": episode,
                "start_index": int(reset_info["start_index"]),
                "start_timestamp": str(reset_info["timestamp"]),
                "total_cost": total_cost,
                "investment_cost": investment_cost,
                "annualized_investment_cost": annualized_investment_cost,
                "operating_cost": operating_cost,
                "grid_stress": total_stress,
                "constraint_violation": total_constraint,
                "renewable_share": mean_renewable,
                "renewable_curtailment": total_curtail,
                "slack_generation": total_slack,
                "load_shedding": total_load_shedding,
                "emissions": total_emissions,
                "total_investment": float(action_sum.sum()),
                "reward_vector": reward_sum.tolist(),
                "actions": action_trace,
                "n_steps": decision_count,
                "n_hours": physical_hour_count,
            }
        )

        ep_total_costs.append(total_cost)
        ep_investment_costs.append(investment_cost)
        ep_annualized_investment_costs.append(annualized_investment_cost)
        ep_operating_costs.append(operating_cost)
        ep_grid_stresses.append(total_stress)
        ep_constraint_viols.append(total_constraint)
        ep_renewable_shares.append(mean_renewable)
        ep_renewable_curtails.append(total_curtail)
        ep_slack_gens.append(total_slack)
        ep_load_shedding.append(total_load_shedding)
        ep_emissions.append(total_emissions)
        ep_reward_vecs.append(reward_sum)
        ep_action_sums.append(action_sum)
        ep_steps_count.append(physical_hour_count)

    # ---- aggregate metrics with CIs ----------------------------------------
    def _arr(lst: list[float]) -> np.ndarray:
        return np.asarray(lst, dtype=float)

    total_costs = _arr(ep_total_costs)
    invest_costs = _arr(ep_investment_costs)
    annualized_invest_costs = _arr(ep_annualized_investment_costs)
    oper_costs = _arr(ep_operating_costs)
    stresses = _arr(ep_grid_stresses)
    violations = _arr(ep_constraint_viols)
    renewables = _arr(ep_renewable_shares)
    curtailments = _arr(ep_renewable_curtails)
    slack_gens = _arr(ep_slack_gens)
    load_shedding_arr = _arr(ep_load_shedding)
    emissions_arr = _arr(ep_emissions)

    reward_matrix = (
        np.vstack(ep_reward_vecs)
        if ep_reward_vecs
        else np.zeros((0, agent.env_reward_dim), dtype=np.float32)
    )

    # ---- action distribution -----------------------------------------------
    # ep_action_sums shape: (n_episodes, n_candidates)
    action_matrix = np.vstack(ep_action_sums) if ep_action_sums else np.zeros((0, n_candidates))
    candidate_lines = list(env.candidate_lines)

    action_stats: dict[str, Any] = {}
    if action_matrix.shape[0] > 0:
        action_stats = {
            "mean_mw_per_line": dict(zip(candidate_lines, action_matrix.mean(axis=0).tolist())),
            "std_mw_per_line": dict(zip(candidate_lines, action_matrix.std(axis=0).tolist())),
            "max_mw_per_line": dict(zip(candidate_lines, action_matrix.max(axis=0).tolist())),
            "total_investment_mean": float(action_matrix.sum(axis=1).mean()),
            "fraction_lines_touched": float(
                (action_matrix.mean(axis=0) > 0.1).mean()  # lines with >0.1 MW average upgrade
            ),
        }

    # ---- line-upgrade summary (for Grid Impact chapter) --------------------
    line_upgrades: dict[str, float] = {}
    if track_line_upgrades and action_matrix.shape[0] > 0:
        line_upgrades = {
            line: float(action_matrix[:, i].mean())
            for i, line in enumerate(candidate_lines)
        }

    # ---- backend usage ------------------------------------------------------
    backend_stats: dict[str, float] = {}
    if all_backend_flags:
        full_fraction = float(np.mean(all_backend_flags))
        backend_stats = {
            "full_pypsa_fraction": full_fraction,
            "proxy_fraction": 1.0 - full_fraction,
        }

    # ---- reward summary per objective --------------------------------------
    reward_stats: dict[str, Any] = {}
    if len(reward_matrix) > 0:
        reward_stats = {
            "mean_vector_reward": reward_matrix.mean(axis=0).tolist(),
            "std_vector_reward": reward_matrix.std(axis=0).tolist(),
        }
        for obj_idx in range(reward_matrix.shape[1]):
            s = _scalar_summary(reward_matrix[:, obj_idx], confidence, f"reward_obj{obj_idx}")
            reward_stats.update(s)

    # ---- build final result dict -------------------------------------------
    ci = confidence
    result: dict[str, Any] = {
        # raw episode records
        "episodes": episode_summaries,
        "n_episodes": len(episode_summaries),
        "evaluation_design": "shared_chronology_stratified_windows",
        "episode_start_indices": [int(value) for value in episode_start_indices],
        "confidence_level": ci,

        # --- total cost ---
        "total_cost_mean": float(total_costs.mean()) if len(total_costs) else 0.0,
        "total_cost_std": float(total_costs.std(ddof=1)) if len(total_costs) >= 2 else float("nan"),
        "total_cost_ci_lower": _t_ci(total_costs, ci)[1],
        "total_cost_ci_upper": _t_ci(total_costs, ci)[2],

        # --- investment vs operating cost split ---
        "investment_cost_mean": float(invest_costs.mean()) if len(invest_costs) else 0.0,
        "investment_cost_std": float(invest_costs.std(ddof=1)) if len(invest_costs) >= 2 else float("nan"),
        "annualized_investment_cost_mean": float(annualized_invest_costs.mean()) if len(annualized_invest_costs) else 0.0,
        "annualized_investment_cost_std": float(annualized_invest_costs.std(ddof=1)) if len(annualized_invest_costs) >= 2 else float("nan"),
        "operating_cost_mean": float(oper_costs.mean()) if len(oper_costs) else 0.0,
        "operating_cost_std": float(oper_costs.std(ddof=1)) if len(oper_costs) >= 2 else float("nan"),

        # --- grid stress ---
        "grid_stress_mean": float(stresses.mean()) if len(stresses) else 0.0,
        "grid_stress_std": float(stresses.std(ddof=1)) if len(stresses) >= 2 else float("nan"),
        "grid_stress_ci_lower": _t_ci(stresses, ci)[1],
        "grid_stress_ci_upper": _t_ci(stresses, ci)[2],

        # --- constraint violations ---
        "constraint_violation_mean": float(violations.mean()) if len(violations) else 0.0,
        "constraint_violation_std": float(violations.std(ddof=1)) if len(violations) >= 2 else float("nan"),
        "constraint_violation_ci_lower": _t_ci(violations, ci)[1],
        "constraint_violation_ci_upper": _t_ci(violations, ci)[2],

        # --- renewable share ---
        "renewable_share_mean": float(renewables.mean()) if len(renewables) else 0.0,
        "renewable_share_std": float(renewables.std(ddof=1)) if len(renewables) >= 2 else float("nan"),
        "renewable_share_ci_lower": _t_ci(renewables, ci)[1],
        "renewable_share_ci_upper": _t_ci(renewables, ci)[2],

        # --- renewable curtailment (new) ---
        "renewable_curtailment_mean": float(curtailments.mean()) if len(curtailments) else 0.0,
        "renewable_curtailment_std": float(curtailments.std(ddof=1)) if len(curtailments) >= 2 else float("nan"),

        # --- slack generation (proxy for unmet demand / thermal backup) ---
        "slack_generation_mean": float(slack_gens.mean()) if len(slack_gens) else 0.0,
        "slack_generation_std": float(slack_gens.std(ddof=1)) if len(slack_gens) >= 2 else float("nan"),

        # --- emergency load shedding (strict full-env feasibility rescue) ---
        "load_shedding_mean": float(load_shedding_arr.mean()) if len(load_shedding_arr) else 0.0,
        "load_shedding_std": float(load_shedding_arr.std(ddof=1)) if len(load_shedding_arr) >= 2 else float("nan"),

        # --- emissions ---
        "emissions_mean": float(emissions_arr.mean()) if len(emissions_arr) else 0.0,
        "emissions_std": float(emissions_arr.std(ddof=1)) if len(emissions_arr) >= 2 else float("nan"),
        "emissions_ci_lower": _t_ci(emissions_arr, ci)[1],
        "emissions_ci_upper": _t_ci(emissions_arr, ci)[2],

        # --- sub-dicts ---
        "reward_stats": reward_stats,
        "action_stats": action_stats,
        "line_upgrades": line_upgrades,
        "backend_stats": backend_stats,
    }

    return result



# Model selection


def select_best_checkpoint(
        checkpoint_evaluations: dict[str, dict[str, Any]],
        primary_metric: str = "renewable_share_mean",
        secondary_metric: str = "total_cost_mean",
        maximize_primary: bool = True,
        maximize_secondary: bool = False,
) -> str:
    """
    Choose the best checkpoint from a dict of validation evaluations.

    Selection is done on the *validation* split.  The chosen checkpoint
    is then evaluated *once* on the test split.

    Parameters
    ----------
    checkpoint_evaluations:
        Mapping from checkpoint path / run ID -> evaluation dict returned
        by ``evaluate_agent``.
    primary_metric:
        Main selection criterion.
    secondary_metric:
        Tie-breaker applied when the top-k primary values are identical.
    maximize_primary / maximize_secondary:
        Direction of optimisation for each metric.

    Returns
    -------
    str
        The key (checkpoint path / run ID) of the selected checkpoint.

    Example
    -------
    >>> evals = {
    ...     "results/run_seed7/agent.pt":  evaluate_agent(agent_a, val_env),
    ...     "results/run_seed11/agent.pt": evaluate_agent(agent_b, val_env),
    ... }
    >>> best = select_best_checkpoint(evals, primary_metric="renewable_share_mean")
    """
    if not checkpoint_evaluations:
        raise ValueError("checkpoint_evaluations is empty.")

    rows = []
    for key, ev in checkpoint_evaluations.items():
        if primary_metric not in ev:
            warnings.warn(f"Metric {primary_metric!r} missing from evaluation of {key!r} - skipped.")
            continue
        rows.append(
            {
                "key": key,
                "primary": float(ev[primary_metric]),
                "secondary": float(ev.get(secondary_metric, 0.0)),
            }
        )

    if not rows:
        raise ValueError(f"No checkpoint contained metric {primary_metric!r}.")

    df = pd.DataFrame(rows)
    df["primary_rank"] = df["primary"].rank(ascending=not maximize_primary, method="min")
    df["secondary_rank"] = df["secondary"].rank(ascending=not maximize_secondary, method="min")
    df["combined_rank"] = df["primary_rank"] + df["secondary_rank"] * 1e-6
    best_key = str(df.loc[df["combined_rank"].idxmin(), "key"])
    return best_key



# Thesis table formatter


_METRIC_LABELS: dict[str, str] = {
    "total_cost_mean": "Total Cost",
    "investment_cost_mean": "Investment Cost",
    "operating_cost_mean": "Operating Cost",
    "grid_stress_mean": "Grid Stress",
    "constraint_violation_mean": "Constraint Violations",
    "renewable_share_mean": "Renewable Share",
    "renewable_curtailment_mean": "Renewable Curtailment",
    "slack_generation_mean": "Slack Generation",
    "load_shedding_mean": "Load Shedding",
    "emissions_mean": "Emissions",
}


def format_thesis_table(
        evaluations: dict[str, dict[str, Any]],
        metrics: Optional[Sequence[str]] = None,
        float_fmt: str = ".4f",
        latex: bool = False,
) -> pd.DataFrame:
    """
    Build a comparison table suitable for the thesis results chapter.

    Parameters
    ----------
    evaluations:
        Mapping from agent / condition label -> evaluation dict.
    metrics:
        Which metrics to include.  Defaults to all primary metrics.
    float_fmt:
        Python format string for numeric values.
    latex:
        If True, return a string of the LaTeX table instead of a DataFrame.

    Returns
    -------
    pd.DataFrame (or str if latex=True)
        Rows = metrics, columns = agent labels.
        Cells show ``mean +/- std`` with CI bounds in brackets.

    Example
    -------
    >>> table = format_thesis_table({"PPO": eval_ppo, "MO-PPO": eval_moppo})
    >>> print(table.to_string())
    """
    if metrics is None:
        metrics = list(_METRIC_LABELS.keys())

    std_keys = {m: m.replace("_mean", "_std") for m in metrics}
    ci_lo_keys = {m: m.replace("_mean", "_ci_lower") for m in metrics}
    ci_hi_keys = {m: m.replace("_mean", "_ci_upper") for m in metrics}

    rows = []
    for metric in metrics:
        row: dict[str, str] = {"Metric": _METRIC_LABELS.get(metric, metric)}
        for label, ev in evaluations.items():
            if metric not in ev:
                row[label] = "-"
                continue
            mean_val = ev[metric]
            std_val = ev.get(std_keys[metric], float("nan"))
            ci_lo = ev.get(ci_lo_keys[metric], float("nan"))
            ci_hi = ev.get(ci_hi_keys[metric], float("nan"))

            mean_str = f"{mean_val:{float_fmt}}"
            std_str = f"{std_val:{float_fmt}}" if not np.isnan(std_val) else "-"
            ci_str = (
                f"[{ci_lo:{float_fmt}}, {ci_hi:{float_fmt}}]"
                if not (np.isnan(ci_lo) or np.isnan(ci_hi))
                else ""
            )
            cell = f"{mean_str} +/- {std_str}"
            if ci_str:
                cell += f"  {ci_str}"
            row[label] = cell
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Metric")

    if latex:
        return df.to_latex(
            caption="Evaluation results (mean +/- std, 95\\% CI in brackets).",
            label="tab:evaluation_results",
            escape=False,
        )
    return df



# Pareto dominance


def compute_pareto_front(
        metric_vectors: Sequence[Sequence[float]],
        maximize: Sequence[bool],
) -> list[bool]:
    """
    Return a boolean mask indicating non-dominated (Pareto-efficient) points.

    A point p dominates q if p is weakly better on all objectives and
    strictly better on at least one.

    Parameters
    ----------
    metric_vectors:
        Shape (n_points, n_objectives).
    maximize:
        Per-objective flag. True means higher is better.

    Notes
    -----
    This fixes the original implementation which could mark a point as
    dominated by itself when the outer loop reached it before the inner
    loop reset ``efficient[i]``.
    """
    points = np.asarray(metric_vectors, dtype=float)
    maximize_arr = np.asarray(maximize, dtype=bool)

    # Flip maximization objectives so that "lower is always better"
    transformed = points.copy()
    transformed[:, maximize_arr] *= -1.0

    efficient = np.ones(len(points), dtype=bool)
    for i, point in enumerate(transformed):
        if not efficient[i]:
            continue
        # Find all points that dominate point i.
        dominates_i = (
                np.all(transformed <= point, axis=1)
                & np.any(transformed < point, axis=1)
        )
        dominates_i[i] = False  # a point cannot dominate itself
        efficient[dominates_i] = False

    return efficient.tolist()
