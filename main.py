"""
main.py - CLI entry point for the TEP-RL thesis framework.

Additional thesis commands
--------------------------
  multi-seed   Run repeated-seed PPO or MO-PPO training.
  compare      Run a statistical comparison of two result sets.
  shapley      Run permutation-Shapley attribution and a grouped surrogate.

All existing commands (train, sweep, evaluate, explain, smoke-test) are
retained unchanged in behaviour but now use the improved training loop
that integrates ``seed_everything`` and ``ExperimentLogger``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tep_rl.config import (
    EnvironmentConfig,
    NetworkConfig,
    PPOConfig,
    TrainingConfig,
)
from tep_rl.data import apply_observation_scale_reference, build_toy_dataset, load_austria_case
from tep_rl.envs import ProxyTEPEnv, PyPSATEPEnv
from tep_rl.evaluation import evaluate_agent
from tep_rl.future_scenarios import apply_future_scenario_from_manifest
from tep_rl.ppo import MOPPOAgent, PPOAgent, load_agent
from tep_rl.reproducibility import seed_everything
from tep_rl.shapley import (
    PolicyShapleyExplainer,
    collect_policy_states,
    permutation_feature_importance,
    policy_sensitivity,
    shapley_guided_ridge_surrogate,
)
from tep_rl.line_metadata import aggregate_line_importance, enrich_feature_table, plot_line_importance_map
from tep_rl.statistics import compare_agents, aggregate_seed_results, summarise_experiment
from tep_rl.thesis_formulations import (
    DEFAULT_THESIS_FORMULATION_PRESET,
    THESIS_FORMULATION_PRESETS,
    apply_thesis_formulation_preset,
)
from tep_rl.training import train_agent, train_multi_seed, train_weight_sweep
from tep_rl.visualization import (
    plot_feature_importance,
    plot_pareto_front,
    plot_temporal_attribution,
    plot_training_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_VALIDATION_WEIGHT_GRID: tuple[tuple[float, ...], ...] = (
    (0.70, 0.20, 0.10),
    (0.50, 0.30, 0.20),
    (0.34, 0.33, 0.33),
    (0.20, 0.30, 0.50),
    (0.20, 0.20, 0.60),
)


# Helpers shared across commands


def _parse_weights(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in re.split(r"[\s,]+", str(text).strip()) if part)


def _parse_seeds(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.split(r"[\s,]+", str(text).strip()) if part)


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _importance_series(df: pd.DataFrame, value_column: str, feature_column: str = "feature") -> pd.Series:
    series = (
        df[[feature_column, value_column]]
        .dropna()
        .drop_duplicates(subset=[feature_column], keep="first")
        .set_index(feature_column)[value_column]
        .astype(float)
    )
    return series.sort_values(ascending=False)


def _top_k_overlap(a: pd.Series, b: pd.Series, top_k: int) -> float:
    k = max(int(top_k), 1)
    top_a = set(a.sort_values(ascending=False).head(k).index)
    top_b = set(b.sort_values(ascending=False).head(k).index)
    if not top_a or not top_b:
        return float("nan")
    return float(len(top_a & top_b) / min(len(top_a), len(top_b)))


def _explainability_agreement_table(
    method_series: dict[str, pd.Series],
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for method_a, method_b in itertools.combinations(sorted(method_series), 2):
        series_a = method_series[method_a]
        series_b = method_series[method_b]
        common = series_a.index.intersection(series_b.index)
        if len(common) < 2:
            rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_common_features": int(len(common)),
                    "spearman_rho": float("nan"),
                    "spearman_p": float("nan"),
                    "kendall_tau": float("nan"),
                    "kendall_p": float("nan"),
                    "top_k_overlap": float("nan"),
                }
            )
            continue
        aligned_a = series_a.reindex(common)
        aligned_b = series_b.reindex(common)
        spearman = stats.spearmanr(aligned_a.to_numpy(dtype=float), aligned_b.to_numpy(dtype=float))
        kendall = stats.kendalltau(aligned_a.to_numpy(dtype=float), aligned_b.to_numpy(dtype=float))
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "n_common_features": int(len(common)),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "kendall_tau": float(kendall.statistic),
                "kendall_p": float(kendall.pvalue),
                "top_k_overlap": _top_k_overlap(aligned_a, aligned_b, top_k=top_k),
            }
        )
    return pd.DataFrame(rows)


def _global_shapley_stability(
    agent,
    states: np.ndarray,
    feature_names: list[str],
    n_samples: int,
    top_k: int,
    runs: int,
    base_seed: int,
    state_sample_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    if runs <= 0:
        return pd.DataFrame(), pd.DataFrame(), {}

    rng = np.random.default_rng(base_seed)
    total_states = int(len(states))
    if total_states == 0:
        raise ValueError("At least one state is required for Shapley stability analysis.")
    sample_size = total_states if state_sample_size in (None, 0) else max(1, min(int(state_sample_size), total_states))

    run_frames: list[pd.DataFrame] = []
    for run_idx in range(runs):
        if sample_size < total_states:
            selected_idx = np.sort(rng.choice(total_states, size=sample_size, replace=False))
            sampled_states = states[selected_idx]
        else:
            sampled_states = states
        run_seed = int(base_seed + 1000 + run_idx)
        explainer = PolicyShapleyExplainer(
            agent=agent,
            feature_names=feature_names,
            n_samples=n_samples,
            seed=run_seed,
        )
        run_df = explainer.global_shapley(sampled_states, top_k=None).copy()
        run_df["run"] = int(run_idx)
        run_df["seed"] = run_seed
        run_df["n_states"] = int(len(sampled_states))
        run_df["rank"] = (
            run_df["global_importance"]
            .rank(method="average", ascending=False)
            .astype(float)
        )
        run_frames.append(run_df)

    long_df = pd.concat(run_frames, ignore_index=True)
    summary_df = (
        long_df.groupby("feature", as_index=False)
        .agg(
            importance_mean=("global_importance", "mean"),
            importance_std=("global_importance", "std"),
            importance_min=("global_importance", "min"),
            importance_max=("global_importance", "max"),
            mean_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            n_runs=("run", "nunique"),
        )
        .sort_values(["importance_mean", "mean_rank"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary_df["importance_cv"] = np.where(
        summary_df["importance_mean"].abs() > 1e-12,
        summary_df["importance_std"] / summary_df["importance_mean"].abs(),
        np.nan,
    )

    pairwise_rows: list[dict[str, float | int]] = []
    run_series = {
        int(run_df["run"].iloc[0]): _importance_series(run_df, "global_importance")
        for run_df in run_frames
    }
    for run_a, run_b in itertools.combinations(sorted(run_series), 2):
        series_a = run_series[run_a]
        series_b = run_series[run_b]
        common = series_a.index.intersection(series_b.index)
        aligned_a = series_a.reindex(common)
        aligned_b = series_b.reindex(common)
        spearman = stats.spearmanr(aligned_a.to_numpy(dtype=float), aligned_b.to_numpy(dtype=float))
        kendall = stats.kendalltau(aligned_a.to_numpy(dtype=float), aligned_b.to_numpy(dtype=float))
        pairwise_rows.append(
            {
                "run_a": int(run_a),
                "run_b": int(run_b),
                "n_common_features": int(len(common)),
                "spearman_rho": float(spearman.statistic),
                "kendall_tau": float(kendall.statistic),
                "top_k_overlap": _top_k_overlap(aligned_a, aligned_b, top_k=top_k),
            }
        )
    pairwise_df = pd.DataFrame(pairwise_rows)
    metrics = {
        "runs": int(runs),
        "states_per_run": int(sample_size),
        "pairwise_mean_spearman_rho": float(pairwise_df["spearman_rho"].mean()) if not pairwise_df.empty else float("nan"),
        "pairwise_min_spearman_rho": float(pairwise_df["spearman_rho"].min()) if not pairwise_df.empty else float("nan"),
        "pairwise_mean_kendall_tau": float(pairwise_df["kendall_tau"].mean()) if not pairwise_df.empty else float("nan"),
        "pairwise_mean_top_k_overlap": float(pairwise_df["top_k_overlap"].mean()) if not pairwise_df.empty else float("nan"),
    }
    return long_df, summary_df, metrics


def _build_dataset_for_window(args, start=None, end=None):
    if args.toy:
        return build_toy_dataset(num_steps=args.toy_steps, seed=args.seed)
    config = NetworkConfig(
        network_path=Path(args.network),
        load_path=Path(args.load),
        wind_path=Path(args.wind),
        solar_path=Path(args.solar),
        year=args.year,
        start=start,
        end=end,
        candidate_line_limit=args.candidate_lines,
    )
    dataset = load_austria_case(config)
    manifest_path = getattr(args, "preprocessing_manifest", None)
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        dataset = apply_future_scenario_from_manifest(dataset, manifest)
        candidate_lines = [str(value) for value in manifest["candidate_lines"]]
        missing = set(candidate_lines).difference(dataset.network.lines.index)
        if missing:
            raise ValueError(f"Preprocessing manifest contains unknown candidate lines: {sorted(missing)}")
        dataset.candidate_lines = candidate_lines
        dataset.demand_scale = pd.Series(manifest["demand_scale"], dtype=float).reindex(dataset.network.buses.index).fillna(1.0)
        dataset.renewable_scale = pd.Series(manifest["renewable_scale"], dtype=float).reindex(dataset.network.buses.index).fillna(1.0)
        dataset.total_demand_scale = float(manifest["total_demand_scale"])
    return dataset


def _freeze_preprocessing(reference, target) -> None:
    target.candidate_lines = list(reference.candidate_lines)
    target.candidate_line_scores = reference.candidate_line_scores
    apply_observation_scale_reference(target, reference)


def _build_env(dataset, args):
    default_env_config = EnvironmentConfig()
    env_config = EnvironmentConfig(
        episode_length=args.episode_length,
        max_line_upgrade_mw=args.max_upgrade_mw,
        total_upgrade_budget_mw=args.budget_mw,
        decision_interval=args.decision_interval,
        temporal_mode=getattr(args, "temporal_mode", default_env_config.temporal_mode),
        budget_release=args.budget_release,
        action_mode=args.action_mode,
        allocation_sharpness=args.allocation_sharpness,
        allocation_sparsity_cutoff=args.allocation_sparsity_cutoff,
        stability_margin=args.stability_margin,
        proxy_balance_mode=getattr(args, "proxy_balance_mode", default_env_config.proxy_balance_mode),
        proxy_dispatch_limit=getattr(args, "proxy_dispatch_limit", default_env_config.proxy_dispatch_limit),
        load_shedding_cost=getattr(args, "load_shedding_cost", default_env_config.load_shedding_cost),
        line_investment_cost_eur_per_mw_km_year=getattr(
            args,
            "line_investment_cost_eur_per_mw_km_year",
            default_env_config.line_investment_cost_eur_per_mw_km_year,
        ),
        cost_reward_scale=args.cost_reward_scale,
        overload_reward_scale=args.overload_reward_scale,
        third_objective_mode=args.third_objective_mode,
        curtailment_reward_scale=args.curtailment_reward_scale,
        emissions_reward_scale=args.emissions_reward_scale,
        seed=args.seed,
        solver_name=args.solver,
        full_env_fallback_to_proxy=not getattr(args, "disable_full_env_fallback_to_proxy", False),
    )
    if args.env == "proxy":
        return ProxyTEPEnv(dataset, env_config)
    return PyPSATEPEnv(dataset, env_config)


def _ppo_config(args) -> PPOConfig:
    default_ppo_config = PPOConfig()
    moppo_training_mode = getattr(args, "moppo_training_mode", "conditioned")
    conditioned_preferences = moppo_training_mode == "conditioned"
    return PPOConfig(
        hidden_sizes=(args.hidden_size, args.hidden_size),
        learning_rate=args.learning_rate,
        learning_rate_schedule=getattr(args, "learning_rate_schedule", default_ppo_config.learning_rate_schedule),
        final_learning_rate=getattr(args, "final_learning_rate", default_ppo_config.final_learning_rate),
        clip_epsilon=getattr(args, "clip_epsilon", default_ppo_config.clip_epsilon),
        gamma=getattr(args, "gamma", default_ppo_config.gamma),
        gae_lambda=getattr(args, "gae_lambda", default_ppo_config.gae_lambda),
        entropy_coef=getattr(args, "entropy_coef", default_ppo_config.entropy_coef),
        entropy_coef_schedule=getattr(args, "entropy_coef_schedule", default_ppo_config.entropy_coef_schedule),
        final_entropy_coef=getattr(args, "final_entropy_coef", default_ppo_config.final_entropy_coef),
        target_kl=getattr(args, "target_kl", default_ppo_config.target_kl),
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        normalize_rewards=getattr(args, "normalize_rewards", default_ppo_config.normalize_rewards),
        normalize_objective_advantages=not getattr(
            args,
            "disable_objective_advantage_normalization",
            False,
        ),
        device=args.device,
        seed=args.seed,
        scalarization_weights=_parse_weights(args.weights),
        moppo_preference_conditioning=(
            conditioned_preferences
            and not getattr(args, "disable_moppo_preference_conditioning", False)
        ),
        moppo_sample_preferences=(
            conditioned_preferences
            and not getattr(args, "disable_moppo_preference_sampling", False)
        ),
        moppo_dirichlet_alpha=getattr(args, "moppo_dirichlet_alpha", default_ppo_config.moppo_dirichlet_alpha),
    )


def _training_config(args, output_dir: Path | None = None) -> TrainingConfig:
    validation_weight_grid = None
    raw_validation_grid = getattr(args, "validation_weight_grid", None)
    if raw_validation_grid:
        validation_weight_grid = tuple(_parse_weights(entry) for entry in raw_validation_grid)
    elif getattr(args, "moppo_training_mode", "conditioned") == "conditioned":
        validation_weight_grid = DEFAULT_VALIDATION_WEIGHT_GRID
    return TrainingConfig(
        total_timesteps=args.timesteps,
        eval_every_updates=args.eval_every,
        eval_episodes=args.eval_episodes,
        show_progress=True,
        early_stopping_patience_evals=args.early_stopping_patience_evals,
        early_stopping_min_evals=args.early_stopping_min_evals,
        early_stopping_min_delta=args.early_stopping_min_delta,
        restore_best_model_at_end=not getattr(args, "disable_restore_best_model", False),
        validation_weight_grid=validation_weight_grid,
        output_dir=output_dir,
    )


def _write_outputs(output_dir: Path, agent, history, evaluation) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    agent.save(output_dir / "agent.pt")
    with (output_dir / "history.pkl").open("wb") as handle:
        pickle.dump(history, handle)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2, default=_json_default)
    best_validation = history.get("best_validation")
    if isinstance(best_validation, dict):
        with (output_dir / "best_validation.json").open("w", encoding="utf-8") as handle:
            json.dump(best_validation, handle, indent=2, default=_json_default)
    plot_training_curves(history, output_dir / "training_curves.png")



# train  (single run, backward-compatible)


def command_train(args) -> None:
    seed_everything(args.seed)

    train_dataset = _build_dataset_for_window(args, start=args.start, end=args.end)
    eval_dataset = (
        _build_dataset_for_window(
            args,
            start=args.eval_start or args.start,
            end=args.eval_end or args.end,
        )
        if (args.eval_start or args.eval_end)
        else train_dataset
    )
    _freeze_preprocessing(train_dataset, eval_dataset)

    env = _build_env(train_dataset, args)
    eval_env = _build_env(eval_dataset, args)
    ppo_config = _ppo_config(args)

    agent = (
        PPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=len(env.config.objective_names),
            config=ppo_config,
        )
        if args.agent == "ppo"
        else MOPPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=len(env.config.objective_names),
            config=ppo_config,
        )
    )

    output_dir = Path(args.output_dir)
    history = train_agent(agent, env, training_config=_training_config(args, output_dir), eval_env=eval_env)
    evaluation = evaluate_agent(agent, eval_env, episodes=args.eval_episodes)
    _write_outputs(output_dir, agent, history, evaluation)



# multi-seed  (Experiments 2 + 3)


def command_multi_seed(args) -> None:
    """
    Train n independent seeds and aggregate statistics.

    Outputs per seed:
      results/<experiment_name>/<run_id>/agent.pt
      results/<experiment_name>/<run_id>/run_log.jsonl
      results/<experiment_name>/<run_id>/config_snapshot.json

    Aggregate outputs:
      results/<experiment_name>/multi_seed_summary.json
      results/<experiment_name>/multi_seed_summary.csv
    """
    seeds = _parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)

    train_dataset = _build_dataset_for_window(args, start=args.start, end=args.end)
    eval_dataset = (
        _build_dataset_for_window(
            args,
            start=args.eval_start or args.start,
            end=args.eval_end or args.end,
        )
        if (args.eval_start or args.eval_end)
        else train_dataset
    )
    _freeze_preprocessing(train_dataset, eval_dataset)

    def env_factory():
        return _build_env(train_dataset, args)

    def eval_factory():
        return _build_env(eval_dataset, args)

    all_results = train_multi_seed(
        env_factory=env_factory,
        base_ppo_config=_ppo_config(args),
        training_config=_training_config(args, output_dir),
        seeds=seeds,
        mode=args.agent,
        eval_env_factory=eval_factory,
        output_dir=output_dir,
        experiment_name=args.experiment_name,
    )

    # Aggregate
    evaluations = [r["evaluation"] for r in all_results]
    summary_df = summarise_experiment(evaluations)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "multi_seed_summary.csv", index=False)
    with (output_dir / "multi_seed_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "seeds": list(seeds),
                "agent": args.agent,
                "summary": summary_df.to_dict(orient="records"),
                "per_seed": [
                    {"seed": r["seed"], "run_id": r["run_id"], "evaluation": r["evaluation"]}
                    for r in all_results
                ],
            },
            fh,
            indent=2,
            default=_json_default,
        )
    logger.info("Multi-seed summary written to %s", output_dir)
    print(summary_df.to_string(index=False))



# compare: statistical tests


def command_compare(args) -> None:
    """
    Load two multi-seed result JSONs and run statistical tests.
    """
    with open(args.result_a, encoding="utf-8") as fh:
        data_a = json.load(fh)
    with open(args.result_b, encoding="utf-8") as fh:
        data_b = json.load(fh)

    evals_a = [entry["evaluation"] for entry in data_a["per_seed"]]
    evals_b = [entry["evaluation"] for entry in data_b["per_seed"]]

    df_a = aggregate_seed_results(evals_a)
    df_b = aggregate_seed_results(evals_b)

    comparison = compare_agents(
        df_a, df_b,
        label_a=args.label_a,
        label_b=args.label_b,
        alpha=args.alpha,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    print(comparison.to_string(index=False))
    logger.info("Statistical comparison written to %s", output_path)



# Scalarisation sweep


def command_sweep(args) -> None:
    train_dataset = _build_dataset_for_window(args, start=args.start, end=args.end)
    eval_dataset = (
        _build_dataset_for_window(
            args,
            start=args.eval_start or args.start,
            end=args.eval_end or args.end,
        )
        if (args.eval_start or args.eval_end)
        else train_dataset
    )
    _freeze_preprocessing(train_dataset, eval_dataset)

    def env_factory():
        return _build_env(train_dataset, args)

    def eval_env_factory():
        return _build_env(eval_dataset, args)

    output_dir = Path(args.output_dir)
    weight_grid = [_parse_weights(weight_string) for weight_string in args.weight_grid]
    sweep_results = train_weight_sweep(
        env_factory=env_factory,
        eval_env_factory=eval_env_factory,
        training_config=_training_config(args, output_dir),
        ppo_config=_ppo_config(args),
        weight_grid=weight_grid,
        mode=args.agent,
        output_dir=output_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sweep_results.pkl").open("wb") as handle:
        pickle.dump(sweep_results, handle)
    summary = {
        "/".join(f"{weight:.2f}" for weight in key): value["evaluation"]
        for key, value in sweep_results.items()
    }
    with (output_dir / "sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    plot_pareto_front(sweep_results, output_dir / "pareto_front.png")



# evaluate


def command_evaluate(args) -> None:
    dataset = _build_dataset_for_window(args, start=args.start, end=args.end)
    env = _build_env(dataset, args)
    agent = load_agent(args.checkpoint, device=args.device)
    if hasattr(agent, "set_eval_preferences") and args.weights is not None:
        agent.set_eval_preferences(_parse_weights(args.weights))
    evaluation = evaluate_agent(agent, env, episodes=args.eval_episodes)
    print(json.dumps(evaluation, indent=2, default=_json_default))



def command_shapley(args) -> None:
    """
    Run permutation-Shapley attribution and an episode-grouped ridge surrogate.

    Outputs:
      global_shapley.csv        - top global feature importance
      global_shapley_full.csv   - full global feature importance table
      shapley_surrogate_*.csv   - explanation-regression surrogate policy
      line_importance_*.csv/png - line-level mapped policy importance
      local_shapley_step0.csv  - local explanation for the first decision step
      temporal_attribution.csv - importance across decision stages
      permutation_importance*.csv/png  - fast baseline (permutation)
      policy_sensitivity*.csv/png      - gradient-based sensitivity
      global_shapley.png        - raw importance bar chart
      global_shapley_labeled.png - readable importance bar chart
    """
    dataset = _build_dataset_for_window(args, start=args.start, end=args.end)
    env = _build_env(dataset, args)
    agent = load_agent(args.checkpoint, device=args.device)
    if hasattr(agent, "set_eval_preferences") and args.weights is not None:
        agent.set_eval_preferences(_parse_weights(args.weights))

    states, episode_ids = collect_policy_states(
        env,
        agent,
        episodes=args.eval_episodes,
        return_episode_ids=True,
    )
    feature_names = env.get_feature_names()

    explainer = PolicyShapleyExplainer(
        agent=agent,
        feature_names=feature_names,
        n_samples=args.shapley_samples,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "shapley_manifest.json",
        {
            "checkpoint": str(args.checkpoint),
            "eval_episodes": int(args.eval_episodes),
            "shapley_samples": int(args.shapley_samples),
            "top_k": int(args.top_k),
            "shapley_surrogate_top_k": int(args.shapley_surrogate_top_k),
            "shapley_ridge_alpha": float(args.shapley_ridge_alpha),
            "shapley_stability_runs": int(args.shapley_stability_runs),
            "shapley_stability_state_samples": None if args.shapley_stability_state_samples is None else int(args.shapley_stability_state_samples),
            "weights": None if args.weights is None else list(_parse_weights(args.weights)),
            "method": "interventional permutation Shapley with observed-state references",
            "surrogate_validation": "episode-grouped holdout",
        },
    )

    # Global permutation-Shapley importance. Keep the full table for line-level
    # aggregation and write a top-k compatibility table for existing reports.
    global_full_df = explainer.global_shapley(states, top_k=None)
    global_full_df.to_csv(output_dir / "global_shapley_full.csv", index=False)
    global_df = global_full_df.head(args.top_k)
    global_df.to_csv(output_dir / "global_shapley.csv", index=False)
    plot_feature_importance(global_df.rename(columns={"global_importance": "importance"}),
                            output_dir / "global_shapley.png", top_k=args.top_k)

    global_labeled_df = enrich_feature_table(global_full_df, dataset.network)
    global_labeled_df.head(args.top_k).to_csv(output_dir / "global_shapley_labeled.csv", index=False)
    plot_feature_importance(
        global_labeled_df.rename(columns={"global_importance": "importance"}),
        output_dir / "global_shapley_labeled.png",
        top_k=args.top_k,
    )

    line_importance_df = aggregate_line_importance(global_full_df, dataset.network)
    line_importance_df.to_csv(output_dir / "line_importance_shapley.csv", index=False)
    plot_line_importance_map(
        dataset.network,
        line_importance_df,
        output_dir / "line_importance_shapley_map.png",
        top_k=args.top_k,
    )

    surrogate_metrics, surrogate_coefficients, surrogate_predictions = shapley_guided_ridge_surrogate(
        agent=agent,
        states=states,
        feature_names=feature_names,
        global_importance=global_full_df,
        action_names=env.get_action_names(),
        top_k=args.shapley_surrogate_top_k,
        ridge_alpha=args.shapley_ridge_alpha,
        seed=args.seed,
        group_ids=episode_ids,
    )
    surrogate_metrics.to_csv(output_dir / "shapley_surrogate_metrics.csv", index=False)
    surrogate_coefficients.to_csv(output_dir / "shapley_surrogate_coefficients.csv", index=False)
    surrogate_predictions.to_csv(output_dir / "shapley_surrogate_predictions.csv", index=False)

    coef_summary = (
        surrogate_coefficients[surrogate_coefficients["feature"] != "__intercept__"]
        .groupby("feature", as_index=False)["abs_coefficient"]
        .mean()
        .rename(columns={"abs_coefficient": "importance"})
        .sort_values("importance", ascending=False)
    )
    coef_summary_labeled = enrich_feature_table(coef_summary.rename(columns={"feature": "feature"}), dataset.network)
    coef_summary_labeled.to_csv(output_dir / "shapley_surrogate_feature_importance.csv", index=False)
    plot_feature_importance(
        coef_summary_labeled,
        output_dir / "shapley_surrogate_feature_importance.png",
        top_k=args.top_k,
    )

    # Local explanation for first state
    local_df = explainer.local_shapley(states[0])
    local_df.to_csv(output_dir / "local_shapley_step0.csv", index=False)
    enrich_feature_table(local_df, dataset.network).to_csv(output_dir / "local_shapley_step0_labeled.csv", index=False)

    # Temporal attribution (first episode)
    obs, _ = env.reset(seed=0)
    episode_states = []
    done = False
    while not done:
        episode_states.append(obs.copy())
        action, _, _ = agent.act(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    temporal_df = explainer.temporal_attribution(np.array(episode_states))
    temporal_df.to_csv(output_dir / "temporal_attribution.csv", index=False)
    plot_temporal_attribution(temporal_df, output_dir / "temporal_attribution.png", top_k=args.top_k)

    # Baseline methods for comparison
    perm_df = permutation_feature_importance(agent, states, feature_names)
    perm_df.to_csv(output_dir / "permutation_importance.csv", index=False)
    plot_feature_importance(perm_df, output_dir / "permutation_importance.png")
    perm_labeled_df = enrich_feature_table(perm_df, dataset.network)
    perm_labeled_df.to_csv(output_dir / "permutation_importance_labeled.csv", index=False)
    plot_feature_importance(perm_labeled_df, output_dir / "permutation_importance_labeled.png")

    sens_df = policy_sensitivity(agent, states, feature_names)
    sens_df.to_csv(output_dir / "policy_sensitivity.csv", index=False)
    plot_feature_importance(sens_df, output_dir / "policy_sensitivity.png")
    sens_labeled_df = enrich_feature_table(sens_df.rename(columns={"sensitivity": "importance"}), dataset.network)
    sens_labeled_df.to_csv(output_dir / "policy_sensitivity_labeled.csv", index=False)
    plot_feature_importance(sens_labeled_df, output_dir / "policy_sensitivity_labeled.png")

    agreement_df = _explainability_agreement_table(
        {
            "global_shapley": _importance_series(global_full_df, "global_importance"),
            "permutation_importance": _importance_series(perm_df, "importance"),
            "policy_sensitivity": _importance_series(sens_df, "sensitivity"),
            "shapley_surrogate": _importance_series(coef_summary, "importance"),
        },
        top_k=args.top_k,
    )
    agreement_df.to_csv(output_dir / "explainability_agreement.csv", index=False)
    _write_json(
        output_dir / "explainability_agreement_summary.json",
        {
            "top_k": int(args.top_k),
            "mean_spearman_rho": float(agreement_df["spearman_rho"].mean()) if not agreement_df.empty else float("nan"),
            "mean_kendall_tau": float(agreement_df["kendall_tau"].mean()) if not agreement_df.empty else float("nan"),
            "mean_top_k_overlap": float(agreement_df["top_k_overlap"].mean()) if not agreement_df.empty else float("nan"),
        },
    )

    if int(args.shapley_stability_runs) > 0:
        stability_long_df, stability_summary_df, stability_metrics = _global_shapley_stability(
            agent=agent,
            states=states,
            feature_names=feature_names,
            n_samples=int(args.shapley_stability_samples or args.shapley_samples),
            top_k=int(args.top_k),
            runs=int(args.shapley_stability_runs),
            base_seed=int(args.seed),
            state_sample_size=args.shapley_stability_state_samples,
        )
        stability_long_df.to_csv(output_dir / "global_shapley_stability_runs.csv", index=False)
        stability_summary_df.to_csv(output_dir / "global_shapley_stability.csv", index=False)
        _write_json(output_dir / "global_shapley_stability_summary.json", stability_metrics)

    logger.info("Policy-attribution outputs written to %s", output_dir)



# smoke-test


def command_smoke_test(args) -> None:
    args.toy = True
    args.toy_steps = 48
    dataset = _build_dataset_for_window(args, start=args.start, end=args.end)
    env = _build_env(dataset, args)
    eval_env = _build_env(dataset, args)
    ppo_config = _ppo_config(args)
    agent = MOPPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        env_reward_dim=len(env.config.objective_names),
        config=ppo_config,
    )
    training_config = _training_config(args)
    history = train_agent(agent, env, training_config=training_config, eval_env=eval_env)
    evaluation = evaluate_agent(agent, eval_env, episodes=args.eval_episodes)
    print(json.dumps(
        {"last_update": history["updates"][-1], "evaluation": evaluation},
        indent=2, default=_json_default
    ))



# Argument parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RL framework for transmission expansion planning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    default_env_config = EnvironmentConfig()
    default_ppo_config = PPOConfig()

    def add_common(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument(
            "--formulation-preset",
            choices=sorted(THESIS_FORMULATION_PRESETS),
            default=DEFAULT_THESIS_FORMULATION_PRESET,
            help="Apply a shared thesis formulation preset unless the specific budget/candidate flags are overridden.",
        )
        command_parser.add_argument("--env", choices=["proxy", "full"], default="proxy")
        command_parser.add_argument("--agent", choices=["ppo", "moppo"], default="moppo")
        command_parser.add_argument("--network", default="derived/austria_net_physical_ratings.nc")
        command_parser.add_argument("--load", default="data/entsoe_at_load_2015_2024_opsd.csv")
        command_parser.add_argument("--wind", default="data/wind_at_2015_2024.csv")
        command_parser.add_argument("--solar", default="data/solar_at_2015_2024.csv")
        command_parser.add_argument("--year", type=int, default=2020)
        command_parser.add_argument("--start", default=None)
        command_parser.add_argument("--end", default=None)
        command_parser.add_argument("--eval-start", default=None)
        command_parser.add_argument("--eval-end", default=None)
        command_parser.add_argument("--episode-length", type=int, default=24)
        command_parser.add_argument("--max-upgrade-mw", type=float, default=60.0)
        command_parser.add_argument("--budget-mw", type=float, default=240.0)
        command_parser.add_argument("--decision-interval", type=int, default=6)
        command_parser.add_argument("--temporal-mode", choices=["decision_block", "hourly"], default=default_env_config.temporal_mode)
        command_parser.add_argument("--budget-release", choices=["linear", "all_at_once"], default="linear")
        command_parser.add_argument("--action-mode", choices=["budgeted", "direct"], default=default_env_config.action_mode)
        command_parser.add_argument("--allocation-sharpness", type=float, default=default_env_config.allocation_sharpness)
        command_parser.add_argument("--allocation-sparsity-cutoff", type=float, default=default_env_config.allocation_sparsity_cutoff)
        command_parser.add_argument("--stability-margin", type=float, default=default_env_config.stability_margin)
        command_parser.add_argument(
            "--proxy-balance-mode",
            choices=["demand_proportional", "single_slack"],
            default=default_env_config.proxy_balance_mode,
        )
        command_parser.add_argument("--proxy-dispatch-limit", type=float, default=default_env_config.proxy_dispatch_limit)
        command_parser.add_argument("--load-shedding-cost", type=float, default=default_env_config.load_shedding_cost)
        command_parser.add_argument("--cost-reward-scale", type=float, default=default_env_config.cost_reward_scale)
        command_parser.add_argument(
            "--line-investment-cost-eur-per-mw-km-year",
            type=float,
            default=default_env_config.line_investment_cost_eur_per_mw_km_year,
        )
        command_parser.add_argument("--overload-reward-scale", type=float, default=default_env_config.overload_reward_scale)
        command_parser.add_argument(
            "--third-objective-mode",
            choices=["renewable_share", "curtailment", "emissions"],
            default=default_env_config.third_objective_mode,
        )
        command_parser.add_argument("--curtailment-reward-scale", type=float, default=default_env_config.curtailment_reward_scale)
        command_parser.add_argument("--emissions-reward-scale", type=float, default=default_env_config.emissions_reward_scale)
        command_parser.add_argument("--candidate-lines", type=int, default=60)
        command_parser.add_argument(
            "--preprocessing-manifest",
            default=None,
            help="Training-fitted candidate lines and observation scales to freeze for validation/test.",
        )
        command_parser.add_argument("--solver", default="highs")
        command_parser.add_argument(
            "--disable-full-env-fallback-to-proxy",
            action="store_true",
            help="In full-environment mode, fail on PyPSA solve errors instead of silently falling back to the proxy model.",
        )
        command_parser.add_argument("--weights", default=",".join(str(weight) for weight in default_ppo_config.scalarization_weights))
        command_parser.add_argument("--timesteps", type=int, default=4096)
        command_parser.add_argument("--rollout-steps", type=int, default=256)
        command_parser.add_argument("--minibatch-size", type=int, default=64)
        command_parser.add_argument("--update-epochs", type=int, default=6)
        command_parser.add_argument("--learning-rate", type=float, default=3e-4)
        command_parser.add_argument("--learning-rate-schedule", choices=["constant", "linear"], default=default_ppo_config.learning_rate_schedule)
        command_parser.add_argument("--final-learning-rate", type=float, default=default_ppo_config.final_learning_rate)
        command_parser.add_argument("--gamma", type=float, default=default_ppo_config.gamma)
        command_parser.add_argument("--gae-lambda", type=float, default=default_ppo_config.gae_lambda)
        command_parser.add_argument("--clip-epsilon", type=float, default=default_ppo_config.clip_epsilon)
        command_parser.add_argument("--entropy-coef", type=float, default=default_ppo_config.entropy_coef)
        command_parser.add_argument("--entropy-coef-schedule", choices=["constant", "linear"], default=default_ppo_config.entropy_coef_schedule)
        command_parser.add_argument("--final-entropy-coef", type=float, default=default_ppo_config.final_entropy_coef)
        command_parser.add_argument("--target-kl", type=float, default=default_ppo_config.target_kl)
        command_parser.add_argument("--normalize-rewards", action="store_true", default=default_ppo_config.normalize_rewards)
        command_parser.add_argument(
            "--moppo-training-mode",
            choices=["fixed", "conditioned"],
            default="conditioned",
            help="`conditioned` trains a preference-conditioned MO-PPO policy over the weight simplex; `fixed` reduces MO-PPO to one scalarisation.",
        )
        command_parser.add_argument(
            "--validation-weight-grid",
            nargs="+",
            default=None,
            help="Optional validation preference grid for conditioned MO-PPO checkpoint selection. When omitted, the thesis default grid is used.",
        )
        command_parser.add_argument("--disable-objective-advantage-normalization", action="store_true")
        command_parser.add_argument("--disable-moppo-preference-conditioning", action="store_true")
        command_parser.add_argument("--disable-moppo-preference-sampling", action="store_true")
        command_parser.add_argument("--moppo-dirichlet-alpha", type=float, default=default_ppo_config.moppo_dirichlet_alpha)
        command_parser.add_argument("--hidden-size", type=int, default=128)
        command_parser.add_argument("--eval-every", type=int, default=2)
        command_parser.add_argument("--eval-episodes", type=int, default=2)
        command_parser.add_argument("--early-stopping-patience-evals", type=int, default=8)
        command_parser.add_argument("--early-stopping-min-evals", type=int, default=20)
        command_parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
        command_parser.add_argument("--disable-restore-best-model", action="store_true")
        command_parser.add_argument("--device", default=default_ppo_config.device)
        command_parser.add_argument("--seed", type=int, default=7)
        command_parser.add_argument("--toy", action="store_true")
        command_parser.add_argument("--toy-steps", type=int, default=72)

    # train
    p = subparsers.add_parser("train", help="Train a single PPO / MO-PPO agent")
    add_common(p);
    p.add_argument("--output-dir", default="results/run")
    p.set_defaults(func=command_train)

    # multi-seed
    p = subparsers.add_parser("multi-seed", help="Repeated-seed thesis training")
    add_common(p)
    p.add_argument("--output-dir", default="results/multi_seed")
    p.add_argument("--seeds", default="7,11,19,23,31",
                   help="Comma-separated seed list, e.g. '7,11,19,23,31'")
    p.add_argument("--experiment-name", default="thesis_run")
    p.set_defaults(func=command_multi_seed)

    # compare
    p = subparsers.add_parser("compare", help="Statistical comparison of two agents")
    p.add_argument("result_a", help="Path to multi_seed_summary.json for agent A")
    p.add_argument("result_b", help="Path to multi_seed_summary.json for agent B")
    p.add_argument("--label-a", default="PPO")
    p.add_argument("--label-b", default="MO-PPO")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output", default="results/comparison.csv")
    p.set_defaults(func=command_compare)

    # sweep
    p = subparsers.add_parser("sweep", help="Pareto preference sweep")
    add_common(p)
    p.add_argument("--output-dir", default="results/sweep")
    p.add_argument("--weight-grid", nargs="+",
                   default=["0.70,0.20,0.10", "0.50,0.30,0.20", "0.34,0.33,0.33", "0.20,0.30,0.50", "0.20,0.20,0.60"])
    p.set_defaults(func=command_sweep)

    # evaluate
    p = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint")
    add_common(p);
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(weights=None)
    p.set_defaults(func=command_evaluate)

    p = subparsers.add_parser(
        "shapley",
        help="Permutation-Shapley policy attribution and episode-grouped ridge surrogate",
    )
    add_common(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="results/shapley")
    p.add_argument("--shapley-samples", type=int, default=100,
                   help="Permutation samples per Shapley estimate")
    p.add_argument("--shapley-surrogate-top-k", type=int, default=12,
                   help="Number of Shapley-selected features used by the ridge surrogate")
    p.add_argument("--shapley-ridge-alpha", type=float, default=1e-3,
                   help="Ridge penalty for the Shapley-guided surrogate")
    p.add_argument("--shapley-stability-runs", type=int, default=0,
                   help="Optional number of repeated global Shapley runs for stability diagnostics")
    p.add_argument("--shapley-stability-state-samples", type=int, default=None,
                   help="Optional number of policy states sampled per Shapley stability run")
    p.add_argument("--shapley-stability-samples", type=int, default=None,
                   help="Optional permutation samples per stability run; defaults to --shapley-samples")
    p.add_argument("--top-k", type=int, default=15,
                   help="Number of top features to visualise")
    p.set_defaults(weights=None)
    p.set_defaults(func=command_shapley)

    # smoke-test
    p = subparsers.add_parser("smoke-test", help="Quick end-to-end smoke test")
    add_common(p)
    p.set_defaults(func=command_smoke_test)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    raw_argv = sys.argv[1:]
    arguments = parser.parse_args(raw_argv)
    apply_thesis_formulation_preset(arguments, raw_argv)
    arguments.func(arguments)
