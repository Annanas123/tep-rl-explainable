"""
training.py - Rollout collection, PPO updates, and experiment runners.

The thesis workflow relies on three guarantees from this module:
1. every run is seeded independently,
2. periodic validation checkpoints are saved for model selection, and
3. per-run artifacts are written in a consistent format for later analysis.
"""

from __future__ import annotations

import copy
import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from tqdm.auto import tqdm

from .config import PPOConfig, TrainingConfig
from .evaluation import evaluate_agent, evaluate_moppo_preference_grid
from .experiment_logger import ExperimentLogger
from .ppo import MOPPOAgent, PPOAgent, RolloutBuffer
from .reproducibility import get_run_id, seed_everything
from .visualization import plot_training_curves

logger = logging.getLogger(__name__)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full_like(weights, 1.0 / max(len(weights), 1))
    return weights / total


def _selection_score(agent, evaluation: dict[str, object]) -> float:
    reward_vector = evaluation.get("reward_stats", {}).get("mean_vector_reward")
    if reward_vector is None:
        raise ValueError("Evaluation is missing reward_stats.mean_vector_reward.")
    vector = np.asarray(reward_vector, dtype=np.float32)
    if agent.agent_kind == "ppo":
        weights = _normalize_weights(np.asarray(agent.scalarization_weights[: len(vector)], dtype=np.float32))
    else:
        weights = _normalize_weights(np.asarray(agent.eval_preference_weights[: len(vector)], dtype=np.float32))
    return float(np.dot(weights, vector))


def _selection_score_for_weights(weights: np.ndarray | tuple[float, ...] | list[float], evaluation: dict[str, object]) -> float:
    reward_vector = evaluation.get("reward_stats", {}).get("mean_vector_reward")
    if reward_vector is None:
        raise ValueError("Evaluation is missing reward_stats.mean_vector_reward.")
    vector = np.asarray(reward_vector, dtype=np.float32)
    normalized = _normalize_weights(np.asarray(weights[: len(vector)], dtype=np.float32))
    return float(np.dot(normalized, vector))


def _validation_reference_utility_weights(agent) -> np.ndarray:
    # Keep MO-PPO checkpoint selection on one fixed utility vector so scores
    # remain comparable across updates.
    config_weights = getattr(getattr(agent, "config", None), "scalarization_weights", None)
    if config_weights is not None:
        return np.asarray(config_weights, dtype=np.float32).copy()
    if hasattr(agent, "scalarization_weights"):
        return np.asarray(agent.scalarization_weights, dtype=np.float32).copy()
    if hasattr(agent, "eval_preference_weights"):
        return np.asarray(agent.eval_preference_weights, dtype=np.float32).copy()
    return np.ones(int(agent.env_reward_dim), dtype=np.float32)


def _evaluate_validation_suite(agent, eval_env, training_config: TrainingConfig) -> dict[str, object]:
    if agent.agent_kind != "moppo" or not training_config.validation_weight_grid:
        agent.start_rollout(training=False)
        evaluation = evaluate_agent(
            agent,
            eval_env,
            episodes=training_config.eval_episodes,
            deterministic=training_config.deterministic_eval,
        )
        return {
            "evaluation": evaluation,
            "selection_score": _selection_score(agent, evaluation),
            "selection_details": None,
        }

    if not hasattr(agent, "set_eval_preferences"):
        raise AttributeError("Conditioned MO-PPO validation requires set_eval_preferences on the agent.")

    reference_weights = _validation_reference_utility_weights(agent)
    grid_result = evaluate_moppo_preference_grid(
        agent,
        eval_env,
        preference_grid=training_config.validation_weight_grid,
        utility_weights=reference_weights,
        episodes=training_config.eval_episodes,
        deterministic=training_config.deterministic_eval,
    )
    selection_details = dict(grid_result["selection_details"])
    return {
        "evaluation": grid_result["evaluation"],
        "selection_score": float(grid_result["selection_score"]),
        "selection_details": selection_details,
    }


def _is_better_evaluation(candidate: dict[str, object], incumbent: dict[str, object] | None, min_delta: float) -> bool:
    if incumbent is None:
        return True
    candidate_score = float(candidate["selection_score"])
    incumbent_score = float(incumbent["selection_score"])
    if candidate_score > incumbent_score + float(min_delta):
        return True
    if candidate_score < incumbent_score + float(min_delta):
        return False

    candidate_eval = candidate["evaluation"]
    incumbent_eval = incumbent["evaluation"]
    candidate_investment = float(candidate_eval.get("action_stats", {}).get("total_investment_mean", float("inf")))
    incumbent_investment = float(incumbent_eval.get("action_stats", {}).get("total_investment_mean", float("inf")))
    candidate_key = (
        float(candidate_eval.get("renewable_share_mean", float("-inf"))),
        -float(candidate_eval.get("renewable_curtailment_mean", float("inf"))),
        -float(candidate_eval.get("load_shedding_mean", float("inf"))),
        -float(candidate_eval.get("total_cost_mean", float("inf"))),
        -float(candidate_eval.get("grid_stress_mean", float("inf"))),
        -candidate_investment,
    )
    incumbent_key = (
        float(incumbent_eval.get("renewable_share_mean", float("-inf"))),
        -float(incumbent_eval.get("renewable_curtailment_mean", float("inf"))),
        -float(incumbent_eval.get("load_shedding_mean", float("inf"))),
        -float(incumbent_eval.get("total_cost_mean", float("inf"))),
        -float(incumbent_eval.get("grid_stress_mean", float("inf"))),
        -incumbent_investment,
    )
    return candidate_key > incumbent_key


def _clone_state_dict_to_cpu(agent) -> dict[str, object]:
    return {key: value.detach().cpu().clone() for key, value in agent.network.state_dict().items()}


def _make_buffer(agent, env) -> RolloutBuffer:
    return RolloutBuffer(
        rollout_steps=agent.config.rollout_steps,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        objective_dim=agent.objective_dim,
        context_dim=agent.context_dim,
    )


def _build_agent(mode: str, env, config: PPOConfig):
    kwargs = dict(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        env_reward_dim=len(env.config.objective_names),
        config=config,
    )
    if mode == "ppo":
        return PPOAgent(**kwargs)
    return MOPPOAgent(**kwargs)


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _persist_run_artifacts(run_dir: Path, agent, history: dict[str, object], evaluation: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    agent.save(run_dir / "agent.pt")
    with (run_dir / "history.pkl").open("wb") as handle:
        pickle.dump(history, handle)
    with (run_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2, default=_json_default)
    best_validation = history.get("best_validation")
    if isinstance(best_validation, dict):
        with (run_dir / "best_validation.json").open("w", encoding="utf-8") as handle:
            json.dump(best_validation, handle, indent=2, default=_json_default)
    plot_training_curves(history, run_dir / "training_curves.png")


def _persist_validation_checkpoint(
    output_dir: Path,
    update_idx: int,
    agent,
    evaluation: dict[str, object],
    selection_details: dict[str, object] | None = None,
) -> dict[str, object]:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"update_{update_idx:04d}.pt"
    evaluation_path = checkpoint_dir / f"update_{update_idx:04d}.evaluation.json"
    selection_path = checkpoint_dir / f"update_{update_idx:04d}.selection.json"
    agent.save(checkpoint_path)
    with evaluation_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2, default=_json_default)
    if selection_details is not None:
        with selection_path.open("w", encoding="utf-8") as handle:
            json.dump(selection_details, handle, indent=2, default=_json_default)
    return {
        "update": update_idx,
        "checkpoint": str(checkpoint_path),
        "evaluation_path": str(evaluation_path),
        "selection_path": str(selection_path) if selection_details is not None else None,
        "evaluation": evaluation,
    }


def train_agent(
    agent,
    env,
    training_config: TrainingConfig,
    eval_env=None,
    exp_logger: Optional[ExperimentLogger] = None,
) -> dict[str, object]:
    """
    Run the PPO/MO-PPO training loop for ``total_timesteps`` environment steps.
    """
    agent.start_episode(training=True)
    observation, _ = env.reset(seed=agent.config.seed)
    total_steps = 0
    update_idx = 0
    terminated = False

    history: dict[str, object] = {"updates": [], "episodes": []}
    best_validation: dict[str, object] | None = None
    best_state_dict: dict[str, object] | None = None
    evaluations_without_improvement = 0
    evaluation_count = 0
    early_stop_reason: str | None = None

    episode_reward = np.zeros(agent.env_reward_dim, dtype=np.float32)
    episode_cost = 0.0
    episode_stress = 0.0
    episode_renewable_sum = 0.0
    episode_hours = 0

    with tqdm(
        total=training_config.total_timesteps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        disable=not training_config.show_progress,
    ) as progress:
        while total_steps < training_config.total_timesteps:
            agent.start_rollout(training=True)
            buffer = _make_buffer(agent, env)

            for _ in range(agent.config.rollout_steps):
                action, log_prob, value = agent.act(observation, deterministic=False)
                next_observation, reward_vector, terminated, _, info = env.step(action)

                reward_vector = np.asarray(reward_vector, dtype=np.float32)
                training_reward = agent.prepare_reward(reward_vector)
                buffer.add(
                    observation=observation,
                    action=action,
                    log_prob=log_prob,
                    reward=training_reward,
                    value=value,
                    done=terminated,
                    context=agent.current_context(),
                )

                episode_reward[: len(reward_vector)] += reward_vector
                episode_cost += float(info["total_cost"])
                episode_stress += float(info["grid_stress"])
                hours = int(info.get("n_hours", 1))
                episode_renewable_sum += float(info["renewable_share"]) * hours
                episode_hours += hours

                observation = next_observation
                total_steps += 1
                progress.update(1)

                if terminated:
                    episode_record = {
                        "episode": len(history["episodes"]) + 1,  # type: ignore[arg-type]
                        "total_steps": total_steps,
                        "reward_vector": episode_reward.tolist(),
                        "total_cost": episode_cost,
                        "grid_stress": episode_stress,
                        "renewable_share": episode_renewable_sum / max(episode_hours, 1),
                    }
                    history["episodes"].append(episode_record)  # type: ignore[union-attr]
                    if exp_logger is not None:
                        exp_logger.log_episode(episode_record)

                    agent.start_episode(training=True)
                    observation, _ = env.reset()
                    episode_reward = np.zeros(agent.env_reward_dim, dtype=np.float32)
                    episode_cost = 0.0
                    episode_stress = 0.0
                    episode_renewable_sum = 0.0
                    episode_hours = 0

                if total_steps >= training_config.total_timesteps:
                    break

            last_value = (
                np.zeros(agent.objective_dim, dtype=np.float32)
                if terminated
                else agent.value(observation)
            )
            agent.set_training_progress(total_steps / max(training_config.total_timesteps, 1))
            buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=agent.config.gamma,
                gae_lambda=agent.config.gae_lambda,
                normalize_rewards=agent.config.normalize_rewards,
            )
            losses = agent.update(buffer)
            update_idx += 1

            update_record = {
                "update": update_idx,
                "timesteps": total_steps,
                **losses,
            }

            if eval_env is not None and update_idx % max(training_config.eval_every_updates, 1) == 0:
                validation_result = _evaluate_validation_suite(agent, eval_env, training_config)
                eval_result = validation_result["evaluation"]
                selection_score = float(validation_result["selection_score"])
                selection_details = validation_result.get("selection_details")
                update_record["evaluation"] = eval_result
                update_record["selection_score"] = selection_score
                if selection_details is not None:
                    update_record["selection_details"] = selection_details
                evaluation_count += 1

                if training_config.output_dir is not None:
                    checkpoint_record = _persist_validation_checkpoint(
                        Path(training_config.output_dir),
                        update_idx,
                        agent,
                        eval_result,
                        selection_details=selection_details,
                    )
                    update_record["checkpoint"] = checkpoint_record
                    history.setdefault("checkpoints", []).append(checkpoint_record)  # type: ignore[union-attr]

                candidate = {
                    "update": update_idx,
                    "selection_score": selection_score,
                    "evaluation": eval_result,
                    "checkpoint": update_record.get("checkpoint"),
                    "selection_details": selection_details,
                }
                if _is_better_evaluation(
                    candidate,
                    best_validation,
                    min_delta=training_config.early_stopping_min_delta,
                ):
                    best_validation = candidate
                    best_state_dict = _clone_state_dict_to_cpu(agent)
                    evaluations_without_improvement = 0
                    logger.info(
                        "New best validation checkpoint at update %d with selection score %.6f",
                        update_idx,
                        selection_score,
                    )
                else:
                    evaluations_without_improvement += 1

                if (
                    training_config.early_stopping_patience_evals > 0
                    and evaluation_count >= training_config.early_stopping_min_evals
                    and evaluations_without_improvement >= training_config.early_stopping_patience_evals
                ):
                    early_stop_reason = (
                        "validation_plateau:"
                        f"{evaluations_without_improvement}_evals_without_improvement"
                    )

                if exp_logger is not None:
                    exp_logger.log_evaluation(eval_result, split="val")

            history["updates"].append(update_record)  # type: ignore[union-attr]
            if exp_logger is not None:
                exp_logger.log_update(update_record)

            progress.set_postfix(
                update=update_idx,
                policy=f"{losses['policy_loss']:.3f}",
                value=f"{losses['value_loss']:.1f}",
                entropy=f"{losses['entropy']:.2f}",
            )

            if early_stop_reason is not None:
                logger.info("Stopping training early at update %d (%s)", update_idx, early_stop_reason)
                break

    if best_validation is not None:
        history["best_validation"] = best_validation
    if early_stop_reason is not None:
        history["stopped_early"] = True
        history["early_stop_reason"] = early_stop_reason
        history["final_update"] = update_idx
    else:
        history["stopped_early"] = False
        history["final_update"] = update_idx

    if training_config.restore_best_model_at_end and best_state_dict is not None:
        agent.network.load_state_dict(best_state_dict)
        agent.network.to(agent.device)
        if hasattr(agent, "set_eval_preferences") and best_validation is not None:
            selection_details = best_validation.get("selection_details") or {}
            selected_weights = selection_details.get("selected_eval_preference_weights")
            if selected_weights is not None:
                agent.set_eval_preferences(selected_weights)
        logger.info(
            "Restored best validation model from update %s (selection score %.6f)",
            best_validation.get("update") if best_validation else "?",
            float(best_validation.get("selection_score", float("nan"))) if best_validation else float("nan"),
        )

    return history


def train_multi_seed(
    env_factory: Callable[[], object],
    base_ppo_config: PPOConfig,
    training_config: TrainingConfig,
    seeds: tuple[int, ...],
    mode: str = "moppo",
    eval_env_factory: Optional[Callable[[], object]] = None,
    output_dir: Optional[Path] = None,
    experiment_name: str = "experiment",
) -> list[dict[str, object]]:
    """
    Train one agent per seed and return per-seed results.
    """
    resolved_eval = eval_env_factory or env_factory
    all_results: list[dict[str, object]] = []

    for seed in tqdm(seeds, desc=f"Seeds ({experiment_name})", unit="seed"):
        seed_everything(seed)
        run_id = get_run_id(experiment_name, seed)

        run_dir: Optional[Path] = None
        if output_dir is not None:
            run_dir = output_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

        local_config = copy.deepcopy(base_ppo_config)
        local_config.seed = seed

        local_training_config = copy.deepcopy(training_config)
        local_training_config.output_dir = run_dir

        env = env_factory()
        eval_env = resolved_eval()
        agent = _build_agent(mode, env, local_config)

        log_dir = run_dir or Path(".")
        with ExperimentLogger(output_dir=log_dir, run_id=run_id) as exp_logger:
            exp_logger.log_config(
                {
                    "experiment_name": experiment_name,
                    "seed": seed,
                    "mode": mode,
                    "ppo_config": asdict(local_config),
                    "training_config": asdict(local_training_config),
                }
            )

            logger.info("Starting seed %d / run %s", seed, run_id)
            history = train_agent(
                agent,
                env,
                training_config=local_training_config,
                eval_env=eval_env,
                exp_logger=exp_logger,
            )
            agent.start_rollout(training=False)
            evaluation = evaluate_agent(
                agent,
                eval_env,
                episodes=local_training_config.eval_episodes,
                deterministic=local_training_config.deterministic_eval,
            )
            exp_logger.log_evaluation(evaluation, split="val")

        if run_dir is not None:
            _persist_run_artifacts(run_dir, agent, history, evaluation)
            logger.info("Run artifacts saved to %s", run_dir)

        all_results.append(
            {
                "seed": seed,
                "run_id": run_id,
                "history": history,
                "evaluation": evaluation,
                "output_dir": str(run_dir) if run_dir else None,
            }
        )

    return all_results


def train_weight_sweep(
    env_factory: Callable[[], object],
    training_config: TrainingConfig,
    ppo_config: PPOConfig,
    weight_grid: list[tuple[float, ...]],
    mode: str = "moppo",
    output_dir: Path | None = None,
    eval_env_factory: Callable[[], object] | None = None,
) -> dict[tuple[float, ...], dict[str, object]]:
    """
    Train one agent per scalarisation weight vector.
    """
    results: dict[tuple[float, ...], dict[str, object]] = {}
    resolved_eval = eval_env_factory or env_factory

    for weights in tqdm(
        weight_grid,
        desc="Sweep",
        unit="config",
        dynamic_ncols=True,
        disable=not training_config.show_progress,
    ):
        seed_everything(ppo_config.seed)

        local_config = copy.deepcopy(ppo_config)
        local_config.scalarization_weights = tuple(weights)

        weight_label = "_".join(f"{w:.2f}" for w in weights)
        run_dir: Optional[Path] = None
        if output_dir is not None:
            run_dir = output_dir / f"weights_{weight_label}"
            run_dir.mkdir(parents=True, exist_ok=True)

        local_training_config = copy.deepcopy(training_config)
        local_training_config.output_dir = run_dir

        env = env_factory()
        eval_env = resolved_eval()
        agent = _build_agent(mode, env, local_config)

        log_dir = run_dir or Path(".")
        run_id = get_run_id(f"sweep_{weight_label}", seed=local_config.seed)
        with ExperimentLogger(output_dir=log_dir, run_id=run_id) as exp_logger:
            exp_logger.log_config(
                {
                    "weights": list(weights),
                    "ppo_config": asdict(local_config),
                    "training_config": asdict(local_training_config),
                }
            )
            history = train_agent(
                agent,
                env,
                training_config=local_training_config,
                eval_env=eval_env,
                exp_logger=exp_logger,
            )
            agent.start_rollout(training=False)
            evaluation = evaluate_agent(
                agent,
                eval_env,
                episodes=local_training_config.eval_episodes,
                deterministic=local_training_config.deterministic_eval,
            )
            exp_logger.log_evaluation(evaluation, split="val")

        if run_dir is not None:
            _persist_run_artifacts(run_dir, agent, history, evaluation)

        results[tuple(weights)] = {"history": history, "evaluation": evaluation}

    return results
