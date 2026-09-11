from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Independent

from .config import PPOConfig


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def _normalize_weight_grid(weight_grid: list[list[float]] | tuple[tuple[float, ...], ...] | None, objective_dim: int) -> tuple[np.ndarray, ...]:
    if not weight_grid:
        return ()
    normalized: list[np.ndarray] = []
    for weights in weight_grid:
        array = _normalize_weights(np.asarray(weights[:objective_dim], dtype=np.float32))
        normalized.append(array)
    return tuple(normalized)


def _build_mlp(input_dim: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(previous, hidden))
        layers.append(nn.Tanh())
        previous = hidden
    return nn.Sequential(*layers)


def _resolve_device(device_name: str) -> torch.device:
    name = str(device_name).strip().lower()
    if name in {"", "auto", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


class ActorCriticNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        objective_dim: int,
        hidden_sizes: tuple[int, ...],
        context_dim: int = 0,
    ):
        super().__init__()
        self.context_dim = max(int(context_dim), 0)
        self.actor_encoder = _build_mlp(obs_dim + self.context_dim, hidden_sizes)
        self.critic_encoder = _build_mlp(obs_dim + self.context_dim, hidden_sizes)
        actor_hidden = hidden_sizes[-1] if hidden_sizes else obs_dim + self.context_dim
        critic_hidden = hidden_sizes[-1] if hidden_sizes else obs_dim + self.context_dim
        self.alpha_head = nn.Linear(actor_hidden, action_dim)
        self.beta_head = nn.Linear(actor_hidden, action_dim)
        self.value_heads = nn.ModuleList(nn.Linear(critic_hidden, 1) for _ in range(objective_dim))

    def _actor_latent(self, observations: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.context_dim > 0:
            if context is None:
                context = torch.zeros(
                    (observations.shape[0], self.context_dim),
                    dtype=observations.dtype,
                    device=observations.device,
                )
            latent_input = torch.cat([observations, context], dim=-1)
        else:
            latent_input = observations
        return self.actor_encoder(latent_input)

    def _critic_latent(self, observations: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.context_dim > 0:
            if context is None:
                context = torch.zeros(
                    (observations.shape[0], self.context_dim),
                    dtype=observations.dtype,
                    device=observations.device,
                )
            critic_input = torch.cat([observations, context], dim=-1)
        else:
            critic_input = observations
        return self.critic_encoder(critic_input)

    def policy_distribution(self, observations: torch.Tensor, context: torch.Tensor | None = None) -> Independent:
        latent = self._actor_latent(observations, context=context)
        alpha = F.softplus(self.alpha_head(latent)) + 1.01
        beta = F.softplus(self.beta_head(latent)) + 1.01
        return Independent(Beta(alpha, beta), 1)

    def values(self, observations: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        latent = self._critic_latent(observations, context=context)
        values = [head(latent) for head in self.value_heads]
        return torch.cat(values, dim=-1)

    def policy_mean(self, observations: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        latent = self._actor_latent(observations, context=context)
        alpha = F.softplus(self.alpha_head(latent)) + 1.01
        beta = F.softplus(self.beta_head(latent)) + 1.01
        return alpha / (alpha + beta)

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = actions.clamp(1e-5, 1.0 - 1e-5)
        dist = self.policy_distribution(observations, context=context)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.values(observations, context=context)
        return log_prob, entropy, values


class RolloutBuffer:
    def __init__(self, rollout_steps: int, obs_dim: int, action_dim: int, objective_dim: int, context_dim: int = 0):
        self.rollout_steps = rollout_steps
        self.context_dim = max(int(context_dim), 0)
        self.obs = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(rollout_steps, dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, objective_dim), dtype=np.float32)
        self.values = np.zeros((rollout_steps, objective_dim), dtype=np.float32)
        self.dones = np.zeros(rollout_steps, dtype=np.float32)
        self.advantages = np.zeros((rollout_steps, objective_dim), dtype=np.float32)
        self.returns = np.zeros((rollout_steps, objective_dim), dtype=np.float32)
        self.contexts = (
            np.zeros((rollout_steps, self.context_dim), dtype=np.float32)
            if self.context_dim > 0
            else None
        )
        self.position = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: np.ndarray,
        value: np.ndarray,
        done: bool,
        context: np.ndarray | None = None,
    ) -> None:
        self.obs[self.position] = observation
        self.actions[self.position] = action
        self.log_probs[self.position] = log_prob
        self.rewards[self.position] = reward
        self.values[self.position] = value
        self.dones[self.position] = float(done)
        if self.contexts is not None:
            if context is None:
                self.contexts[self.position] = 0.0
            else:
                self.contexts[self.position] = np.asarray(context, dtype=np.float32)
        self.position += 1

    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,
        gamma: float,
        gae_lambda: float,
        normalize_rewards: bool,
    ) -> None:
        rewards = self.rewards.copy()
        if normalize_rewards:
            reward_mean = rewards.mean(axis=0, keepdims=True)
            reward_std = rewards.std(axis=0, keepdims=True) + 1e-8
            rewards = (rewards - reward_mean) / reward_std

        advantage = np.zeros_like(last_value, dtype=np.float32)
        for step in reversed(range(self.position)):
            next_non_terminal = 1.0 - self.dones[step]
            next_value = last_value if step == self.position - 1 else self.values[step + 1]
            delta = rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            self.advantages[step] = advantage
            self.returns[step] = self.advantages[step] + self.values[step]

    def minibatches(self, batch_size: int):
        indices = np.arange(self.position)
        np.random.shuffle(indices)
        for start in range(0, self.position, batch_size):
            yield indices[start : start + batch_size]


class BasePPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        objective_dim: int,
        env_reward_dim: int,
        config: PPOConfig,
        agent_kind: str,
        context_dim: int = 0,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.objective_dim = objective_dim
        self.env_reward_dim = env_reward_dim
        self.config = config
        self.agent_kind = agent_kind
        self.context_dim = max(int(context_dim), 0)
        self.device = _resolve_device(config.device)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        network = ActorCriticNetwork(
            obs_dim,
            action_dim,
            objective_dim,
            config.hidden_sizes,
            context_dim=self.context_dim,
        )
        try:
            self.network = network.to(self.device)
        except Exception as exc:
            if self.device.type != "cuda":
                raise
            self.device = torch.device("cpu")
            self.config.device = "cpu"
            self.network = network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.current_learning_rate = float(config.learning_rate)
        self.current_entropy_coef = float(config.entropy_coef)

    def _is_cuda_runtime_error(self, exc: Exception) -> bool:
        if self.device.type != "cuda":
            return False
        return "cuda" in str(exc).lower()

    def _move_optimizer_state(self, device: torch.device) -> None:
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    def _fallback_to_cpu(self) -> None:
        if self.device.type == "cpu":
            return
        self.device = torch.device("cpu")
        self.config.device = "cpu"
        self.network = self.network.to(self.device)
        self._move_optimizer_state(self.device)

    def prepare_reward(self, reward_vector: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _scalarize_advantages(
        self,
        advantages: torch.Tensor,
        contexts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def start_rollout(self, training: bool = False) -> None:
        del training

    def start_episode(self, training: bool = False) -> None:
        del training

    def current_context(self) -> np.ndarray | None:
        if self.context_dim <= 0:
            return None
        return np.zeros(self.context_dim, dtype=np.float32)

    def set_training_progress(self, progress_fraction: float) -> None:
        progress = float(np.clip(progress_fraction, 0.0, 1.0))

        if str(self.config.learning_rate_schedule).lower() == "linear":
            start_lr = float(self.config.learning_rate)
            end_lr = float(self.config.final_learning_rate)
            self.current_learning_rate = start_lr + (end_lr - start_lr) * progress
        else:
            self.current_learning_rate = float(self.config.learning_rate)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_learning_rate

        if str(self.config.entropy_coef_schedule).lower() == "linear":
            start_entropy = float(self.config.entropy_coef)
            end_entropy = float(self.config.final_entropy_coef)
            self.current_entropy_coef = start_entropy + (end_entropy - start_entropy) * progress
        else:
            self.current_entropy_coef = float(self.config.entropy_coef)

    def _current_context_tensor(self, batch_size: int = 1) -> torch.Tensor | None:
        context = self.current_context()
        if context is None:
            return None
        context_array = np.asarray(context, dtype=np.float32)
        context_batch = np.repeat(context_array[None, :], batch_size, axis=0)
        return torch.as_tensor(context_batch, dtype=torch.float32, device=self.device)

    def act(self, observation: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, np.ndarray]:
        try:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            context_tensor = self._current_context_tensor(batch_size=1)
            with torch.no_grad():
                if deterministic:
                    action_tensor = self.network.policy_mean(observation_tensor, context=context_tensor)
                    dist = self.network.policy_distribution(observation_tensor, context=context_tensor)
                    log_prob = dist.log_prob(action_tensor)
                else:
                    dist = self.network.policy_distribution(observation_tensor, context=context_tensor)
                    action_tensor = dist.sample()
                    log_prob = dist.log_prob(action_tensor)
                value_tensor = self.network.values(observation_tensor, context=context_tensor)
        except Exception as exc:
            if not self._is_cuda_runtime_error(exc):
                raise
            self._fallback_to_cpu()
            return self.act(observation, deterministic=deterministic)

        action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        return action, float(log_prob.item()), value_tensor.squeeze(0).cpu().numpy().astype(np.float32)

    def value(self, observation: np.ndarray) -> np.ndarray:
        try:
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            context_tensor = self._current_context_tensor(batch_size=1)
            with torch.no_grad():
                values = self.network.values(observation_tensor, context=context_tensor)
        except Exception as exc:
            if not self._is_cuda_runtime_error(exc):
                raise
            self._fallback_to_cpu()
            return self.value(observation)
        return values.squeeze(0).cpu().numpy().astype(np.float32)

    def policy_mean(self, observations: np.ndarray) -> np.ndarray:
        try:
            observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            if observation_tensor.dim() == 1:
                observation_tensor = observation_tensor.unsqueeze(0)
            context_tensor = self._current_context_tensor(batch_size=observation_tensor.shape[0])
            with torch.no_grad():
                means = self.network.policy_mean(observation_tensor, context=context_tensor)
        except Exception as exc:
            if not self._is_cuda_runtime_error(exc):
                raise
            self._fallback_to_cpu()
            return self.policy_mean(observations)
        return means.cpu().numpy()

    def update(self, buffer: RolloutBuffer) -> dict[str, float | bool]:
        try:
            observations = torch.as_tensor(buffer.obs[: buffer.position], dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(buffer.actions[: buffer.position], dtype=torch.float32, device=self.device)
            old_log_probs = torch.as_tensor(buffer.log_probs[: buffer.position], dtype=torch.float32, device=self.device)
            returns = torch.as_tensor(buffer.returns[: buffer.position], dtype=torch.float32, device=self.device)
            advantages = torch.as_tensor(buffer.advantages[: buffer.position], dtype=torch.float32, device=self.device)
            contexts = None
            if buffer.contexts is not None:
                contexts = torch.as_tensor(buffer.contexts[: buffer.position], dtype=torch.float32, device=self.device)

            scalar_advantages = self._scalarize_advantages(advantages, contexts=contexts)
            # Objective-wise standardisation already places MO-PPO advantages
            # on comparable scales. A second scalar standardisation would make
            # the documented preference scalarisation harder to interpret.
            normalize_scalar = self.config.normalize_advantages and not (
                self.agent_kind == "moppo" and self.config.normalize_objective_advantages
            )
            if normalize_scalar:
                scalar_advantages = (scalar_advantages - scalar_advantages.mean()) / (scalar_advantages.std() + 1e-8)

            actor_losses = []
            value_losses = []
            entropies = []
            clip_fractions = []
            approx_kls = []
            stop_early = False

            for _ in range(self.config.update_epochs):
                for batch_indices in buffer.minibatches(self.config.minibatch_size):
                    batch_obs = observations[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_scalar_advantages = scalar_advantages[batch_indices]
                    batch_contexts = contexts[batch_indices] if contexts is not None else None

                    new_log_probs, entropy, values = self.network.evaluate_actions(
                        batch_obs,
                        batch_actions,
                        context=batch_contexts,
                    )
                    log_ratios = new_log_probs - batch_old_log_probs
                    ratios = torch.exp(log_ratios)
                    clipped_ratios = torch.clamp(ratios, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon)
                    policy_loss = -torch.min(
                        ratios * batch_scalar_advantages,
                        clipped_ratios * batch_scalar_advantages,
                    ).mean()

                    value_loss = F.mse_loss(values, batch_returns)
                    entropy_loss = entropy.mean()
                    loss = policy_loss + self.config.value_coef * value_loss - self.current_entropy_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                    actor_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy_loss.item()))
                    clip_fraction = ((ratios - 1.0).abs() > self.config.clip_epsilon).float().mean()
                    approx_kl = ((ratios - 1.0) - log_ratios).mean()
                    clip_fractions.append(float(clip_fraction.item()))
                    approx_kls.append(float(approx_kl.item()))

                    if self.config.target_kl is not None and float(approx_kl.item()) > float(self.config.target_kl):
                        stop_early = True
                        break
                if stop_early:
                    break

            return {
                "policy_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
                "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
                "entropy": float(np.mean(entropies)) if entropies else 0.0,
                "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
                "learning_rate": float(self.current_learning_rate),
                "entropy_coef": float(self.current_entropy_coef),
                "stopped_early": stop_early,
            }
        except Exception as exc:
            if not self._is_cuda_runtime_error(exc):
                raise
            self._fallback_to_cpu()
            return self.update(buffer)

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "state_dict": self.network.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "objective_dim": self.objective_dim,
            "env_reward_dim": self.env_reward_dim,
            "agent_kind": self.agent_kind,
            "config": asdict(self.config),
        }
        if self.agent_kind == "moppo":
            eval_weights = getattr(self, "eval_preference_weights", None)
            current_weights = getattr(self, "current_preference_weights", None)
            checkpoint["eval_preference_weights"] = None if eval_weights is None else np.asarray(eval_weights, dtype=np.float32).tolist()
            checkpoint["current_preference_weights"] = None if current_weights is None else np.asarray(current_weights, dtype=np.float32).tolist()
        torch.save(checkpoint, Path(path))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "BasePPOAgent":
        resolved_device = _resolve_device(device or "cpu")
        checkpoint = torch.load(Path(path), map_location=resolved_device, weights_only=False)
        config_data = dict(checkpoint["config"])
        if checkpoint["agent_kind"] == "moppo":
            config_data.setdefault("moppo_preference_conditioning", False)
            config_data.setdefault("moppo_sample_preferences", False)
            config_data.setdefault("moppo_preference_sampling_mode", "dirichlet")
            config_data.setdefault("moppo_preference_grid", None)
            config_data.setdefault("moppo_dirichlet_alpha", 1.0)
            config_data.setdefault("normalize_objective_advantages", False)
        config = PPOConfig(**config_data)
        if device is not None:
            config.device = str(resolved_device)

        if checkpoint["agent_kind"] == "ppo":
            agent = PPOAgent(
                obs_dim=checkpoint["obs_dim"],
                action_dim=checkpoint["action_dim"],
                env_reward_dim=checkpoint["env_reward_dim"],
                config=config,
            )
        else:
            agent = MOPPOAgent(
                obs_dim=checkpoint["obs_dim"],
                action_dim=checkpoint["action_dim"],
                env_reward_dim=checkpoint["env_reward_dim"],
                config=config,
            )

        state_dict = checkpoint["state_dict"]
        if any(key.startswith("encoder.") for key in state_dict):
            upgraded_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("encoder."):
                    suffix = key.removeprefix("encoder.")
                    upgraded_state_dict[f"actor_encoder.{suffix}"] = value
                    upgraded_state_dict[f"critic_encoder.{suffix}"] = value.clone()
                else:
                    upgraded_state_dict[key] = value
            state_dict = upgraded_state_dict

        agent.network.load_state_dict(state_dict)
        agent.network.to(agent.device)
        if checkpoint["agent_kind"] == "moppo":
            stored_eval_weights = checkpoint.get("eval_preference_weights")
            if stored_eval_weights is not None:
                agent.set_eval_preferences(stored_eval_weights)
            stored_current_weights = checkpoint.get("current_preference_weights")
            if stored_current_weights is not None:
                agent.current_preference_weights = _normalize_weights(np.asarray(stored_current_weights, dtype=np.float32))
        return agent


class PPOAgent(BasePPOAgent):
    def __init__(self, obs_dim: int, action_dim: int, env_reward_dim: int, config: PPOConfig):
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            objective_dim=1,
            env_reward_dim=env_reward_dim,
            config=config,
            agent_kind="ppo",
            context_dim=0,
        )
        self.scalarization_weights = _normalize_weights(
            np.asarray(config.scalarization_weights[:env_reward_dim], dtype=np.float32)
        )

    def prepare_reward(self, reward_vector: np.ndarray) -> np.ndarray:
        scalar_reward = float(np.dot(self.scalarization_weights, reward_vector[: self.env_reward_dim]))
        return np.asarray([scalar_reward], dtype=np.float32)

    def _scalarize_advantages(
        self,
        advantages: torch.Tensor,
        contexts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del contexts
        return advantages.squeeze(-1)


class MOPPOAgent(BasePPOAgent):
    def __init__(self, obs_dim: int, action_dim: int, env_reward_dim: int, config: PPOConfig):
        self.eval_preference_weights = _normalize_weights(
            np.asarray(config.scalarization_weights[:env_reward_dim], dtype=np.float32)
        )
        self.current_preference_weights = self.eval_preference_weights.copy()
        self.training_preference_grid = _normalize_weight_grid(config.moppo_preference_grid, env_reward_dim)
        self.preference_grid_cursor = 0
        context_dim = env_reward_dim if config.moppo_preference_conditioning else 0
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            objective_dim=env_reward_dim,
            env_reward_dim=env_reward_dim,
            config=config,
            agent_kind="moppo",
            context_dim=context_dim,
        )

    def set_eval_preferences(self, weights: np.ndarray | list[float] | tuple[float, ...]) -> None:
        normalized = _normalize_weights(np.asarray(weights[: self.env_reward_dim], dtype=np.float32))
        self.eval_preference_weights = normalized
        self.current_preference_weights = normalized.copy()

    def start_rollout(self, training: bool = False) -> None:
        if not training:
            self.current_preference_weights = self.eval_preference_weights.copy()

    def start_episode(self, training: bool = False) -> None:
        if training and self.config.moppo_sample_preferences:
            sampling_mode = str(getattr(self.config, "moppo_preference_sampling_mode", "dirichlet")).lower()
            if sampling_mode == "grid" and self.training_preference_grid:
                index = self.preference_grid_cursor % len(self.training_preference_grid)
                self.current_preference_weights = self.training_preference_grid[index].astype(np.float32, copy=True)
                self.preference_grid_cursor += 1
            else:
                concentration = np.full(
                    self.env_reward_dim,
                    max(float(self.config.moppo_dirichlet_alpha), 1e-3),
                    dtype=np.float32,
                )
                self.current_preference_weights = _normalize_weights(
                    np.random.dirichlet(concentration).astype(np.float32)
                )
        else:
            self.current_preference_weights = self.eval_preference_weights.copy()

    def current_context(self) -> np.ndarray | None:
        if self.context_dim <= 0:
            return None
        return self.current_preference_weights.astype(np.float32, copy=True)

    def prepare_reward(self, reward_vector: np.ndarray) -> np.ndarray:
        return reward_vector[: self.env_reward_dim].astype(np.float32)

    def _scalarize_advantages(
        self,
        advantages: torch.Tensor,
        contexts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_advantages = advantages
        if self.config.normalize_objective_advantages:
            adv_mean = normalized_advantages.mean(dim=0, keepdim=True)
            adv_std = normalized_advantages.std(dim=0, keepdim=True, unbiased=False) + 1e-8
            normalized_advantages = (normalized_advantages - adv_mean) / adv_std

        if contexts is not None and contexts.numel() > 0:
            weights = contexts
        else:
            weights = torch.as_tensor(
                self.eval_preference_weights,
                dtype=torch.float32,
                device=advantages.device,
            ).expand(normalized_advantages.shape[0], -1)
        return (normalized_advantages * weights).sum(dim=-1)


def load_agent(path: str | Path, device: str | None = None) -> BasePPOAgent:
    return BasePPOAgent.load(path, device=device)
