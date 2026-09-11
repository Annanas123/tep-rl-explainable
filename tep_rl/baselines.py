from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .envs import ProxyTEPEnv


BASELINE_LABELS = {
    "zero": "Zero",
    "uniform": "Uniform",
    "top_1_excess": "Top-1 Excess",
    "top_3_excess": "Top-3 Excess",
    "proportional_excess": "Proportional Excess",
    "myopic_proxy": "Myopic One-Step (Proxy)",
}


def _normalize_weights(weights) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return np.asarray([1.0], dtype=np.float32)
    array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0.0:
        return np.full(array.shape, 1.0 / float(array.size), dtype=np.float32)
    return array / total


class BaselineAgent:
    def __init__(self, env):
        self.env = env
        self.action_dim = int(env.action_space.shape[0])
        self.env_reward_dim = len(env.config.objective_names)

    def start_rollout(self, training: bool = False) -> None:
        del training

    def start_episode(self, training: bool = False) -> None:
        del training

    def _candidate_count(self) -> int:
        return len(self.env.candidate_lines)

    def _is_budgeted_action(self) -> bool:
        return self.action_dim == self._candidate_count() + 1

    def _zero_value(self) -> np.ndarray:
        return np.zeros(self.env_reward_dim, dtype=np.float32)

    def _candidate_loading(self) -> np.ndarray:
        return (
            self.env.preview_state.simulation.raw_line_loading
            .reindex(self.env.candidate_lines)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

    def _candidate_excess(self) -> np.ndarray:
        return np.clip(self._candidate_loading() - float(self.env.config.stability_margin), 0.0, None)

    def _encode_scores(self, scores: np.ndarray, spend_fraction: float = 1.0) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.size != self._candidate_count():
            raise ValueError(f"Expected {self._candidate_count()} scores, got {scores.size}.")

        positive = np.clip(scores, 0.0, None)
        if not np.any(positive > 0.0):
            return np.zeros(self.action_dim, dtype=np.float32)

        max_score = float(positive.max())
        if max_score > 0.0:
            positive = positive / max_score

        spend_fraction = float(np.clip(spend_fraction, 0.0, 1.0))
        if self._is_budgeted_action():
            action = np.zeros(self.action_dim, dtype=np.float32)
            action[0] = spend_fraction
            action[1:] = np.clip(positive, 0.0, 1.0)
            return action

        weights = positive / max(float(positive.sum()), 1e-6)
        scaled = np.clip(weights * float(scores.size) * spend_fraction, 0.0, 1.0)
        return scaled.astype(np.float32)

    def _encode_direct_increments(self, increments: pd.Series) -> np.ndarray:
        remaining_cap = self.env._remaining_line_upgrade_cap().reindex(self.env.candidate_lines).fillna(0.0)
        fractions = np.zeros(self._candidate_count(), dtype=np.float32)
        for idx, line in enumerate(self.env.candidate_lines):
            cap = float(remaining_cap.loc[line])
            if cap <= 1e-8:
                fractions[idx] = 0.0
            else:
                fractions[idx] = float(np.clip(float(increments.loc[line]) / cap, 0.0, 1.0))
        return fractions


class ZeroAgent(BaselineAgent):
    def act(self, observation, deterministic: bool = True):
        del observation, deterministic
        action = np.zeros(self.action_dim, dtype=np.float32)
        return action, 0.0, self._zero_value()


class UniformAgent(BaselineAgent):
    def __init__(self, env, spend_fraction: float = 1.0):
        super().__init__(env)
        self.spend_fraction = float(spend_fraction)

    def act(self, observation, deterministic: bool = True):
        del observation, deterministic
        scores = np.ones(self._candidate_count(), dtype=np.float32)
        action = self._encode_scores(scores, spend_fraction=self.spend_fraction)
        return action, 0.0, self._zero_value()


class TopKExcessAgent(BaselineAgent):
    def __init__(self, env, top_k: int, spend_fraction: float = 1.0):
        super().__init__(env)
        self.top_k = max(int(top_k), 1)
        self.spend_fraction = float(spend_fraction)

    def act(self, observation, deterministic: bool = True):
        del observation, deterministic
        excess = self._candidate_excess()
        if not np.any(excess > 0.0):
            action = np.zeros(self.action_dim, dtype=np.float32)
        else:
            active = np.flatnonzero(excess > 0.0)
            ranked = active[np.argsort(excess[active])[::-1]]
            chosen = ranked[: self.top_k]
            scores = np.zeros_like(excess)
            scores[chosen] = excess[chosen]
            action = self._encode_scores(scores, spend_fraction=self.spend_fraction)
        return action, 0.0, self._zero_value()


class ProportionalExcessAgent(BaselineAgent):
    def __init__(self, env, spend_fraction: float = 1.0):
        super().__init__(env)
        self.spend_fraction = float(spend_fraction)

    def act(self, observation, deterministic: bool = True):
        del observation, deterministic
        action = self._encode_scores(self._candidate_excess(), spend_fraction=self.spend_fraction)
        return action, 0.0, self._zero_value()


@dataclass
class _SelectorState:
    current_step: int
    start_index: int
    current_line_capacities: pd.Series
    cumulative_upgrades: pd.Series
    preview_state: object


class MyopicOneStepProxyAgent(BaselineAgent):
    def __init__(
        self,
        env,
        objective_weights: tuple[float, ...] | list[float] | np.ndarray = (0.34, 0.33, 0.33),
        chunk_mw: float | None = None,
        max_chunks: int = 12,
        min_improvement: float = 1e-6,
    ):
        super().__init__(env)
        self.objective_weights = _normalize_weights(objective_weights)
        self.chunk_mw = None if chunk_mw is None else float(chunk_mw)
        self.max_chunks = max(int(max_chunks), 1)
        self.min_improvement = float(min_improvement)
        self.selector_env = ProxyTEPEnv(env.dataset, env.config)

    def _sync_selector(self) -> None:
        self.selector_env.start_index = int(self.env.start_index)
        self.selector_env.current_step = int(self.env.current_step)
        self.selector_env.current_line_capacities = self.env.current_line_capacities.copy()
        self.selector_env.cumulative_upgrades = self.env.cumulative_upgrades.copy()
        self.selector_env.preview_state = self.env.preview_state

    def _capture_selector_state(self) -> _SelectorState:
        return _SelectorState(
            current_step=int(self.selector_env.current_step),
            start_index=int(self.selector_env.start_index),
            current_line_capacities=self.selector_env.current_line_capacities.copy(),
            cumulative_upgrades=self.selector_env.cumulative_upgrades.copy(),
            preview_state=self.selector_env.preview_state,
        )

    def _restore_selector_state(self, state: _SelectorState) -> None:
        self.selector_env.current_step = state.current_step
        self.selector_env.start_index = state.start_index
        self.selector_env.current_line_capacities = state.current_line_capacities.copy()
        self.selector_env.cumulative_upgrades = state.cumulative_upgrades.copy()
        self.selector_env.preview_state = state.preview_state

    def _immediate_scalarized_score(self, increments: pd.Series) -> float:
        snapshot = self._capture_selector_state()
        try:
            updated_upgrades = snapshot.cumulative_upgrades.add(increments, fill_value=0.0)
            updated_capacities = snapshot.current_line_capacities.copy()
            updated_capacities.loc[self.env.candidate_lines] = (
                self.env.base_line_capacities.reindex(self.env.candidate_lines).fillna(0.0) + updated_upgrades
            )
            self.selector_env.cumulative_upgrades = updated_upgrades
            self.selector_env.current_line_capacities = updated_capacities

            horizon = self.selector_env._transition_horizon(self.selector_env.current_step)
            simulation, renewable_share_reward = self.selector_env._simulate_transition_window(horizon)
            third_reward = self.selector_env._third_objective_reward(simulation, renewable_share_reward)
            investment_cost = self.env._episode_investment_cost(increments)
            total_cost = investment_cost + float(simulation.operating_cost)
            reward_vector = np.asarray(
                [
                    -total_cost / max(float(self.env.config.cost_reward_scale), 1e-6),
                    -float(simulation.grid_stress) / max(float(self.env.config.overload_reward_scale), 1e-6),
                    float(third_reward),
                ],
                dtype=np.float32,
            )
            weights = self.objective_weights[: reward_vector.size]
            weights = _normalize_weights(weights)
            return float(np.dot(weights, reward_vector))
        finally:
            self._restore_selector_state(snapshot)

    def _chunk_size(self, available_budget: float) -> float:
        if self.chunk_mw is not None:
            return min(max(self.chunk_mw, 1e-6), available_budget)
        return min(max(float(self.env.config.max_line_upgrade_mw) / 2.0, 1e-6), available_budget)

    def act(self, observation, deterministic: bool = True):
        del observation, deterministic
        if not self.env._is_decision_step():
            return np.zeros(self._candidate_count(), dtype=np.float32), 0.0, self._zero_value()

        self._sync_selector()
        available_budget = float(self.env._available_budget())
        remaining_cap = self.env._remaining_line_upgrade_cap().reindex(self.env.candidate_lines).fillna(0.0)
        if available_budget <= 1e-8 or float(remaining_cap.sum()) <= 1e-8:
            return np.zeros(self._candidate_count(), dtype=np.float32), 0.0, self._zero_value()

        increments = pd.Series(0.0, index=self.env.candidate_lines, dtype=float)
        current_score = self._immediate_scalarized_score(increments)
        remaining_budget = min(available_budget, float(remaining_cap.sum()))
        max_chunks = self.max_chunks

        while remaining_budget > 1e-8 and max_chunks > 0:
            max_chunks -= 1
            chunk_size = self._chunk_size(remaining_budget)
            best_line = None
            best_score = current_score
            best_step = 0.0
            best_gain_per_mw = self.min_improvement

            residual_cap = (remaining_cap - increments).clip(lower=0.0)
            for line in self.env.candidate_lines:
                feasible = float(residual_cap.loc[line])
                if feasible <= 1e-8:
                    continue
                trial_step = min(chunk_size, feasible, remaining_budget)
                if trial_step <= 1e-8:
                    continue
                trial_increments = increments.copy()
                trial_increments.loc[line] += trial_step
                trial_score = self._immediate_scalarized_score(trial_increments)
                gain_per_mw = (trial_score - current_score) / trial_step
                if gain_per_mw > best_gain_per_mw:
                    best_gain_per_mw = gain_per_mw
                    best_score = trial_score
                    best_line = line
                    best_step = trial_step

            if best_line is None or best_step <= 0.0:
                break

            increments.loc[best_line] += best_step
            remaining_budget = max(remaining_budget - best_step, 0.0)
            current_score = best_score

        action = self._encode_direct_increments(increments)
        return action, 0.0, self._zero_value()


def build_baseline_agents(
    env,
    uniform_value: float = 1.0,
    heuristic_top_k: int = 3,
    myopic_weights: tuple[float, ...] | list[float] | np.ndarray = (0.34, 0.33, 0.33),
    myopic_chunk_mw: float | None = None,
) -> dict[str, BaselineAgent]:
    return {
        "zero": ZeroAgent(env),
        "uniform": UniformAgent(env, spend_fraction=uniform_value),
        "top_1_excess": TopKExcessAgent(env, top_k=1, spend_fraction=1.0),
        "top_3_excess": TopKExcessAgent(env, top_k=heuristic_top_k, spend_fraction=1.0),
        "proportional_excess": ProportionalExcessAgent(env, spend_fraction=1.0),
        "myopic_proxy": MyopicOneStepProxyAgent(
            env,
            objective_weights=myopic_weights,
            chunk_mw=myopic_chunk_mw,
        ),
    }
