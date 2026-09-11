from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .config import EnvironmentConfig
from .data import TEPDataset
from .simulation import DCPowerFlowModel, SimulationResult, run_dc_proxy, run_pypsa_lopf


def _max_frame_row_sum(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 1.0
    values = frame.to_numpy(dtype=np.float32, copy=False)
    row_sums = np.nansum(values, axis=1, dtype=np.float32)
    if row_sums.size == 0:
        return 1.0
    return max(float(np.nanmax(row_sums)), 1.0)


@dataclass
class ObservationPreview:
    timestamp: pd.Timestamp
    horizon_hours: int
    demand_by_bus: pd.Series
    renewable_by_bus: pd.Series
    simulation: SimulationResult


class BaseTEPEnv(gym.Env, ABC):
    metadata = {"render_modes": []}

    def __init__(self, dataset: TEPDataset, config: EnvironmentConfig):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.bus_names = list(dataset.network.buses.index)
        self.line_names = list(dataset.network.lines.index)
        self.candidate_lines = list(dataset.candidate_lines)

        self.base_line_capacities = dataset.network.lines["s_nom"].reindex(self.line_names).fillna(1.0).clip(lower=1.0)
        self.current_line_capacities = self.base_line_capacities.copy()
        self.cumulative_upgrades = pd.Series(0.0, index=self.candidate_lines)
        self.max_upgrade_vector = pd.Series(config.max_line_upgrade_mw, index=self.candidate_lines, dtype=float)

        lengths = dataset.network.lines["length"].reindex(self.candidate_lines).fillna(0.0).clip(lower=0.0)
        if (lengths <= 0.0).any():
            invalid = list(lengths[lengths <= 0.0].index)
            raise ValueError(f"Candidate lines require positive route lengths for cost calculation: {invalid}")
        # EUR/(MW year). Existing PyPSA capital_cost values are deliberately
        # not reused because their price year and annuitisation are not audited
        # for this extracted case.
        self.line_upgrade_cost = (
            lengths * float(config.line_investment_cost_eur_per_mw_km_year)
        ).astype(float)

        self.demand_scale = (
            dataset.demand_scale.reindex(self.bus_names).fillna(1.0)
            if dataset.demand_scale is not None
            else dataset.demand_by_bus.max().replace(0.0, 1.0)
        )
        self.renewable_scale = (
            dataset.renewable_scale.reindex(self.bus_names).fillna(1.0)
            if dataset.renewable_scale is not None
            else dataset.renewable_by_bus.max().replace(0.0, 1.0).reindex(self.bus_names).fillna(1.0)
        )
        self.total_demand_scale = (
            float(dataset.total_demand_scale)
            if dataset.total_demand_scale is not None
            else _max_frame_row_sum(dataset.demand_by_bus)
        )

        self.start_index = 0
        self.current_step = 0
        self.history: list[dict[str, Any]] = []
        self.last_result = self._empty_result()
        self.preview_state = self._empty_preview()

        self.feature_names = (
            [f"demand::{bus}" for bus in self.bus_names]
            + [f"renewable::{bus}" for bus in self.bus_names]
            + [f"line_loading_raw::{line}" for line in self.line_names]
            + [f"candidate_loading_raw::{line}" for line in self.candidate_lines]
            + [f"upgrade::{line}" for line in self.candidate_lines]
            + [
                "emissions_norm",
                "renewable_share",
                "grid_stress_norm",
                "remaining_budget_fraction",
                "available_budget_fraction",
                "decision_step_flag",
                "progress_fraction",
            ]
        )

        obs_dim = len(self.feature_names)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        if self.config.action_mode == "budgeted":
            self.action_labels = ["spend_fraction"] + [f"allocation::{line}" for line in self.candidate_lines]
        else:
            self.action_labels = [f"upgrade::{line}" for line in self.candidate_lines]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(len(self.action_labels),), dtype=np.float32)

    def _empty_result(self) -> SimulationResult:
        zeros = pd.Series(0.0, index=self.line_names, dtype=float)
        return SimulationResult(
            raw_line_flows=zeros.copy(),
            raw_line_loading=zeros.copy(),
            line_flows=zeros.copy(),
            line_loading=zeros.copy(),
            renewable_potential=0.0,
            renewable_served=0.0,
            renewable_share=0.0,
            renewable_curtailment=0.0,
            grid_stress=0.0,
            constraint_violation=0.0,
            slack_generation=0.0,
            load_shedding=0.0,
            emissions=0.0,
            operating_cost=0.0,
            backend="none",
        )

    def _empty_preview(self) -> ObservationPreview:
        timestamp = pd.Timestamp(self.dataset.snapshots[0]) if len(self.dataset.snapshots) else pd.Timestamp(0)
        zeros = pd.Series(0.0, index=self.bus_names, dtype=float)
        return ObservationPreview(
            timestamp=timestamp,
            horizon_hours=1,
            demand_by_bus=zeros.copy(),
            renewable_by_bus=zeros.copy(),
            simulation=self._empty_result(),
        )

    def _remaining_budget(self) -> float:
        return max(self.config.total_upgrade_budget_mw - float(self.cumulative_upgrades.sum()), 0.0)

    def _decision_interval(self) -> int:
        return max(int(self.config.decision_interval), 1)

    def _remaining_line_upgrade_cap(self) -> pd.Series:
        remaining = self.max_upgrade_vector - self.cumulative_upgrades
        return remaining.clip(lower=0.0)

    def _annualized_investment_cost(self, increments: pd.Series) -> float:
        """Return the annualised corridor cost in EUR/year."""
        return float((increments * self.line_upgrade_cost).sum())

    def _episode_investment_cost(self, increments: pd.Series) -> float:
        """Put annualised CAPEX on the same time basis as one episode's OPEX."""
        reference_hours = max(float(self.config.investment_cost_reference_hours), 1e-6)
        episode_fraction = float(self.config.episode_length) / reference_hours
        return self._annualized_investment_cost(increments) * episode_fraction

    def _decision_stage_count(self) -> int:
        return max(((self.config.episode_length - 1) // self._decision_interval()) + 1, 1)

    def _decision_stage_index(self) -> int:
        return min(self.current_step // self._decision_interval(), self._decision_stage_count() - 1)

    def _is_decision_step(self) -> bool:
        if self.config.temporal_mode == "decision_block":
            return True
        return self.current_step % self._decision_interval() == 0

    def _released_budget(self) -> float:
        total_budget = max(float(self.config.total_upgrade_budget_mw), 0.0)
        if self.config.budget_release == "all_at_once":
            return total_budget
        if self.config.budget_release == "linear":
            return total_budget * float(self._decision_stage_index() + 1) / float(self._decision_stage_count())
        raise ValueError(f"Unsupported budget release mode: {self.config.budget_release!r}")

    def _available_budget(self) -> float:
        released_budget = min(self._released_budget(), float(self.config.total_upgrade_budget_mw))
        return max(released_budget - float(self.cumulative_upgrades.sum()), 0.0)

    def _step_index(self) -> int:
        return self.start_index + self.current_step

    def _remaining_episode_hours(self, step_offset: int | None = None) -> int:
        step = self.current_step if step_offset is None else int(step_offset)
        return max(int(self.config.episode_length) - step, 0)

    def _remaining_dataset_hours(self, step_offset: int | None = None) -> int:
        row_idx = self.start_index + (self.current_step if step_offset is None else int(step_offset))
        return max(len(self.dataset.snapshots) - row_idx, 0)

    def _transition_horizon(self, step_offset: int | None = None) -> int:
        remaining_episode = self._remaining_episode_hours(step_offset)
        remaining_dataset = self._remaining_dataset_hours(step_offset)
        if remaining_episode <= 0 or remaining_dataset <= 0:
            return 0
        if self.config.temporal_mode == "hourly":
            return 1
        return max(min(self._decision_interval(), remaining_episode, remaining_dataset), 1)

    def _direct_action_dim(self) -> int:
        return len(self.candidate_lines)

    def _budgeted_action_dim(self) -> int:
        return len(self.candidate_lines) + 1

    def _decode_direct_action(
        self,
        action: np.ndarray,
        available_budget: float,
        remaining_line_cap: pd.Series,
    ) -> pd.Series:
        clipped = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
        proposed = pd.Series(clipped, index=self.candidate_lines, dtype=float) * remaining_line_cap
        total_increment = float(proposed.sum())

        if total_increment > available_budget > 0.0:
            proposed = proposed * (available_budget / total_increment)
        elif available_budget <= 0.0:
            proposed = proposed * 0.0

        return proposed.clip(lower=0.0)

    def _allocation_weights(self, scores: np.ndarray) -> pd.Series:
        weights = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
        if weights.size != len(self.candidate_lines):
            raise ValueError(
                f"Expected {len(self.candidate_lines)} allocation scores, got {weights.size}."
            )

        if weights.size == 0:
            return pd.Series(0.0, index=self.candidate_lines, dtype=float)

        cutoff = max(float(self.config.allocation_sparsity_cutoff), 0.0)
        max_score = float(weights.max(initial=0.0))
        if cutoff > 0.0 and max_score > 0.0:
            weights = np.where(weights >= cutoff * max_score, weights, 0.0)

        if float(weights.sum()) <= 0.0:
            return pd.Series(0.0, index=self.candidate_lines, dtype=float)

        sharpness = max(float(self.config.allocation_sharpness), 1.0)
        logits = sharpness * weights
        shifted = logits - float(np.max(logits))
        sorted_logits = np.sort(shifted)[::-1]
        cumulative = np.cumsum(sorted_logits)
        support = np.nonzero(1.0 + np.arange(1, len(sorted_logits) + 1) * sorted_logits > cumulative)[0]
        if support.size == 0:
            return pd.Series(0.0, index=self.candidate_lines, dtype=float)
        k = int(support[-1]) + 1
        tau = (float(cumulative[k - 1]) - 1.0) / float(k)
        sparse = np.maximum(shifted - tau, 0.0)
        total = float(sparse.sum())
        if total <= 0.0:
            return pd.Series(0.0, index=self.candidate_lines, dtype=float)
        return pd.Series(sparse / total, index=self.candidate_lines, dtype=float)

    def _allocate_budget(
        self,
        weights: pd.Series,
        remaining_line_cap: pd.Series,
        desired_spend: float,
    ) -> pd.Series:
        allocations = pd.Series(0.0, index=self.candidate_lines, dtype=float)
        residual_cap = remaining_line_cap.clip(lower=0.0).astype(float)
        residual_weights = weights.clip(lower=0.0).astype(float)
        remaining_budget = min(float(desired_spend), float(residual_cap.sum()))

        while remaining_budget > 1e-8:
            active = (residual_cap > 1e-8) & (residual_weights > 0.0)
            if not active.any():
                break

            active_weights = residual_weights.loc[active]
            weight_total = float(active_weights.sum())
            if weight_total <= 0.0:
                break

            proposed = active_weights / weight_total * remaining_budget
            capped = np.minimum(
                proposed.to_numpy(dtype=float),
                residual_cap.loc[active].to_numpy(dtype=float),
            )
            capped_series = pd.Series(capped, index=active_weights.index, dtype=float)
            spent = float(capped_series.sum())
            if spent <= 1e-10:
                break

            allocations.loc[capped_series.index] = allocations.loc[capped_series.index] + capped_series
            residual_cap.loc[capped_series.index] = residual_cap.loc[capped_series.index] - capped_series
            remaining_budget = max(remaining_budget - spent, 0.0)

        return allocations.clip(lower=0.0)

    def _decode_budgeted_action(
        self,
        action: np.ndarray,
        available_budget: float,
        remaining_line_cap: pd.Series,
    ) -> pd.Series:
        clipped = np.clip(np.asarray(action, dtype=np.float32), 0.0, 1.0)
        spend_fraction = float(clipped[0])
        weights = self._allocation_weights(clipped[1:])

        feasible_budget = min(float(available_budget), float(remaining_line_cap.sum()))
        desired_spend = spend_fraction * feasible_budget
        if desired_spend <= 0.0 or weights.sum() <= 0.0:
            return pd.Series(0.0, index=self.candidate_lines, dtype=float)
        return self._allocate_budget(weights, remaining_line_cap, desired_spend)

    def _decode_action(
        self,
        action: np.ndarray,
        available_budget: float | None = None,
        remaining_line_cap: pd.Series | None = None,
    ) -> pd.Series:
        if not self._is_decision_step():
            return pd.Series(0.0, index=self.candidate_lines, dtype=float)

        available_budget = self._available_budget() if available_budget is None else float(available_budget)
        remaining_line_cap = (
            self._remaining_line_upgrade_cap()
            if remaining_line_cap is None
            else remaining_line_cap.reindex(self.candidate_lines).fillna(0.0)
        )
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size == self._direct_action_dim():
            return self._decode_direct_action(action, available_budget, remaining_line_cap)
        if self.config.action_mode == "budgeted" and action.size == self._budgeted_action_dim():
            return self._decode_budgeted_action(action, available_budget, remaining_line_cap)
        raise ValueError(
            f"Unexpected action dimension {action.size} for mode {self.config.action_mode!r}. "
            f"Expected {self._direct_action_dim()} (direct) or {self._budgeted_action_dim()} (budgeted)."
        )

    def _build_action_metadata(
        self,
        action: np.ndarray,
        increments: pd.Series,
        available_budget: float,
        remaining_line_cap: pd.Series,
    ) -> dict[str, float | int | str]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        applied_spend_mw = float(increments.sum())
        active_lines = int((increments > 0.1).sum())

        if action.size == self._budgeted_action_dim():
            spend_fraction = float(np.clip(action[0], 0.0, 1.0))
            requested_spend_mw = spend_fraction * min(float(available_budget), float(remaining_line_cap.sum()))
            mode_used = "budgeted"
        else:
            clipped = np.clip(action, 0.0, 1.0)
            requested_spend_mw = float((pd.Series(clipped, index=self.candidate_lines) * remaining_line_cap).sum())
            spend_fraction = (
                min(requested_spend_mw / max(float(available_budget), 1e-6), 1.0)
                if available_budget > 0.0
                else 0.0
            )
            mode_used = "direct"

        return {
            "action_mode": mode_used,
            "requested_spend_fraction": spend_fraction,
            "requested_spend_mw": requested_spend_mw,
            "applied_spend_mw": applied_spend_mw,
            "active_lines": active_lines,
        }

    def _state_series(self, step_offset: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Timestamp]:
        row_idx = self.start_index + int(step_offset)
        timestamp = pd.Timestamp(self.dataset.snapshots[row_idx])
        demand_by_bus = self.dataset.demand_by_bus.iloc[row_idx].reindex(self.bus_names).fillna(0.0)
        demand_by_load = self.dataset.demand_by_load.iloc[row_idx].reindex(self.dataset.network.loads.index).fillna(0.0)
        renewable_by_bus = self.dataset.renewable_by_bus.iloc[row_idx].reindex(self.bus_names).fillna(0.0)
        generator_availability = self.dataset.generator_availability.iloc[row_idx].reindex(self.dataset.network.generators.index).fillna(1.0)
        return demand_by_bus, demand_by_load, renewable_by_bus, generator_availability, timestamp

    def _simulate_at_step(self, step_offset: int) -> tuple[pd.Series, pd.Series, SimulationResult, pd.Timestamp]:
        demand_by_bus, demand_by_load, renewable_by_bus, generator_availability, timestamp = self._state_series(step_offset)
        simulation = self._simulate(
            timestamp=timestamp,
            demand_by_bus=demand_by_bus,
            demand_by_load=demand_by_load,
            renewable_by_bus=renewable_by_bus,
            generator_availability=generator_availability,
        )
        return demand_by_bus, renewable_by_bus, simulation, timestamp

    def _build_preview_state(self, step_offset: int) -> ObservationPreview:
        horizon = self._transition_horizon(step_offset)
        if horizon <= 0:
            return self._empty_preview()

        demand_acc = np.zeros(len(self.bus_names), dtype=float)
        renewable_acc = np.zeros(len(self.bus_names), dtype=float)
        raw_flow_acc = np.zeros(len(self.line_names), dtype=float)
        flow_acc = np.zeros(len(self.line_names), dtype=float)
        raw_loading_max = np.zeros(len(self.line_names), dtype=float)
        loading_max = np.zeros(len(self.line_names), dtype=float)

        renewable_potential = 0.0
        renewable_served = 0.0
        renewable_share = 0.0
        renewable_curtailment = 0.0
        grid_stress = 0.0
        constraint_violation = 0.0
        slack_generation = 0.0
        load_shedding = 0.0
        emissions = 0.0
        operating_cost = 0.0
        timestamp: pd.Timestamp | None = None
        backend = "none"

        for local_step in range(horizon):
            demand_by_bus, renewable_by_bus, simulation, current_timestamp = self._simulate_at_step(step_offset + local_step)
            if timestamp is None:
                timestamp = current_timestamp
            demand_acc += demand_by_bus.reindex(self.bus_names).to_numpy(dtype=float)
            renewable_acc += renewable_by_bus.reindex(self.bus_names).to_numpy(dtype=float)
            raw_flow_acc += simulation.raw_line_flows.reindex(self.line_names).fillna(0.0).to_numpy(dtype=float)
            flow_acc += simulation.line_flows.reindex(self.line_names).fillna(0.0).to_numpy(dtype=float)
            raw_loading_max = np.maximum(
                raw_loading_max,
                simulation.raw_line_loading.reindex(self.line_names).fillna(0.0).to_numpy(dtype=float),
            )
            loading_max = np.maximum(
                loading_max,
                simulation.line_loading.reindex(self.line_names).fillna(0.0).to_numpy(dtype=float),
            )
            renewable_potential += float(simulation.renewable_potential)
            renewable_served += float(simulation.renewable_served)
            renewable_share += float(simulation.renewable_share)
            renewable_curtailment += float(simulation.renewable_curtailment)
            grid_stress += float(simulation.grid_stress)
            constraint_violation += float(simulation.constraint_violation)
            slack_generation += float(simulation.slack_generation)
            load_shedding += float(simulation.load_shedding)
            emissions += float(simulation.emissions)
            operating_cost += float(simulation.operating_cost)
            backend = simulation.backend

        inv_horizon = 1.0 / float(horizon)
        preview_result = SimulationResult(
            raw_line_flows=pd.Series(raw_flow_acc * inv_horizon, index=self.line_names, dtype=float),
            raw_line_loading=pd.Series(raw_loading_max, index=self.line_names, dtype=float),
            line_flows=pd.Series(flow_acc * inv_horizon, index=self.line_names, dtype=float),
            line_loading=pd.Series(loading_max, index=self.line_names, dtype=float),
            renewable_potential=renewable_potential * inv_horizon,
            renewable_served=renewable_served * inv_horizon,
            renewable_share=renewable_share * inv_horizon,
            renewable_curtailment=renewable_curtailment * inv_horizon,
            grid_stress=grid_stress * inv_horizon,
            constraint_violation=constraint_violation * inv_horizon,
            slack_generation=slack_generation * inv_horizon,
            load_shedding=load_shedding * inv_horizon,
            emissions=emissions * inv_horizon,
            operating_cost=operating_cost * inv_horizon,
            backend=backend,
        )
        return ObservationPreview(
            timestamp=timestamp if timestamp is not None else pd.Timestamp(0),
            horizon_hours=horizon,
            demand_by_bus=pd.Series(demand_acc * inv_horizon, index=self.bus_names, dtype=float),
            renewable_by_bus=pd.Series(renewable_acc * inv_horizon, index=self.bus_names, dtype=float),
            simulation=preview_result,
        )

    def _build_observation(self) -> np.ndarray:
        demand = self.preview_state.demand_by_bus.reindex(self.bus_names).fillna(0.0) / self.demand_scale
        renewable = self.preview_state.renewable_by_bus.reindex(self.bus_names).fillna(0.0) / self.renewable_scale
        loading = self.preview_state.simulation.raw_line_loading.reindex(self.line_names).fillna(0.0).clip(0.0, 2.5)
        candidate_loading = self.preview_state.simulation.raw_line_loading.reindex(self.candidate_lines).fillna(0.0).clip(0.0, 2.5)
        upgrades = (self.cumulative_upgrades / self.max_upgrade_vector).reindex(self.candidate_lines).fillna(0.0).clip(0.0, 5.0)

        extras = np.array(
            [
                self.preview_state.simulation.emissions / self.total_demand_scale,
                self.preview_state.simulation.renewable_share,
                self.preview_state.simulation.grid_stress / max(self.config.overload_reward_scale, 1e-6),
                self._remaining_budget() / max(self.config.total_upgrade_budget_mw, 1.0),
                self._available_budget() / max(self.config.total_upgrade_budget_mw, 1.0),
                float(self._is_decision_step()),
                self.current_step / max(self.config.episode_length - 1, 1),
            ],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                demand.to_numpy(dtype=np.float32),
                renewable.to_numpy(dtype=np.float32),
                loading.to_numpy(dtype=np.float32),
                candidate_loading.to_numpy(dtype=np.float32),
                upgrades.to_numpy(dtype=np.float32),
                extras,
            ]
        )
        return obs.astype(np.float32)

    def _simulate_transition_window(self, horizon: int) -> tuple[SimulationResult, float]:
        raw_flow_acc = np.zeros(len(self.line_names), dtype=float)
        flow_acc = np.zeros(len(self.line_names), dtype=float)
        raw_loading_max = np.zeros(len(self.line_names), dtype=float)
        loading_max = np.zeros(len(self.line_names), dtype=float)

        renewable_potential_total = 0.0
        renewable_served_total = 0.0
        renewable_share_sum = 0.0
        renewable_curtailment_total = 0.0
        grid_stress_total = 0.0
        constraint_violation_total = 0.0
        slack_generation_total = 0.0
        load_shedding_total = 0.0
        emissions_total = 0.0
        operating_cost_total = 0.0
        backend = "none"

        for local_step in range(horizon):
            _, _, simulation, _ = self._simulate_at_step(self.current_step + local_step)
            raw_flow_acc += simulation.raw_line_flows.reindex(self.line_names).fillna(0.0).to_numpy(dtype=float)
            flow_acc += simulation.line_flows.reindex(self.line_names).fillna(0.0).to_numpy(dtype=float)
            raw_loading_max = np.maximum(
                raw_loading_max,
                simulation.raw_line_loading.reindex(self.line_names).fillna(0.0).to_numpy(dtype=float),
            )
            loading_max = np.maximum(
                loading_max,
                simulation.line_loading.reindex(self.line_names).fillna(0.0).to_numpy(dtype=float),
            )
            renewable_potential_total += float(simulation.renewable_potential)
            renewable_served_total += float(simulation.renewable_served)
            renewable_share_sum += float(simulation.renewable_share)
            renewable_curtailment_total += float(simulation.renewable_curtailment)
            grid_stress_total += float(simulation.grid_stress)
            constraint_violation_total += float(simulation.constraint_violation)
            slack_generation_total += float(simulation.slack_generation)
            load_shedding_total += float(simulation.load_shedding)
            emissions_total += float(simulation.emissions)
            operating_cost_total += float(simulation.operating_cost)
            backend = simulation.backend

        inv_horizon = 1.0 / float(max(horizon, 1))
        result = SimulationResult(
            raw_line_flows=pd.Series(raw_flow_acc * inv_horizon, index=self.line_names, dtype=float),
            raw_line_loading=pd.Series(raw_loading_max, index=self.line_names, dtype=float),
            line_flows=pd.Series(flow_acc * inv_horizon, index=self.line_names, dtype=float),
            line_loading=pd.Series(loading_max, index=self.line_names, dtype=float),
            renewable_potential=renewable_potential_total,
            renewable_served=renewable_served_total,
            renewable_share=renewable_share_sum * inv_horizon,
            renewable_curtailment=renewable_curtailment_total,
            grid_stress=grid_stress_total,
            constraint_violation=constraint_violation_total,
            slack_generation=slack_generation_total,
            load_shedding=load_shedding_total,
            emissions=emissions_total,
            operating_cost=operating_cost_total,
            backend=backend,
        )
        return result, renewable_share_sum

    def _third_objective_reward(
        self,
        simulation: SimulationResult,
        renewable_share_reward: float,
    ) -> float:
        mode = str(self.config.third_objective_mode).strip().lower()
        if mode == "renewable_share":
            return float(renewable_share_reward)
        if mode == "curtailment":
            return -float(simulation.renewable_curtailment) / max(float(self.config.curtailment_reward_scale), 1e-6)
        if mode == "emissions":
            return -float(simulation.emissions) / max(float(self.config.emissions_reward_scale), 1e-6)
        raise ValueError(f"Unsupported third objective mode: {self.config.third_objective_mode!r}")

    def get_feature_names(self) -> list[str]:
        return list(self.feature_names)

    def get_action_names(self) -> list[str]:
        return list(self.action_labels)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        max_start = max(len(self.dataset.snapshots) - self.config.episode_length, 0)
        if options and "start_index" in options:
            self.start_index = int(np.clip(options["start_index"], 0, max_start))
        elif self.config.random_start and max_start > 0:
            self.start_index = int(self.rng.integers(0, max_start + 1))
        else:
            self.start_index = 0

        self.current_step = 0
        self.current_line_capacities = self.base_line_capacities.copy()
        self.cumulative_upgrades = pd.Series(0.0, index=self.candidate_lines)
        self.history = []
        self.last_result = self._empty_result()
        self.preview_state = self._build_preview_state(self.current_step)

        info = {
            "start_index": self.start_index,
            "timestamp": str(self.preview_state.timestamp),
            "preview_horizon_hours": self.preview_state.horizon_hours,
        }
        return self._build_observation(), info

    def step(self, action: np.ndarray):
        available_budget_before = self._available_budget()
        remaining_line_cap_before = self._remaining_line_upgrade_cap()
        increments = self._decode_action(
            action,
            available_budget=available_budget_before,
            remaining_line_cap=remaining_line_cap_before,
        )
        action_metadata = self._build_action_metadata(
            action,
            increments,
            available_budget=available_budget_before,
            remaining_line_cap=remaining_line_cap_before,
        )
        self.cumulative_upgrades = self.cumulative_upgrades + increments
        self.current_line_capacities.loc[self.candidate_lines] = (
            self.base_line_capacities.reindex(self.candidate_lines) + self.cumulative_upgrades
        )

        horizon = self._transition_horizon(self.current_step)
        if horizon <= 0:
            raise RuntimeError("Environment stepped after termination.")

        block_timestamp = pd.Timestamp(self.dataset.snapshots[self._step_index()])
        annualized_investment_cost = self._annualized_investment_cost(increments)
        step_investment_cost = self._episode_investment_cost(increments)
        simulation, renewable_share_reward = self._simulate_transition_window(horizon)
        self.last_result = simulation
        third_objective_reward = self._third_objective_reward(simulation, renewable_share_reward)

        total_cost = step_investment_cost + simulation.operating_cost
        reward_vector = np.array(
            [
                -total_cost / max(self.config.cost_reward_scale, 1e-6),
                -simulation.grid_stress / max(self.config.overload_reward_scale, 1e-6),
                third_objective_reward,
            ],
            dtype=np.float32,
        )

        info = {
            "timestamp": str(block_timestamp),
            "n_hours": horizon,
            "preview_horizon_hours": self.preview_state.horizon_hours,
            "investment_cost": step_investment_cost,
            "annualized_investment_cost": annualized_investment_cost,
            "investment_cost_time_basis_hours": self.config.episode_length,
            "operating_cost": simulation.operating_cost,
            "total_cost": total_cost,
            "released_budget": self._released_budget(),
            "available_budget": self._available_budget(),
            "remaining_budget": self._remaining_budget(),
            "decision_step": self._is_decision_step(),
            "grid_stress": simulation.grid_stress,
            "constraint_violation": simulation.constraint_violation,
            "renewable_share": simulation.renewable_share,
            "renewable_served": simulation.renewable_served,
            "renewable_potential": simulation.renewable_potential,
            "renewable_curtailment": simulation.renewable_curtailment,
            "emissions": simulation.emissions,
            "slack_generation": simulation.slack_generation,
            "load_shedding": simulation.load_shedding,
            "reward_vector": reward_vector.tolist(),
            "third_objective_mode": self.config.third_objective_mode,
            "third_objective_reward": third_objective_reward,
            "action_mw": increments.tolist(),
            "backend": simulation.backend,
            **action_metadata,
        }
        self.history.append(info)

        self.current_step += horizon
        terminated = (
            self.current_step >= self.config.episode_length
            or self._step_index() >= len(self.dataset.snapshots)
        )
        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            self.preview_state = self._build_preview_state(self.current_step)
            observation = self._build_observation()

        return observation, reward_vector, terminated, False, info

    @abstractmethod
    def _simulate(
        self,
        timestamp: pd.Timestamp,
        demand_by_bus: pd.Series,
        demand_by_load: pd.Series,
        renewable_by_bus: pd.Series,
        generator_availability: pd.Series,
    ) -> SimulationResult:
        raise NotImplementedError


class ProxyTEPEnv(BaseTEPEnv):
    def __init__(self, dataset: TEPDataset, config: EnvironmentConfig):
        super().__init__(dataset=dataset, config=config)
        self.dc_model = DCPowerFlowModel.from_network(dataset.network)

    def _simulate(
        self,
        timestamp: pd.Timestamp,
        demand_by_bus: pd.Series,
        demand_by_load: pd.Series,
        renewable_by_bus: pd.Series,
        generator_availability: pd.Series,
    ) -> SimulationResult:
        del timestamp, demand_by_load, generator_availability
        return run_dc_proxy(
            model=self.dc_model,
            demand_by_bus=demand_by_bus,
            renewable_by_bus=renewable_by_bus,
            line_capacities=self.current_line_capacities,
            slack_cost=self.config.slack_marginal_cost,
            slack_emission_factor=self.config.slack_emission_factor,
            stability_margin=self.config.stability_margin,
            dispatch_loading_limit=self.config.proxy_dispatch_limit,
            balance_mode=self.config.proxy_balance_mode,
        )


class PyPSATEPEnv(BaseTEPEnv):
    def __init__(self, dataset: TEPDataset, config: EnvironmentConfig):
        super().__init__(dataset=dataset, config=config)
        self.dc_model = DCPowerFlowModel.from_network(dataset.network)

    def _simulate(
        self,
        timestamp: pd.Timestamp,
        demand_by_bus: pd.Series,
        demand_by_load: pd.Series,
        renewable_by_bus: pd.Series,
        generator_availability: pd.Series,
    ) -> SimulationResult:
        try:
            return run_pypsa_lopf(
                base_network=self.dataset.network,
                snapshot=timestamp,
                demand_by_load=demand_by_load,
                demand_by_bus=demand_by_bus,
                renewable_by_bus=renewable_by_bus,
                generator_availability=generator_availability,
                line_capacities=self.current_line_capacities,
                renewable_carriers=self.dataset.renewable_carriers,
                marginal_costs=self.config.marginal_costs,
                emission_factors=self.config.emission_factors,
                solver_name=self.config.solver_name,
                stability_margin=self.config.stability_margin,
                load_shedding_cost=self.config.load_shedding_cost,
            )
        except Exception:
            if not self.config.full_env_fallback_to_proxy:
                raise
            return run_dc_proxy(
                model=self.dc_model,
                demand_by_bus=demand_by_bus,
                renewable_by_bus=renewable_by_bus,
                line_capacities=self.current_line_capacities,
                slack_cost=self.config.slack_marginal_cost,
                slack_emission_factor=self.config.slack_emission_factor,
                stability_margin=self.config.stability_margin,
                dispatch_loading_limit=self.config.proxy_dispatch_limit,
                balance_mode=self.config.proxy_balance_mode,
            )
