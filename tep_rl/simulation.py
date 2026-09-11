from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pypsa


@dataclass
class SimulationResult:
    raw_line_flows: pd.Series
    raw_line_loading: pd.Series
    line_flows: pd.Series
    line_loading: pd.Series
    renewable_potential: float
    renewable_served: float
    renewable_share: float
    renewable_curtailment: float
    grid_stress: float
    constraint_violation: float
    slack_generation: float
    load_shedding: float
    emissions: float
    operating_cost: float
    backend: str


@dataclass
class DCPowerFlowModel:
    buses: pd.Index
    lines: pd.Index
    incidence: np.ndarray
    susceptance: np.ndarray
    line_bus0: np.ndarray
    line_bus1: np.ndarray
    slack_bus: str
    bus_lookup: dict[str, int]
    flow_matrix: np.ndarray
    component_bus_indices: tuple[np.ndarray, ...]
    component_slack_indices: tuple[int, ...]

    @classmethod
    def from_network(cls, network: pypsa.Network, slack_bus: str | None = None) -> "DCPowerFlowModel":
        working = network.copy()
        working.calculate_dependent_values()
        buses = pd.Index(working.buses.index)
        lines = pd.Index(working.lines.index)
        bus_lookup = {bus: idx for idx, bus in enumerate(buses)}

        if slack_bus is None:
            slack_bus = str(buses[0])

        n_lines = len(lines)
        n_transformers = len(working.transformers)
        n_branches = n_lines + n_transformers
        branch_incidence = np.zeros((n_branches, len(buses)), dtype=float)
        line_bus0 = np.zeros(len(lines), dtype=int)
        line_bus1 = np.zeros(len(lines), dtype=int)

        for line_idx, (_, row) in enumerate(working.lines.iterrows()):
            i = bus_lookup[str(row.bus0)]
            j = bus_lookup[str(row.bus1)]
            branch_incidence[line_idx, i] = 1.0
            branch_incidence[line_idx, j] = -1.0
            line_bus0[line_idx] = i
            line_bus1[line_idx] = j

        transformer_offset = n_lines
        for transformer_idx, (_, row) in enumerate(working.transformers.iterrows()):
            branch_idx = transformer_offset + transformer_idx
            i = bus_lookup[str(row.bus0)]
            j = bus_lookup[str(row.bus1)]
            branch_incidence[branch_idx, i] = 1.0
            branch_incidence[branch_idx, j] = -1.0

        line_x = working.lines["x_pu_eff"].replace(0.0, np.nan).to_numpy(dtype=float)
        transformer_x = working.transformers["x_pu_eff"].replace(0.0, np.nan).to_numpy(dtype=float)
        branch_x = np.concatenate([line_x, transformer_x])
        if not np.isfinite(branch_x).all():
            raise ValueError("All lines and transformers require finite non-zero reactance for the DC proxy.")
        branch_susceptance = 1.0 / np.maximum(np.abs(branch_x), 1e-8)

        if n_transformers and "phase_shift" in working.transformers:
            phase_shift = working.transformers["phase_shift"].fillna(0.0).to_numpy(dtype=float)
            if not np.allclose(phase_shift, 0.0, atol=1e-10):
                raise ValueError("The DC proxy currently requires zero transformer phase shifts.")

        adjacency: list[set[int]] = [set() for _ in buses]
        for branch in branch_incidence:
            endpoints = np.flatnonzero(branch)
            if len(endpoints) != 2:
                continue
            i, j = int(endpoints[0]), int(endpoints[1])
            adjacency[i].add(j)
            adjacency[j].add(i)

        components: list[np.ndarray] = []
        unseen = set(range(len(buses)))
        while unseen:
            root = min(unseen)
            stack = [root]
            members: list[int] = []
            unseen.remove(root)
            while stack:
                node = stack.pop()
                members.append(node)
                neighbours = adjacency[node].intersection(unseen)
                unseen.difference_update(neighbours)
                stack.extend(neighbours)
            components.append(np.asarray(sorted(members), dtype=int))
        components.sort(key=len, reverse=True)

        requested_slack = bus_lookup[str(slack_bus)]
        component_slacks: list[int] = []
        branch_flow_matrix = np.zeros((n_branches, len(buses)), dtype=float)
        laplacian = branch_incidence.T @ np.diag(branch_susceptance) @ branch_incidence
        weighted_incidence = np.diag(branch_susceptance) @ branch_incidence
        for component in components:
            component_set = set(component.tolist())
            component_slack = requested_slack if requested_slack in component_set else int(component[0])
            component_slacks.append(component_slack)
            reduced = component[component != component_slack]
            if len(reduced) == 0:
                continue
            reduced_laplacian = laplacian[np.ix_(reduced, reduced)]
            try:
                inverse = np.linalg.inv(reduced_laplacian)
            except np.linalg.LinAlgError as exc:
                names = [str(buses[index]) for index in component]
                raise ValueError(f"Singular DC component after transformer inclusion: {names}") from exc
            branch_flow_matrix[:, reduced] = weighted_incidence[:, reduced] @ inverse

        incidence = branch_incidence[:n_lines]
        susceptance = branch_susceptance[:n_lines]
        flow_matrix = branch_flow_matrix[:n_lines]

        return cls(
            buses=buses,
            lines=lines,
            incidence=incidence,
            susceptance=susceptance,
            line_bus0=line_bus0,
            line_bus1=line_bus1,
            slack_bus=slack_bus,
            bus_lookup=bus_lookup,
            flow_matrix=flow_matrix,
            component_bus_indices=tuple(components),
            component_slack_indices=tuple(component_slacks),
        )

    def solve(self, injections: pd.Series) -> pd.Series:
        power = injections.reindex(self.buses).fillna(0.0).to_numpy(dtype=float)
        for component in self.component_bus_indices:
            mismatch = float(power[component].sum())
            tolerance = 1e-7 * max(float(np.abs(power[component]).sum()), 1.0)
            if abs(mismatch) > tolerance:
                names = [str(self.buses[index]) for index in component]
                raise ValueError(
                    f"DC injections are not balanced within component {names}: mismatch={mismatch:.6g}."
                )
        flows = self.flow_matrix @ power
        return pd.Series(flows, index=self.lines)

    def balance_dispatch(
        self,
        renewable_dispatch: pd.Series,
        demand: pd.Series,
        balance_mode: str = "demand_proportional",
    ) -> tuple[pd.Series, pd.Series, float]:
        """Balance generation and demand independently in every connected component."""
        dispatch = renewable_dispatch.reindex(self.buses).fillna(0.0).clip(lower=0.0).copy()
        demand = demand.reindex(self.buses).fillna(0.0).clip(lower=0.0)
        balancing_supply = pd.Series(0.0, index=self.buses, dtype=float)
        slack_generation = 0.0

        for component, slack_idx in zip(self.component_bus_indices, self.component_slack_indices):
            component_buses = self.buses[component]
            component_demand = float(demand.reindex(component_buses).sum())
            component_renewable = float(dispatch.reindex(component_buses).sum())
            if component_renewable > component_demand and component_renewable > 0.0:
                dispatch.loc[component_buses] *= component_demand / component_renewable
                component_renewable = component_demand

            deficit = max(component_demand - component_renewable, 0.0)
            slack_generation += deficit
            slack_name = str(self.buses[slack_idx])
            if deficit > 0.0 and str(balance_mode).strip().lower() == "demand_proportional":
                weights = demand.reindex(component_buses)
                if float(weights.sum()) > 0.0:
                    balancing_supply.loc[component_buses] = weights * (deficit / float(weights.sum()))
                else:
                    balancing_supply.loc[slack_name] = deficit
            else:
                balancing_supply.loc[slack_name] = deficit

        injections = dispatch + balancing_supply - demand
        for component, slack_idx in zip(self.component_bus_indices, self.component_slack_indices):
            component_buses = self.buses[component]
            residual = float(injections.reindex(component_buses).sum())
            injections.loc[str(self.buses[slack_idx])] -= residual
        return dispatch, injections, slack_generation


def _balanced_injection(
    model: DCPowerFlowModel,
    renewable_dispatch: pd.Series,
    demand: pd.Series,
    balance_mode: str = "demand_proportional",
) -> tuple[pd.Series, pd.Series, float]:
    return model.balance_dispatch(renewable_dispatch, demand, balance_mode=balance_mode)


def run_dc_proxy(
    model: DCPowerFlowModel,
    demand_by_bus: pd.Series,
    renewable_by_bus: pd.Series,
    line_capacities: pd.Series,
    slack_cost: float,
    slack_emission_factor: float,
    stability_margin: float,
    dispatch_loading_limit: float | None = None,
    balance_mode: str = "demand_proportional",
) -> SimulationResult:
    demand = demand_by_bus.reindex(model.buses).fillna(0.0).clip(lower=0.0)
    renewable_potential = renewable_by_bus.reindex(model.buses).fillna(0.0).clip(lower=0.0)
    total_demand = float(demand.sum())
    total_renewable = float(renewable_potential.sum())

    if total_renewable > total_demand and total_renewable > 0.0:
        base_dispatch = renewable_potential * (total_demand / total_renewable)
    else:
        base_dispatch = renewable_potential.copy()

    capacities = line_capacities.reindex(model.lines).fillna(1.0).clip(lower=1.0)
    effective_dispatch_limit = stability_margin if dispatch_loading_limit is None else float(dispatch_loading_limit)
    effective_dispatch_limit = float(np.clip(effective_dispatch_limit, 1e-3, 1.0))

    base_dispatch, raw_injections, raw_slack = _balanced_injection(
        model,
        base_dispatch,
        demand,
        balance_mode=balance_mode,
    )
    raw_flows = model.solve(raw_injections)
    raw_loading = raw_flows.abs() / capacities

    # Compute grid stress from the uncorrected load flow. The corrected flow can
    # hide overloads after curtailment, while raw loading exposes the candidate
    # lines that the agent should learn to reinforce.
    grid_stress = float(np.maximum(raw_loading.to_numpy() - stability_margin, 0.0).sum())
    constraint_violation_raw = float(np.maximum(raw_loading.to_numpy() - 1.0, 0.0).sum())

    lower, upper = 0.0, 1.0
    final_dispatch = base_dispatch.copy()
    final_flows = raw_flows.copy()
    final_loading = raw_loading.copy()
    final_slack = raw_slack

    if raw_loading.gt(effective_dispatch_limit + 1e-4).any():
        for _ in range(22):
            alpha = 0.5 * (lower + upper)
            trial_dispatch = base_dispatch * alpha
            trial_dispatch, trial_injections, trial_slack = _balanced_injection(
                model,
                trial_dispatch,
                demand,
                balance_mode=balance_mode,
            )
            trial_flows = model.solve(trial_injections)
            trial_loading = trial_flows.abs() / capacities

            if trial_loading.le(effective_dispatch_limit + 1e-4).all():
                lower = alpha
                final_dispatch = trial_dispatch
                final_flows = trial_flows
                final_loading = trial_loading
                final_slack = trial_slack
            else:
                upper = alpha

    renewable_served = float(final_dispatch.sum())
    renewable_share = renewable_served / max(total_demand, 1e-6)
    renewable_curtailment = max(total_renewable - renewable_served, 0.0)
    #grid_stress = float(np.maximum(final_loading.to_numpy() - stability_margin, 0.0).sum())
    #constraint_violation = float(np.maximum(final_loading.to_numpy() - 1.0, 0.0).sum())
    operating_cost = float(final_slack * slack_cost)
    emissions = float(final_slack * slack_emission_factor)

    return SimulationResult(
        raw_line_flows=raw_flows,
        raw_line_loading=raw_loading,
        line_flows=final_flows,
        line_loading=final_loading,
        renewable_potential=total_renewable,
        renewable_served=renewable_served,
        renewable_share=renewable_share,
        renewable_curtailment=renewable_curtailment,
        grid_stress=grid_stress,
        constraint_violation=constraint_violation_raw,
        slack_generation=float(final_slack),
        load_shedding=0.0,
        emissions=emissions,
        operating_cost=operating_cost,
        backend="proxy-dc",
    )


def _ensure_backstop_generator(
    network: pypsa.Network,
    slack_bus: str,
    marginal_costs: Mapping[str, float],
) -> list[str]:
    """Add a distributed high-cost adequacy reserve at load buses.

    A single unlimited slack generator creates an artificial bottleneck at its
    connection corridor.  Distributing the same high-cost reserve over load
    buses is consistent with the proxy model's demand-proportional balancing:
    reinforcement is then rewarded for transporting low-cost renewable power,
    not for reaching an arbitrary numerical slack bus.
    """
    existing_slack = network.generators.index[
        network.generators.get("carrier", pd.Series(index=network.generators.index, dtype=str))
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("slack")
    ]
    if len(existing_slack):
        network.generators.loc[existing_slack, "p_nom"] = 0.0
        if "p_nom_extendable" in network.generators.columns:
            network.generators.loc[existing_slack, "p_nom_extendable"] = False

    if "slack" not in network.carriers.index:
        network.add("Carrier", "slack")
    load_buses = pd.Index(network.loads["bus"].dropna().astype(str).unique())
    if len(load_buses) == 0:
        load_buses = pd.Index([str(slack_bus)])
    names: list[str] = []
    for sequence, bus in enumerate(load_buses, start=1):
        candidate = f"RL_Backstop::{sequence:03d}"
        if candidate not in network.generators.index:
            network.add(
                "Generator",
                candidate,
                bus=str(bus),
                p_nom=1_000_000.0,
                marginal_cost=float(marginal_costs.get("slack", 220.0)),
                carrier="slack",
            )
        names.append(candidate)
    return names


def _ensure_load_shedding_generators(
    network: pypsa.Network,
    load_shedding_cost: float,
) -> list[str]:
    names: list[str] = []
    if "load_shedding" not in network.carriers.index:
        network.add("Carrier", "load_shedding")
    for bus in network.buses.index:
        candidate = f"LL::{bus}"
        if candidate not in network.generators.index:
            network.add(
                "Generator",
                candidate,
                bus=bus,
                p_nom=1_000_000.0,
                marginal_cost=float(load_shedding_cost),
                carrier="load_shedding",
            )
        names.append(candidate)
    return names


def _solve_network(
    network: pypsa.Network,
    solver_name: str,
) -> tuple[str, str, str]:
    solver_options = None
    if solver_name.lower() == "highs":
        solver_options = {"log_to_console": False}

    optimize = getattr(network, "optimize", None)
    if callable(optimize):
        optimize_kwargs = {
            "solver_name": solver_name,
            "solver_options": solver_options,
            "include_objective_constant": False,
        }
        try:
            status, condition = optimize(snapshots=network.snapshots, **optimize_kwargs)
        except TypeError:
            status, condition = optimize(
                snapshots=network.snapshots,
                solver_name=solver_name,
                solver_options=solver_options,
            )
        return str(status), str(condition), "pypsa-optimize"

    if solver_options is None:
        status = network.lopf(network.snapshots, solver_name=solver_name, pyomo=False)
    else:
        status = network.lopf(
            network.snapshots,
            solver_name=solver_name,
            solver_options=solver_options,
            pyomo=False,
        )
    if isinstance(status, tuple) and len(status) == 2:
        return str(status[0]), str(status[1]), "pypsa-lopf"
    return str(status), "", "pypsa-lopf"


def run_pypsa_lopf(
    base_network: pypsa.Network,
    snapshot: pd.Timestamp,
    demand_by_load: pd.Series,
    demand_by_bus: pd.Series,
    renewable_by_bus: pd.Series,
    generator_availability: pd.Series,
    line_capacities: pd.Series,
    renewable_carriers: Sequence[str],
    marginal_costs: Mapping[str, float],
    emission_factors: Mapping[str, float],
    solver_name: str,
    stability_margin: float,
    load_shedding_cost: float,
) -> SimulationResult:
    network = base_network.copy()
    network.set_snapshots(pd.DatetimeIndex([snapshot]))

    slack_bus = str(network.buses.index[0])
    _ensure_backstop_generator(network, slack_bus=slack_bus, marginal_costs=marginal_costs)
    load_shedding_generators = _ensure_load_shedding_generators(
        network,
        load_shedding_cost=load_shedding_cost,
    )

    line_capacities = line_capacities.reindex(network.lines.index).fillna(network.lines["s_nom"]).clip(lower=1.0)
    network.lines.loc[:, "s_nom"] = line_capacities

    demand_frame = pd.DataFrame(
        [demand_by_load.reindex(network.loads.index).fillna(0.0).to_numpy(dtype=float)],
        index=network.snapshots,
        columns=network.loads.index,
    )
    network.loads_t.p_set = demand_frame

    availability = generator_availability.reindex(network.generators.index).fillna(1.0).clip(lower=0.0, upper=1.0)
    availability_frame = pd.DataFrame(
        [availability.to_numpy(dtype=float)],
        index=network.snapshots,
        columns=network.generators.index,
    )
    network.generators_t.p_max_pu = availability_frame

    for generator, row in network.generators.iterrows():
        carrier = str(row.carrier)
        if carrier in marginal_costs:
            network.generators.at[generator, "marginal_cost"] = float(marginal_costs[carrier])
        elif carrier.lower() in marginal_costs:
            network.generators.at[generator, "marginal_cost"] = float(marginal_costs[carrier.lower()])

    status, condition, backend = _solve_network(network, solver_name=solver_name)
    if status != "ok" or condition != "optimal":
        raise RuntimeError(f"PyPSA optimisation failed with status {(status, condition)}")

    flows = network.lines_t.p0.loc[snapshot].reindex(network.lines.index).fillna(0.0)
    loading = flows.abs() / line_capacities
    dispatch = network.generators_t.p.loc[snapshot].reindex(network.generators.index).fillna(0.0)
    renewable_set = {carrier.lower() for carrier in renewable_carriers}

    renewable_served = 0.0
    emissions = 0.0
    for generator, output in dispatch.items():
        carrier = str(network.generators.at[generator, "carrier"])
        carrier_lower = carrier.lower()
        if carrier_lower in renewable_set or any(tag in carrier_lower for tag in ("wind", "solar", "ror")):
            renewable_served += float(max(output, 0.0))
        factor = emission_factors.get(carrier, emission_factors.get(carrier_lower, 0.0))
        emissions += float(max(output, 0.0) * factor)

    renewable_potential = float(renewable_by_bus.sum())
    total_demand = float(demand_by_bus.sum())
    renewable_share = renewable_served / max(total_demand, 1e-6)
    renewable_curtailment = max(renewable_potential - renewable_served, 0.0)
    grid_stress = float(np.maximum(loading.to_numpy() - stability_margin, 0.0).sum())
    constraint_violation = float(np.maximum(loading.to_numpy() - 1.0, 0.0).sum())
    operating_cost = float(network.objective)
    slack_generation = float(dispatch[network.generators.carrier == "slack"].clip(lower=0.0).sum())
    load_shedding = float(dispatch.reindex(load_shedding_generators).fillna(0.0).clip(lower=0.0).sum())

    return SimulationResult(
        raw_line_flows=flows,
        raw_line_loading=loading,
        line_flows=flows,
        line_loading=loading,
        renewable_potential=renewable_potential,
        renewable_served=renewable_served,
        renewable_share=renewable_share,
        renewable_curtailment=renewable_curtailment,
        grid_stress=grid_stress,
        constraint_violation=constraint_violation,
        slack_generation=slack_generation,
        load_shedding=load_shedding,
        emissions=emissions,
        operating_cost=operating_cost,
        backend=backend,
    )
