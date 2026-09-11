"""Solve a transparent multi-snapshot linear DC reinforcement benchmark.

The benchmark is not an official Austrian expansion plan.  It provides a
mathematical lower-bound reference on the same synthetic isolated-Austria case,
candidate corridors, per-line limits, and total MW budget as the RL agents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tep_rl.config import EnvironmentConfig, NetworkConfig
from tep_rl.data import load_austria_case
from tep_rl.evaluation import stratified_episode_start_indices
from tep_rl.future_scenarios import apply_future_scenario_from_manifest, scenario_audit
from tep_rl.simulation import (
    DCPowerFlowModel,
    _ensure_backstop_generator,
    _ensure_load_shedding_generators,
)


def parse_args() -> argparse.Namespace:
    defaults = EnvironmentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="derived/austria_net_physical_ratings.nc")
    parser.add_argument("--load", default="data/entsoe_at_load_2015_2024_opsd.csv")
    parser.add_argument("--wind", default="data/wind_at_2015_2024.csv")
    parser.add_argument("--solar", default="data/solar_at_2015_2024.csv")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--preprocessing-manifest", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Episode length; defaults to the preprocessing manifest or 24.",
    )
    parser.add_argument(
        "--budget-mw",
        type=float,
        default=None,
        help="Total reinforcement budget; defaults to the preprocessing manifest or 500 MW.",
    )
    parser.add_argument(
        "--max-upgrade-mw",
        type=float,
        default=None,
        help="Per-corridor cap; defaults to the preprocessing manifest or 500 MW.",
    )
    parser.add_argument(
        "--line-investment-cost-eur-per-mw-km-year",
        type=float,
        default=None,
        help="Annualised corridor cost in EUR/(MW km year); defaults to the preprocessing manifest.",
    )
    parser.add_argument("--load-shedding-cost", type=float, default=None)
    parser.add_argument("--solver", default="highs")
    parser.add_argument(
        "--annual-hours",
        type=float,
        default=8760.0,
        help="Total objective/energy weight represented by the selected operating windows; use 0 to keep unit weights.",
    )
    parser.add_argument(
        "--binding-tolerance-pu",
        type=float,
        default=1e-6,
        help="A line is reported as binding when loading is within this tolerance of s_max_pu.",
    )
    parser.add_argument("--output-dir", default="results/dc_tep_baseline")
    return parser.parse_args()


def resolve_manifest_defaults(args: argparse.Namespace, manifest: dict[str, object]) -> argparse.Namespace:
    """Use the frozen RL formulation unless the caller explicitly overrides it."""
    defaults = EnvironmentConfig()
    environment = manifest.get("environment", {})
    if not isinstance(environment, dict):
        environment = {}
    fallbacks = {
        "episode_length": 24,
        "budget_mw": 500.0,
        "max_upgrade_mw": 500.0,
        "line_investment_cost_eur_per_mw_km_year": defaults.line_investment_cost_eur_per_mw_km_year,
        "load_shedding_cost": defaults.load_shedding_cost,
    }
    for name, fallback in fallbacks.items():
        if getattr(args, name) is None:
            setattr(args, name, environment.get(name, fallback))
    args.episode_length = int(args.episode_length)
    for name in fallbacks:
        if name != "episode_length":
            setattr(args, name, float(getattr(args, name)))
    return args


def _dynamic_frame(container, attribute: str, snapshots: pd.Index, columns: pd.Index) -> pd.DataFrame:
    frame = getattr(container, attribute, None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(0.0, index=snapshots, columns=columns)
    return frame.reindex(index=snapshots, columns=columns).fillna(0.0).astype(float)


def _window_lookup(
    dataset_snapshots: pd.DatetimeIndex,
    starts: list[int],
    episode_length: int,
) -> tuple[pd.Series, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    lookup: dict[pd.Timestamp, int] = {}
    for window_id, start in enumerate(starts, start=1):
        end = min(start + episode_length, len(dataset_snapshots))
        window_snapshots = pd.DatetimeIndex(dataset_snapshots[start:end])
        if len(window_snapshots) != episode_length:
            raise ValueError(
                f"Window {window_id} contains {len(window_snapshots)} snapshots; expected {episode_length}."
            )
        for snapshot in window_snapshots:
            if snapshot in lookup:
                raise ValueError(f"Diagnostic windows overlap at {snapshot}.")
            lookup[pd.Timestamp(snapshot)] = window_id
        rows.append(
            {
                "window_id": window_id,
                "start_timestamp": str(window_snapshots[0]),
                "end_timestamp": str(window_snapshots[-1]),
                "n_snapshots": int(len(window_snapshots)),
            }
        )
    return pd.Series(lookup, name="window_id", dtype=int), rows


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.preprocessing_manifest).read_text(encoding="utf-8"))
    args = resolve_manifest_defaults(args, manifest)
    candidate_lines = [str(value) for value in manifest["candidate_lines"]]
    dataset = load_austria_case(
        NetworkConfig(
            network_path=Path(args.network),
            load_path=Path(args.load),
            wind_path=Path(args.wind),
            solar_path=Path(args.solar),
            start=args.start,
            end=args.end,
            candidate_line_limit=None,
        )
    )
    dataset = apply_future_scenario_from_manifest(dataset, manifest)
    missing = set(candidate_lines).difference(dataset.network.lines.index)
    if missing:
        raise ValueError(f"Unknown candidate lines in preprocessing manifest: {sorted(missing)}")

    starts = stratified_episode_start_indices(dataset.snapshots, args.episode_length, args.episodes)
    positions = sorted(
        {
            position
            for start in starts
            for position in range(start, min(start + args.episode_length, len(dataset.snapshots)))
        }
    )
    snapshots = pd.DatetimeIndex(dataset.snapshots[positions])
    network = dataset.network.copy()
    network.set_snapshots(snapshots)
    if float(args.annual_hours) > 0.0:
        representative_weight = float(args.annual_hours) / float(len(snapshots))
        for column in network.snapshot_weightings.columns:
            network.snapshot_weightings.loc[:, column] = representative_weight
    else:
        representative_weight = 1.0
    network.loads_t.p_set = dataset.demand_by_load.loc[snapshots].reindex(columns=network.loads.index).fillna(0.0)
    network.generators_t.p_max_pu = (
        dataset.generator_availability.loc[snapshots]
        .reindex(columns=network.generators.index)
        .fillna(1.0)
        .clip(lower=0.0, upper=1.0)
    )

    defaults = EnvironmentConfig()
    slack_bus = str(network.buses.index[0])
    backstop_generators = _ensure_backstop_generator(network, slack_bus, defaults.marginal_costs)
    shedding = _ensure_load_shedding_generators(network, args.load_shedding_cost)
    for generator, row in network.generators.iterrows():
        carrier = str(row.carrier).lower()
        if carrier in defaults.marginal_costs:
            network.generators.at[generator, "marginal_cost"] = float(defaults.marginal_costs[carrier])

    base_capacity = network.lines["s_nom"].astype(float).copy()
    network.lines.loc[:, "s_nom_extendable"] = False
    network.lines.loc[candidate_lines, "s_nom_extendable"] = True
    network.lines.loc[candidate_lines, "s_nom_min"] = base_capacity.loc[candidate_lines]
    network.lines.loc[candidate_lines, "s_nom_max"] = base_capacity.loc[candidate_lines] + float(args.max_upgrade_mw)

    lengths = network.lines.loc[candidate_lines, "length"].fillna(0.0).clip(lower=0.0)
    if (lengths <= 0.0).any():
        raise ValueError("All candidate corridors require positive route lengths for cost calculation.")
    investment_cost = float(args.line_investment_cost_eur_per_mw_km_year) * lengths
    network.lines.loc[:, "capital_cost"] = 0.0
    network.lines.loc[candidate_lines, "capital_cost"] = investment_cost

    def extra_functionality(n, _snapshots) -> None:
        capacity = n.model.variables["Line-s_nom"]
        base = base_capacity.reindex(candidate_lines)
        n.model.add_constraints(
            (capacity - base).sum() <= float(args.budget_mw),
            name="global_reinforcement_budget_mw",
        )

    dc_model = DCPowerFlowModel.from_network(network)
    status, condition = network.optimize(
        snapshots=snapshots,
        solver_name=args.solver,
        solver_options={"log_to_console": False} if args.solver.lower() == "highs" else None,
        extra_functionality=extra_functionality,
        assign_all_duals=True,
        # PyPSA subtracts the already-built capacity cost only when this is
        # enabled.  The resulting objective therefore contains operating cost
        # plus incremental reinforcement cost, not the historical grid's cost.
        include_objective_constant=True,
    )
    backend = "pypsa-optimize-linear-dc-tep"
    if str(status) != "ok" or str(condition) != "optimal":
        raise RuntimeError(f"DC-TEP baseline failed with status {(status, condition)}")

    upgrades = (network.lines.loc[candidate_lines, "s_nom_opt"] - base_capacity.loc[candidate_lines]).clip(lower=0.0)
    dispatch = network.generators_t.p.reindex(index=snapshots, columns=network.generators.index).fillna(0.0)
    energy_weights = network.snapshot_weightings.get(
        "generators", pd.Series(1.0, index=snapshots)
    ).reindex(snapshots).fillna(1.0).astype(float)
    objective_weights = network.snapshot_weightings.get(
        "objective", pd.Series(1.0, index=snapshots)
    ).reindex(snapshots).fillna(1.0).astype(float)
    weighted_dispatch = dispatch.clip(lower=0.0).mul(energy_weights, axis=0)
    load_shedding_by_snapshot = dispatch.reindex(columns=shedding).clip(lower=0.0).sum(axis=1)
    backstop_by_snapshot = (
        dispatch.reindex(columns=backstop_generators).fillna(0.0).clip(lower=0.0).sum(axis=1)
    )
    load_shedding = float(load_shedding_by_snapshot.mul(energy_weights).sum())
    slack_generation = float(backstop_by_snapshot.mul(energy_weights).sum())
    line_flows = network.lines_t.p0.reindex(columns=network.lines.index).fillna(0.0)
    line_loading = line_flows.abs().divide(network.lines["s_nom_opt"], axis=1)
    investment_total = float((upgrades * investment_cost).sum())

    generator_carriers = network.generators["carrier"].fillna("").astype(str)
    renewable_set = {carrier.lower() for carrier in dataset.renewable_carriers}
    renewable_generators = pd.Index(
        [
            generator
            for generator, carrier in generator_carriers.items()
            if carrier.lower() in renewable_set
            or any(tag in carrier.lower() for tag in ("wind", "solar", "ror"))
        ]
    )
    renewable_availability = (
        network.generators_t.p_max_pu.reindex(index=snapshots, columns=renewable_generators)
        .fillna(1.0)
        .clip(lower=0.0, upper=1.0)
    )
    renewable_capacity = network.generators.loc[renewable_generators, "p_nom"].astype(float)
    renewable_available = renewable_availability.mul(renewable_capacity, axis=1)
    renewable_dispatch = dispatch.reindex(columns=renewable_generators).clip(lower=0.0)
    renewable_curtailment = (renewable_available - renewable_dispatch).clip(lower=0.0)

    marginal_cost = network.generators["marginal_cost"].fillna(0.0).astype(float)
    variable_cost_by_generator = dispatch.mul(objective_weights, axis=0).mul(marginal_cost, axis=1)
    operating_cost = float(variable_cost_by_generator.to_numpy().sum())
    carrier_cost_rows = pd.DataFrame(
        {
            "generator": network.generators.index,
            "carrier": generator_carriers.reindex(network.generators.index).values,
            "generation_mwh": weighted_dispatch.sum(axis=0).reindex(network.generators.index).values,
            "variable_cost": variable_cost_by_generator.sum(axis=0).reindex(network.generators.index).values,
        }
    )
    carrier_costs = (
        carrier_cost_rows.groupby("carrier", dropna=False)[["generation_mwh", "variable_cost"]]
        .sum()
        .reset_index()
        .sort_values("variable_cost", ascending=False)
    )

    allowed_loading = network.lines["s_max_pu"].fillna(1.0).astype(float)
    binding_mask = line_loading.ge(allowed_loading - float(args.binding_tolerance_pu), axis=1)
    mu_upper = _dynamic_frame(network.lines_t, "mu_upper", snapshots, network.lines.index)
    mu_lower = _dynamic_frame(network.lines_t, "mu_lower", snapshots, network.lines.index)
    marginal_prices = _dynamic_frame(network.buses_t, "marginal_price", snapshots, network.buses.index)
    component_price_rows: list[dict[str, object]] = []
    component_price_spreads: dict[int, pd.Series] = {}
    for component_id, component in enumerate(dc_model.component_bus_indices, start=1):
        component_buses = pd.Index([str(dc_model.buses[index]) for index in component])
        component_prices = marginal_prices.reindex(columns=component_buses)
        component_spread = component_prices.max(axis=1) - component_prices.min(axis=1)
        component_price_spreads[component_id] = component_spread
        component_price_rows.append(
            {
                "component_id": component_id,
                "n_buses": int(len(component_buses)),
                "buses": "|".join(component_buses),
                "mean_price": float(component_prices.to_numpy().mean()),
                "min_price": float(component_prices.to_numpy().min()),
                "max_price": float(component_prices.to_numpy().max()),
                "mean_within_component_price_spread": float(component_spread.mean()),
                "max_within_component_price_spread": float(component_spread.max()),
            }
        )
    component_price_summary = pd.DataFrame(component_price_rows)
    largest_component_spread = component_price_spreads[1]
    candidate_set = set(candidate_lines)
    window_lookup, window_rows = _window_lookup(dataset.snapshots, starts, args.episode_length)
    window_ids = window_lookup.reindex(snapshots)
    if window_ids.isna().any():
        raise ValueError("At least one optimised snapshot is not assigned to a diagnostic window.")

    binding_rows: list[dict[str, object]] = []
    for snapshot_position, line_position in np.argwhere(binding_mask.to_numpy(dtype=bool)):
        snapshot = pd.Timestamp(snapshots[snapshot_position])
        line = str(network.lines.index[line_position])
        flow = float(line_flows.iat[snapshot_position, line_position])
        capacity = float(network.lines.at[line, "s_nom_opt"])
        limit_pu = float(allowed_loading.at[line])
        binding_rows.append(
            {
                "window_id": int(window_ids.loc[snapshot]),
                "timestamp": str(snapshot),
                "line": line,
                "is_candidate": line in candidate_set,
                "flow_mw": flow,
                "absolute_flow_mw": abs(flow),
                "capacity_mw": capacity,
                "allowed_loading_pu": limit_pu,
                "loading_pu": float(line_loading.iat[snapshot_position, line_position]),
                "utilisation_of_allowed_limit": float(
                    line_loading.iat[snapshot_position, line_position] / max(limit_pu, 1e-12)
                ),
                "headroom_mw": float(capacity * limit_pu - abs(flow)),
                "mu_upper": float(mu_upper.iat[snapshot_position, line_position]),
                "mu_lower": float(mu_lower.iat[snapshot_position, line_position]),
                "has_nonzero_shadow_price": bool(
                    abs(float(mu_upper.iat[snapshot_position, line_position]))
                    + abs(float(mu_lower.iat[snapshot_position, line_position]))
                    > 1e-6
                ),
            }
        )
    binding_events = pd.DataFrame(
        binding_rows,
        columns=[
            "window_id",
            "timestamp",
            "line",
            "is_candidate",
            "flow_mw",
            "absolute_flow_mw",
            "capacity_mw",
            "allowed_loading_pu",
            "loading_pu",
            "utilisation_of_allowed_limit",
            "headroom_mw",
            "mu_upper",
            "mu_lower",
            "has_nonzero_shadow_price",
        ],
    )

    line_diagnostics = pd.DataFrame(
        {
            "line": network.lines.index.astype(str),
            "is_candidate": [str(line) in candidate_set for line in network.lines.index],
            "base_s_nom_mw": base_capacity.reindex(network.lines.index).values,
            "s_nom_opt_mw": network.lines["s_nom_opt"].reindex(network.lines.index).values,
            "allowed_loading_pu": allowed_loading.reindex(network.lines.index).values,
            "mean_loading_pu": line_loading.mean(axis=0).reindex(network.lines.index).values,
            "p95_loading_pu": line_loading.quantile(0.95, axis=0).reindex(network.lines.index).values,
            "max_loading_pu": line_loading.max(axis=0).reindex(network.lines.index).values,
            "binding_hours": binding_mask.sum(axis=0).reindex(network.lines.index).astype(int).values,
            "absolute_shadow_value_sum": (
                mu_upper.abs().add(mu_lower.abs(), fill_value=0.0).sum(axis=0)
                .reindex(network.lines.index)
                .fillna(0.0)
                .values
            ),
        }
    ).sort_values(["max_loading_pu", "absolute_shadow_value_sum"], ascending=False)

    transformer_flows = _dynamic_frame(
        network.transformers_t, "p0", snapshots, network.transformers.index
    )
    transformer_capacity_column = (
        "s_nom_opt" if "s_nom_opt" in network.transformers.columns else "s_nom"
    )
    transformer_capacity = network.transformers[transformer_capacity_column].astype(float)
    transformer_allowed_loading = network.transformers["s_max_pu"].fillna(1.0).astype(float)
    transformer_loading = transformer_flows.abs().divide(transformer_capacity, axis=1).fillna(0.0)
    transformer_binding_mask = transformer_loading.ge(
        transformer_allowed_loading - float(args.binding_tolerance_pu), axis=1
    )
    transformer_mu_upper = _dynamic_frame(
        network.transformers_t, "mu_upper", snapshots, network.transformers.index
    )
    transformer_mu_lower = _dynamic_frame(
        network.transformers_t, "mu_lower", snapshots, network.transformers.index
    )
    transformer_diagnostics = pd.DataFrame(
        {
            "transformer": network.transformers.index.astype(str),
            "s_nom_mva": transformer_capacity.reindex(network.transformers.index).values,
            "allowed_loading_pu": transformer_allowed_loading.reindex(network.transformers.index).values,
            "mean_loading_pu": transformer_loading.mean(axis=0).reindex(network.transformers.index).values,
            "p95_loading_pu": transformer_loading.quantile(0.95, axis=0)
            .reindex(network.transformers.index)
            .values,
            "max_loading_pu": transformer_loading.max(axis=0).reindex(network.transformers.index).values,
            "binding_hours": transformer_binding_mask.sum(axis=0)
            .reindex(network.transformers.index)
            .astype(int)
            .values,
            "absolute_shadow_value_sum": (
                transformer_mu_upper.abs()
                .add(transformer_mu_lower.abs(), fill_value=0.0)
                .sum(axis=0)
                .reindex(network.transformers.index)
                .fillna(0.0)
                .values
            ),
        }
    ).sort_values(["max_loading_pu", "absolute_shadow_value_sum"], ascending=False)

    demand_by_snapshot = network.loads_t.p_set.reindex(index=snapshots).fillna(0.0).sum(axis=1)
    renewable_available_by_snapshot = renewable_available.sum(axis=1)
    renewable_dispatch_by_snapshot = renewable_dispatch.sum(axis=1)
    renewable_curtailment_by_snapshot = renewable_curtailment.sum(axis=1)
    generation_by_snapshot = dispatch.clip(lower=0.0).sum(axis=1)
    operating_cost_by_snapshot = variable_cost_by_generator.sum(axis=1)
    cross_component_price_range = marginal_prices.max(axis=1) - marginal_prices.min(axis=1)

    for row in window_rows:
        window_id = int(row["window_id"])
        selected = window_ids.index[window_ids.eq(window_id)]
        selected_binding = (
            binding_events[binding_events["window_id"].eq(window_id)]
            if not binding_events.empty
            else binding_events
        )
        selected_loading = line_loading.loc[selected]
        row.update(
            {
                "peak_demand_mw": float(demand_by_snapshot.loc[selected].max()),
                "demand_mwh": float(demand_by_snapshot.loc[selected].mul(energy_weights.loc[selected]).sum()),
                "total_generation_mwh": float(
                    generation_by_snapshot.loc[selected].mul(energy_weights.loc[selected]).sum()
                ),
                "renewable_available_mwh": float(
                    renewable_available_by_snapshot.loc[selected].mul(energy_weights.loc[selected]).sum()
                ),
                "renewable_dispatch_mwh": float(
                    renewable_dispatch_by_snapshot.loc[selected].mul(energy_weights.loc[selected]).sum()
                ),
                "renewable_curtailment_mwh": float(
                    renewable_curtailment_by_snapshot.loc[selected].mul(energy_weights.loc[selected]).sum()
                ),
                "load_shedding_mwh": float(
                    load_shedding_by_snapshot.loc[selected].mul(energy_weights.loc[selected]).sum()
                ),
                "backstop_generation_mwh": float(
                    backstop_by_snapshot.loc[selected].mul(energy_weights.loc[selected]).sum()
                ),
                "operating_cost": float(operating_cost_by_snapshot.loc[selected].sum()),
                "mean_line_loading": float(selected_loading.to_numpy().mean()),
                "p95_line_loading": float(np.quantile(selected_loading.to_numpy(), 0.95)),
                "max_line_loading": float(selected_loading.to_numpy().max()),
                "binding_line_hours": int(len(selected_binding)),
                "unique_binding_lines": int(selected_binding["line"].nunique()) if not selected_binding.empty else 0,
                "binding_candidate_hours": int(selected_binding["is_candidate"].sum())
                if not selected_binding.empty
                else 0,
                "unique_binding_candidate_lines": int(
                    selected_binding.loc[selected_binding["is_candidate"], "line"].nunique()
                )
                if not selected_binding.empty
                else 0,
                "mean_cross_component_price_range": float(
                    cross_component_price_range.loc[selected].mean()
                ),
                "max_cross_component_price_range": float(
                    cross_component_price_range.loc[selected].max()
                ),
                "mean_largest_component_price_spread": float(
                    largest_component_spread.loc[selected].mean()
                ),
                "max_largest_component_price_spread": float(
                    largest_component_spread.loc[selected].max()
                ),
            }
        )
    window_summary = pd.DataFrame(window_rows)

    max_flat_index = int(np.nanargmax(line_loading.to_numpy()))
    max_snapshot_position, max_line_position = np.unravel_index(max_flat_index, line_loading.shape)
    max_snapshot = pd.Timestamp(snapshots[max_snapshot_position])
    max_line = str(network.lines.index[max_line_position])
    unique_binding_lines = sorted(binding_events["line"].unique().tolist()) if not binding_events.empty else []
    unique_binding_candidates = (
        sorted(binding_events.loc[binding_events["is_candidate"], "line"].unique().tolist())
        if not binding_events.empty
        else []
    )
    economically_binding_events = (
        binding_events[binding_events["has_nonzero_shadow_price"]]
        if not binding_events.empty
        else binding_events
    )
    unique_economically_binding_lines = (
        sorted(economically_binding_events["line"].unique().tolist())
        if not economically_binding_events.empty
        else []
    )
    unique_economically_binding_candidates = (
        sorted(
            economically_binding_events.loc[
                economically_binding_events["is_candidate"], "line"
            ].unique().tolist()
        )
        if not economically_binding_events.empty
        else []
    )
    unique_binding_transformers = sorted(
        transformer_binding_mask.columns[transformer_binding_mask.any(axis=0)].astype(str).tolist()
    )

    if load_shedding > 1e-6:
        gate_classification = "adequacy_failure"
    elif float(upgrades.sum()) > 1e-6:
        gate_classification = "expansion_selected"
    elif unique_economically_binding_candidates:
        gate_classification = "economic_candidate_congestion_without_expansion"
    elif unique_economically_binding_lines:
        gate_classification = "economic_non_candidate_congestion_without_expansion"
    elif unique_binding_lines:
        gate_classification = "capacity_limit_reached_without_economic_shadow"
    elif unique_binding_transformers:
        gate_classification = "binding_transformer_without_expansion"
    else:
        gate_classification = "no_expansion_or_binding_signal"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "line": upgrades.index,
            "base_s_nom_mw": base_capacity.loc[upgrades.index].values,
            "upgrade_mw": upgrades.values,
            "s_nom_opt_mw": network.lines.loc[upgrades.index, "s_nom_opt"].values,
            "investment_cost_per_mw": investment_cost.loc[upgrades.index].values,
        }
    ).to_csv(output_dir / "dc_tep_upgrades.csv", index=False)
    binding_events.to_csv(output_dir / "dc_tep_binding_events.csv", index=False)
    line_diagnostics.to_csv(output_dir / "dc_tep_line_diagnostics.csv", index=False)
    transformer_diagnostics.to_csv(output_dir / "dc_tep_transformer_diagnostics.csv", index=False)
    window_summary.to_csv(output_dir / "dc_tep_window_summary.csv", index=False)
    carrier_costs.to_csv(output_dir / "dc_tep_cost_by_carrier.csv", index=False)
    marginal_prices.to_csv(output_dir / "dc_tep_bus_marginal_prices.csv", index_label="timestamp")
    component_price_summary.to_csv(output_dir / "dc_tep_component_price_summary.csv", index=False)
    summary = {
        "scope": "multi-snapshot linear DC reinforcement lower-bound on the synthetic isolated-Austria case",
        "future_scenario": scenario_audit(dataset),
        "backend": backend,
        "status": str(status),
        "condition": str(condition),
        "n_shared_windows": int(args.episodes),
        "n_snapshots": int(len(snapshots)),
        "represented_annual_hours": float(args.annual_hours) if float(args.annual_hours) > 0.0 else float(len(snapshots)),
        "snapshot_weight": representative_weight,
        "start_timestamps": [str(dataset.snapshots[index]) for index in starts],
        "reinforcement_constraints": {
            "candidate_corridors": int(len(candidate_lines)),
            "max_upgrade_per_corridor_mw": float(args.max_upgrade_mw),
            "total_budget_mw": float(args.budget_mw),
            "source": "preprocessing_manifest_unless_explicitly_overridden",
        },
        "total_upgrade_mw": float(upgrades.sum()),
        "active_lines": int((upgrades > 1e-6).sum()),
        "investment_cost": investment_total,
        "investment_cost_basis": {
            "units": "EUR_per_MW_year_by_corridor",
            "annualized_eur_per_mw_km_year": float(args.line_investment_cost_eur_per_mw_km_year),
            "capital_cost_source": "European Commission JRC (2012), 400-kV HVAC overhead-line range",
            "annuity_convention": "ENTSO-E CBA: 4 percent real discount rate, 40-year lifetime",
        },
        "objective_incremental_cost": float(network.objective),
        "pypsa_existing_capacity_objective_constant": float(network.objective_constant),
        "operating_cost_reconstructed": operating_cost,
        "objective_reconstruction_residual": float(network.objective - operating_cost - investment_total),
        "load_shedding_mwh": load_shedding,
        "backstop_generation_mwh": slack_generation,
        "backstop_design": "distributed_high_cost_adequacy_reserve_at_load_buses",
        "renewable_available_mwh": float(
            renewable_available_by_snapshot.mul(energy_weights).sum()
        ),
        "renewable_dispatch_mwh": float(
            renewable_dispatch_by_snapshot.mul(energy_weights).sum()
        ),
        "renewable_curtailment_mwh": float(
            renewable_curtailment_by_snapshot.mul(energy_weights).sum()
        ),
        "mean_line_loading": float(line_loading.to_numpy().mean()),
        "p95_line_loading": float(np.quantile(line_loading.to_numpy(), 0.95)),
        "max_line_loading": float(line_loading.to_numpy().max()),
        "max_loading_line": max_line,
        "max_loading_timestamp": str(max_snapshot),
        "max_loading_line_is_candidate": max_line in candidate_set,
        "max_loading_allowed_pu": float(allowed_loading.at[max_line]),
        "binding_tolerance_pu": float(args.binding_tolerance_pu),
        "binding_line_hours": int(len(binding_events)),
        "unique_binding_lines": unique_binding_lines,
        "binding_candidate_hours": int(binding_events["is_candidate"].sum())
        if not binding_events.empty
        else 0,
        "unique_binding_candidate_lines": unique_binding_candidates,
        "economically_binding_line_hours": int(len(economically_binding_events)),
        "unique_economically_binding_lines": unique_economically_binding_lines,
        "unique_economically_binding_candidate_lines": unique_economically_binding_candidates,
        "max_transformer_loading": float(transformer_loading.to_numpy().max())
        if transformer_loading.size
        else 0.0,
        "binding_transformer_hours": int(transformer_binding_mask.to_numpy().sum()),
        "unique_binding_transformers": unique_binding_transformers,
        "mean_cross_component_price_range": float(cross_component_price_range.mean()),
        "max_cross_component_price_range": float(cross_component_price_range.max()),
        "mean_largest_component_price_spread": float(largest_component_spread.mean()),
        "max_largest_component_price_spread": float(largest_component_spread.max()),
        "gate_classification": gate_classification,
        "expansion_signal_present": bool(float(upgrades.sum()) > 1e-6),
        "diagnostic_files": {
            "upgrades": "dc_tep_upgrades.csv",
            "binding_events": "dc_tep_binding_events.csv",
            "line_diagnostics": "dc_tep_line_diagnostics.csv",
            "transformer_diagnostics": "dc_tep_transformer_diagnostics.csv",
            "window_summary": "dc_tep_window_summary.csv",
            "cost_by_carrier": "dc_tep_cost_by_carrier.csv",
            "bus_marginal_prices": "dc_tep_bus_marginal_prices.csv",
            "component_price_summary": "dc_tep_component_price_summary.csv",
        },
    }
    (output_dir / "dc_tep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
