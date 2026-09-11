from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx
import pandas as pd
import pypsa

from .config import NetworkConfig
from .simulation import DCPowerFlowModel


@dataclass
class TEPDataset:
    network: pypsa.Network
    snapshots: pd.DatetimeIndex
    demand_by_bus: pd.DataFrame
    demand_by_load: pd.DataFrame
    renewable_by_bus: pd.DataFrame
    generator_availability: pd.DataFrame
    candidate_lines: list[str]
    renewable_carriers: tuple[str, ...]
    candidate_line_scores: pd.DataFrame | None = None
    demand_scale: pd.Series | None = None
    renewable_scale: pd.Series | None = None
    total_demand_scale: float | None = None
    scenario_metadata: dict[str, object] | None = None


def make_naive_index(index_like: pd.Index) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(index_like))
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index


def align_series_to_snapshots(series: pd.Series, snapshots: pd.DatetimeIndex) -> pd.Series:
    aligned = series.copy()
    aligned.index = make_naive_index(aligned.index)
    return aligned.sort_index().reindex(snapshots, method="nearest").ffill().bfill()


def _index_overlaps_snapshots(index_like: pd.Index, snapshots: pd.DatetimeIndex) -> bool:
    if len(index_like) == 0 or len(snapshots) == 0:
        return False
    index = make_naive_index(index_like)
    return index.min() <= snapshots.max() and index.max() >= snapshots.min()


def _slice_time_window(
    data: pd.Series | pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series | pd.DataFrame:
    if data.empty:
        return data

    sliced = data.sort_index()
    if start is not None:
        start_ts = pd.Timestamp(start)
        sliced = sliced.loc[sliced.index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end)
        # Treat date-only end bounds as inclusive whole-day windows so
        # CLI arguments like ``--end 2020-12-31`` keep all hourly samples
        # on that day instead of only the midnight snapshot.
        if (
            isinstance(end, str)
            and len(end.strip()) <= 10
            and end_ts == end_ts.normalize()
        ):
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        sliced = sliced.loc[sliced.index <= end_ts]
    return sliced


def _load_ninja_csv(path: Path, year: Optional[int]) -> pd.DataFrame:
    header_row = None
    with path.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            if line.lower().startswith("time,"):
                header_row = row_idx
                break

    if header_row is None:
        raise ValueError(f"Could not detect CSV header in {path}")

    frame = pd.read_csv(path, skiprows=header_row, parse_dates=["time"], index_col="time")
    frame.index = make_naive_index(frame.index)
    frame.columns = [str(col).strip() for col in frame.columns]
    if year is not None:
        year_mask = frame.index.year == year
        if year_mask.any():
            frame = frame.loc[year_mask]
    return frame


def _load_opsd_load(path: Path, year: Optional[int], country: str) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["utc_timestamp"])
    frame = frame.set_index("utc_timestamp")
    frame.index = make_naive_index(frame.index)
    country = country.upper()
    candidates = [
        f"{country}_load_actual_entsoe",
        f"{country}_load_actual_entsoe_transparency",
        f"{country}_load_forecast_entsoe_transparency",
    ]
    for column in candidates:
        if column in frame.columns:
            series = frame[column].astype(float)
            if year is not None:
                series = series.loc[series.index.year == year]
            if series.notna().any():
                return series.ffill().bfill()
    raise KeyError(f"No OPSD load column found for {country} in {path}")


def _ensure_bus_countries(network: pypsa.Network, default_country: str = "AT") -> None:
    if "country" not in network.buses.columns:
        network.buses["country"] = default_country
    network.buses["country"] = network.buses["country"].fillna(default_country)


def _ensure_defined_carriers(network: pypsa.Network) -> None:
    component_defaults = {
        "buses": "electricity",
        "lines": "AC",
        "links": "DC",
    }

    for component_name, default_carrier in component_defaults.items():
        component = getattr(network, component_name, None)
        if component is None or component.empty:
            continue
        if "carrier" not in component.columns:
            component["carrier"] = default_carrier
        carriers = component["carrier"].copy()
        carriers = carriers.where(carriers.notna(), default_carrier)
        carriers = carriers.astype(str).str.strip()
        component["carrier"] = carriers.mask(carriers == "", default_carrier)

    carrier_names: set[str] = set()
    for component_name in (
        "buses",
        "generators",
        "loads",
        "lines",
        "links",
        "storage_units",
        "transformers",
    ):
        component = getattr(network, component_name, None)
        if component is None or component.empty or "carrier" not in component.columns:
            continue
        values = component["carrier"].dropna().astype(str).str.strip()
        carrier_names.update(value for value in values if value)

    existing = set(network.carriers.index.astype(str))
    for carrier_name in sorted(carrier_names - existing):
        network.add("Carrier", carrier_name)


def _subset_component_timeseries(network: pypsa.Network) -> None:
    if hasattr(network.generators_t, "p_max_pu") and not network.generators_t.p_max_pu.empty:
        valid = [column for column in network.generators_t.p_max_pu.columns if column in network.generators.index]
        network.generators_t.p_max_pu = network.generators_t.p_max_pu.loc[:, valid]

    if hasattr(network.loads_t, "p_set") and not network.loads_t.p_set.empty:
        valid = [column for column in network.loads_t.p_set.columns if column in network.loads.index]
        network.loads_t.p_set = network.loads_t.p_set.loc[:, valid]

    if hasattr(network.storage_units_t, "inflow") and not network.storage_units_t.inflow.empty:
        valid = [column for column in network.storage_units_t.inflow.columns if column in network.storage_units.index]
        network.storage_units_t.inflow = network.storage_units_t.inflow.loc[:, valid]


def cleanup_network(network: pypsa.Network, default_country: str = "AT") -> pypsa.Network:
    _ensure_bus_countries(network, default_country=default_country)

    # The OSM-derived Austrian extract contains one completely disconnected
    # single-bus artefact.  It cannot be reached by any reinforcement action
    # and would otherwise create unavoidable load shedding in future cases.
    # Keep the largest line/transformer-connected component before attaching
    # national and regional profiles.
    graph = nx.Graph()
    graph.add_nodes_from(network.buses.index)
    if not network.lines.empty:
        graph.add_edges_from(network.lines[["bus0", "bus1"]].itertuples(index=False, name=None))
    if hasattr(network, "transformers") and not network.transformers.empty:
        graph.add_edges_from(network.transformers[["bus0", "bus1"]].itertuples(index=False, name=None))
    components = list(nx.connected_components(graph))
    if components:
        largest_component = max(components, key=len)
        network.buses = network.buses.loc[network.buses.index.isin(largest_component)].copy()

    valid_buses = set(network.buses.index)
    network.lines = network.lines[
        network.lines.bus0.isin(valid_buses) & network.lines.bus1.isin(valid_buses)
    ].copy()
    network.loads = network.loads[network.loads.bus.isin(valid_buses)].copy()
    network.generators = network.generators[network.generators.bus.isin(valid_buses)].copy()
    if "carrier" in network.generators.columns:
        # ``AT_Slack`` is an artificial feasibility generator from the model
        # preparation, not an observed Austrian plant.  Operational solves add
        # an explicitly documented distributed adequacy reserve instead.
        network.generators = network.generators[
            ~network.generators["carrier"].fillna("").astype(str).str.lower().eq("slack")
        ].copy()

    if hasattr(network, "transformers") and not network.transformers.empty:
        network.transformers = network.transformers[
            network.transformers.bus0.isin(valid_buses) & network.transformers.bus1.isin(valid_buses)
        ].copy()
    if hasattr(network, "links") and not network.links.empty:
        network.links = network.links[
            network.links.bus0.isin(valid_buses) & network.links.bus1.isin(valid_buses)
        ].copy()

    if hasattr(network, "storage_units") and not network.storage_units.empty:
        network.storage_units = network.storage_units[network.storage_units.bus.isin(valid_buses)].copy()

    if "carrier" not in network.buses.columns:
        network.buses["carrier"] = "electricity"
    network.buses["carrier"] = network.buses["carrier"].fillna("electricity")
    _ensure_defined_carriers(network)

    network.snapshots = make_naive_index(network.snapshots)
    _subset_component_timeseries(network)

    if "s_nom" in network.lines.columns:
        network.lines["s_nom"] = network.lines["s_nom"].fillna(0.0).clip(lower=1.0)
    if "x" in network.lines.columns:
        network.lines["x"] = network.lines["x"].replace(0.0, np.nan).fillna(0.1)
    if "r" in network.lines.columns:
        network.lines["r"] = network.lines["r"].replace(0.0, np.nan).fillna(0.01)

    return network


def _build_bus_demand_profiles(
    network: pypsa.Network,
    snapshots: pd.DatetimeIndex,
    country_load: Optional[pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        hasattr(network.loads_t, "p_set")
        and not network.loads_t.p_set.empty
        and _index_overlaps_snapshots(network.loads_t.p_set.index, snapshots)
    ):
        load_profiles = network.loads_t.p_set.copy()
        load_profiles.index = make_naive_index(load_profiles.index)
        load_profiles = load_profiles.reindex(index=snapshots, method="nearest").ffill().bfill()
        load_profiles = load_profiles.reindex(columns=network.loads.index).fillna(0.0)
        bus_profiles = pd.DataFrame(0.0, index=snapshots, columns=network.buses.index)
        for load_name, row in network.loads.iterrows():
            bus_profiles[row.bus] = bus_profiles[row.bus] + load_profiles[load_name]
        if bus_profiles.to_numpy().sum() > 0:
            return bus_profiles, load_profiles

    if country_load is None:
        country_load = pd.Series(1.0, index=snapshots)

    aligned_country_load = align_series_to_snapshots(country_load, snapshots)
    bus_weights = network.loads.groupby("bus")["p_set"].sum().reindex(network.buses.index).fillna(0.0)
    if bus_weights.sum() <= 0:
        bus_weights = pd.Series(1.0, index=network.buses.index)
    bus_weights = bus_weights / bus_weights.sum()

    bus_profiles = pd.DataFrame(index=snapshots, columns=network.buses.index, dtype=float)
    for bus in network.buses.index:
        bus_profiles[bus] = aligned_country_load.values * float(bus_weights.loc[bus])

    load_profiles = pd.DataFrame(index=snapshots, columns=network.loads.index, dtype=float)
    for load_name, row in network.loads.iterrows():
        sibling_loads = network.loads.index[network.loads.bus == row.bus]
        weight = 1.0 / max(len(sibling_loads), 1)
        load_profiles[load_name] = bus_profiles[row.bus].values * weight

    return bus_profiles.fillna(0.0), load_profiles.fillna(0.0)


def _build_generator_availability(
    network: pypsa.Network,
    snapshots: pd.DatetimeIndex,
    wind_cf: pd.Series,
    solar_cf: pd.Series,
) -> pd.DataFrame:
    availability = pd.DataFrame(1.0, index=snapshots, columns=network.generators.index, dtype=float)
    existing = getattr(network.generators_t, "p_max_pu", pd.DataFrame(index=snapshots))
    if not existing.empty and _index_overlaps_snapshots(existing.index, snapshots):
        existing = existing.copy()
        existing.index = make_naive_index(existing.index)
        existing = existing.reindex(index=snapshots, method="nearest")
        existing = existing.reindex(columns=network.generators.index)
    else:
        existing = pd.DataFrame(index=snapshots)

    wind_series = align_series_to_snapshots(wind_cf, snapshots)
    solar_series = align_series_to_snapshots(solar_cf, snapshots)

    for generator, row in network.generators.iterrows():
        carrier = str(row.carrier).lower()
        if generator in existing.columns and existing[generator].notna().any():
            availability[generator] = existing[generator].ffill().bfill().fillna(1.0)
        elif "wind" in carrier:
            availability[generator] = wind_series.values
        elif "solar" in carrier:
            availability[generator] = solar_series.values
        elif carrier == "ror":
            availability[generator] = 0.55
        else:
            availability[generator] = 1.0

    return availability.clip(lower=0.0, upper=1.0)


def build_renewable_bus_profiles(
    network: pypsa.Network,
    availability: pd.DataFrame,
    renewable_carriers: tuple[str, ...],
) -> pd.DataFrame:
    generator_capacities = network.generators["p_nom"].fillna(0.0).clip(lower=0.0)
    renewable = pd.DataFrame(0.0, index=availability.index, columns=network.buses.index)
    renewable_set = {carrier.lower() for carrier in renewable_carriers}

    for generator, row in network.generators.iterrows():
        carrier = str(row.carrier).lower()
        if carrier not in renewable_set and not any(tag in carrier for tag in ("wind", "solar", "ror")):
            continue
        renewable[row.bus] = renewable[row.bus] + availability[generator] * float(generator_capacities.loc[generator])

    return renewable.fillna(0.0)


def _candidate_line_ranking(
    network: pypsa.Network,
    demand_by_bus: pd.DataFrame | None = None,
    renewable_by_bus: pd.DataFrame | None = None,
    sample_count: int = 336,
) -> pd.DataFrame:
    """Rank reinforcement candidates by sampled electrical loading pressure.

    Screening is deliberately based on operational criticality rather than the
    former ``capital_cost + length / s_nom`` heuristic.  The latter collapses
    to a pure length ranking whenever capital costs are absent and ratings are
    uniform.  A small normalised length penalty is retained only as a
    transparent construction-cost proxy when project-level costs are absent.
    """
    ranking = network.lines[["length", "s_nom", "capital_cost"]].copy()
    ranking["length"] = ranking["length"].fillna(ranking["length"].median()).clip(lower=0.0)
    ranking["s_nom"] = ranking["s_nom"].fillna(0.0).clip(lower=1.0)

    if demand_by_bus is None or renewable_by_bus is None or demand_by_bus.empty:
        inverse_capacity = 1.0 / ranking["s_nom"]
        ranking["mean_loading"] = inverse_capacity / max(float(inverse_capacity.max()), 1e-12)
        ranking["p95_loading"] = ranking["mean_loading"]
        ranking["overload_frequency"] = 0.0
    else:
        common = demand_by_bus.index.intersection(renewable_by_bus.index)
        if common.empty:
            raise ValueError("Candidate screening requires aligned demand and renewable timestamps.")
        positions = np.unique(
            np.linspace(0, len(common) - 1, num=min(max(int(sample_count), 1), len(common)), dtype=int)
        )
        model = DCPowerFlowModel.from_network(network)
        loadings: list[np.ndarray] = []
        capacities = ranking["s_nom"].reindex(model.lines).to_numpy(dtype=float)
        for position in positions:
            timestamp = common[position]
            _, injections, _ = model.balance_dispatch(
                renewable_by_bus.loc[timestamp],
                demand_by_bus.loc[timestamp],
                balance_mode="demand_proportional",
            )
            loading = model.solve(injections).abs().to_numpy(dtype=float) / capacities
            loadings.append(loading)
        loading_matrix = np.vstack(loadings)
        ranking["mean_loading"] = loading_matrix.mean(axis=0)
        ranking["p95_loading"] = np.quantile(loading_matrix, 0.95, axis=0)
        ranking["overload_frequency"] = (loading_matrix > 1.0).mean(axis=0)

    def _unit_scale(series: pd.Series) -> pd.Series:
        minimum, maximum = float(series.min()), float(series.max())
        if np.isclose(minimum, maximum):
            return pd.Series(0.0, index=series.index)
        return (series - minimum) / (maximum - minimum)

    criticality = (
        0.50 * _unit_scale(ranking["p95_loading"])
        + 0.35 * _unit_scale(ranking["mean_loading"])
        + 0.15 * _unit_scale(ranking["overload_frequency"])
    )
    capital_cost = ranking["capital_cost"].fillna(0.0).clip(lower=0.0)
    if (capital_cost > 0.0).any():
        cost_proxy = _unit_scale(capital_cost)
        ranking["cost_proxy_source"] = "capital_cost"
    else:
        cost_proxy = _unit_scale(ranking["length"])
        ranking["cost_proxy_source"] = "normalised_length"
    ranking["criticality_score"] = criticality
    ranking["screening_score"] = criticality - 0.05 * cost_proxy
    return ranking.sort_values(
        ["screening_score", "p95_loading", "mean_loading"],
        ascending=False,
    )


def _select_candidate_lines(
    network: pypsa.Network,
    limit: Optional[int],
    demand_by_bus: pd.DataFrame | None = None,
    renewable_by_bus: pd.DataFrame | None = None,
) -> list[str]:
    if limit is None or limit >= len(network.lines):
        return list(network.lines.index)
    ranking = _candidate_line_ranking(
        network,
        demand_by_bus=demand_by_bus,
        renewable_by_bus=renewable_by_bus,
    )
    return list(ranking.head(limit).index)


def refit_candidate_line_screening(dataset: TEPDataset, limit: Optional[int]) -> None:
    """Refit the candidate ranking after a scenario transformation.

    Future-scenario generation and load targets change the electrical pressure
    used for candidate screening.  The ranking must therefore be fitted on the
    transformed training split, not inherited from the historical input case.
    """
    ranking = _candidate_line_ranking(
        dataset.network,
        demand_by_bus=dataset.demand_by_bus,
        renewable_by_bus=dataset.renewable_by_bus,
    )
    dataset.candidate_line_scores = ranking
    if limit is None or limit >= len(dataset.network.lines):
        dataset.candidate_lines = list(dataset.network.lines.index)
    else:
        dataset.candidate_lines = list(ranking.head(limit).index)


def fit_observation_scales(dataset: TEPDataset) -> None:
    """Fit observation scales on a reference (normally training) split."""
    dataset.demand_scale = dataset.demand_by_bus.max().replace(0.0, 1.0)
    dataset.renewable_scale = dataset.renewable_by_bus.max().replace(0.0, 1.0)
    row_sums = dataset.demand_by_bus.sum(axis=1).to_numpy(dtype=float)
    dataset.total_demand_scale = max(float(np.nanmax(row_sums)) if row_sums.size else 1.0, 1.0)


def apply_observation_scale_reference(dataset: TEPDataset, reference: TEPDataset) -> None:
    """Apply training-fitted scales to validation/test datasets without refitting."""
    if reference.demand_scale is None or reference.renewable_scale is None or reference.total_demand_scale is None:
        fit_observation_scales(reference)
    dataset.demand_scale = reference.demand_scale.reindex(dataset.network.buses.index).fillna(1.0).copy()
    dataset.renewable_scale = reference.renewable_scale.reindex(dataset.network.buses.index).fillna(1.0).copy()
    dataset.total_demand_scale = float(reference.total_demand_scale)


def _build_common_snapshots(
    country_load: pd.Series,
    wind_frame: pd.DataFrame,
    solar_frame: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
) -> pd.DatetimeIndex:
    load_window = _slice_time_window(country_load, start=start, end=end)
    wind_window = _slice_time_window(wind_frame, start=start, end=end)
    solar_window = _slice_time_window(solar_frame, start=start, end=end)

    common_index = pd.DatetimeIndex(load_window.index)
    common_index = common_index.intersection(pd.DatetimeIndex(wind_window.index))
    common_index = common_index.intersection(pd.DatetimeIndex(solar_window.index))
    common_index = make_naive_index(common_index)
    common_index = common_index.sort_values()

    if common_index.empty:
        raise ValueError("No common timestamps remain after aligning load, wind, and solar data.")
    return common_index


def load_austria_case(config: NetworkConfig) -> TEPDataset:
    network = pypsa.Network(config.network_path)
    network = cleanup_network(network, default_country=config.country)

    line_ratings = network.lines["s_nom"].astype(float)
    if len(line_ratings) > 10 and np.isclose(float(line_ratings.std(ddof=0)), 0.0, atol=1e-8):
        raise ValueError(
            "All line ratings are identical. This calibrated synthetic network is no longer accepted; "
            "run scripts/restore_physical_line_ratings.py and use its output."
        )

    if network.loads.empty or network.generators.empty:
        raise ValueError(
            "The selected network does not contain the load and generator components required for RL training. "
            "Use a PyPSA electricity network with assets already attached, or add a preprocessing step that "
            "constructs Austria's load and generator fleet on top of the extracted topology."
        )

    use_year_filter = config.year if config.start is None and config.end is None else None
    wind_frame = _load_ninja_csv(config.wind_path, year=use_year_filter)
    solar_frame = _load_ninja_csv(config.solar_path, year=use_year_filter)
    country_load = _load_opsd_load(config.load_path, year=use_year_filter, country=config.country)

    snapshots = _build_common_snapshots(
        country_load=country_load,
        wind_frame=wind_frame,
        solar_frame=solar_frame,
        start=config.start,
        end=config.end,
    )

    wind_cf = wind_frame["NATIONAL"] if "NATIONAL" in wind_frame.columns else wind_frame.iloc[:, 0]
    solar_cf = solar_frame["NATIONAL"] if "NATIONAL" in solar_frame.columns else solar_frame.iloc[:, 0]
    country_load = country_load.reindex(snapshots)
    wind_cf = wind_cf.reindex(snapshots)
    solar_cf = solar_cf.reindex(snapshots)

    demand_by_bus, demand_by_load = _build_bus_demand_profiles(network, snapshots, country_load)
    generator_availability = _build_generator_availability(network, snapshots, wind_cf, solar_cf)
    renewable_by_bus = build_renewable_bus_profiles(network, generator_availability, config.renewable_carriers)
    candidate_ranking = _candidate_line_ranking(
        network,
        demand_by_bus=demand_by_bus,
        renewable_by_bus=renewable_by_bus,
    )
    if config.candidate_line_limit is None or config.candidate_line_limit >= len(network.lines):
        candidate_lines = list(network.lines.index)
    else:
        candidate_lines = list(candidate_ranking.head(config.candidate_line_limit).index)

    dataset = TEPDataset(
        network=network,
        snapshots=snapshots,
        demand_by_bus=demand_by_bus,
        demand_by_load=demand_by_load,
        renewable_by_bus=renewable_by_bus,
        generator_availability=generator_availability,
        candidate_lines=candidate_lines,
        renewable_carriers=config.renewable_carriers,
        candidate_line_scores=candidate_ranking,
    )
    fit_observation_scales(dataset)
    return dataset


def build_toy_dataset(num_steps: int = 72, seed: int = 7) -> TEPDataset:
    rng = np.random.default_rng(seed)
    snapshots = pd.date_range("2020-01-01", periods=num_steps, freq="h")

    network = pypsa.Network()
    network.set_snapshots(snapshots)

    for name, x_coord, y_coord in (("A", 13.4, 47.1), ("B", 14.3, 48.0), ("C", 16.3, 48.2)):
        network.add("Bus", name, x=x_coord, y=y_coord)

    network.buses["country"] = "AT"

    network.add("Line", "L_AB", bus0="A", bus1="B", x=0.08, r=0.01, s_nom=90, length=120)
    network.add("Line", "L_BC", bus0="B", bus1="C", x=0.06, r=0.01, s_nom=70, length=90)
    network.add("Line", "L_AC", bus0="A", bus1="C", x=0.11, r=0.01, s_nom=55, length=150)

    network.add("Load", "Load_A", bus="A", p_set=30)
    network.add("Load", "Load_B", bus="B", p_set=40)
    network.add("Load", "Load_C", bus="C", p_set=32)

    network.add("Generator", "Wind_A", bus="A", p_nom=55, marginal_cost=0.0, carrier="onwind")
    network.add("Generator", "Solar_C", bus="C", p_nom=45, marginal_cost=0.0, carrier="solar")
    network.add("Generator", "Gas_B", bus="B", p_nom=120, marginal_cost=70.0, carrier="CCGT")
    network.add("Generator", "Slack_B", bus="B", p_nom=1_000, marginal_cost=220.0, carrier="slack")

    seasonal = np.sin(np.linspace(0, 6 * np.pi, num_steps))
    demand_by_bus = pd.DataFrame(
        {
            "A": 24 + 7 * (seasonal + 1.0) + rng.normal(0.0, 1.2, num_steps),
            "B": 32 + 9 * (np.roll(seasonal, 4) + 1.0) + rng.normal(0.0, 1.3, num_steps),
            "C": 26 + 6 * (np.roll(seasonal, 8) + 1.0) + rng.normal(0.0, 1.0, num_steps),
        },
        index=snapshots,
    ).clip(lower=8.0)

    demand_by_load = pd.DataFrame(index=snapshots, columns=network.loads.index, dtype=float)
    demand_by_load["Load_A"] = demand_by_bus["A"].values
    demand_by_load["Load_B"] = demand_by_bus["B"].values
    demand_by_load["Load_C"] = demand_by_bus["C"].values

    daylight = np.maximum(np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, num_steps)), 0.0)
    wind_cf = (0.55 + 0.25 * np.sin(np.linspace(0, 10 * np.pi, num_steps) + 0.7) + rng.normal(0.0, 0.03, num_steps)).clip(0.05, 0.95)
    solar_cf = (0.8 * daylight + rng.normal(0.0, 0.01, num_steps)).clip(0.0, 0.95)

    generator_availability = pd.DataFrame(
        {
            "Wind_A": wind_cf,
            "Solar_C": solar_cf,
            "Gas_B": np.ones(num_steps),
            "Slack_B": np.ones(num_steps),
        },
        index=snapshots,
    )

    renewable_by_bus = pd.DataFrame(0.0, index=snapshots, columns=network.buses.index)
    renewable_by_bus["A"] = generator_availability["Wind_A"].values * network.generators.at["Wind_A", "p_nom"]
    renewable_by_bus["C"] = generator_availability["Solar_C"].values * network.generators.at["Solar_C", "p_nom"]

    network.loads_t.p_set = demand_by_load.copy()
    network.generators_t.p_max_pu = generator_availability.copy()

    return TEPDataset(
        network=cleanup_network(network),
        snapshots=snapshots,
        demand_by_bus=demand_by_bus,
        demand_by_load=demand_by_load,
        renewable_by_bus=renewable_by_bus,
        generator_availability=generator_availability,
        candidate_lines=list(network.lines.index),
        renewable_carriers=("onwind", "solar", "ror"),
    )
