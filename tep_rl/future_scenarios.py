from __future__ import annotations

"""Source-based future-scenario transformations for the Austrian case.

The scenario changes exogenous demand and renewable-capacity inputs.  It does
not add transmission corridors and it does not let the RL agent choose wind or
PV sites.  APG NUTS-2 targets are mapped to model buses before candidate-line
screening; the RL action remains reinforcement of existing corridors.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd

from .data import TEPDataset, build_renewable_bus_profiles, fit_observation_scales


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_CATALOG = PROJECT_ROOT / "config" / "official_future_scenarios.json"
DEFAULT_NUTS2_BOUNDARIES = PROJECT_ROOT / "data" / "NUTS_RG_01M_2024_4326_LEVL_2.geojson"


def load_scenario_catalog(path: str | Path = DEFAULT_SCENARIO_CATALOG) -> dict[str, Any]:
    catalog_path = Path(path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if int(catalog.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported future-scenario catalog schema in {catalog_path}.")
    if not isinstance(catalog.get("scenarios"), dict):
        raise ValueError(f"Future-scenario catalog {catalog_path} has no scenario mapping.")
    return catalog


def available_scenarios(path: str | Path = DEFAULT_SCENARIO_CATALOG) -> tuple[str, ...]:
    return tuple(sorted(load_scenario_catalog(path)["scenarios"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_buses_to_nuts2(
    dataset: TEPDataset,
    boundary_path: str | Path = DEFAULT_NUTS2_BOUNDARIES,
) -> pd.DataFrame:
    """Map every model bus to an Austrian NUTS-2 region.

    Boundary/interface nodes that lie just outside the polygon are assigned to
    the nearest Austrian region and their distance is recorded.  This is
    required for the synthetic boundary nodes retained by the network extract.
    """
    path = Path(boundary_path)
    if not path.exists():
        raise FileNotFoundError(
            f"NUTS-2 boundary file not found: {path}. Download the official GISCO file listed in the scenario catalog."
        )
    nuts = gpd.read_file(path)
    required = {"NUTS_ID", "LEVL_CODE", "CNTR_CODE", "geometry"}
    missing = required.difference(nuts.columns)
    if missing:
        raise ValueError(f"NUTS boundary file is missing columns: {sorted(missing)}")
    nuts = nuts[(nuts["CNTR_CODE"] == "AT") & (nuts["LEVL_CODE"] == 2)].copy()
    if set(nuts["NUTS_ID"]) != {"AT11", "AT12", "AT13", "AT21", "AT22", "AT31", "AT32", "AT33", "AT34"}:
        raise ValueError("The boundary file does not contain the nine expected Austrian NUTS-2 regions.")

    buses = dataset.network.buses[["x", "y"]].astype(float).copy()
    points = gpd.GeoDataFrame(
        buses,
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        points,
        nuts[["NUTS_ID", "NUTS_NAME", "geometry"]],
        how="left",
        predicate="within",
    )
    result = joined[["x", "y", "NUTS_ID", "NUTS_NAME"]].copy()
    result["assignment_method"] = np.where(result["NUTS_ID"].notna(), "within", "nearest_boundary")
    result["nearest_distance_km"] = 0.0

    if result["NUTS_ID"].isna().any():
        nuts_metric = nuts.to_crs("EPSG:3035")
        points_metric = points.to_crs("EPSG:3035")
        for bus in result.index[result["NUTS_ID"].isna()]:
            distances = nuts_metric.geometry.distance(points_metric.loc[bus, "geometry"])
            nearest_index = distances.idxmin()
            result.at[bus, "NUTS_ID"] = str(nuts_metric.at[nearest_index, "NUTS_ID"])
            result.at[bus, "NUTS_NAME"] = str(nuts_metric.at[nearest_index, "NUTS_NAME"])
            result.at[bus, "nearest_distance_km"] = float(distances.loc[nearest_index]) / 1000.0

    if result["NUTS_ID"].isna().any():
        raise ValueError("At least one model bus could not be assigned to an Austrian NUTS-2 region.")
    result.index = result.index.astype(str)
    result.index.name = "bus"
    return result


def _regional_peak(dataset: TEPDataset, bus_to_region: pd.Series, region: str) -> float:
    buses = bus_to_region.index[bus_to_region.eq(region)].intersection(dataset.demand_by_bus.columns)
    if len(buses) == 0:
        return 0.0
    return float(dataset.demand_by_bus.loc[:, buses].sum(axis=1).max())


def _capacity_allocation(
    dataset: TEPDataset,
    bus_to_region: pd.Series,
    carrier: str,
    regional_targets: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, str]]:
    network = dataset.network
    generator_carrier = network.generators["carrier"].fillna("").astype(str).str.lower()
    existing = (
        network.generators.loc[generator_carrier.eq(carrier), ["bus", "p_nom"]]
        .assign(p_nom=lambda frame: pd.to_numeric(frame["p_nom"], errors="coerce").fillna(0.0).clip(lower=0.0))
        .groupby("bus")["p_nom"]
        .sum()
    )
    load_weight = (
        network.loads.assign(p_set=pd.to_numeric(network.loads["p_set"], errors="coerce").fillna(0.0).clip(lower=0.0))
        .groupby("bus")["p_set"]
        .sum()
    )

    allocations: dict[str, float] = {}
    methods: dict[str, str] = {}
    for region, raw_target in regional_targets.items():
        target = float(raw_target)
        region_buses = pd.Index(bus_to_region.index[bus_to_region.eq(region)], dtype=str)
        if len(region_buses) == 0:
            raise ValueError(f"No network bus maps to scenario region {region}.")
        existing_weights = existing.reindex(region_buses).fillna(0.0)
        if float(existing_weights.sum()) > 0.0:
            weights = existing_weights
            methods[region] = "existing_carrier_capacity_share"
        else:
            weights = load_weight.reindex(region_buses).fillna(0.0)
            if float(weights.sum()) > 0.0:
                methods[region] = "static_load_share_fallback"
            else:
                weights = pd.Series(1.0, index=region_buses)
                methods[region] = "equal_bus_share_fallback"
        weights = weights / float(weights.sum())
        for bus, share in weights.items():
            amount = target * float(share)
            if amount > 0.0:
                allocations[str(bus)] = allocations.get(str(bus), 0.0) + amount
    return allocations, methods


def fit_future_scenario(
    reference: TEPDataset,
    scenario_id: str,
    catalog_path: str | Path = DEFAULT_SCENARIO_CATALOG,
    boundary_path: str | Path = DEFAULT_NUTS2_BOUNDARIES,
) -> dict[str, Any]:
    """Fit leakage-safe scenario mappings on the training split."""
    catalog_file = Path(catalog_path)
    boundary_file = Path(boundary_path)
    catalog = load_scenario_catalog(catalog_file)
    if scenario_id not in catalog["scenarios"]:
        raise ValueError(f"Unknown future scenario {scenario_id!r}; choose from {sorted(catalog['scenarios'])}.")
    scenario = catalog["scenarios"][scenario_id]
    sources = {source_id: catalog["sources"][source_id] for source_id in scenario.get("source_ids", [])}
    policy_context_sources = {
        source_id: catalog["sources"][source_id]
        for source_id in scenario.get("policy_context_source_ids", [])
    }
    calibration: dict[str, Any] = {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "label": scenario["label"],
        "target_year": scenario.get("target_year"),
        "identity": bool(scenario.get("identity", False)),
        "fit_snapshots": [str(reference.snapshots.min()), str(reference.snapshots.max())],
        "catalog_path": str(catalog_file),
        "catalog_sha256": _sha256(catalog_file),
        "sources": sources,
        "policy_context_sources": policy_context_sources,
        "methodology": {
            "temporal_profiles": "Historical ENTSO-E load and Renewables.ninja wind/PV capacity-factor profiles are retained.",
            "load": "Within each historical calendar year, the regional load shape is normalised to the official APG NUTS-2 peak target. This creates alternative historical weather/load-shape years for one fixed future system.",
            "generation": "APG NUTS-2 wind/PV capacity targets are exogenous. Within-region shares preserve existing carrier siting; regions without a carrier use static load shares.",
            "transmission_action": "The RL agent reinforces existing candidate corridors only; it does not build new routes or site generation.",
        },
        "limitations": [
            "This is a source-based renewable transport-stress scenario, not a reproduction of APG's full European market model.",
            "APG's published run-of-river, pumped-storage, battery, power-to-gas and European exchange assumptions are not operationally reconstructed in this 24-hour isolated-Austria model.",
            "A distributed high-cost adequacy reserve keeps strict dispatch feasible; its energy is a model-omission diagnostic, not forecast Austrian generation.",
            "Project-level renewable siting is outside the action space; the APG NUTS-2 capacities are imposed exogenously.",
            "All generators of one renewable carrier use the same national historical capacity-factor series.",
        ],
    }
    if calibration["identity"]:
        calibration.update(
            {
                "bus_to_nuts2": {},
                "bus_assignment": {},
                "load_scale_by_nuts2": {},
                "generation_capacity_by_bus_mw": {},
            }
        )
        return calibration

    mapping = assign_buses_to_nuts2(reference, boundary_file)
    bus_to_region = mapping["NUTS_ID"].astype(str)
    targets = scenario["regional_targets_mw"]
    required_target_keys = {"onwind", "solar", "peak_load"}
    missing_targets = required_target_keys.difference(targets)
    if missing_targets:
        raise ValueError(f"Scenario {scenario_id} is missing target groups: {sorted(missing_targets)}")

    reference_peaks: dict[str, float] = {}
    load_scales: dict[str, float] = {}
    for region, target in targets["peak_load"].items():
        peak = _regional_peak(reference, bus_to_region, region)
        if peak <= 0.0:
            raise ValueError(f"Training data have no positive load in scenario region {region}.")
        reference_peaks[region] = peak
        load_scales[region] = float(target) / peak

    capacity_by_bus: dict[str, dict[str, float]] = {}
    allocation_methods: dict[str, dict[str, str]] = {}
    for carrier in ("onwind", "solar"):
        allocations, methods = _capacity_allocation(
            reference,
            bus_to_region,
            carrier,
            targets[carrier],
        )
        capacity_by_bus[carrier] = allocations
        allocation_methods[carrier] = methods

    regional_sums = {key: float(sum(float(value) for value in targets[key].values())) for key in required_target_keys}
    declared = {key: float(value) for key, value in scenario["declared_national_targets_mw"].items()}
    rounding_delta = {key: regional_sums[key] - declared[key] for key in required_target_keys}
    calibration.update(
        {
            "boundary_dataset": {
                **catalog["boundary_dataset"],
                "resolved_path": str(boundary_file),
                "sha256": _sha256(boundary_file),
            },
            "bus_to_nuts2": bus_to_region.to_dict(),
            "bus_assignment": mapping[["assignment_method", "nearest_distance_km"]].to_dict(orient="index"),
            "regional_targets_mw": targets,
            "declared_national_targets_mw": declared,
            "regional_target_sums_mw": regional_sums,
            "regional_rounding_delta_mw": rounding_delta,
            "reference_regional_peak_mw": reference_peaks,
            "load_scale_by_nuts2": load_scales,
            "generation_capacity_by_bus_mw": capacity_by_bus,
            "generation_allocation_method_by_nuts2": allocation_methods,
        }
    )
    return calibration


def _representative_profile(dataset: TEPDataset, carrier: str) -> pd.Series:
    network_carriers = dataset.network.generators["carrier"].fillna("").astype(str).str.lower()
    generators = network_carriers.index[network_carriers.eq(carrier)]
    columns = dataset.generator_availability.columns.intersection(generators)
    if len(columns) == 0:
        raise ValueError(f"No historical availability profile is available for carrier {carrier}.")
    return dataset.generator_availability.loc[:, columns].mean(axis=1).clip(lower=0.0, upper=1.0)


def apply_future_scenario(dataset: TEPDataset, calibration: Mapping[str, Any]) -> TEPDataset:
    """Apply one fitted scenario calibration without refitting on the target split."""
    if int(calibration.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported fitted future-scenario schema.")
    if bool(calibration.get("identity", False)):
        dataset.scenario_metadata = dict(calibration)
        return dataset

    network = dataset.network.copy()
    bus_to_region = pd.Series(calibration["bus_to_nuts2"], dtype=str).reindex(network.buses.index.astype(str))
    if bus_to_region.isna().any():
        missing = list(bus_to_region.index[bus_to_region.isna()])
        raise ValueError(f"Fitted scenario has no NUTS-2 assignment for buses: {missing}")

    demand_by_bus = dataset.demand_by_bus.copy()
    load_bus = network.loads["bus"].astype(str)
    load_region = load_bus.map(bus_to_region)
    demand_by_load = dataset.demand_by_load.copy()
    applied_load_scales: dict[str, dict[str, float]] = {}
    regional_targets = calibration.get("regional_targets_mw", {}).get("peak_load")
    if regional_targets:
        years = pd.Index(dataset.snapshots.year).unique().sort_values()
        for year in years:
            row_mask = dataset.snapshots.year == int(year)
            applied_load_scales[str(int(year))] = {}
            for region, raw_target in regional_targets.items():
                buses = bus_to_region.index[bus_to_region.eq(str(region))].intersection(demand_by_bus.columns)
                loads = load_region.index[load_region.eq(str(region))].intersection(demand_by_load.columns)
                peak = float(demand_by_bus.loc[row_mask, buses].sum(axis=1).max()) if len(buses) else 0.0
                if peak <= 0.0:
                    raise ValueError(f"Dataset year {year} has no positive load in scenario region {region}.")
                scale = float(raw_target) / peak
                applied_load_scales[str(int(year))][str(region)] = scale
                demand_by_bus.loc[row_mask, buses] = demand_by_bus.loc[row_mask, buses] * scale
                demand_by_load.loc[row_mask, loads] = demand_by_load.loc[row_mask, loads] * scale
    else:
        # Backwards-compatible path for hand-built/test calibrations.
        load_scales_by_region = {str(key): float(value) for key, value in calibration["load_scale_by_nuts2"].items()}
        bus_scale = bus_to_region.map(load_scales_by_region)
        if bus_scale.isna().any():
            missing_regions = sorted(set(bus_to_region[bus_scale.isna()]))
            raise ValueError(f"Fitted scenario has no load scale for regions: {missing_regions}")
        demand_by_bus = demand_by_bus.mul(bus_scale.reindex(demand_by_bus.columns), axis=1)
        load_scale = load_region.map(load_scales_by_region)
        if load_scale.isna().any():
            missing_loads = list(load_scale.index[load_scale.isna()])
            raise ValueError(f"Fitted scenario could not scale loads: {missing_loads}")
        demand_by_load = demand_by_load.mul(load_scale.reindex(demand_by_load.columns), axis=1)

    availability = dataset.generator_availability.copy()
    weather_profiles = {carrier: _representative_profile(dataset, carrier) for carrier in ("onwind", "solar")}
    new_availability: dict[str, np.ndarray] = {}
    generator_carriers = network.generators["carrier"].fillna("").astype(str).str.lower()
    for carrier in ("onwind", "solar"):
        existing = generator_carriers.index[generator_carriers.eq(carrier)]
        original_capacity = pd.to_numeric(
            network.generators.loc[existing, "p_nom"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        network.generators.loc[existing, "p_nom"] = 0.0
        if "p_nom_min" in network.generators.columns:
            network.generators.loc[existing, "p_nom_min"] = 0.0
        if "p_nom_extendable" in network.generators.columns:
            network.generators.loc[existing, "p_nom_extendable"] = False
        for sequence, (bus, capacity) in enumerate(
            sorted(calibration["generation_capacity_by_bus_mw"][carrier].items()),
            start=1,
        ):
            at_bus = existing[network.generators.loc[existing, "bus"].astype(str).eq(str(bus))]
            if len(at_bus):
                shares = original_capacity.reindex(at_bus).fillna(0.0)
                if float(shares.sum()) <= 0.0:
                    shares = pd.Series(1.0, index=at_bus)
                shares = shares / float(shares.sum())
                network.generators.loc[at_bus, "p_nom"] = shares * float(capacity)
                continue
            name = f"scenario::{calibration['scenario_id']}::{carrier}::{sequence:03d}"
            network.add(
                "Generator",
                name,
                bus=str(bus),
                p_nom=float(capacity),
                p_nom_extendable=False,
                marginal_cost=0.0,
                carrier=carrier,
            )
            new_availability[name] = weather_profiles[carrier].to_numpy(dtype=float)

    if new_availability:
        availability = pd.concat(
            [availability, pd.DataFrame(new_availability, index=dataset.snapshots)],
            axis=1,
        )
    availability = availability.reindex(columns=network.generators.index).fillna(1.0).clip(lower=0.0, upper=1.0)
    renewable_by_bus = build_renewable_bus_profiles(network, availability, dataset.renewable_carriers)
    applied_metadata = dict(calibration)
    applied_metadata["applied_load_scale_by_year_nuts2"] = applied_load_scales
    transformed = TEPDataset(
        network=network,
        snapshots=dataset.snapshots.copy(),
        demand_by_bus=demand_by_bus,
        demand_by_load=demand_by_load,
        renewable_by_bus=renewable_by_bus,
        generator_availability=availability,
        candidate_lines=list(dataset.candidate_lines),
        renewable_carriers=dataset.renewable_carriers,
        candidate_line_scores=dataset.candidate_line_scores,
        scenario_metadata=applied_metadata,
    )
    fit_observation_scales(transformed)
    return transformed


def apply_future_scenario_from_manifest(dataset: TEPDataset, manifest: Mapping[str, Any]) -> TEPDataset:
    fitted = manifest.get("future_scenario")
    if not fitted:
        return dataset
    return apply_future_scenario(dataset, fitted)


def scenario_audit(dataset: TEPDataset) -> dict[str, Any]:
    """Return achieved split-level capacities and regional demand peaks."""
    metadata = dataset.scenario_metadata or {}
    scenario_id = str(metadata.get("scenario_id", "historical"))
    audit: dict[str, Any] = {"scenario_id": scenario_id, "snapshots": int(len(dataset.snapshots))}
    carrier = dataset.network.generators["carrier"].fillna("").astype(str).str.lower()
    audit["installed_capacity_mw"] = {
        name: float(dataset.network.generators.loc[carrier.eq(name), "p_nom"].sum())
        for name in ("onwind", "solar")
    }
    if metadata.get("bus_to_nuts2"):
        bus_to_region = pd.Series(metadata["bus_to_nuts2"], dtype=str)
        audit["regional_peak_load_mw"] = {
            region: _regional_peak(dataset, bus_to_region, region)
            for region in sorted(set(bus_to_region))
        }
        audit["national_peak_load_mw"] = float(dataset.demand_by_bus.sum(axis=1).max())
    return audit
