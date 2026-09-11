from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import pypsa


def _normalise_bus_countries(buses: pd.DataFrame, default_country: str) -> pd.DataFrame:
    normalised = buses.copy()
    if "country" not in normalised.columns:
        normalised["country"] = default_country
    normalised["country"] = normalised["country"].fillna(default_country).astype(str).str.upper()
    return normalised


def _sorted_bus_columns(columns: Iterable[str]) -> list[str]:
    def sort_key(column: str) -> tuple[int, str]:
        suffix = column[3:]
        if suffix.isdigit():
            return int(suffix), column
        return 999, column

    return sorted((column for column in columns if column.startswith("bus")), key=sort_key)


def _connected_buses(row: pd.Series, bus_columns: list[str]) -> list[str]:
    buses: list[str] = []
    for column in bus_columns:
        value = row.get(column)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            buses.append(text)
    return buses


def _subset_component_timeseries(network: pypsa.Network, component_name: str) -> None:
    if not hasattr(network, component_name) or not hasattr(network, f"{component_name}_t"):
        return

    component = getattr(network, component_name)
    dynamic = getattr(network, f"{component_name}_t")
    valid = set(component.index)

    for attribute in list(dynamic.keys()):
        value = dynamic[attribute]
        if isinstance(value, pd.DataFrame) and not value.empty:
            columns = [column for column in value.columns if column in valid]
            dynamic[attribute] = value.loc[:, columns].copy()


def _build_border_records(
    component_name: str,
    frame: pd.DataFrame,
    bus_columns: list[str],
    internal_buses: set[str],
    bus_countries: dict[str, str],
) -> tuple[list[str], list[dict[str, object]]]:
    internal_index: list[str] = []
    border_records: list[dict[str, object]] = []

    for name, row in frame.iterrows():
        buses = _connected_buses(row, bus_columns)
        if not buses:
            continue

        internal = [bus for bus in buses if bus in internal_buses]
        if len(internal) == len(buses):
            internal_index.append(name)
            continue

        if not internal:
            continue

        external = [bus for bus in buses if bus not in internal_buses]
        capacity_column = "s_nom" if "s_nom" in row.index else "p_nom" if "p_nom" in row.index else ""
        nominal_capacity = float(row.get(capacity_column, 0.0)) if capacity_column else 0.0
        external_countries = sorted({bus_countries.get(bus, "") for bus in external if bus_countries.get(bus, "")})

        border_records.append(
            {
                "component_type": component_name,
                "component_name": name,
                "internal_buses": "|".join(internal),
                "external_buses": "|".join(external),
                "external_countries": "|".join(external_countries),
                "carrier": str(row.get("carrier", "")),
                "capacity_column": capacity_column,
                "nominal_capacity": nominal_capacity,
                "length": float(row.get("length", 0.0)) if "length" in row.index and pd.notna(row.get("length")) else 0.0,
                "bus0": str(row.get("bus0", "")),
                "bus1": str(row.get("bus1", "")),
            }
        )

    return internal_index, border_records


def _subset_bus_component(frame: pd.DataFrame, bus_column: str, internal_buses: set[str]) -> list[str]:
    if bus_column not in frame.columns:
        return list(frame.index)
    return list(frame.index[frame[bus_column].isin(internal_buses)])


def extract_country_subnetwork(
    network: pypsa.Network,
    country: str = "AT",
    default_country: str | None = None,
) -> tuple[pypsa.Network, pd.DataFrame, dict[str, object]]:
    country_code = country.upper()
    bus_frame = _normalise_bus_countries(network.buses, default_country or country_code)
    bus_country_map = bus_frame["country"].to_dict()

    country_buses = list(bus_frame.index[bus_frame["country"] == country_code])
    if not country_buses:
        raise ValueError(f"No buses found for country {country_code!r}.")

    internal_buses = set(country_buses)
    border_records: list[dict[str, object]] = []
    border_component_counts: dict[str, int] = {}

    line_index, line_borders = _build_border_records(
        "lines",
        network.lines,
        ["bus0", "bus1"],
        internal_buses,
        bus_country_map,
    )
    border_records.extend(line_borders)
    border_component_counts["lines"] = len(line_borders)

    transformer_index, transformer_borders = _build_border_records(
        "transformers",
        network.transformers,
        ["bus0", "bus1"],
        internal_buses,
        bus_country_map,
    )
    border_records.extend(transformer_borders)
    border_component_counts["transformers"] = len(transformer_borders)

    link_bus_columns = _sorted_bus_columns(network.links.columns)
    link_index, link_borders = _build_border_records(
        "links",
        network.links,
        link_bus_columns,
        internal_buses,
        bus_country_map,
    )
    border_records.extend(link_borders)
    border_component_counts["links"] = len(link_borders)

    generator_index = _subset_bus_component(network.generators, "bus", internal_buses)
    load_index = _subset_bus_component(network.loads, "bus", internal_buses)
    storage_index = _subset_bus_component(network.storage_units, "bus", internal_buses)
    store_index = _subset_bus_component(network.stores, "bus", internal_buses)
    shunt_index = _subset_bus_component(network.shunt_impedances, "bus", internal_buses)

    sub = network.copy()
    sub.buses = bus_frame.loc[country_buses].copy()
    sub.lines = network.lines.loc[line_index].copy()
    sub.transformers = network.transformers.loc[transformer_index].copy()
    sub.links = network.links.loc[link_index].copy()
    sub.generators = network.generators.loc[generator_index].copy()
    sub.loads = network.loads.loc[load_index].copy()
    sub.storage_units = network.storage_units.loc[storage_index].copy()
    sub.stores = network.stores.loc[store_index].copy()
    sub.shunt_impedances = network.shunt_impedances.loc[shunt_index].copy()

    for component_name in (
        "lines",
        "transformers",
        "links",
        "generators",
        "loads",
        "storage_units",
        "stores",
        "shunt_impedances",
    ):
        _subset_component_timeseries(sub, component_name)

    border_frame = pd.DataFrame(border_records)
    if border_frame.empty:
        border_frame = pd.DataFrame(
            columns=[
                "component_type",
                "component_name",
                "internal_buses",
                "external_buses",
                "external_countries",
                "carrier",
                "capacity_column",
                "nominal_capacity",
                "length",
                "bus0",
                "bus1",
            ]
        )
    else:
        border_frame = border_frame.sort_values(["component_type", "component_name"]).reset_index(drop=True)

    summary = {
        "country": country_code,
        "counts": {
            "buses": len(sub.buses),
            "lines": len(sub.lines),
            "transformers": len(sub.transformers),
            "links": len(sub.links),
            "generators": len(sub.generators),
            "loads": len(sub.loads),
            "storage_units": len(sub.storage_units),
            "stores": len(sub.stores),
            "shunt_impedances": len(sub.shunt_impedances),
        },
        "border_component_counts": border_component_counts,
        "neighbour_countries": sorted(
            {
                country_name
                for entry in border_frame["external_countries"].tolist()
                for country_name in str(entry).split("|")
                if country_name
            }
        ),
    }
    return sub, border_frame, summary


def export_country_subnetwork(
    input_path: Path | str,
    output_path: Path | str,
    country: str = "AT",
    border_output_path: Path | str | None = None,
    summary_output_path: Path | str | None = None,
    default_country: str | None = None,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    border_output_path = Path(border_output_path) if border_output_path else output_path.with_name(f"{output_path.stem}_border_interfaces.csv")
    summary_output_path = Path(summary_output_path) if summary_output_path else output_path.with_name(f"{output_path.stem}_summary.json")

    network = pypsa.Network(input_path)
    subnet, border_frame, summary = extract_country_subnetwork(
        network=network,
        country=country,
        default_country=default_country,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    border_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    subnet.export_to_netcdf(output_path)
    border_frame.to_csv(border_output_path, index=False)

    summary_payload = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "border_output_path": str(border_output_path),
        "summary_output_path": str(summary_output_path),
        **summary,
    }
    summary_output_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_payload
