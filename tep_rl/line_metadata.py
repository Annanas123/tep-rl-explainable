from __future__ import annotations

import re
from pathlib import Path
from textwrap import fill
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
from math import cos, radians


CITY_ANCHORS = [
    {"name": "Vienna", "state": "Vienna", "lon": 16.3725, "lat": 48.2084},
    {"name": "St. Poelten", "state": "Lower Austria", "lon": 15.6230, "lat": 48.2048},
    {"name": "Linz", "state": "Upper Austria", "lon": 14.2858, "lat": 48.3069},
    {"name": "Graz", "state": "Styria", "lon": 15.4395, "lat": 47.0707},
    {"name": "Salzburg", "state": "Salzburg", "lon": 13.0550, "lat": 47.8095},
    {"name": "Innsbruck", "state": "Tyrol", "lon": 11.4041, "lat": 47.2692},
    {"name": "Klagenfurt", "state": "Carinthia", "lon": 14.3050, "lat": 46.6365},
    {"name": "Villach", "state": "Carinthia", "lon": 13.8500, "lat": 46.6150},
    {"name": "Bregenz", "state": "Vorarlberg", "lon": 9.7423, "lat": 47.5031},
    {"name": "Wiener Neustadt", "state": "Lower Austria", "lon": 16.2468, "lat": 47.8150},
    {"name": "Eisenstadt", "state": "Burgenland", "lon": 16.5270, "lat": 47.8457},
]


def parse_linestring(value: object) -> list[tuple[float, float]]:
    match = re.search(r"LINESTRING\s*\((.*)\)", str(value))
    if not match:
        return []
    coords: list[tuple[float, float]] = []
    for item in match.group(1).split(","):
        parts = item.strip().split()
        if len(parts) >= 2:
            coords.append((float(parts[0]), float(parts[1])))
    return coords


def draw_country_outline(
    ax: plt.Axes,
    network: pypsa.Network,
    country_code: str = "AT",
    facecolor: str = "#f4efe6",
    edgecolor: str = "0.35",
    alpha: float = 0.65,
    linewidth: float = 0.8,
) -> None:
    """Draw a country polygon from PyPSA-Eur shapes when available."""
    shapes = getattr(network, "shapes", pd.DataFrame())
    if shapes.empty or "geometry" not in shapes.columns:
        return
    mask = pd.Series(True, index=shapes.index)
    if "idx" in shapes.columns:
        mask &= shapes["idx"].astype(str).str.upper().eq(country_code.upper())
    if "type" in shapes.columns:
        mask &= shapes["type"].astype(str).str.lower().eq("country")
    selected = shapes.loc[mask]
    if selected.empty:
        return
    try:
        from shapely import wkt
    except Exception:
        return

    def _draw_polygon(poly) -> None:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth, zorder=0)
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, facecolor="white", edgecolor=edgecolor, alpha=1.0, linewidth=0.3, zorder=0)

    for geometry in selected["geometry"]:
        try:
            geom = wkt.loads(str(geometry))
        except Exception:
            continue
        if geom.geom_type == "Polygon":
            _draw_polygon(geom)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                _draw_polygon(poly)


def apply_geo_aspect(ax: plt.Axes, network: pypsa.Network) -> None:
    """Approximate distance-preserving aspect ratio for lon/lat maps over Austria."""
    try:
        mean_lat = float(network.buses["y"].mean())
    except Exception:
        return
    ax.set_aspect(1.0 / max(cos(radians(mean_lat)), 1e-6), adjustable="box")


def nearest_city_anchor(lon: float, lat: float) -> str:
    def _distance_sq(entry: dict[str, float]) -> float:
        return (float(entry["lon"]) - lon) ** 2 + (float(entry["lat"]) - lat) ** 2

    match = min(CITY_ANCHORS, key=_distance_sq)
    return str(match["name"])


def nearest_city_record(lon: float, lat: float) -> dict[str, object]:
    def _distance_sq(entry: dict[str, float]) -> float:
        return (float(entry["lon"]) - lon) ** 2 + (float(entry["lat"]) - lat) ** 2

    return min(CITY_ANCHORS, key=_distance_sq)


def point_anchor_label(lon: float, lat: float, precision: int = 2) -> str:
    match = nearest_city_record(lon, lat)
    return f"{match['name']} ({match['state']}; {lon:.{precision}f}E,{lat:.{precision}f}N)"


def _coord_label(lon: float, lat: float, precision: int = 2) -> str:
    return f"{lon:.{precision}f}E,{lat:.{precision}f}N"


def osm_way_url(line_name: str) -> str:
    if not line_name.startswith("way/"):
        return ""
    way_id = line_name.split("/")[1].split("-")[0]
    return f"https://www.openstreetmap.org/way/{way_id}"


def endpoint_label(network: pypsa.Network, line_name: str, precision: int = 2) -> str:
    if line_name not in network.lines.index:
        return line_name
    line = network.lines.loc[line_name]
    bus0 = network.buses.loc[line.bus0]
    bus1 = network.buses.loc[line.bus1]
    voltage = line.get("v_nom", "")
    prefix = f"{float(voltage):.0f} kV " if pd.notna(voltage) and str(voltage) != "" else ""
    start = f"{float(bus0.x):.{precision}f}E,{float(bus0.y):.{precision}f}N"
    end = f"{float(bus1.x):.{precision}f}E,{float(bus1.y):.{precision}f}N"
    return f"{prefix}{start} -> {end}"


def endpoint_anchor_label(network: pypsa.Network, line_name: str) -> str:
    if line_name not in network.lines.index:
        return line_name
    line = network.lines.loc[line_name]
    bus0 = network.buses.loc[line.bus0]
    bus1 = network.buses.loc[line.bus1]
    voltage = line.get("v_nom", "")
    prefix = f"{float(voltage):.0f} kV " if pd.notna(voltage) and str(voltage) != "" else ""
    start = nearest_city_anchor(float(bus0.x), float(bus0.y))
    end = nearest_city_anchor(float(bus1.x), float(bus1.y))
    if start == end:
        start = f"{start} ({_coord_label(float(bus0.x), float(bus0.y))})"
        end = f"{end} ({_coord_label(float(bus1.x), float(bus1.y))})"
    return f"{prefix}{start} -> {end}"


def bus_label(network: pypsa.Network, bus_name: str, precision: int = 2) -> str:
    if bus_name not in network.buses.index:
        return bus_name
    bus = network.buses.loc[bus_name]
    voltage = bus.get("v_nom", "")
    prefix = f"{float(voltage):.0f} kV " if pd.notna(voltage) and str(voltage) != "" else ""
    return f"{prefix}{float(bus.x):.{precision}f}E,{float(bus.y):.{precision}f}N"


def bus_anchor_label(network: pypsa.Network, bus_name: str, precision: int = 2) -> str:
    if bus_name not in network.buses.index:
        return bus_name
    bus = network.buses.loc[bus_name]
    voltage = bus.get("v_nom", "")
    prefix = f"{float(voltage):.0f} kV " if pd.notna(voltage) and str(voltage) != "" else ""
    match = nearest_city_record(float(bus.x), float(bus.y))
    return f"{prefix}{match['name']} ({match['state']})"


def line_metadata_records(network: pypsa.Network, lines: Iterable[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_name in lines:
        if line_name not in network.lines.index:
            continue
        line = network.lines.loc[line_name]
        bus0 = network.buses.loc[line.bus0]
        bus1 = network.buses.loc[line.bus1]
        records.append(
            {
                "line": line_name,
                "line_label": endpoint_label(network, line_name),
                "line_anchor_label": endpoint_anchor_label(network, line_name),
                "voltage_kv": line.get("v_nom", np.nan),
                "bus0": line.bus0,
                "bus0_lon": bus0.get("x", np.nan),
                "bus0_lat": bus0.get("y", np.nan),
                "bus0_anchor": point_anchor_label(float(bus0.get("x", np.nan)), float(bus0.get("y", np.nan))),
                "bus1": line.bus1,
                "bus1_lon": bus1.get("x", np.nan),
                "bus1_lat": bus1.get("y", np.nan),
                "bus1_anchor": point_anchor_label(float(bus1.get("x", np.nan)), float(bus1.get("y", np.nan))),
                "length_km": line.get("length", np.nan),
                "base_capacity_mw": line.get("s_nom", np.nan),
                "osm_way_url": osm_way_url(line_name),
            }
        )
    return records


def line_metadata_frame(network: pypsa.Network, lines: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(line_metadata_records(network, lines))


def anchor_points_for_lines(
    network: pypsa.Network,
    lines: Iterable[str],
    max_unique_anchors: int | None = None,
) -> list[dict[str, float | str]]:
    points: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for line_name in lines:
        if line_name not in network.lines.index:
            continue
        line = network.lines.loc[line_name]
        for bus_name in [line.bus0, line.bus1]:
            if bus_name not in network.buses.index:
                continue
            bus = network.buses.loc[bus_name]
            label = nearest_city_anchor(float(bus.x), float(bus.y))
            if label in seen:
                continue
            seen.add(label)
            points.append({"label": label, "lon": float(bus.x), "lat": float(bus.y)})
            if max_unique_anchors is not None and len(points) >= max_unique_anchors:
                return points
    return points


def annotate_anchor_points(
    ax: plt.Axes,
    anchor_points: Iterable[dict[str, float | str]],
    fontsize: int = 7,
) -> None:
    for item in anchor_points:
        lon = float(item["lon"])
        lat = float(item["lat"])
        label = str(item["label"])
        ax.scatter(lon, lat, s=20, color="black", edgecolor="white", linewidth=0.6, zorder=4)
        ax.annotate(
            label,
            (lon, lat),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=fontsize,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
            zorder=5,
        )


def feature_display_name(feature: str, network: pypsa.Network) -> str:
    global_features = {
        "progress_fraction": "Episode Progress",
        "available_budget_fraction": "Available Budget",
        "remaining_budget_fraction": "Remaining Budget",
        "spent_budget_fraction": "Used Budget",
    }
    if feature in global_features:
        return global_features[feature]
    if "::" not in feature:
        return feature.replace("_", " ").title()
    prefix, value = feature.split("::", 1)
    readable_prefix = {
        "upgrade": "Prior Upgrade State",
        "line_loading_raw": "Line Loading",
        "candidate_loading_raw": "Candidate-Line Loading",
        "demand": "Bus Demand",
        "renewable": "Renewable Availability",
    }.get(prefix, prefix.replace("_", " ").title())
    if value in network.lines.index:
        return f"{readable_prefix}: {endpoint_anchor_label(network, value)}"
    if value in network.buses.index:
        return f"{readable_prefix}: {bus_anchor_label(network, value)}"
    return feature


def enrich_feature_table(frame: pd.DataFrame, network: pypsa.Network) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["feature_raw"] = enriched["feature"]
    enriched["feature"] = [feature_display_name(feature, network) for feature in enriched["feature_raw"]]
    return enriched


def aggregate_line_importance(
    feature_importance: pd.DataFrame,
    network: pypsa.Network,
    value_col: str = "global_importance",
) -> pd.DataFrame:
    allowed_prefixes = {"upgrade", "line_loading_raw", "candidate_loading_raw"}
    rows: dict[str, dict[str, float]] = {}
    for _, row in feature_importance.iterrows():
        feature = str(row["feature"])
        if "::" not in feature:
            continue
        prefix, line_name = feature.split("::", 1)
        # Bus and line identifiers can share the same OSM-style text.  Only
        # features that semantically describe a physical line may contribute
        # to corridor importance; otherwise demand/renewable bus features can
        # be misclassified as line features by identifier coincidence.
        if prefix not in allowed_prefixes:
            continue
        if line_name not in network.lines.index:
            continue
        value = float(row.get(value_col, 0.0))
        record = rows.setdefault(
            line_name,
            {
                "line_importance": 0.0,
                "upgrade_importance": 0.0,
                "raw_loading_importance": 0.0,
                "candidate_loading_importance": 0.0,
            },
        )
        record["line_importance"] += value
        if prefix == "upgrade":
            record["upgrade_importance"] += value
        elif prefix == "line_loading_raw":
            record["raw_loading_importance"] += value
        elif prefix == "candidate_loading_raw":
            record["candidate_loading_importance"] += value

    metadata = line_metadata_frame(network, rows.keys())
    if metadata.empty:
        return metadata
    values = pd.DataFrame([{"line": line, **payload} for line, payload in rows.items()])
    return (
        metadata.merge(values, on="line", how="left")
        .fillna(0.0)
        .sort_values("line_importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_line_importance_map(
    network: pypsa.Network,
    line_importance: pd.DataFrame,
    output_path: Path | str,
    top_k: int = 8,
    value_col: str = "line_importance",
) -> None:
    if line_importance.empty:
        return
    output_path = Path(output_path)
    top = line_importance.sort_values(value_col, ascending=False).head(top_k)
    max_value = max(float(top[value_col].max()), 1e-12)

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    draw_country_outline(ax, network)
    for _, line in network.lines.iterrows():
        coords = parse_linestring(line.get("geometry", ""))
        if len(coords) >= 2:
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color="black", linewidth=0.45, alpha=0.38, zorder=1)

    highlight_colors = [
        "#e5a35c",
        "#6f9fcf",
        "#79b96f",
        "#3f8f86",
        "#b9c1ca",
        "#c98242",
        "#4f82b7",
        "#5f9f58",
        "#276b66",
        "#7f858c",
    ]
    for rank, (color, (_, row)) in enumerate(zip(highlight_colors, top.iterrows()), start=1):
        line_name = row["line"]
        if line_name not in network.lines.index:
            continue
        coords = parse_linestring(network.lines.loc[line_name].get("geometry", ""))
        if len(coords) < 2:
            continue
        xs, ys = zip(*coords)
        width = 1.2 + 4.8 * float(row[value_col]) / max_value
        corridor_label = fill(f"C{rank}: {endpoint_anchor_label(network, line_name)}", width=44, subsequent_indent="    ")
        ax.plot(xs, ys, color=color, linewidth=width, label=corridor_label, zorder=3)

    annotate_anchor_points(ax, anchor_points_for_lines(network, top["line"].tolist(), max_unique_anchors=6), fontsize=9)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    apply_geo_aspect(ax, network)
    ax.grid(alpha=0.25)
    ax.legend(
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        borderaxespad=0.0,
        ncol=2,
        columnspacing=1.2,
        handlelength=2.2,
        title="Top corridors",
    )
    fig.tight_layout(rect=(0.0, 0.30, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"}
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path, **save_kwargs)
    else:
        fig.savefig(output_path, dpi=300, **save_kwargs)
        fig.savefig(output_path.with_suffix(".pdf"), **save_kwargs)
    plt.close(fig)
