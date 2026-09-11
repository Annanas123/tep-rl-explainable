from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pypsa

from tep_rl.line_metadata import endpoint_label, osm_way_url, parse_linestring


def _row_for_line(network: pypsa.Network, line_name: str) -> dict[str, object]:
    if line_name not in network.lines.index:
        matches = [name for name in network.lines.index if line_name in str(name)]
        raise KeyError(f"Line {line_name!r} not found. Partial matches: {matches[:10]}")

    line = network.lines.loc[line_name]
    bus0 = str(line.bus0)
    bus1 = str(line.bus1)
    bus0_row = network.buses.loc[bus0]
    bus1_row = network.buses.loc[bus1]
    return {
        "line": line_name,
        "tags": line.get("tags", ""),
        "carrier": line.get("carrier", ""),
        "voltage_kv": line.get("v_nom", ""),
        "type": line.get("type", ""),
        "bus0": bus0,
        "bus0_lon": bus0_row.get("x", ""),
        "bus0_lat": bus0_row.get("y", ""),
        "bus1": bus1,
        "bus1_lon": bus1_row.get("x", ""),
        "bus1_lat": bus1_row.get("y", ""),
        "length_km": line.get("length", ""),
        "capacity_mw": line.get("s_nom", ""),
        "thermal_limit_pu": line.get("s_max_pu", ""),
        "num_parallel": line.get("num_parallel", ""),
        "line_label": endpoint_label(network, line_name),
        "osm_way_url": osm_way_url(line_name),
    }


def _plot_line(network: pypsa.Network, line_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for _, line in network.lines.iterrows():
        coords = parse_linestring(line.get("geometry", ""))
        if len(coords) >= 2:
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color="0.75", linewidth=0.6, alpha=0.45)

    target = network.lines.loc[line_name]
    coords = parse_linestring(target.get("geometry", ""))
    if len(coords) >= 2:
        xs, ys = zip(*coords)
        ax.plot(xs, ys, color="#c43c39", linewidth=3.0, label=line_name)

    for bus_col, color in (("bus0", "#1f77b4"), ("bus1", "#2ca02c")):
        bus = str(target[bus_col])
        bus_row = network.buses.loc[bus]
        ax.scatter([bus_row.x], [bus_row.y], s=45, color=color, zorder=5)
        ax.annotate(bus, (bus_row.x, bus_row.y), xytext=(5, 5), textcoords="offset points", fontsize=7)

    ax.set_title(f"Network location of {line_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a PyPSA line and export human-readable metadata.")
    parser.add_argument("--network", default="derived/austria_net_physical_ratings.nc")
    parser.add_argument("--line", required=True)
    parser.add_argument("--output-dir", default="results/line_inspection")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    network = pypsa.Network(args.network)
    row = _row_for_line(network, args.line)

    csv_path = output_dir / f"{args.line.replace('/', '_')}_metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    png_path = output_dir / f"{args.line.replace('/', '_')}_map.png"
    _plot_line(network, args.line, png_path)

    print(f"Line metadata written to {csv_path}")
    print(f"Line map written to {png_path}")
    for key, value in row.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
