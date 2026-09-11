from __future__ import annotations

"""Create a compact dataset and network report for the thesis case study."""

import argparse
from pathlib import Path

import pypsa

from thesis_pipeline_utils import build_dataset, network_summary_frame, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a thesis-ready dataset/network sanity report.")
    parser.add_argument("--network", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--wind", required=True)
    parser.add_argument("--solar", required=True)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--candidate-lines", type=int, default=None)
    parser.add_argument("--output-dir", default="results/stage2_dataset_sanity")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    network_path = Path(args.network)
    load_path = Path(args.load)
    wind_path = Path(args.wind)
    solar_path = Path(args.solar)

    network = pypsa.Network(network_path)
    dataset = build_dataset(
        network=network_path,
        load=load_path,
        wind=wind_path,
        solar=solar_path,
        start=args.start,
        end=args.end,
        candidate_lines=args.candidate_lines,
    )

    demand_totals = dataset.demand_by_bus.sum(axis=1)
    renewable_totals = dataset.renewable_by_bus.sum(axis=1)
    generator_capacities = (
        network.generators.assign(carrier=network.generators["carrier"].astype(str))
        .groupby("carrier")["p_nom"]
        .sum()
        .sort_values(ascending=False)
    )

    summary = {
        "network_path": str(network_path),
        "load_path": str(load_path),
        "wind_path": str(wind_path),
        "solar_path": str(solar_path),
        "time_window": {
            "start": str(dataset.snapshots.min()),
            "end": str(dataset.snapshots.max()),
            "n_snapshots": int(len(dataset.snapshots)),
        },
        "network_counts": {
            "buses": int(len(network.buses)),
            "lines": int(len(network.lines)),
            "transformers": int(len(network.transformers)),
            "generators": int(len(network.generators)),
            "loads": int(len(network.loads)),
            "storage_units": int(len(network.storage_units)),
        },
        "candidate_lines": {
            "count": int(len(dataset.candidate_lines)),
            "sample": dataset.candidate_lines[:10],
        },
        "demand_mw": {
            "mean": float(demand_totals.mean()),
            "max": float(demand_totals.max()),
            "min": float(demand_totals.min()),
        },
        "renewable_potential_mw": {
            "mean": float(renewable_totals.mean()),
            "max": float(renewable_totals.max()),
            "min": float(renewable_totals.min()),
        },
        "generator_capacity_by_carrier_mw": {
            str(carrier): float(value) for carrier, value in generator_capacities.items()
        },
        "line_capacity_mw": {
            "mean": float(network.lines["s_nom"].fillna(0.0).mean()),
            "median": float(network.lines["s_nom"].fillna(0.0).median()),
            "max": float(network.lines["s_nom"].fillna(0.0).max()),
            "min": float(network.lines["s_nom"].fillna(0.0).min()),
        },
    }

    write_json(output_dir / "dataset_summary.json", summary)
    network_summary_frame(network).to_csv(output_dir / "network_component_counts.csv", index=False)

    markdown_lines = [
        "# Dataset And Network Summary",
        "",
        f"- Network: `{network_path}`",
        f"- Load: `{load_path}`",
        f"- Wind: `{wind_path}`",
        f"- Solar: `{solar_path}`",
        f"- Snapshots: `{summary['time_window']['n_snapshots']}`",
        f"- Coverage: `{summary['time_window']['start']}` to `{summary['time_window']['end']}`",
        f"- Candidate lines: `{summary['candidate_lines']['count']}`",
        "",
        "## Network Counts",
        "",
    ]
    for component, count in summary["network_counts"].items():
        markdown_lines.append(f"- {component}: `{count}`")
    markdown_lines.extend(
        [
            "",
            "## Aggregate Power Levels",
            "",
            f"- Mean demand: `{summary['demand_mw']['mean']:.2f}` MW",
            f"- Peak demand: `{summary['demand_mw']['max']:.2f}` MW",
            f"- Mean renewable potential: `{summary['renewable_potential_mw']['mean']:.2f}` MW",
            f"- Peak renewable potential: `{summary['renewable_potential_mw']['max']:.2f}` MW",
            "",
            "## Generator Capacity By Carrier",
            "",
        ]
    )
    for carrier, value in summary["generator_capacity_by_carrier_mw"].items():
        markdown_lines.append(f"- {carrier}: `{value:.2f}` MW")

    (output_dir / "dataset_summary.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"Wrote dataset summary to {output_dir}")


if __name__ == "__main__":
    main()
