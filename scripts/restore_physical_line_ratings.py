"""Restore PyPSA-Eur line ratings in the model-ready Austrian case.

The previous stress-calibration step replaced every thermal rating with the
same synthetic value.  This script keeps the attached loads and generators
from the model-ready case, but restores ``s_nom`` from the unmodified Austrian
PyPSA-Eur extraction.  It writes a new file and never overwrites either input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pypsa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ready", default="derived/austria_net_ready.nc")
    parser.add_argument("--ratings-source", default="derived/austria_from_base.nc")
    parser.add_argument("--output", default="derived/austria_net_physical_ratings.nc")
    parser.add_argument("--report", default="derived/austria_net_physical_ratings_report.json")
    return parser.parse_args()


def _components(network: pypsa.Network) -> list[list[str]]:
    adjacency = {str(bus): set() for bus in network.buses.index}
    for table in (network.lines, network.transformers):
        for row in table.itertuples():
            bus0, bus1 = str(row.bus0), str(row.bus1)
            adjacency[bus0].add(bus1)
            adjacency[bus1].add(bus0)

    unseen = set(adjacency)
    components: list[list[str]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        members: list[str] = []
        while stack:
            node = stack.pop()
            members.append(node)
            neighbours = adjacency[node].intersection(unseen)
            unseen.difference_update(neighbours)
            stack.extend(neighbours)
        components.append(sorted(members))
    return sorted(components, key=len, reverse=True)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_ready)
    source_path = Path(args.ratings_source)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if output_path.resolve() in {model_path.resolve(), source_path.resolve()}:
        raise ValueError("--output must differ from both input files.")

    network = pypsa.Network(model_path)
    ratings_source = pypsa.Network(source_path)
    missing = network.lines.index.difference(ratings_source.lines.index)
    extra = ratings_source.lines.index.difference(network.lines.index)
    if len(missing) or len(extra):
        raise ValueError(
            f"Line identifiers differ (missing={missing.tolist()}, extra={extra.tolist()})."
        )

    restored = ratings_source.lines["s_nom"].reindex(network.lines.index).astype(float)
    if restored.isna().any() or (restored <= 0.0).any():
        raise ValueError("The ratings source contains missing or non-positive s_nom values.")
    if np.isclose(float(restored.std(ddof=0)), 0.0):
        raise ValueError("The ratings source is uniform and cannot restore physical diversity.")

    previous = network.lines["s_nom"].astype(float).copy()
    network.lines.loc[:, "s_nom"] = restored
    network.meta = dict(getattr(network, "meta", {}) or {})
    network.meta.update(
        {
            "line_rating_provenance": str(source_path),
            "line_rating_correction": "Restored unmodified PyPSA-Eur extraction s_nom values",
            "case_scope": "synthetic isolated-Austria reinforcement benchmark; not an official expansion model",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    network.export_to_netcdf(output_path)

    components = _components(network)
    report = {
        "model_ready_input": str(model_path),
        "ratings_source": str(source_path),
        "output": str(output_path),
        "line_count": int(len(network.lines)),
        "transformer_count": int(len(network.transformers)),
        "previous_s_nom": previous.describe().to_dict(),
        "restored_s_nom": restored.describe().to_dict(),
        "connected_component_sizes_with_transformers": [len(item) for item in components],
        "isolated_components": [item for item in components if len(item) == 1],
        "cross_border_interfaces_modelled": False,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
