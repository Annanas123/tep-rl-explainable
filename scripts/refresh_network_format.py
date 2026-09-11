from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pypsa

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tep_rl.data import cleanup_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a PyPSA network with the current PyPSA version and re-export it to a fresh .nc file."
    )
    parser.add_argument("--input", required=True, help="Input PyPSA network (.nc)")
    parser.add_argument("--output", required=True, help="Output path for the refreshed network (.nc)")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Apply the repo's cleanup step before export to drop invalid component references.",
    )
    parser.add_argument(
        "--default-country",
        default="AT",
        help="Fallback country code used only when --cleanup is enabled and buses miss a country value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    network = pypsa.Network(input_path)
    if args.cleanup:
        network = cleanup_network(network, default_country=args.default_country)

    meta = dict(getattr(network, "meta", {}) or {})
    meta["refreshed_from"] = str(input_path.resolve())
    meta["refreshed_with_pypsa"] = pypsa.__version__
    meta["cleanup_applied"] = bool(args.cleanup)
    network.meta = meta

    output_path.parent.mkdir(parents=True, exist_ok=True)
    network.export_to_netcdf(output_path)

    summary = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "pypsa_version": pypsa.__version__,
        "buses": int(len(network.buses)),
        "lines": int(len(network.lines)),
        "generators": int(len(network.generators)),
        "loads": int(len(network.loads)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
