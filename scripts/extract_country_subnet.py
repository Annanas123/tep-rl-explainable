from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tep_rl.subnetwork import export_country_subnetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a country-specific PyPSA subnet and save border-interface metadata."
    )
    parser.add_argument("--input", required=True, help="Input PyPSA network (.nc)")
    parser.add_argument("--output", required=True, help="Output path for the extracted subnet (.nc)")
    parser.add_argument("--country", default="AT", help="Country code to extract, e.g. AT")
    parser.add_argument(
        "--border-output",
        default=None,
        help="Optional CSV path for removed cross-border interfaces. Defaults next to --output.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON path for extraction summary. Defaults next to --output.",
    )
    parser.add_argument(
        "--default-country",
        default=None,
        help="Fallback country code for networks whose buses do not have a country column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_country_subnetwork(
        input_path=Path(args.input),
        output_path=Path(args.output),
        country=args.country,
        border_output_path=Path(args.border_output) if args.border_output else None,
        summary_output_path=Path(args.summary_output) if args.summary_output else None,
        default_country=args.default_country,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
