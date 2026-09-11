from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = (
    "config_snapshot.json",
    "run_log.jsonl",
    "history.pkl",
    "evaluation.json",
    "agent.pt",
)


def _load_last_event(run_log_path: Path) -> dict | None:
    if not run_log_path.exists():
        return None
    last_line = ""
    with run_log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    return json.loads(last_line) if last_line else None


def validate_results(results_root: Path) -> dict[str, object]:
    runs = []
    for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        present = {name: (run_dir / name).exists() for name in REQUIRED_FILES}
        last_event = _load_last_event(run_dir / "run_log.jsonl")
        run_ok = all(present.values()) and last_event is not None and last_event.get("event") == "run_end" and bool(last_event.get("success"))
        runs.append(
            {
                "run_dir": str(run_dir),
                "status": "ok" if run_ok else "incomplete",
                "files": present,
                "last_event": last_event,
            }
        )

    return {
        "results_root": str(results_root),
        "n_runs": len(runs),
        "n_complete": sum(1 for run in runs if run["status"] == "ok"),
        "n_incomplete": sum(1 for run in runs if run["status"] != "ok"),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate per-seed thesis experiment outputs.")
    parser.add_argument("results_root", help="Directory containing one subdirectory per run/seed.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    summary = validate_results(Path(args.results_root))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
