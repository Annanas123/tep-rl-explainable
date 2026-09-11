"""
experiment_logger.py - Structured logging for reproducible RL experiments.

Each run gets its own ``ExperimentLogger`` instance that writes:
  * A machine-readable ``run_log.jsonl`` (one JSON object per event)
  * Human-readable ``run.log`` via Python's ``logging`` module
  * A frozen ``config_snapshot.json`` at the start of every run

The JSONL format makes downstream aggregation (e.g. pandas read_json) trivial.

Usage
-----
    from tep_rl.experiment_logger import ExperimentLogger

    with ExperimentLogger(output_dir=run_dir, run_id=run_id) as log:
        log.log_config(config_dict)
        for update in training_loop():
            log.log_update(update)
        log.log_evaluation(eval_results)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    """Fallback serializer for types not natively supported by json."""
    if hasattr(value, "tolist"):          # numpy arrays / tensors
        return value.tolist()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ExperimentLogger:
    """
    Context-manager logger that writes structured experiment records.

    Parameters
    ----------
    output_dir:
        Directory where log files are written.  Created if it does not exist.
    run_id:
        Human-readable identifier for this run (see ``get_run_id``).
    console_level:
        Verbosity of the Python logger attached to this run.
    """

    def __init__(
        self,
        output_dir: Path,
        run_id: str,
        console_level: int = logging.INFO,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self._start_time: float = 0.0
        self._jsonl_path: Path = self.output_dir / "run_log.jsonl"
        self._log_path: Path = self.output_dir / "run.log"
        self._console_level = console_level
        self._file_handler: Optional[logging.FileHandler] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentLogger":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.perf_counter()

        # Attach a file handler to the root logger so all module-level
        # loggers write to this run's log file as well.
        self._file_handler = logging.FileHandler(self._log_path, encoding="utf-8")
        self._file_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
        )
        logging.getLogger().addHandler(self._file_handler)

        self._write({"event": "run_start", "run_id": self.run_id, "time": _now_iso()})
        logger.info("Run %s started - output dir: %s", self.run_id, self.output_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.perf_counter() - self._start_time
        self._write(
            {
                "event": "run_end",
                "run_id": self.run_id,
                "time": _now_iso(),
                "elapsed_seconds": round(elapsed, 2),
                "success": exc_type is None,
            }
        )
        if self._file_handler is not None:
            logging.getLogger().removeHandler(self._file_handler)
            self._file_handler.close()
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_config(self, config: dict[str, Any]) -> None:
        """Freeze and write the complete configuration for this run."""
        snapshot_path = self.output_dir / "config_snapshot.json"
        with snapshot_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, default=_json_default)
        self._write({"event": "config", "config": config})
        logger.info("Config snapshot written to %s", snapshot_path)

    def log_update(self, update: dict[str, Any]) -> None:
        """Record one PPO update step (losses, timesteps, optional eval)."""
        self._write({"event": "update", **update})

    def log_episode(self, episode: dict[str, Any]) -> None:
        """Record one completed training episode."""
        self._write({"event": "episode", **episode})

    def log_evaluation(self, evaluation: dict[str, Any], split: str = "val") -> None:
        """
        Record an evaluation result.

        Parameters
        ----------
        evaluation:
            Dict returned by ``evaluate_agent``.
        split:
            ``"val"`` for validation evaluations during training,
            ``"test"`` for the final held-out evaluation.
        """
        self._write({"event": f"eval_{split}", **evaluation})
        logger.info(
            "Eval [%s] cost=%.2f  renewable=%.6f  curtail=%.3f  shed=%.3f  stress=%.3f",
            split,
            evaluation.get("total_cost_mean", float("nan")),
            evaluation.get("renewable_share_mean", float("nan")),
            evaluation.get("renewable_curtailment_mean", float("nan")),
            evaluation.get("load_shedding_mean", float("nan")),
            evaluation.get("grid_stress_mean", float("nan")),
        )

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Record an arbitrary scalar metric."""
        record: dict[str, Any] = {"event": "metric", "name": name, "value": value}
        if step is not None:
            record["step"] = step
        self._write(record)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> None:
        """Append one JSON record to the JSONL log file."""
        with self._jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")
