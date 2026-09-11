from __future__ import annotations

"""Central result-directory layout for thesis experiments."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentDirectory:
    """Describe one top-level result directory."""

    key: str
    preferred_name: str
    description: str


DATASET_SANITY = ExperimentDirectory(
    key="dataset_sanity",
    preferred_name="stage2_dataset_sanity",
    description="Dataset and network sanity report generated before training.",
)
PPO_VALIDATION = ExperimentDirectory(
    key="ppo_validation",
    preferred_name="stage4_ppo_validation",
    description="Multi-seed PPO validation training runs in the proxy environment.",
)
MOPPO_VALIDATION = ExperimentDirectory(
    key="moppo_validation",
    preferred_name="stage5_moppo_validation",
    description="Multi-seed MO-PPO validation training runs in the proxy environment.",
)
SCALARIZATION_SWEEP = ExperimentDirectory(
    key="scalarization_sweep",
    preferred_name="stage6_scalarization_sweep",
    description="Scalarised PPO weight sweep used for candidate-set trade-off analysis.",
)
BASELINE_BENCHMARK = ExperimentDirectory(
    key="baseline_benchmark",
    preferred_name="stage7_baseline_runtime",
    description="Runtime benchmark for deterministic baseline policies.",
)
PROXY_EVALUATION = ExperimentDirectory(
    key="proxy_evaluation",
    preferred_name="stage8_proxy_evaluation",
    description="Held-out proxy test evaluation for learned and baseline policies.",
)
FULLENV_EVALUATION = ExperimentDirectory(
    key="fullenv_evaluation",
    preferred_name="stage9_full_environment",
    description="Strict full-environment reranking and test evaluation.",
)
EXPLAINABILITY = ExperimentDirectory(
    key="explainability",
    preferred_name="stage9_explainability",
    description="Explainability outputs for the selected MO-PPO checkpoint.",
)
AGENT_COMPARISON = ExperimentDirectory(
    key="agent_comparison",
    preferred_name="stage8_9_agent_comparison",
    description="Cross-stage PPO versus MO-PPO comparison tables.",
)
PARETO_ARCHIVE = ExperimentDirectory(
    key="pareto_archive",
    preferred_name="supplement_pareto_archive",
    description="Supplementary scalarised PPO candidate set used for empirical Pareto analysis.",
)
PREFERENCE_RESPONSE = ExperimentDirectory(
    key="preference_response",
    preferred_name="supplement_moppo_preference_response",
    description="Supplementary conditioned MO-PPO preference-response diagnostic.",
)


EXPERIMENT_DIRECTORIES: dict[str, ExperimentDirectory] = {
    spec.key: spec
    for spec in (
        DATASET_SANITY,
        PPO_VALIDATION,
        MOPPO_VALIDATION,
        SCALARIZATION_SWEEP,
        BASELINE_BENCHMARK,
        PROXY_EVALUATION,
        FULLENV_EVALUATION,
        EXPLAINABILITY,
        AGENT_COMPARISON,
        PARETO_ARCHIVE,
        PREFERENCE_RESPONSE,
    )
}


def get_experiment_directory(key: str) -> ExperimentDirectory:
    """Return the registered directory specification for ``key``."""
    try:
        return EXPERIMENT_DIRECTORIES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown experiment directory key {key!r}.") from exc


def preferred_results_dir(results_root: Path, key: str) -> Path:
    """Return the descriptive directory path used for new outputs."""
    spec = get_experiment_directory(key)
    return Path(results_root) / spec.preferred_name


def locate_results_dir(results_root: Path, key: str) -> Path:
    """Return the configured result directory for ``key``."""
    return preferred_results_dir(results_root, key)
