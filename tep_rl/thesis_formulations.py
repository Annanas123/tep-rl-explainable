from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ThesisFormulationPreset:
    name: str
    candidate_lines: int
    max_upgrade_mw: float
    budget_mw: float
    description: str


SOURCE_BASED_V12 = ThesisFormulationPreset(
    name="source_based_v12",
    candidate_lines=60,
    max_upgrade_mw=500.0,
    budget_mw=500.0,
    description=(
        "Source-based future-stress formulation: 60 training-screened existing corridors, "
        "a 500 MW per-corridor cap and a 500 MW total envelope. The envelope is approximately "
        "one additional circuit at the smallest restored physical 220-kV rating (491.6 MW)."
    ),
)


THESIS_FORMULATION_PRESETS: dict[str, ThesisFormulationPreset] = {
    SOURCE_BASED_V12.name: SOURCE_BASED_V12,
}

DEFAULT_THESIS_FORMULATION_PRESET = SOURCE_BASED_V12.name


def get_thesis_formulation_preset(name: str) -> ThesisFormulationPreset:
    try:
        return THESIS_FORMULATION_PRESETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown thesis formulation preset {name!r}. "
            f"Choose one of {sorted(THESIS_FORMULATION_PRESETS)}."
        ) from exc


def _flag_present(argv: Iterable[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def apply_thesis_formulation_preset(args, argv: Iterable[str]):
    preset_name = getattr(args, "formulation_preset", DEFAULT_THESIS_FORMULATION_PRESET)
    preset = get_thesis_formulation_preset(str(preset_name))

    if not _flag_present(argv, "--candidate-lines"):
        args.candidate_lines = preset.candidate_lines
    if not _flag_present(argv, "--max-upgrade-mw"):
        args.max_upgrade_mw = preset.max_upgrade_mw
    if not _flag_present(argv, "--budget-mw"):
        args.budget_mw = preset.budget_mw

    setattr(args, "formulation_description", preset.description)
    return preset
