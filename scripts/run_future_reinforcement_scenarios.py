from __future__ import annotations

"""Evaluate finished thesis policies under source-based future scenarios.

The script deliberately reuses the policies trained for the thesis 2040 case.
It is therefore a cross-scenario transfer/sensitivity experiment, not a literal
multi-period investment process.  Candidate corridors and observation scales
are frozen to the original 2040 training preprocessing so action semantics do
not change between scenario years.
"""

import argparse
import gc
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tep_rl.data import TEPDataset
from tep_rl.evaluation import evaluate_agent
from tep_rl.future_scenarios import (
    DEFAULT_NUTS2_BOUNDARIES,
    DEFAULT_SCENARIO_CATALOG,
    apply_future_scenario,
    fit_future_scenario,
    load_scenario_catalog,
    scenario_audit,
)
from tep_rl.line_metadata import line_metadata_frame
from tep_rl.ppo import load_agent
from thesis_pipeline_utils import build_dataset, build_env, read_json, write_json


DEFAULT_RESULTS_ROOT = ROOT / "results" / "thesis_final_v12_apg_nt2040"
DEFAULT_SCENARIOS = (
    "apg_tyndp_nt_2030",
    "apg_tyndp_nt_2035_midpoint",
    "apg_tyndp_nt_2040",
)
METRICS = (
    "total_cost_mean",
    "operating_cost_mean",
    "annualized_investment_cost_mean",
    "grid_stress_mean",
    "renewable_curtailment_mean",
    "renewable_share_mean",
    "load_shedding_mean",
    "slack_generation_mean",
    "emissions_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scenario-catalog", type=Path, default=DEFAULT_SCENARIO_CATALOG)
    parser.add_argument("--nuts2-boundaries", type=Path, default=DEFAULT_NUTS2_BOUNDARIES)
    parser.add_argument("--scenario-ids", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--env-modes", nargs="+", choices=("proxy", "full"), default=["proxy", "full"])
    parser.add_argument("--agents", nargs="+", choices=("ppo", "moppo"), default=["ppo", "moppo"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-policies", type=int, default=None)
    parser.add_argument("--run-dc", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _seed_from_selected(item: dict[str, Any]) -> int:
    match = re.search(r"seed(\d+)", str(item.get("run_dir", item.get("checkpoint", ""))))
    if not match:
        raise ValueError(f"Cannot recover seed from selected checkpoint record: {item}")
    return int(match.group(1))


def _selection_file(results_root: Path, agent: str) -> Path:
    return results_root / "stage9_full_environment" / f"selected_checkpoints_{agent}_fullenv_validation.json"


def _load_selected(results_root: Path, agent: str, max_policies: int | None) -> list[dict[str, Any]]:
    path = _selection_file(results_root, agent)
    if not path.exists():
        raise FileNotFoundError(f"Selected {agent.upper()} checkpoints not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty list in {path}")
    selected = sorted(payload, key=_seed_from_selected)
    selected = selected[:max_policies] if max_policies is not None else selected
    missing = [
        item.get("checkpoint")
        for item in selected
        if not Path(item.get("checkpoint", "")).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Selected {agent.upper()} checkpoints are missing: {missing}")
    return selected


def _freeze_original_preprocessing(dataset: TEPDataset, preprocessing: dict[str, Any]) -> None:
    candidates = [str(value) for value in preprocessing["candidate_lines"]]
    missing = sorted(set(candidates).difference(dataset.network.lines.index.astype(str)))
    if missing:
        raise ValueError(f"Original candidate corridors are absent from the reconstructed network: {missing}")
    dataset.candidate_lines = candidates
    dataset.demand_scale = pd.Series(preprocessing["demand_scale"], dtype=float).reindex(
        dataset.network.buses.index
    ).fillna(1.0)
    dataset.renewable_scale = pd.Series(preprocessing["renewable_scale"], dtype=float).reindex(
        dataset.network.buses.index
    ).fillna(1.0)
    dataset.total_demand_scale = float(preprocessing["total_demand_scale"])


def _environment_kwargs(pipeline_manifest: dict[str, Any]) -> dict[str, Any]:
    env = pipeline_manifest["environment"]
    return {
        "episode_length": int(env["episode_length"]),
        "max_upgrade_mw": float(env["max_upgrade_mw"]),
        "budget_mw": float(env["budget_mw"]),
        "decision_interval": int(env["decision_interval"]),
        "temporal_mode": str(env["temporal_mode"]),
        "budget_release": str(env["budget_release"]),
        "action_mode": str(env["action_mode"]),
        "allocation_sharpness": float(env["allocation_sharpness"]),
        "allocation_sparsity_cutoff": float(env["allocation_sparsity_cutoff"]),
        "stability_margin": float(env["stability_margin"]),
        "proxy_balance_mode": str(env["proxy_balance_mode"]),
        "proxy_dispatch_limit": env.get("proxy_dispatch_limit"),
        "load_shedding_cost": float(env["load_shedding_cost"]),
        "line_investment_cost_eur_per_mw_km_year": float(
            env["line_investment_cost_eur_per_mw_km_year"]
        ),
        "cost_reward_scale": float(env["cost_reward_scale"]),
        "overload_reward_scale": float(env["overload_reward_scale"]),
        "third_objective_mode": str(env["third_objective_mode"]),
        "curtailment_reward_scale": float(env["curtailment_reward_scale"]),
        "emissions_reward_scale": float(env["emissions_reward_scale"]),
        "solver": str(env["solver"]),
        "full_env_fallback_to_proxy": False,
    }


def _base_datasets(pipeline_manifest: dict[str, Any]) -> tuple[TEPDataset, TEPDataset]:
    network = _resolve_project_path(pipeline_manifest["network"])
    load = _resolve_project_path(pipeline_manifest["load"])
    wind = _resolve_project_path(pipeline_manifest["wind"])
    solar = _resolve_project_path(pipeline_manifest["solar"])
    train_start, train_end = pipeline_manifest["splits"]["train"]
    test_start, test_end = pipeline_manifest["splits"]["test"]
    train = build_dataset(network, load, wind, solar, train_start, train_end, candidate_lines=None)
    test = build_dataset(network, load, wind, solar, test_start, test_end, candidate_lines=None)
    return train, test


def _scenario_dataset_and_manifest(
    scenario_id: str,
    train_base: TEPDataset,
    test_base: TEPDataset,
    pipeline_manifest: dict[str, Any],
    preprocessing: dict[str, Any],
    catalog_path: Path,
    boundaries: Path,
    output_dir: Path,
) -> tuple[TEPDataset, dict[str, Any]]:
    calibration = fit_future_scenario(
        train_base,
        scenario_id,
        catalog_path=catalog_path,
        boundary_path=boundaries,
    )
    calibration["input_profiles"] = {
        "load": pipeline_manifest["load"],
        "wind_capacity_factor": pipeline_manifest["wind"],
        "solar_capacity_factor": pipeline_manifest["solar"],
    }
    transformed = apply_future_scenario(test_base, calibration)
    _freeze_original_preprocessing(transformed, preprocessing)

    scenario_manifest = dict(preprocessing)
    scenario_manifest["future_scenario"] = calibration
    scenario_manifest["future_scenario_audit"] = {"test": scenario_audit(transformed)}
    scenario_manifest["transfer_evaluation"] = {
        "training_scenario_id": pipeline_manifest["future_scenario"]["requested_id"],
        "evaluation_scenario_id": scenario_id,
        "candidate_corridors": "frozen_from_original_2040_training_preprocessing",
        "observation_scales": "frozen_from_original_2040_training_preprocessing",
        "interpretation": (
            "Cross-scenario transfer/sensitivity evaluation of policies trained for the 2040 thesis case; "
            "not an independently trained target-year policy or a cumulative multi-period build schedule."
        ),
    }
    manifest_path = output_dir / "manifests" / f"{scenario_id}_preprocessing_manifest.json"
    write_json(manifest_path, scenario_manifest)
    return transformed, scenario_manifest


def _original_evaluation(
    results_root: Path,
    env_mode: str,
    agent: str,
    selected: dict[str, Any],
    episodes: int,
) -> tuple[dict[str, Any] | None, Path | None]:
    run_id = Path(selected["run_dir"]).name
    stage = "stage8_proxy_evaluation" if env_mode == "proxy" else "stage9_full_environment"
    prefix = results_root / stage
    if env_mode == "proxy":
        candidate = prefix / "proxy" / agent / run_id / "test_evaluation.json"
    else:
        candidate = prefix / agent / run_id / "test_evaluation.json"
    if not candidate.exists():
        return None, None
    evaluation = read_json(candidate)
    if int(evaluation.get("n_episodes", -1)) != int(episodes):
        return None, None
    return evaluation, candidate


def _metric(evaluation: dict[str, Any], name: str) -> float:
    value = evaluation.get(name)
    if value is not None:
        return float(value)
    if name == "annualized_investment_cost_mean":
        return float(evaluation.get("investment_cost_mean", np.nan)) * 365.0
    return float("nan")


def _evaluate_one(
    dataset: TEPDataset,
    selected: dict[str, Any],
    agent_name: str,
    env_mode: str,
    scenario_id: str,
    scenario_year: int,
    env_kwargs: dict[str, Any],
    episodes: int,
    device: str,
    output_dir: Path,
    original_results_root: Path,
    reuse_original_2040: bool,
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = _seed_from_selected(selected)
    run_id = Path(selected["run_dir"]).name
    destination = output_dir / "rl" / env_mode / scenario_id / agent_name / run_id
    evaluation_path = destination / "test_evaluation.json"
    metadata_path = destination / "evaluation_metadata.json"
    if not force and evaluation_path.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        cached = read_json(evaluation_path)
        if (
            metadata.get("checkpoint") == selected["checkpoint"]
            and metadata.get("scenario_id") == scenario_id
            and metadata.get("env_mode") == env_mode
            and int(cached.get("n_episodes", -1)) == int(episodes)
        ):
            return cached, metadata

    reused_from: Path | None = None
    evaluation: dict[str, Any] | None = None
    if reuse_original_2040 and not force:
        evaluation, reused_from = _original_evaluation(
            original_results_root, env_mode, agent_name, selected, episodes
        )
    if evaluation is None:
        local_kwargs = dict(env_kwargs)
        local_kwargs["seed"] = seed
        env = build_env(dataset, env_mode=env_mode, **local_kwargs)
        agent = load_agent(selected["checkpoint"], device=device)
        preferences = selected.get("eval_preference_weights")
        if hasattr(agent, "set_eval_preferences") and preferences is not None:
            agent.set_eval_preferences(preferences)
        evaluation = evaluate_agent(agent, env, episodes=episodes, deterministic=True)
        del env
        del agent
        gc.collect()

    metadata = {
        "scenario_id": scenario_id,
        "scenario_year": scenario_year,
        "env_mode": env_mode,
        "agent": agent_name,
        "seed": seed,
        "checkpoint": selected["checkpoint"],
        "eval_preference_weights": selected.get("eval_preference_weights"),
        "n_episodes": int(episodes),
        "reused_from": str(reused_from) if reused_from else None,
        "training_scenario_id": "apg_tyndp_nt_2040",
        "interpretation": "2040-trained policy evaluated under a target-year input scenario.",
    }
    write_json(destination / "selection.json", selected)
    write_json(evaluation_path, evaluation)
    write_json(metadata_path, metadata)
    return evaluation, metadata


def _aggregate(
    evaluations: list[tuple[dict[str, Any], dict[str, Any]]],
    network,
    output_dir: Path,
) -> None:
    seed_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    for evaluation, metadata in evaluations:
        row = {key: metadata[key] for key in ("scenario_id", "scenario_year", "env_mode", "agent", "seed")}
        row["n_episodes"] = int(evaluation["n_episodes"])
        row["total_investment_mean"] = float(
            evaluation.get("action_stats", {}).get("total_investment_mean", np.nan)
        )
        row["active_lines_mean"] = float(
            sum(float(value) > 1e-6 for value in evaluation.get("line_upgrades", {}).values())
        )
        for metric in METRICS:
            row[metric] = _metric(evaluation, metric)
        seed_rows.append(row)
        for line, value in evaluation.get("line_upgrades", {}).items():
            line_rows.append(
                {
                    **{
                        key: row[key]
                        for key in ("scenario_id", "scenario_year", "env_mode", "agent", "seed")
                    },
                    "line": str(line),
                    "upgrade_mw": float(value),
                }
            )

    seed_frame = pd.DataFrame(seed_rows).sort_values(
        ["env_mode", "scenario_year", "agent", "seed"]
    )
    line_seed = pd.DataFrame(line_rows)
    seed_frame.to_csv(output_dir / "policy_seed_records.csv", index=False)
    line_seed.to_csv(output_dir / "line_seed_records.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    value_columns = ["total_investment_mean", "active_lines_mean", *METRICS]
    for keys, frame in seed_frame.groupby(["scenario_id", "scenario_year", "env_mode", "agent"], sort=True):
        row = dict(zip(("scenario_id", "scenario_year", "env_mode", "agent"), keys))
        row["n_policies"] = int(len(frame))
        row["episodes_per_policy"] = int(frame["n_episodes"].iloc[0])
        for column in value_columns:
            values = frame[column].to_numpy(dtype=float)
            row[column] = float(np.nanmean(values))
            row[f"{column}_std_across_seeds"] = (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(row)
    pd.DataFrame(summary_rows).sort_values(["env_mode", "scenario_year", "agent"]).to_csv(
        output_dir / "policy_summary.csv", index=False
    )

    if line_seed.empty:
        pd.DataFrame().to_csv(output_dir / "line_summary.csv", index=False)
        return
    line_summary = (
        line_seed.groupby(
            ["scenario_id", "scenario_year", "env_mode", "agent", "line"], sort=True
        )["upgrade_mw"]
        .agg(
            mean_upgrade_mw="mean",
            std_upgrade_mw="std",
            median_upgrade_mw="median",
            min_upgrade_mw="min",
            max_upgrade_mw="max",
            seed_count="count",
            selected_seed_count=lambda values: int((values > 1e-6).sum()),
        )
        .reset_index()
    )
    line_summary["std_upgrade_mw"] = line_summary["std_upgrade_mw"].fillna(0.0)
    line_summary["selection_frequency"] = line_summary["selected_seed_count"] / line_summary["seed_count"]
    metadata = line_metadata_frame(network, line_summary["line"].drop_duplicates())
    line_summary = line_summary.merge(metadata, on="line", how="left")
    line_summary.to_csv(output_dir / "line_summary.csv", index=False)
    (
        line_summary.sort_values(
            ["env_mode", "scenario_year", "agent", "mean_upgrade_mw"],
            ascending=[True, True, True, False],
        )
        .groupby(["scenario_id", "scenario_year", "env_mode", "agent"], sort=False)
        .head(10)
        .to_csv(output_dir / "top_corridors.csv", index=False)
    )


def _collect_cached_evaluations(output_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    collected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for metadata_path in sorted((output_dir / "rl").glob("*/*/*/*/evaluation_metadata.json")):
        evaluation_path = metadata_path.with_name("test_evaluation.json")
        if evaluation_path.exists():
            collected.append((read_json(evaluation_path), read_json(metadata_path)))
    return collected


def _completed_evaluation_groups(
    evaluations: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = [
        {
            "scenario_id": metadata["scenario_id"],
            "scenario_year": int(metadata["scenario_year"]),
            "env_mode": metadata["env_mode"],
            "agent": metadata["agent"],
            "seed": int(metadata["seed"]),
            "episodes": int(evaluation["n_episodes"]),
        }
        for evaluation, metadata in evaluations
    ]
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    groups: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["scenario_id", "scenario_year", "env_mode", "agent", "episodes"], sort=True
    ):
        scenario_id, scenario_year, env_mode, agent, episodes = keys
        groups.append(
            {
                "scenario_id": str(scenario_id),
                "scenario_year": int(scenario_year),
                "env_mode": str(env_mode),
                "agent": str(agent),
                "episodes_per_policy": int(episodes),
                "n_policies": int(group["seed"].nunique()),
                "seeds": sorted(int(seed) for seed in group["seed"].unique()),
            }
        )
    return groups


def _reuse_original_dc(
    results_root: Path,
    destination: Path,
    scenario_id: str,
    episodes: int,
    force: bool,
) -> bool:
    if scenario_id != "apg_tyndp_nt_2040" or episodes != 10 or force:
        return False
    source = results_root / "dc_tep_gate_main"
    summary_path = source / "dc_tep_summary.json"
    if not summary_path.exists():
        return False
    summary = read_json(summary_path)
    if summary.get("future_scenario", {}).get("scenario_id") != scenario_id:
        return False
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob("dc_tep_*"):
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    write_json(
        destination / "reuse_metadata.json",
        {"reused_from": str(source), "reason": "identical 2040 scenario, formulation, and ten-window design"},
    )
    return True


def _run_dc(
    args: argparse.Namespace,
    pipeline_manifest: dict[str, Any],
    scenario_manifests: dict[str, Path],
    output_dir: Path,
) -> None:
    for scenario_id, manifest_path in scenario_manifests.items():
        destination = output_dir / "dc" / scenario_id
        existing = destination / "dc_tep_summary.json"
        if existing.exists() and not args.force:
            continue
        if _reuse_original_dc(args.results_root, destination, scenario_id, args.episodes, args.force):
            continue
        command = [
            sys.executable,
            str(ROOT / "scripts" / "solve_dc_tep_baseline.py"),
            "--network",
            str(_resolve_project_path(pipeline_manifest["network"])),
            "--load",
            str(_resolve_project_path(pipeline_manifest["load"])),
            "--wind",
            str(_resolve_project_path(pipeline_manifest["wind"])),
            "--solar",
            str(_resolve_project_path(pipeline_manifest["solar"])),
            "--start",
            str(pipeline_manifest["splits"]["test"][0]),
            "--end",
            str(pipeline_manifest["splits"]["test"][1]),
            "--preprocessing-manifest",
            str(manifest_path),
            "--episodes",
            str(args.episodes),
            "--output-dir",
            str(destination),
        ]
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.max_policies is not None and args.max_policies < 1:
        raise ValueError("--max-policies must be at least 1 when supplied")
    args.results_root = args.results_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.results_root / "future_reinforcement_scenarios"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_manifest = read_json(args.results_root / "pipeline_manifest.json")
    preprocessing = read_json(args.results_root / "preprocessing_manifest.json")
    catalog = load_scenario_catalog(args.scenario_catalog)
    missing_scenarios = sorted(set(args.scenario_ids).difference(catalog["scenarios"]))
    if missing_scenarios:
        raise ValueError(f"Unknown scenario identifiers: {missing_scenarios}")
    if pipeline_manifest["future_scenario"]["requested_id"] != "apg_tyndp_nt_2040":
        raise ValueError("This transfer pipeline expects the completed 2040 thesis run as its policy source.")

    train_base, test_base = _base_datasets(pipeline_manifest)
    env_kwargs = _environment_kwargs(pipeline_manifest)
    scenario_datasets: dict[str, TEPDataset] = {}
    scenario_manifests: dict[str, Path] = {}
    for scenario_id in args.scenario_ids:
        dataset, _ = _scenario_dataset_and_manifest(
            scenario_id,
            train_base,
            test_base,
            pipeline_manifest,
            preprocessing,
            args.scenario_catalog,
            args.nuts2_boundaries,
            output_dir,
        )
        scenario_datasets[scenario_id] = dataset
        scenario_manifests[scenario_id] = output_dir / "manifests" / f"{scenario_id}_preprocessing_manifest.json"

    if not args.skip_rl:
        selections = {
            agent: _load_selected(args.results_root, agent, args.max_policies) for agent in args.agents
        }
        for env_mode in args.env_modes:
            for scenario_id in args.scenario_ids:
                scenario = catalog["scenarios"][scenario_id]
                scenario_year = int(scenario["target_year"])
                for agent_name in args.agents:
                    for selected in selections[agent_name]:
                        print(
                            f"[{env_mode}] {scenario_year} {agent_name.upper()} seed {_seed_from_selected(selected):03d}",
                            flush=True,
                        )
                        _evaluate_one(
                            scenario_datasets[scenario_id],
                            selected,
                            agent_name,
                            env_mode,
                            scenario_id,
                            scenario_year,
                            env_kwargs,
                            args.episodes,
                            args.device,
                            output_dir,
                            args.results_root,
                            reuse_original_2040=scenario_id == "apg_tyndp_nt_2040",
                            force=args.force,
                        )
        _aggregate(_collect_cached_evaluations(output_dir), test_base.network, output_dir)

    if args.run_dc:
        _run_dc(args, pipeline_manifest, scenario_manifests, output_dir)

    cached_evaluations = _collect_cached_evaluations(output_dir)
    completed_evaluations = _completed_evaluation_groups(cached_evaluations)
    dc_scenarios_available = sorted(
        path.parent.name for path in (output_dir / "dc").glob("*/dc_tep_summary.json")
    )
    run_metadata = {
        "scope": "cross-scenario transfer evaluation of the completed 2040-trained thesis policies",
        "training_scenario_id": "apg_tyndp_nt_2040",
        "completed_evaluations": completed_evaluations,
        "evaluation_scenario_ids_available": sorted(
            {group["scenario_id"] for group in completed_evaluations}
        ),
        "env_modes_available": sorted({group["env_mode"] for group in completed_evaluations}),
        "agents_available": sorted({group["agent"] for group in completed_evaluations}),
        "dc_scenarios_available": dc_scenarios_available,
        "last_invocation": {
            "evaluation_scenario_ids": args.scenario_ids,
            "env_modes": args.env_modes,
            "agents": args.agents,
            "episodes_per_policy": args.episodes,
            "max_policies_per_agent": args.max_policies,
            "skip_rl": bool(args.skip_rl),
            "dc_reference_requested": bool(args.run_dc),
            "force": bool(args.force),
        },
        "candidate_corridors": "frozen from original 2040 training preprocessing",
        "observation_scales": "frozen from original 2040 training preprocessing",
        "limitations": [
            "The learned policies were trained only on the 2040 thesis scenario.",
            "The 2035 inputs are an arithmetic midpoint, not a published APG/ENTSO-E target year.",
            "Each target year is evaluated independently; reinforcements do not persist between years.",
            "Outputs are scenario-specific candidate reinforcements, not necessity findings or an official build schedule.",
        ],
    }
    write_json(output_dir / "run_metadata.json", run_metadata)

    if args.plot:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "plot_future_reinforcement_results.py"),
            "--results-dir",
            str(output_dir),
            "--network",
            str(_resolve_project_path(pipeline_manifest["network"])),
            "--scenario-catalog",
            str(args.scenario_catalog),
            "--nuts2-boundaries",
            str(args.nuts2_boundaries),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"Future reinforcement scenario outputs written to {output_dir}")


if __name__ == "__main__":
    main()
