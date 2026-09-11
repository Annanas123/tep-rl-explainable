from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tep_rl.baselines import build_baseline_agents
from tep_rl.config import EnvironmentConfig, NetworkConfig, PPOConfig
from tep_rl.data import load_austria_case
from tep_rl.envs import ProxyTEPEnv, PyPSATEPEnv
from tep_rl.evaluation import evaluate_agent
from tep_rl.future_scenarios import apply_future_scenario_from_manifest
from tep_rl.thesis_formulations import (
    DEFAULT_THESIS_FORMULATION_PRESET,
    THESIS_FORMULATION_PRESETS,
    apply_thesis_formulation_preset,
)


def _parse_number_tuple(text: str, cast) -> tuple:
    return tuple(cast(part) for part in re.split(r"[\s,]+", str(text).strip()) if part)


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark runtime of heuristic baselines in the proxy or full environment.")
    default_env = EnvironmentConfig()
    default_ppo = PPOConfig()
    parser.add_argument(
        "--formulation-preset",
        choices=sorted(THESIS_FORMULATION_PRESETS),
        default=DEFAULT_THESIS_FORMULATION_PRESET,
        help="Apply a shared thesis formulation preset unless the specific budget/candidate flags are overridden.",
    )
    parser.add_argument("--env", choices=["proxy", "full"], default="full")
    parser.add_argument("--network", default="derived/austria_net_physical_ratings.nc")
    parser.add_argument("--load", default="data/entsoe_at_load_2015_2024_opsd.csv")
    parser.add_argument("--wind", default="data/wind_at_2015_2024.csv")
    parser.add_argument("--solar", default="data/solar_at_2015_2024.csv")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--candidate-lines", type=int, default=60)
    parser.add_argument("--preprocessing-manifest", default=None)
    parser.add_argument("--episode-length", type=int, default=24)
    parser.add_argument("--max-upgrade-mw", type=float, default=60.0)
    parser.add_argument("--budget-mw", type=float, default=240.0)
    parser.add_argument("--decision-interval", type=int, default=6)
    parser.add_argument("--temporal-mode", choices=["decision_block", "hourly"], default=default_env.temporal_mode)
    parser.add_argument("--budget-release", choices=["linear", "all_at_once"], default=default_env.budget_release)
    parser.add_argument("--action-mode", choices=["budgeted", "direct"], default=default_env.action_mode)
    parser.add_argument("--allocation-sharpness", type=float, default=default_env.allocation_sharpness)
    parser.add_argument("--allocation-sparsity-cutoff", type=float, default=default_env.allocation_sparsity_cutoff)
    parser.add_argument("--stability-margin", type=float, default=default_env.stability_margin)
    parser.add_argument(
        "--proxy-balance-mode",
        choices=["demand_proportional", "single_slack"],
        default=default_env.proxy_balance_mode,
    )
    parser.add_argument("--proxy-dispatch-limit", type=float, default=default_env.proxy_dispatch_limit)
    parser.add_argument("--load-shedding-cost", type=float, default=default_env.load_shedding_cost)
    parser.add_argument(
        "--line-investment-cost-eur-per-mw-km-year",
        type=float,
        default=default_env.line_investment_cost_eur_per_mw_km_year,
    )
    parser.add_argument("--cost-reward-scale", type=float, default=default_env.cost_reward_scale)
    parser.add_argument("--overload-reward-scale", type=float, default=default_env.overload_reward_scale)
    parser.add_argument(
        "--third-objective-mode",
        choices=["renewable_share", "curtailment", "emissions"],
        default=default_env.third_objective_mode,
    )
    parser.add_argument("--curtailment-reward-scale", type=float, default=default_env.curtailment_reward_scale)
    parser.add_argument("--emissions-reward-scale", type=float, default=default_env.emissions_reward_scale)
    parser.add_argument("--solver", default="highs")
    parser.add_argument(
        "--disable-full-env-fallback-to-proxy",
        action="store_true",
        help="In full-environment mode, fail on PyPSA solve errors instead of falling back to the proxy backend.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--weights", default=",".join(str(weight) for weight in default_ppo.scalarization_weights))
    parser.add_argument("--myopic-chunk-mw", type=float, default=None)
    parser.add_argument("--baselines", nargs="+", default=["zero", "uniform", "myopic_proxy"])
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    raw_argv = sys.argv[1:]
    args = build_parser().parse_args(raw_argv)
    apply_thesis_formulation_preset(args, raw_argv)
    dataset = load_austria_case(
        NetworkConfig(
            network_path=Path(args.network),
            load_path=Path(args.load),
            wind_path=Path(args.wind),
            solar_path=Path(args.solar),
            year=args.year,
            start=args.start,
            end=args.end,
            candidate_line_limit=args.candidate_lines,
        )
    )
    if args.preprocessing_manifest:
        manifest = json.loads(Path(args.preprocessing_manifest).read_text(encoding="utf-8"))
        dataset = apply_future_scenario_from_manifest(dataset, manifest)
        dataset.candidate_lines = [str(value) for value in manifest["candidate_lines"]]
        dataset.demand_scale = pd.Series(manifest["demand_scale"], dtype=float).reindex(dataset.network.buses.index).fillna(1.0)
        dataset.renewable_scale = pd.Series(manifest["renewable_scale"], dtype=float).reindex(dataset.network.buses.index).fillna(1.0)
        dataset.total_demand_scale = float(manifest["total_demand_scale"])
    env_config = EnvironmentConfig(
        episode_length=args.episode_length,
        max_line_upgrade_mw=args.max_upgrade_mw,
        total_upgrade_budget_mw=args.budget_mw,
        decision_interval=args.decision_interval,
        temporal_mode=args.temporal_mode,
        budget_release=args.budget_release,
        action_mode=args.action_mode,
        allocation_sharpness=args.allocation_sharpness,
        allocation_sparsity_cutoff=args.allocation_sparsity_cutoff,
        stability_margin=args.stability_margin,
        proxy_balance_mode=args.proxy_balance_mode,
        proxy_dispatch_limit=args.proxy_dispatch_limit,
        load_shedding_cost=args.load_shedding_cost,
        line_investment_cost_eur_per_mw_km_year=args.line_investment_cost_eur_per_mw_km_year,
        cost_reward_scale=args.cost_reward_scale,
        overload_reward_scale=args.overload_reward_scale,
        third_objective_mode=args.third_objective_mode,
        curtailment_reward_scale=args.curtailment_reward_scale,
        emissions_reward_scale=args.emissions_reward_scale,
        seed=args.seed,
        solver_name=args.solver,
        full_env_fallback_to_proxy=not args.disable_full_env_fallback_to_proxy,
    )

    weights = _parse_number_tuple(args.weights, float)

    def make_env():
        if args.env == "proxy":
            return ProxyTEPEnv(dataset, env_config)
        return PyPSATEPEnv(dataset, env_config)

    names = list(args.baselines)
    results = {}
    for name in names:
        env = make_env()
        agent = build_baseline_agents(
            env,
            myopic_weights=weights,
            myopic_chunk_mw=args.myopic_chunk_mw,
        )[name]
        start_time = time.perf_counter()
        evaluation = evaluate_agent(agent, env, episodes=args.episodes, deterministic=True)
        elapsed = time.perf_counter() - start_time
        results[name] = {
            "elapsed_seconds": elapsed,
            "episodes": args.episodes,
            "seconds_per_episode": elapsed / max(args.episodes, 1),
            "grid_stress_mean": evaluation.get("grid_stress_mean"),
            "total_cost_mean": evaluation.get("total_cost_mean"),
        }

    text = json.dumps(results, indent=2, default=_json_default)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
