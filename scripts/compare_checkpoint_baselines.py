from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tep_rl.baselines import BASELINE_LABELS, build_baseline_agents
from tep_rl.config import EnvironmentConfig, NetworkConfig, PPOConfig
from tep_rl.data import load_austria_case
from tep_rl.envs import ProxyTEPEnv, PyPSATEPEnv
from tep_rl.evaluation import evaluate_agent
from tep_rl.ppo import load_agent
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


def _build_env_factory(args):
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
        cost_reward_scale=args.cost_reward_scale,
        overload_reward_scale=args.overload_reward_scale,
        third_objective_mode=args.third_objective_mode,
        curtailment_reward_scale=args.curtailment_reward_scale,
        emissions_reward_scale=args.emissions_reward_scale,
        seed=args.seed,
        solver_name=args.solver,
        full_env_fallback_to_proxy=not args.disable_full_env_fallback_to_proxy,
    )

    def make_env():
        if args.env == "proxy":
            return ProxyTEPEnv(dataset, env_config)
        return PyPSATEPEnv(dataset, env_config)

    return make_env


def _compact_summary(evaluation: dict[str, object]) -> dict[str, object]:
    action_stats = evaluation.get("action_stats", {})
    line_upgrades = evaluation.get("line_upgrades", {})
    top_lines = sorted(line_upgrades.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "total_cost_mean": evaluation.get("total_cost_mean"),
        "grid_stress_mean": evaluation.get("grid_stress_mean"),
        "renewable_share_mean": evaluation.get("renewable_share_mean"),
        "constraint_violation_mean": evaluation.get("constraint_violation_mean"),
        "slack_generation_mean": evaluation.get("slack_generation_mean"),
        "emissions_mean": evaluation.get("emissions_mean"),
        "total_investment_mean": action_stats.get("total_investment_mean"),
        "fraction_lines_touched": action_stats.get("fraction_lines_touched"),
        "top_line_upgrades": top_lines,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a saved checkpoint against the heuristic baseline suite.",
    )
    default_env = EnvironmentConfig()
    parser.add_argument(
        "--formulation-preset",
        choices=sorted(THESIS_FORMULATION_PRESETS),
        default=DEFAULT_THESIS_FORMULATION_PRESET,
        help="Apply a shared thesis formulation preset unless the specific budget/candidate flags are overridden.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env", choices=["proxy", "full"], default="proxy")
    parser.add_argument("--network", required=True)
    parser.add_argument("--load", required=True)
    parser.add_argument("--wind", required=True)
    parser.add_argument("--solar", required=True)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--candidate-lines", type=int, default=60)
    parser.add_argument("--episode-length", type=int, default=24)
    parser.add_argument("--max-upgrade-mw", type=float, default=60.0)
    parser.add_argument("--budget-mw", type=float, default=240.0)
    parser.add_argument("--decision-interval", type=int, default=6)
    parser.add_argument("--temporal-mode", choices=["decision_block", "hourly"], default=default_env.temporal_mode)
    parser.add_argument("--budget-release", choices=["linear", "all_at_once"], default="linear")
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
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--uniform-value", type=float, default=1.0)
    parser.add_argument("--heuristic-top-k", type=int, default=3)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--myopic-chunk-mw", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    raw_argv = sys.argv[1:]
    args = build_parser().parse_args(raw_argv)
    apply_thesis_formulation_preset(args, raw_argv)
    make_env = _build_env_factory(args)
    learned = load_agent(args.checkpoint, device=args.device)
    if args.weights:
        weights = _parse_number_tuple(args.weights, float)
    elif hasattr(learned, "scalarization_weights"):
        weights = tuple(float(value) for value in learned.scalarization_weights.tolist())
    elif hasattr(learned, "eval_preference_weights"):
        weights = tuple(float(value) for value in learned.eval_preference_weights.tolist())
    else:
        weights = PPOConfig().scalarization_weights
    if hasattr(learned, "set_eval_preferences"):
        learned.set_eval_preferences(weights)

    results = {}
    learned_eval = evaluate_agent(learned, make_env(), episodes=args.episodes, deterministic=True)
    results["learned"] = _compact_summary(learned_eval)
    results["learned"]["label"] = "Learned"

    baseline_names = list(
        build_baseline_agents(
            make_env(),
            uniform_value=args.uniform_value,
            heuristic_top_k=args.heuristic_top_k,
            myopic_weights=weights,
            myopic_chunk_mw=args.myopic_chunk_mw,
        ).keys()
    )
    for name in baseline_names:
        env = make_env()
        agent = build_baseline_agents(
            env,
            uniform_value=args.uniform_value,
            heuristic_top_k=args.heuristic_top_k,
            myopic_weights=weights,
            myopic_chunk_mw=args.myopic_chunk_mw,
        )[name]
        evaluation = evaluate_agent(agent, env, episodes=args.episodes, deterministic=True)
        results[name] = _compact_summary(evaluation)
        results[name]["label"] = BASELINE_LABELS.get(name, name)

    text = json.dumps(results, indent=2, default=_json_default)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
