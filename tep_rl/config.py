"""Central configuration dataclasses for the TEP-RL framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_OBJECTIVES = ("cost", "overload", "sustainability")
THESIS_SEEDS: tuple[int, ...] = (7, 11, 19, 23, 31)


@dataclass
class NetworkConfig:
    network_path: Path = Path("derived/austria_net_physical_ratings.nc")
    load_path: Path = Path("data/entsoe_at_load_2015_2024_opsd.csv")
    wind_path: Path = Path("data/wind_at_2015_2024.csv")
    solar_path: Path = Path("data/solar_at_2015_2024.csv")
    year: Optional[int] = None
    start: Optional[str] = None
    end: Optional[str] = None
    country: str = "AT"
    candidate_line_limit: Optional[int] = None
    renewable_carriers: tuple[str, ...] = (
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "solar",
        "ror",
    )


@dataclass
class EnvironmentConfig:
    episode_length: int = 24
    decision_interval: int = 6
    temporal_mode: str = "decision_block"  # "decision_block" | "hourly"

    max_line_upgrade_mw: float = 150.0
    total_upgrade_budget_mw: float = 600.0
    budget_release: str = "linear"  # "linear" | "all_at_once"
    action_mode: str = "budgeted"  # "budgeted" | "direct"

    allocation_sharpness: float = 12.0
    allocation_sparsity_cutoff: float = 0.70

    # Annualised 400-kV HVAC overhead-line midpoint in EUR/(MW km year).
    # Derivation: JRC 2012 midpoint 550 kEUR/km for a 1,500 MVA single
    # circuit, a transparent +20 % hilly-terrain factor, and the ENTSO-E
    # 4 % / 40-year annuity convention: 550000*1.20/1500*CRF = 22.22.
    line_investment_cost_eur_per_mw_km_year: float = 22.22
    investment_cost_reference_hours: float = 8760.0
    slack_marginal_cost: float = 140.0
    slack_emission_factor: float = 0.55
    load_shedding_cost: float = 10_000.0
    stability_margin: float = 0.70
    proxy_balance_mode: str = "demand_proportional"  # "demand_proportional" | "single_slack"
    proxy_dispatch_limit: Optional[float] = None
    constraint_threshold: float = 0.0
    cost_reward_scale: float = 10_000_000.0
    overload_reward_scale: float = 175.0
    third_objective_mode: str = "renewable_share"  # "renewable_share" | "curtailment" | "emissions"
    curtailment_reward_scale: float = 10_000.0
    emissions_reward_scale: float = 100_000.0

    random_start: bool = True
    seed: int = 7
    solver_name: str = "highs"
    full_env_fallback_to_proxy: bool = True
    objective_names: tuple[str, ...] = DEFAULT_OBJECTIVES
    emission_factors: dict[str, float] = field(
        default_factory=lambda: {
            "solar": 0.0,
            "onwind": 0.0,
            "offwind-ac": 0.0,
            "offwind-dc": 0.0,
            "ror": 0.0,
            "hydro": 0.0,
            "CCGT": 0.37,
            "OCGT": 0.50,
            "coal": 0.95,
            "oil": 0.78,
            "lignite": 1.05,
            "slack": 0.70,
        }
    )
    marginal_costs: dict[str, float] = field(
        default_factory=lambda: {
            "solar": 0.0,
            "onwind": 0.0,
            "offwind-ac": 0.0,
            "offwind-dc": 0.0,
            "ror": 5.0,
            "hydro": 8.0,
            "CCGT": 70.0,
            "OCGT": 95.0,
            "coal": 120.0,
            "oil": 150.0,
            "lignite": 135.0,
            "slack": 220.0,
        }
    )


@dataclass
class PPOConfig:
    hidden_sizes: tuple[int, ...] = (128, 128)
    learning_rate: float = 3e-4
    learning_rate_schedule: str = "linear"  # "constant" | "linear"
    final_learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    target_kl: Optional[float] = 0.03

    entropy_coef: float = 0.05
    entropy_coef_schedule: str = "linear"  # "constant" | "linear"
    final_entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 256
    minibatch_size: int = 64
    update_epochs: int = 8
    normalize_advantages: bool = True
    normalize_rewards: bool = False
    normalize_objective_advantages: bool = True
    device: str = "auto"
    seed: int = 7
    scalarization_weights: tuple[float, ...] = (0.34, 0.33, 0.33)
    moppo_preference_conditioning: bool = True
    moppo_sample_preferences: bool = True
    moppo_preference_sampling_mode: str = "dirichlet"  # "dirichlet" | "grid"
    moppo_preference_grid: tuple[tuple[float, ...], ...] | None = None
    moppo_dirichlet_alpha: float = 1.0


@dataclass
class TrainingConfig:
    total_timesteps: int = 20_000
    eval_every_updates: int = 5
    eval_episodes: int = 3
    deterministic_eval: bool = True
    show_progress: bool = False
    early_stopping_patience_evals: int = 8
    early_stopping_min_evals: int = 20
    early_stopping_min_delta: float = 1e-4
    restore_best_model_at_end: bool = True
    validation_weight_grid: Optional[tuple[tuple[float, ...], ...]] = None
    output_dir: Optional[Path] = None


@dataclass
class MultiSeedConfig:
    seeds: tuple[int, ...] = THESIS_SEEDS
    agent_mode: str = "moppo"  # "ppo" | "moppo"
    env_mode: str = "proxy"  # "proxy" | "full"
    ppo_config: PPOConfig = field(default_factory=PPOConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    weight_grid: Optional[list[tuple[float, ...]]] = None
    experiment_name: str = "unnamed"
    notes: str = ""
