import json
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pypsa
import torch

from main import _build_env, _parse_seeds, _parse_weights
from scripts.experiment_layout import preferred_results_dir
from scripts.run_moppo_preference_response import approximate_hypervolume, generate_simplex_weight_grid
from scripts.run_thesis_pipeline import rerank_selected_runs_on_target_env
from scripts.solve_dc_tep_baseline import resolve_manifest_defaults
from scripts.thesis_pipeline_utils import list_checkpoint_candidates_from_run_dir, select_best_checkpoint_from_run_dir
from tep_rl.baselines import build_baseline_agents
from tep_rl.config import EnvironmentConfig, NetworkConfig, PPOConfig, TrainingConfig
from tep_rl.data import (
    _build_bus_demand_profiles,
    _build_generator_availability,
    _candidate_line_ranking,
    apply_observation_scale_reference,
    build_toy_dataset,
    fit_observation_scales,
    load_austria_case,
)
from tep_rl.envs import ProxyTEPEnv, PyPSATEPEnv
from tep_rl.evaluation import evaluate_agent, metric_selection_score, stratified_episode_start_indices
from tep_rl.future_scenarios import apply_future_scenario, load_scenario_catalog, scenario_audit
from tep_rl.line_metadata import aggregate_line_importance
from tep_rl.ppo import MOPPOAgent, PPOAgent, load_agent
from tep_rl.simulation import DCPowerFlowModel, run_dc_proxy, run_pypsa_lopf
from tep_rl.shapley import (
    PolicyShapleyExplainer,
    capacity_preserving_corridor_ablation,
    episode_grouped_train_test_indices,
)
from tep_rl.statistics import compare_agents, compare_paired_evaluations
from tep_rl.subnetwork import extract_country_subnetwork
from tep_rl.thesis_formulations import DEFAULT_THESIS_FORMULATION_PRESET, apply_thesis_formulation_preset
from tep_rl.training import _evaluate_validation_suite, train_agent


class FrameworkSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._logger_levels = {}
        for logger_name in (
            "pypsa.network.io",
            "pypsa.networks",
            "pypsa.consistency",
            "pypsa.optimization.optimize",
            "linopy.io",
            "linopy.model",
            "linopy.solvers",
            "linopy.constants",
        ):
            logger = logging.getLogger(logger_name)
            cls._logger_levels[logger_name] = logger.level
            logger.setLevel(logging.ERROR)

    @classmethod
    def tearDownClass(cls):
        for logger_name, level in cls._logger_levels.items():
            logging.getLogger(logger_name).setLevel(level)

    def setUp(self):
        self.dataset = build_toy_dataset(num_steps=36, seed=3)
        self.env_config = EnvironmentConfig(
            episode_length=6,
            decision_interval=1,
            temporal_mode="hourly",
            max_line_upgrade_mw=30.0,
            total_upgrade_budget_mw=120.0,
            seed=3,
            solver_name="highs",
        )

    def test_proxy_environment_step(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        observation, _ = env.reset(seed=1)
        self.assertEqual(observation.shape[0], env.observation_space.shape[0])

        next_observation, reward, terminated, _, info = env.step(env.action_space.sample())
        self.assertEqual(next_observation.shape[0], env.observation_space.shape[0])
        self.assertEqual(reward.shape[0], 3)
        self.assertIn("renewable_share", info)
        self.assertFalse(terminated)

    def test_weight_and_seed_parsers_accept_commas_and_spaces(self):
        self.assertEqual(_parse_weights("0.34,0.33,0.33"), (0.34, 0.33, 0.33))
        self.assertEqual(_parse_weights("0.34 0.33 0.33"), (0.34, 0.33, 0.33))
        self.assertEqual(_parse_seeds("7,11,19"), (7, 11, 19))
        self.assertEqual(_parse_seeds("7 11 19"), (7, 11, 19))

    def test_source_based_formulation_is_default_and_uses_one_circuit_envelope(self):
        args = SimpleNamespace(
            formulation_preset=DEFAULT_THESIS_FORMULATION_PRESET,
            candidate_lines=60,
            max_upgrade_mw=60.0,
            budget_mw=240.0,
        )
        preset = apply_thesis_formulation_preset(args, [])
        self.assertEqual(preset.name, "source_based_v12")
        self.assertEqual(args.candidate_lines, 60)
        self.assertAlmostEqual(args.max_upgrade_mw, 500.0)
        self.assertAlmostEqual(args.budget_mw, 500.0)

    def test_experiment_layout_prefers_descriptive_name_for_new_outputs(self):
        root = Path("results") / "temporary_layout_check"
        self.assertEqual(
            preferred_results_dir(root, "ppo_validation"),
            root / "stage4_ppo_validation",
        )

    def test_explicit_cli_budget_flags_override_formulation_preset(self):
        args = SimpleNamespace(
            formulation_preset="source_based_v12",
            candidate_lines=72,
            max_upgrade_mw=45.0,
            budget_mw=180.0,
        )
        apply_thesis_formulation_preset(args, ["--candidate-lines", "72", "--max-upgrade-mw", "45", "--budget-mw", "180"])
        self.assertEqual(args.candidate_lines, 72)
        self.assertAlmostEqual(args.max_upgrade_mw, 45.0)
        self.assertAlmostEqual(args.budget_mw, 180.0)

    def test_train_agent_restores_best_validation_model_and_stops_early(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        eval_env = ProxyTEPEnv(self.dataset, self.env_config)
        agent = PPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=len(env.config.objective_names),
            config=PPOConfig(
                rollout_steps=1,
                minibatch_size=1,
                update_epochs=1,
                device="cpu",
                seed=3,
            ),
        )
        initial_value = float(next(agent.network.parameters()).detach().flatten()[0].item())

        update_counter = {"value": 0}

        def fake_update(_buffer):
            update_counter["value"] += 1
            with torch.no_grad():
                next(agent.network.parameters()).add_(1.0)
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "clip_fraction": 0.0,
                "approx_kl": 0.0,
                "stopped_early": False,
            }

        agent.update = fake_update

        evaluation_side_effects = [
            {
                "reward_stats": {"mean_vector_reward": [0.10, 0.0, 0.0]},
                "renewable_share_mean": 0.10,
                "total_cost_mean": 10.0,
                "grid_stress_mean": 10.0,
            },
            {
                "reward_stats": {"mean_vector_reward": [0.20, 0.0, 0.0]},
                "renewable_share_mean": 0.10,
                "total_cost_mean": 9.0,
                "grid_stress_mean": 9.0,
            },
            {
                "reward_stats": {"mean_vector_reward": [0.19, 0.0, 0.0]},
                "renewable_share_mean": 0.10,
                "total_cost_mean": 9.5,
                "grid_stress_mean": 9.5,
            },
            {
                "reward_stats": {"mean_vector_reward": [0.18, 0.0, 0.0]},
                "renewable_share_mean": 0.10,
                "total_cost_mean": 9.6,
                "grid_stress_mean": 9.6,
            },
        ]

        training_config = TrainingConfig(
            total_timesteps=10,
            eval_every_updates=1,
            eval_episodes=1,
            show_progress=False,
            early_stopping_patience_evals=2,
            early_stopping_min_evals=2,
            early_stopping_min_delta=1e-4,
            restore_best_model_at_end=True,
        )

        with patch("tep_rl.training.evaluate_agent", side_effect=evaluation_side_effects):
            history = train_agent(agent, env, training_config=training_config, eval_env=eval_env)

        self.assertTrue(history["stopped_early"])
        self.assertEqual(history["best_validation"]["update"], 2)
        self.assertEqual(len(history["updates"]), 4)
        restored_value = float(next(agent.network.parameters()).detach().flatten()[0].item())
        self.assertAlmostEqual(restored_value, initial_value + 2.0, places=5)

    def test_ppo_training_progress_anneals_learning_rate_and_entropy(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        agent = PPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=len(env.config.objective_names),
            config=PPOConfig(
                learning_rate=3e-4,
                learning_rate_schedule="linear",
                final_learning_rate=1e-4,
                entropy_coef=0.05,
                entropy_coef_schedule="linear",
                final_entropy_coef=0.01,
                device="cpu",
                seed=3,
            ),
        )

        agent.set_training_progress(0.5)
        self.assertAlmostEqual(agent.current_learning_rate, 2e-4, places=8)
        self.assertAlmostEqual(agent.current_entropy_coef, 0.03, places=8)
        self.assertAlmostEqual(agent.optimizer.param_groups[0]["lr"], 2e-4, places=8)

    def test_staged_budget_release_prevents_front_loading(self):
        env = ProxyTEPEnv(
            self.dataset,
            EnvironmentConfig(
                episode_length=8,
                max_line_upgrade_mw=30.0,
                total_upgrade_budget_mw=120.0,
                decision_interval=4,
                temporal_mode="hourly",
                budget_release="linear",
                seed=3,
                solver_name="highs",
            ),
        )
        env.reset(seed=1)

        _, _, _, _, first_info = env.step(np.ones(env.action_space.shape[0], dtype=np.float32))
        self.assertAlmostEqual(sum(first_info["action_mw"]), 60.0, places=4)
        self.assertTrue(first_info["decision_step"])

        _, _, _, _, second_info = env.step(np.ones(env.action_space.shape[0], dtype=np.float32))
        self.assertAlmostEqual(sum(second_info["action_mw"]), 0.0, places=4)
        self.assertFalse(second_info["decision_step"])

    def test_decision_block_mode_advances_to_next_control_stage(self):
        env = ProxyTEPEnv(
            self.dataset,
            EnvironmentConfig(
                episode_length=8,
                max_line_upgrade_mw=30.0,
                total_upgrade_budget_mw=120.0,
                decision_interval=4,
                temporal_mode="decision_block",
                budget_release="linear",
                seed=3,
                solver_name="highs",
            ),
        )
        env.reset(seed=1)
        _, _, terminated, _, info = env.step(np.ones(env.action_space.shape[0], dtype=np.float32))
        self.assertEqual(info["n_hours"], 4)
        self.assertFalse(terminated)
        self.assertEqual(env.current_step, 4)
        self.assertTrue(info["decision_step"])

    def test_main_build_env_accepts_reward_scaling_overrides(self):
        args = SimpleNamespace(
            episode_length=6,
            max_upgrade_mw=30.0,
            budget_mw=120.0,
            decision_interval=3,
            temporal_mode="decision_block",
            budget_release="all_at_once",
            action_mode="budgeted",
            allocation_sharpness=8.0,
            allocation_sparsity_cutoff=0.1,
            stability_margin=0.8,
            proxy_balance_mode="single_slack",
            proxy_dispatch_limit=0.65,
            load_shedding_cost=9876.0,
            cost_reward_scale=123.0,
            overload_reward_scale=45.0,
            third_objective_mode="emissions",
            curtailment_reward_scale=321.0,
            emissions_reward_scale=654.0,
            seed=3,
            solver="highs",
            env="proxy",
            disable_full_env_fallback_to_proxy=True,
        )
        env = _build_env(self.dataset, args)
        self.assertEqual(env.config.action_mode, "budgeted")
        self.assertAlmostEqual(env.config.allocation_sharpness, 8.0)
        self.assertAlmostEqual(env.config.allocation_sparsity_cutoff, 0.1)
        self.assertAlmostEqual(env.config.stability_margin, 0.8)
        self.assertEqual(env.config.proxy_balance_mode, "single_slack")
        self.assertAlmostEqual(env.config.proxy_dispatch_limit, 0.65)
        self.assertAlmostEqual(env.config.load_shedding_cost, 9876.0)
        self.assertAlmostEqual(env.config.cost_reward_scale, 123.0)
        self.assertAlmostEqual(env.config.overload_reward_scale, 45.0)
        self.assertEqual(env.config.third_objective_mode, "emissions")
        self.assertAlmostEqual(env.config.curtailment_reward_scale, 321.0)
        self.assertAlmostEqual(env.config.emissions_reward_scale, 654.0)
        self.assertFalse(env.config.full_env_fallback_to_proxy)

    def test_proxy_dispatch_limit_makes_renewable_serving_action_sensitive(self):
        network = pypsa.Network()
        network.add("Bus", "wind_bus")
        network.add("Bus", "load_bus")
        network.add("Line", "line_1", bus0="wind_bus", bus1="load_bus", x=0.1, s_nom=50.0)
        model = DCPowerFlowModel.from_network(network, slack_bus="load_bus")

        demand = pd.Series({"wind_bus": 0.0, "load_bus": 100.0})
        renewable = pd.Series({"wind_bus": 100.0, "load_bus": 0.0})
        base_capacity = pd.Series({"line_1": 50.0})
        upgraded_capacity = pd.Series({"line_1": 100.0})

        constrained = run_dc_proxy(
            model=model,
            demand_by_bus=demand,
            renewable_by_bus=renewable,
            line_capacities=base_capacity,
            slack_cost=140.0,
            slack_emission_factor=0.55,
            stability_margin=0.7,
            dispatch_loading_limit=0.7,
        )
        upgraded = run_dc_proxy(
            model=model,
            demand_by_bus=demand,
            renewable_by_bus=renewable,
            line_capacities=upgraded_capacity,
            slack_cost=140.0,
            slack_emission_factor=0.55,
            stability_margin=0.7,
            dispatch_loading_limit=0.7,
        )

        self.assertGreater(upgraded.renewable_served, constrained.renewable_served)
        self.assertGreater(upgraded.renewable_share, constrained.renewable_share)
        self.assertLess(upgraded.renewable_curtailment, constrained.renewable_curtailment)

    def test_run_pypsa_lopf_uses_distributed_backstop_to_avoid_infeasibility(self):
        network = pypsa.Network()
        network.add("Bus", "slack_bus")
        network.add("Bus", "isolated_load_bus")
        network.add("Load", "load_1", bus="isolated_load_bus", p_set=0.0)

        snapshot = pd.Timestamp("2024-01-01 00:00:00")
        result = run_pypsa_lopf(
            base_network=network,
            snapshot=snapshot,
            demand_by_load=pd.Series({"load_1": 50.0}),
            demand_by_bus=pd.Series({"slack_bus": 0.0, "isolated_load_bus": 50.0}),
            renewable_by_bus=pd.Series({"slack_bus": 0.0, "isolated_load_bus": 0.0}),
            generator_availability=pd.Series(dtype=float),
            line_capacities=pd.Series(dtype=float),
            renewable_carriers=(),
            marginal_costs={"slack": 220.0},
            emission_factors={"slack": 0.7},
            solver_name="highs",
            stability_margin=0.7,
            load_shedding_cost=10_000.0,
        )

        self.assertAlmostEqual(result.load_shedding, 0.0)
        self.assertAlmostEqual(result.slack_generation, 50.0)
        self.assertEqual(result.backend, "pypsa-optimize")

    def test_conditioned_moppo_selects_best_eval_preference_for_reference_utility(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        eval_env = ProxyTEPEnv(self.dataset, self.env_config)
        agent = MOPPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=len(env.config.objective_names),
            config=PPOConfig(
                rollout_steps=1,
                minibatch_size=1,
                update_epochs=1,
                device="cpu",
                seed=3,
                scalarization_weights=(0.70, 0.20, 0.10),
                moppo_preference_conditioning=True,
                moppo_sample_preferences=True,
            ),
        )

        def fake_update(_buffer):
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "clip_fraction": 0.0,
                "approx_kl": 0.0,
                "stopped_early": False,
            }

        agent.update = fake_update

        baseline_eval = {
            "reward_stats": {"mean_vector_reward": [0.0, 0.0, 0.0]},
            "renewable_share_mean": 0.10,
            "renewable_curtailment_mean": 100.0,
            "total_cost_mean": 12.0,
            "grid_stress_mean": 12.0,
            "load_shedding_mean": 0.0,
        }
        eval_a = {
            "reward_stats": {"mean_vector_reward": [0.9, 0.9, 0.9]},
            "renewable_share_mean": 0.105,
            "renewable_curtailment_mean": 95.0,
            "total_cost_mean": 11.8,
            "grid_stress_mean": 11.5,
            "load_shedding_mean": 0.0,
        }
        eval_b = {
            "reward_stats": {"mean_vector_reward": [0.1, 0.1, 0.1]},
            "renewable_share_mean": 0.14,
            "renewable_curtailment_mean": 60.0,
            "total_cost_mean": 9.0,
            "grid_stress_mean": 8.0,
            "load_shedding_mean": 0.0,
        }

        training_config = TrainingConfig(
            total_timesteps=1,
            eval_every_updates=1,
            eval_episodes=1,
            show_progress=False,
            validation_weight_grid=((0.70, 0.20, 0.10), (0.20, 0.70, 0.10)),
        )

        with patch("tep_rl.evaluation.evaluate_agent", side_effect=[baseline_eval, eval_a, eval_b]):
            history = train_agent(agent, env, training_config=training_config, eval_env=eval_env)

        expected_score = metric_selection_score(
            eval_b,
            utility_weights=(0.70, 0.20, 0.10),
            reference_evaluation=baseline_eval,
            third_objective_mode=env.config.third_objective_mode,
        )
        self.assertAlmostEqual(history["best_validation"]["selection_score"], expected_score, places=6)
        self.assertEqual(len(history["best_validation"]["selection_details"]["grid"]), 2)
        self.assertEqual(
            history["best_validation"]["evaluation"]["total_cost_mean"],
            eval_b["total_cost_mean"],
        )
        np.testing.assert_allclose(
            history["best_validation"]["selection_details"]["selected_eval_preference_weights"],
            np.array([0.2, 0.7, 0.1], dtype=np.float32),
        )
        self.assertEqual(
            history["best_validation"]["selection_details"]["selection_score_mode"],
            "metric_relative_to_zero_upgrade_reference",
        )

    def test_moppo_validation_suite_uses_fixed_scalarization_reference_weights(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        eval_env = ProxyTEPEnv(self.dataset, self.env_config)
        agent = MOPPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=len(env.config.objective_names),
            config=PPOConfig(
                device="cpu",
                seed=3,
                scalarization_weights=(0.70, 0.20, 0.10),
                moppo_preference_conditioning=True,
                moppo_sample_preferences=True,
            ),
        )
        agent.set_eval_preferences((0.20, 0.70, 0.10))

        training_config = TrainingConfig(
            total_timesteps=1,
            eval_every_updates=1,
            eval_episodes=1,
            show_progress=False,
            validation_weight_grid=((0.70, 0.20, 0.10), (0.20, 0.70, 0.10)),
        )

        fake_result = {
            "evaluation": {"reward_stats": {"mean_vector_reward": [0.0, 0.0, 0.0]}},
            "selection_score": 0.0,
            "selection_details": {"selection_score_mode": "metric_relative_to_zero_upgrade_reference"},
        }

        with patch("tep_rl.training.evaluate_moppo_preference_grid", return_value=fake_result) as mocked_grid_eval:
            _evaluate_validation_suite(agent, eval_env, training_config)

        np.testing.assert_allclose(
            mocked_grid_eval.call_args.kwargs["utility_weights"],
            np.array([0.70, 0.20, 0.10], dtype=np.float32),
        )

    def test_moppo_checkpoint_preserves_selected_eval_preferences(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        agent = MOPPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=len(env.config.objective_names),
            config=PPOConfig(
                device="cpu",
                seed=3,
                scalarization_weights=(0.70, 0.20, 0.10),
                moppo_preference_conditioning=True,
                moppo_sample_preferences=True,
            ),
        )
        agent.set_eval_preferences((0.2, 0.7, 0.1))

        with TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "agent.pt"
            agent.save(checkpoint)
            loaded = load_agent(checkpoint, device="cpu")

        np.testing.assert_allclose(loaded.eval_preference_weights, np.array([0.2, 0.7, 0.1], dtype=np.float32))
        np.testing.assert_allclose(loaded.current_preference_weights, np.array([0.2, 0.7, 0.1], dtype=np.float32))

    def test_compare_agents_treats_curtailment_as_lower_is_better(self):
        a = pd.DataFrame({"renewable_curtailment_mean": [10.0, 12.0, 11.0]})
        b = pd.DataFrame({"renewable_curtailment_mean": [20.0, 22.0, 21.0]})

        comparison = compare_agents(
            a,
            b,
            label_a="Agent A",
            label_b="Agent B",
            metrics=("renewable_curtailment_mean",),
        )

        self.assertEqual(len(comparison), 1)
        self.assertTrue(bool(comparison.iloc[0]["significant"]))
        self.assertEqual(comparison.iloc[0]["better"], "Agent A")

    def test_checkpoint_candidate_listing_prefers_stored_best_validation_score(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_a"
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)

            (run_dir / "config_snapshot.json").write_text(
                json.dumps(
                    {
                        "seed": 7,
                        "ppo_config": {
                            "scalarization_weights": [0.34, 0.33, 0.33],
                        },
                    }
                ),
                encoding="utf-8",
            )

            eval_4 = {
                "reward_stats": {"mean_vector_reward": [0.2, 0.0, 0.0]},
                "renewable_share_mean": 0.10,
                "total_cost_mean": 9.0,
                "grid_stress_mean": 2.0,
                "load_shedding_mean": 5.0,
            }
            eval_8 = {
                "reward_stats": {"mean_vector_reward": [0.1, 0.0, 0.0]},
                "renewable_share_mean": 0.20,
                "total_cost_mean": 8.0,
                "grid_stress_mean": 1.0,
                "load_shedding_mean": 4.0,
            }
            (checkpoints_dir / "update_0004.evaluation.json").write_text(pd.Series(eval_4).to_json(), encoding="utf-8")
            (checkpoints_dir / "update_0008.evaluation.json").write_text(pd.Series(eval_8).to_json(), encoding="utf-8")
            (checkpoints_dir / "update_0004.pt").write_text("", encoding="utf-8")
            (checkpoints_dir / "update_0008.pt").write_text("", encoding="utf-8")
            (checkpoints_dir / "update_0008.selection.json").write_text(
                '{"mode": "weight_grid_mean", "mean_selection_score": 0.9}',
                encoding="utf-8",
            )
            (run_dir / "best_validation.json").write_text(
                json.dumps(
                    {
                        "selection_score": 0.9,
                        "update": 8,
                        "checkpoint": {
                            "checkpoint": str(checkpoints_dir / "update_0008.pt"),
                            "evaluation_path": str(checkpoints_dir / "update_0008.evaluation.json"),
                        },
                        "selection_details": {
                            "mode": "weight_grid_mean",
                            "mean_selection_score": 0.9,
                        },
                    }
                ),
                encoding="utf-8",
            )

            candidates = list_checkpoint_candidates_from_run_dir(run_dir)
            best = select_best_checkpoint_from_run_dir(run_dir)

            self.assertEqual(Path(candidates[0]["checkpoint"]).name, "update_0008.pt")
            self.assertEqual(Path(best["checkpoint"]).name, "update_0008.pt")
            self.assertAlmostEqual(float(best["selection_score"]), 0.9, places=6)

    def test_checkpoint_candidate_listing_uses_best_selection_score_before_mean(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_b"
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)

            (run_dir / "config_snapshot.json").write_text(
                json.dumps(
                    {
                        "seed": 11,
                        "ppo_config": {
                            "scalarization_weights": [0.34, 0.33, 0.33],
                        },
                    }
                ),
                encoding="utf-8",
            )

            evaluation = {
                "reward_stats": {"mean_vector_reward": [0.2, 0.1, 0.0]},
                "renewable_share_mean": 0.15,
                "renewable_curtailment_mean": 200.0,
                "total_cost_mean": 8.5,
                "grid_stress_mean": 0.8,
                "load_shedding_mean": 0.0,
            }
            (checkpoints_dir / "update_0004.evaluation.json").write_text(pd.Series(evaluation).to_json(), encoding="utf-8")
            (checkpoints_dir / "update_0004.pt").write_text("", encoding="utf-8")
            (checkpoints_dir / "update_0004.selection.json").write_text(
                json.dumps(
                    {
                        "mode": "weight_grid_best_for_reference_utility",
                        "best_selection_score": 0.42,
                        "mean_selection_score": 0.11,
                    }
                ),
                encoding="utf-8",
            )

            candidates = list_checkpoint_candidates_from_run_dir(run_dir)

            self.assertEqual(len(candidates), 1)
            self.assertAlmostEqual(float(candidates[0]["selection_score"]), 0.42, places=6)

    def test_fullenv_rerank_can_replace_proxy_best_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_b"
            checkpoints_dir = run_dir / "checkpoints"
            output_dir = Path(tmpdir) / "out"
            checkpoints_dir.mkdir(parents=True)

            (run_dir / "config_snapshot.json").write_text(
                json.dumps(
                    {
                        "seed": 7,
                        "ppo_config": {
                            "scalarization_weights": [0.34, 0.33, 0.33],
                        },
                    }
                ),
                encoding="utf-8",
            )

            checkpoint_4 = checkpoints_dir / "update_0004.pt"
            checkpoint_8 = checkpoints_dir / "update_0008.pt"
            checkpoint_4.write_text("", encoding="utf-8")
            checkpoint_8.write_text("", encoding="utf-8")
            eval_template = {
                "reward_stats": {"mean_vector_reward": [0.1, 0.0, 0.0]},
                "renewable_share_mean": 0.10,
                "total_cost_mean": 10.0,
                "grid_stress_mean": 2.0,
                "load_shedding_mean": 5.0,
            }
            (checkpoints_dir / "update_0004.evaluation.json").write_text(pd.Series(eval_template).to_json(), encoding="utf-8")
            (checkpoints_dir / "update_0008.evaluation.json").write_text(pd.Series(eval_template).to_json(), encoding="utf-8")
            (run_dir / "best_validation.json").write_text(
                json.dumps(
                    {
                        "selection_score": 0.8,
                        "update": 4,
                        "checkpoint": {
                            "checkpoint": str(checkpoint_4),
                            "evaluation_path": str(checkpoints_dir / "update_0004.evaluation.json"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            selected_runs = [
                {
                    "run_dir": str(run_dir),
                    "weights": [0.34, 0.33, 0.33],
                    "checkpoint": str(checkpoint_4),
                    "selection_score": 0.8,
                }
            ]

            def fake_load_agent(checkpoint, device="cpu"):
                return SimpleNamespace(checkpoint=str(checkpoint), device=device)

            def fake_build_env(dataset, env_mode, **kwargs):
                return SimpleNamespace(dataset=dataset, env_mode=env_mode, kwargs=kwargs)

            def fake_evaluate(agent, env, episodes, deterministic=True):
                if "0008" in agent.checkpoint:
                    return {
                        "reward_stats": {"mean_vector_reward": [1.0, 0.0, 0.0]},
                        "renewable_share_mean": 0.20,
                        "total_cost_mean": 8.0,
                        "grid_stress_mean": 1.0,
                        "load_shedding_mean": 2.0,
                    }
                return {
                    "reward_stats": {"mean_vector_reward": [0.2, 0.0, 0.0]},
                    "renewable_share_mean": 0.10,
                    "total_cost_mean": 9.0,
                    "grid_stress_mean": 1.5,
                    "load_shedding_mean": 4.0,
                }

            with patch("scripts.run_thesis_pipeline.load_agent", side_effect=fake_load_agent), patch(
                "scripts.run_thesis_pipeline.build_env",
                side_effect=fake_build_env,
            ), patch("scripts.run_thesis_pipeline.evaluate_agent", side_effect=fake_evaluate):
                reranked = rerank_selected_runs_on_target_env(
                    selected_runs=selected_runs,
                    env_mode="full",
                    dataset=object(),
                    output_dir=output_dir,
                    env_kwargs={"seed": 7},
                    episodes=2,
                    rerank_top_k=2,
                    device="cpu",
                )

            self.assertEqual(len(reranked), 1)
            self.assertEqual(Path(reranked[0]["checkpoint"]).name, "update_0008.pt")
            self.assertEqual(reranked[0]["selection_details"]["mode"], "target_env_validation_rerank")

    def test_budgeted_action_space_adds_spend_gate(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        self.assertEqual(env.action_space.shape[0], len(env.candidate_lines) + 1)
        self.assertEqual(env.get_action_names()[0], "spend_fraction")

    def test_budgeted_action_can_target_single_line_without_spending_full_budget(self):
        env = ProxyTEPEnv(
            self.dataset,
            EnvironmentConfig(
                episode_length=6,
                max_line_upgrade_mw=30.0,
                total_upgrade_budget_mw=60.0,
                decision_interval=1,
                temporal_mode="decision_block",
                budget_release="all_at_once",
                action_mode="budgeted",
                allocation_sharpness=8.0,
                seed=3,
                solver_name="highs",
            ),
        )
        env.reset(seed=1)
        action = np.array([0.5, 1.0, 0.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        self.assertAlmostEqual(sum(info["action_mw"]), 30.0, places=4)
        self.assertAlmostEqual(info["action_mw"][0], 30.0, places=4)
        self.assertEqual(info["active_lines"], 1)

    def test_sparse_budget_decoder_prunes_low_score_lines(self):
        env = ProxyTEPEnv(
            self.dataset,
            EnvironmentConfig(
                episode_length=6,
                max_line_upgrade_mw=30.0,
                total_upgrade_budget_mw=60.0,
                decision_interval=1,
                temporal_mode="decision_block",
                budget_release="all_at_once",
                action_mode="budgeted",
                allocation_sharpness=8.0,
                allocation_sparsity_cutoff=0.5,
                seed=3,
                solver_name="highs",
            ),
        )
        env.reset(seed=1)
        action = np.array([1.0, 1.0, 0.55, 0.1], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        self.assertGreater(info["action_mw"][0], 0.0)
        self.assertAlmostEqual(info["action_mw"][2], 0.0, places=6)
        self.assertLessEqual(info["active_lines"], 2)

    def test_baseline_suite_returns_expected_actions(self):
        env = ProxyTEPEnv(
            self.dataset,
            EnvironmentConfig(
                episode_length=6,
                max_line_upgrade_mw=30.0,
                total_upgrade_budget_mw=60.0,
                decision_interval=1,
                temporal_mode="decision_block",
                budget_release="all_at_once",
                action_mode="budgeted",
                stability_margin=0.0,
                seed=3,
                solver_name="highs",
            ),
        )
        observation, _ = env.reset(seed=1)
        baselines = build_baseline_agents(env, uniform_value=1.0, heuristic_top_k=3)

        self.assertEqual(
            set(baselines.keys()),
            {"zero", "uniform", "top_1_excess", "top_3_excess", "proportional_excess", "myopic_proxy"},
        )

        zero_action, _, _ = baselines["zero"].act(observation, deterministic=True)
        uniform_action, _, _ = baselines["uniform"].act(observation, deterministic=True)
        top1_action, _, _ = baselines["top_1_excess"].act(observation, deterministic=True)

        self.assertTrue(np.allclose(zero_action, 0.0))
        self.assertEqual(uniform_action.shape[0], env.action_space.shape[0])
        self.assertAlmostEqual(float(uniform_action[0]), 1.0, places=6)
        self.assertAlmostEqual(float(top1_action[0]), 1.0, places=6)
        self.assertLessEqual(int(np.sum(top1_action[1:] > 0.0)), 1)

        myopic_action, _, _ = baselines["myopic_proxy"].act(observation, deterministic=True)
        self.assertEqual(myopic_action.shape[0], len(env.candidate_lines))
        self.assertTrue(np.all(myopic_action >= 0.0))
        self.assertTrue(np.all(myopic_action <= 1.0))

    def test_excess_baselines_hold_budget_when_no_candidate_is_over_margin(self):
        env = ProxyTEPEnv(
            self.dataset,
            EnvironmentConfig(
                episode_length=6,
                max_line_upgrade_mw=30.0,
                total_upgrade_budget_mw=60.0,
                decision_interval=1,
                temporal_mode="decision_block",
                budget_release="all_at_once",
                action_mode="budgeted",
                stability_margin=10.0,
                seed=3,
                solver_name="highs",
            ),
        )
        observation, _ = env.reset(seed=1)
        baselines = build_baseline_agents(env, uniform_value=1.0, heuristic_top_k=3)

        for name in ("top_1_excess", "top_3_excess", "proportional_excess"):
            action, _, _ = baselines[name].act(observation, deterministic=True)
            self.assertTrue(np.allclose(action, 0.0), msg=name)

    def test_observation_preview_uses_current_block_loading_signal(self):
        env = ProxyTEPEnv(
            self.dataset,
            EnvironmentConfig(
                episode_length=6,
                max_line_upgrade_mw=30.0,
                total_upgrade_budget_mw=120.0,
                decision_interval=2,
                temporal_mode="decision_block",
                budget_release="all_at_once",
                action_mode="budgeted",
                stability_margin=0.7,
                seed=3,
                solver_name="highs",
            ),
        )
        observation, _ = env.reset(seed=1)

        demand_offset = len(env.bus_names)
        renewable_offset = demand_offset + len(env.bus_names)
        raw_loading_offset = renewable_offset
        candidate_loading_offset = raw_loading_offset + len(env.line_names)

        raw_loading_obs = observation[raw_loading_offset:candidate_loading_offset]
        candidate_loading_obs = observation[
            candidate_loading_offset:candidate_loading_offset + len(env.candidate_lines)
        ]

        expected_raw = env.preview_state.simulation.raw_line_loading.reindex(env.line_names).fillna(0.0).clip(0.0, 2.5).to_numpy(dtype=np.float32)
        expected_candidate = env.preview_state.simulation.raw_line_loading.reindex(env.candidate_lines).fillna(0.0).clip(0.0, 2.5).to_numpy(dtype=np.float32)

        np.testing.assert_allclose(raw_loading_obs, expected_raw)
        np.testing.assert_allclose(candidate_loading_obs, expected_candidate)
        self.assertGreaterEqual(float(env.preview_state.simulation.raw_line_loading.max()), float(env.preview_state.simulation.line_loading.max()))

    def test_emissions_third_objective_matches_info_payload(self):
        env = ProxyTEPEnv(
            self.dataset,
            EnvironmentConfig(
                episode_length=6,
                decision_interval=1,
                temporal_mode="decision_block",
                max_line_upgrade_mw=30.0,
                total_upgrade_budget_mw=120.0,
                third_objective_mode="emissions",
                emissions_reward_scale=100.0,
                seed=3,
                solver_name="highs",
            ),
        )
        env.reset(seed=1)
        _, reward, _, _, info = env.step(np.zeros(env.action_space.shape[0], dtype=np.float32))
        self.assertEqual(info["third_objective_mode"], "emissions")
        self.assertAlmostEqual(float(reward[2]), float(info["third_objective_reward"]), places=6)
        self.assertAlmostEqual(float(reward[2]), -float(info["emissions"]) / 100.0, places=6)

    def test_direct_actions_respect_cumulative_line_caps(self):
        env = ProxyTEPEnv(
            self.dataset,
            EnvironmentConfig(
                episode_length=8,
                max_line_upgrade_mw=10.0,
                total_upgrade_budget_mw=120.0,
                decision_interval=4,
                temporal_mode="hourly",
                budget_release="all_at_once",
                action_mode="budgeted",
                seed=3,
                solver_name="highs",
            ),
        )
        env.reset(seed=1)

        direct_action = np.ones(len(env.candidate_lines), dtype=np.float32)
        _, _, _, _, first_info = env.step(direct_action)
        self.assertAlmostEqual(sum(first_info["action_mw"]), 30.0, places=4)
        self.assertEqual(first_info["action_mode"], "direct")

        for _ in range(3):
            env.step(np.zeros(env.action_space.shape[0], dtype=np.float32))
        _, _, _, _, second_info = env.step(direct_action)
        self.assertAlmostEqual(sum(second_info["action_mw"]), 0.0, places=4)

    def test_scalar_ppo_smoke(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        eval_env = ProxyTEPEnv(self.dataset, self.env_config)
        agent = PPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=3,
            config=PPOConfig(
                rollout_steps=32,
                minibatch_size=16,
                update_epochs=2,
                hidden_sizes=(64, 64),
                scalarization_weights=(0.5, 0.3, 0.2),
                device="gpu",
            ),
        )
        history = train_agent(
            agent,
            env,
            training_config=TrainingConfig(total_timesteps=96, eval_every_updates=1, eval_episodes=1),
            eval_env=eval_env,
        )
        evaluation = evaluate_agent(agent, eval_env, episodes=1)
        self.assertGreaterEqual(len(history["updates"]), 1)
        self.assertIn("total_cost_mean", evaluation)

    def test_gpu_device_alias_falls_back_safely(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        agent = PPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=3,
            config=PPOConfig(device="gpu"),
        )
        self.assertIn(agent.device.type, {"cpu", "cuda"})

    def test_moppo_preference_conditioning_samples_training_context(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        agent = MOPPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=3,
            config=PPOConfig(
                device="cpu",
                scalarization_weights=(0.34, 0.33, 0.33),
                moppo_preference_conditioning=True,
                moppo_sample_preferences=True,
                moppo_dirichlet_alpha=0.5,
            ),
        )
        agent.start_episode(training=True)
        sampled = agent.current_context()
        self.assertEqual(sampled.shape, (3,))
        self.assertAlmostEqual(float(np.sum(sampled)), 1.0, places=5)
        agent.start_episode(training=False)
        np.testing.assert_allclose(agent.current_context(), agent.eval_preference_weights)

    def test_moppo_grid_preference_sampling_cycles_configured_grid(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        grid = ((0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8))
        agent = MOPPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=3,
            config=PPOConfig(
                device="cpu",
                scalarization_weights=(0.34, 0.33, 0.33),
                moppo_preference_conditioning=True,
                moppo_sample_preferences=True,
                moppo_preference_sampling_mode="grid",
                moppo_preference_grid=grid,
            ),
        )
        observed = []
        for _ in range(5):
            agent.start_episode(training=True)
            observed.append(tuple(float(value) for value in agent.current_context()))
        np.testing.assert_allclose(observed[0], grid[0], atol=1e-7)
        np.testing.assert_allclose(observed[1], grid[1], atol=1e-7)
        np.testing.assert_allclose(observed[2], grid[2], atol=1e-7)
        np.testing.assert_allclose(observed[3], grid[0], atol=1e-7)
        np.testing.assert_allclose(observed[4], grid[1], atol=1e-7)

    def test_moppo_preference_simplex_weight_grid_generates_expected_triangle(self):
        grid = generate_simplex_weight_grid(0.5)
        self.assertEqual(len(grid), 6)
        self.assertEqual(grid[0], (1.0, 0.0, 0.0))
        self.assertEqual(grid[-1], (0.0, 0.0, 1.0))
        for weights in grid:
            self.assertAlmostEqual(sum(weights), 1.0, places=8)

    def test_moppo_preference_hypervolume_prefers_better_points(self):
        worse_point = np.array([[0.70, 0.70, 0.70]], dtype=float)
        better_point = np.array([[0.40, 0.40, 0.40]], dtype=float)
        self.assertGreater(
            approximate_hypervolume(better_point, ref_point=np.array([1.0, 1.0, 1.0]), samples=12_000, seed=3),
            approximate_hypervolume(worse_point, ref_point=np.array([1.0, 1.0, 1.0]), samples=12_000, seed=3),
        )

    def test_moppo_values_are_preference_conditioned(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        observation, _ = env.reset(seed=0)
        agent = MOPPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=3,
            config=PPOConfig(
                device="cpu",
                scalarization_weights=(0.34, 0.33, 0.33),
                moppo_preference_conditioning=True,
                moppo_sample_preferences=True,
                moppo_dirichlet_alpha=0.5,
            ),
        )
        agent.set_eval_preferences((1.0, 0.0, 0.0))
        agent.start_episode(training=False)
        eval_values = agent.value(observation)
        agent.set_eval_preferences((0.0, 1.0, 0.0))
        agent.start_episode(training=False)
        alternative_values = agent.value(observation)
        self.assertFalse(np.allclose(eval_values, alternative_values))

    def test_moppo_smoke(self):
        env = ProxyTEPEnv(self.dataset, self.env_config)
        eval_env = ProxyTEPEnv(self.dataset, self.env_config)
        agent = MOPPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            env_reward_dim=3,
            config=PPOConfig(
                rollout_steps=32,
                minibatch_size=16,
                update_epochs=2,
                hidden_sizes=(64, 64),
                scalarization_weights=(0.25, 0.25, 0.5),
                device="gpu",
            ),
        )
        history = train_agent(
            agent,
            env,
            training_config=TrainingConfig(total_timesteps=96, eval_every_updates=1, eval_episodes=1),
            eval_env=eval_env,
        )
        evaluation = evaluate_agent(agent, eval_env, episodes=1)
        self.assertGreaterEqual(len(history["updates"]), 1)
        self.assertEqual(len(evaluation["reward_stats"]["mean_vector_reward"]), 3)

    def test_full_environment_one_step(self):
        env = PyPSATEPEnv(self.dataset, self.env_config)
        env.reset(seed=1)
        _, reward, _, _, info = env.step(env.action_space.sample())
        self.assertEqual(reward.shape[0], 3)
        self.assertIn("backend", info)

    def test_extract_country_subnetwork_from_elec(self):
        source_path = Path("elec_s_512.nc")
        network = pypsa.Network()
        network.import_from_netcdf(source_path, skip_time=True)

        subnet, border_frame, summary = extract_country_subnetwork(network, country="AT")

        self.assertEqual(len(subnet.buses), 11)
        self.assertEqual(len(subnet.lines), 14)
        self.assertEqual(len(subnet.generators), 41)
        self.assertEqual(len(subnet.loads), 11)
        self.assertEqual(len(subnet.storage_units), 16)
        self.assertEqual(summary["border_component_counts"]["lines"], 13)
        neighbours = {
            country
            for entry in border_frame["external_countries"].tolist()
            for country in str(entry).split("|")
            if country
        }
        self.assertTrue({"CH", "CZ", "DE", "HU", "IT", "SI"}.issubset(neighbours))

    def test_external_profiles_override_mismatched_network_year(self):
        network = pypsa.Network()
        network.set_snapshots(pd.date_range("2013-01-01", periods=3, freq="h"))
        network.add("Bus", "AT-bus")
        network.buses.loc["AT-bus", "country"] = "AT"
        network.add("Load", "AT-load", bus="AT-bus", p_set=1.0)
        network.add("Generator", "AT-wind", bus="AT-bus", p_nom=10.0, carrier="onwind")
        network.loads_t.p_set = pd.DataFrame({"AT-load": [1.0, 2.0, 3.0]}, index=network.snapshots)
        network.generators_t.p_max_pu = pd.DataFrame({"AT-wind": [0.1, 0.2, 0.3]}, index=network.snapshots)

        target_snapshots = pd.date_range("2020-01-01", periods=3, freq="h")
        country_load = pd.Series([10.0, 20.0, 30.0], index=target_snapshots)
        wind_cf = pd.Series([0.6, 0.7, 0.8], index=target_snapshots)
        solar_cf = pd.Series([0.0, 0.0, 0.0], index=target_snapshots)

        _, load_profiles = _build_bus_demand_profiles(network, target_snapshots, country_load)
        availability = _build_generator_availability(network, target_snapshots, wind_cf, solar_cf)

        self.assertEqual(load_profiles["AT-load"].tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(availability["AT-wind"].tolist(), [0.6, 0.7, 0.8])

    def test_load_austria_case_respects_time_window(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            snapshots = pd.date_range("2020-01-01", periods=96, freq="h")

            network = pypsa.Network()
            network.set_snapshots(snapshots)
            network.add("Bus", "AT-bus")
            network.buses.loc["AT-bus", "country"] = "AT"
            network.add("Load", "AT-load", bus="AT-bus", p_set=10.0)
            network.add("Generator", "AT-wind", bus="AT-bus", p_nom=20.0, carrier="onwind")
            network_path = temp_path / "toy_window.nc"
            network.export_to_netcdf(network_path)

            wind_path = temp_path / "wind.csv"
            solar_path = temp_path / "solar.csv"
            load_path = temp_path / "opsd.csv"

            ninja_frame = pd.DataFrame({"time": snapshots, "NATIONAL": [0.5] * len(snapshots)})
            ninja_frame.to_csv(wind_path, index=False)
            ninja_frame.to_csv(solar_path, index=False)

            load_frame = pd.DataFrame(
                {
                    "utc_timestamp": snapshots,
                    "AT_load_actual_entsoe_transparency": [100.0] * len(snapshots),
                }
            )
            load_frame.to_csv(load_path, index=False)

            dataset = load_austria_case(
                NetworkConfig(
                    network_path=network_path,
                    load_path=load_path,
                    wind_path=wind_path,
                    solar_path=solar_path,
                    year=2020,
                    start="2020-01-02 00:00:00",
                    end="2020-01-03 23:00:00",
                )
            )

            self.assertEqual(str(dataset.snapshots.min()), "2020-01-02 00:00:00")
            self.assertEqual(str(dataset.snapshots.max()), "2020-01-03 23:00:00")
            self.assertEqual(len(dataset.snapshots), 48)

    def test_load_austria_case_rejects_topology_only_network(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            network = pypsa.Network()
            network.set_snapshots(pd.date_range("2020-01-01", periods=24, freq="h"))
            network.add("Bus", "AT-bus")
            network.buses.loc["AT-bus", "country"] = "AT"
            network_path = temp_path / "topology_only.nc"
            network.export_to_netcdf(network_path)

            wind_path = temp_path / "wind.csv"
            solar_path = temp_path / "solar.csv"
            load_path = temp_path / "opsd.csv"
            snapshots = pd.date_range("2020-01-01", periods=24, freq="h")
            pd.DataFrame({"time": snapshots, "NATIONAL": [0.5] * len(snapshots)}).to_csv(wind_path, index=False)
            pd.DataFrame({"time": snapshots, "NATIONAL": [0.5] * len(snapshots)}).to_csv(solar_path, index=False)
            pd.DataFrame(
                {
                    "utc_timestamp": snapshots,
                    "AT_load_actual_entsoe_transparency": [100.0] * len(snapshots),
                }
            ).to_csv(load_path, index=False)

            with self.assertRaisesRegex(ValueError, "does not contain the load and generator components"):
                load_austria_case(
                    NetworkConfig(
                        network_path=network_path,
                        load_path=load_path,
                        wind_path=wind_path,
                        solar_path=solar_path,
                        year=2020,
                    )
                )

    def test_dc_proxy_includes_transformers_and_balances_each_island(self):
        network = pypsa.Network()
        network.add("Bus", "A", v_nom=220.0)
        network.add("Bus", "B", v_nom=220.0)
        network.add("Bus", "C", v_nom=380.0)
        network.add("Bus", "D", v_nom=220.0)
        network.add("Line", "AB", bus0="A", bus1="B", x=0.1, r=0.01, s_nom=100.0)
        network.add("Transformer", "BC", bus0="B", bus1="C", x=0.1, s_nom=100.0)

        model = DCPowerFlowModel.from_network(network, slack_bus="A")
        self.assertEqual(sorted(len(component) for component in model.component_bus_indices), [1, 3])
        dispatch, injections, slack_generation = model.balance_dispatch(
            pd.Series({"A": 100.0, "B": 0.0, "C": 0.0, "D": 0.0}),
            pd.Series({"A": 0.0, "B": 0.0, "C": 80.0, "D": 20.0}),
        )
        self.assertAlmostEqual(float(dispatch.sum()), 80.0)
        self.assertAlmostEqual(float(slack_generation), 20.0)
        for component in model.component_bus_indices:
            self.assertAlmostEqual(float(injections.reindex(model.buses[component]).sum()), 0.0, places=8)
        self.assertGreater(abs(float(model.solve(injections).loc["AB"])), 0.0)

    def test_candidate_screening_uses_loading_not_line_length(self):
        network = pypsa.Network()
        for bus in ("A", "B", "C"):
            network.add("Bus", bus, v_nom=220.0)
        network.add("Line", "short_critical", bus0="A", bus1="B", x=0.1, r=0.01, s_nom=20.0, length=10.0)
        network.add("Line", "long_uncritical", bus0="B", bus1="C", x=0.1, r=0.01, s_nom=1000.0, length=500.0)
        network.add("Line", "return", bus0="A", bus1="C", x=0.1, r=0.01, s_nom=1000.0, length=50.0)
        snapshots = pd.date_range("2023-01-01", periods=24, freq="h")
        demand = pd.DataFrame({"A": 0.0, "B": 100.0, "C": 0.0}, index=snapshots)
        renewable = pd.DataFrame({"A": 100.0, "B": 0.0, "C": 0.0}, index=snapshots)
        ranking = _candidate_line_ranking(network, demand, renewable, sample_count=12)
        self.assertEqual(ranking.index[0], "short_critical")

    def test_evaluation_starts_cover_full_chronology(self):
        snapshots = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="h")
        starts = stratified_episode_start_indices(snapshots, episode_length=24, episodes=10)
        months = {snapshots[index].month for index in starts}
        self.assertEqual(len(starts), 10)
        self.assertGreaterEqual(len(months), 8)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], len(snapshots) - 24)

    def test_validation_and_test_reuse_training_observation_scales(self):
        training = build_toy_dataset(num_steps=48, seed=1)
        testing = build_toy_dataset(num_steps=48, seed=2)
        testing.demand_by_bus *= 10.0
        fit_observation_scales(training)
        fit_observation_scales(testing)
        test_specific_scale = testing.demand_scale.copy()
        apply_observation_scale_reference(testing, training)
        self.assertFalse(np.allclose(test_specific_scale.values, testing.demand_scale.values))
        np.testing.assert_allclose(testing.demand_scale.values, training.demand_scale.values)

    def test_official_future_scenario_catalog_contains_apg_regional_targets(self):
        catalog = load_scenario_catalog()
        scenario = catalog["scenarios"]["apg_tyndp_nt_2040"]
        self.assertEqual(scenario["declared_national_targets_mw"]["onwind"], 16000)
        self.assertEqual(scenario["declared_national_targets_mw"]["solar"], 30000)
        self.assertEqual(scenario["declared_national_targets_mw"]["peak_load"], 28059)
        self.assertEqual(set(scenario["regional_targets_mw"]["solar"]), {
            "AT11", "AT12", "AT13", "AT21", "AT22", "AT31", "AT32", "AT33", "AT34"
        })
        self.assertIn("apg_nep_2025", scenario["source_ids"])
        self.assertAlmostEqual(
            float(catalog["reinforcement_cost_assumption"]["annualized_eur_per_mw_km_year"]),
            22.22,
        )

    def test_2035_future_scenario_is_the_declared_componentwise_midpoint(self):
        catalog = load_scenario_catalog()
        scenario_2030 = catalog["scenarios"]["apg_tyndp_nt_2030"]
        scenario_2035 = catalog["scenarios"]["apg_tyndp_nt_2035_midpoint"]
        scenario_2040 = catalog["scenarios"]["apg_tyndp_nt_2040"]
        self.assertEqual(
            scenario_2035["official_status"],
            "derived_interpolation_not_published_scenario",
        )
        for technology in ("onwind", "solar", "peak_load"):
            for region, value_2035 in scenario_2035["regional_targets_mw"][technology].items():
                expected = (
                    scenario_2030["regional_targets_mw"][technology][region]
                    + scenario_2040["regional_targets_mw"][technology][region]
                ) / 2.0
                self.assertAlmostEqual(value_2035, expected)
            expected_national = (
                scenario_2030["declared_national_targets_mw"][technology]
                + scenario_2040["declared_national_targets_mw"][technology]
            ) / 2.0
            self.assertAlmostEqual(
                scenario_2035["declared_national_targets_mw"][technology],
                expected_national,
            )

    def test_dc_reference_inherits_frozen_rl_formulation(self):
        args = SimpleNamespace(
            episode_length=None,
            budget_mw=None,
            max_upgrade_mw=None,
            line_investment_cost_eur_per_mw_km_year=None,
            load_shedding_cost=None,
        )
        manifest = {
            "environment": {
                "episode_length": 24,
                "budget_mw": 500.0,
                "max_upgrade_mw": 500.0,
                "line_investment_cost_eur_per_mw_km_year": 22.22,
                "load_shedding_cost": 10_000.0,
            }
        }
        resolved = resolve_manifest_defaults(args, manifest)
        self.assertEqual(resolved.episode_length, 24)
        self.assertEqual(resolved.budget_mw, 500.0)
        self.assertEqual(resolved.max_upgrade_mw, 500.0)
        self.assertEqual(resolved.line_investment_cost_eur_per_mw_km_year, 22.22)

    def test_line_cost_uses_documented_mw_km_annuity_and_episode_time_basis(self):
        dataset = build_toy_dataset(num_steps=24, seed=3)
        config = EnvironmentConfig(episode_length=24, decision_interval=6)
        env = ProxyTEPEnv(dataset, config)
        self.assertAlmostEqual(
            float(env.line_upgrade_cost.loc["L_AB"]),
            120.0 * config.line_investment_cost_eur_per_mw_km_year,
        )
        increments = pd.Series(0.0, index=env.candidate_lines)
        increments.loc["L_AB"] = 10.0
        expected_annual = 10.0 * 120.0 * config.line_investment_cost_eur_per_mw_km_year
        self.assertAlmostEqual(env._annualized_investment_cost(increments), expected_annual)
        self.assertAlmostEqual(
            env._episode_investment_cost(increments),
            expected_annual * config.episode_length / config.investment_cost_reference_hours,
        )

    def test_future_scenario_keeps_weather_shapes_and_applies_frozen_load_scale(self):
        base = build_toy_dataset(num_steps=36, seed=4)
        wind_profile = base.generator_availability["Wind_A"].copy()
        solar_profile = base.generator_availability["Solar_C"].copy()
        calibration = {
            "schema_version": 1,
            "scenario_id": "test_official_projection",
            "identity": False,
            "bus_to_nuts2": {"A": "AT11", "B": "AT12", "C": "AT13"},
            "load_scale_by_nuts2": {"AT11": 2.0, "AT12": 3.0, "AT13": 4.0},
            "generation_capacity_by_bus_mw": {
                "onwind": {"A": 160.0},
                "solar": {"C": 300.0},
            },
        }
        transformed = apply_future_scenario(base, calibration)
        np.testing.assert_allclose(transformed.demand_by_bus["A"], base.demand_by_bus["A"] * 2.0)
        np.testing.assert_allclose(transformed.demand_by_bus["B"], base.demand_by_bus["B"] * 3.0)
        np.testing.assert_allclose(transformed.demand_by_bus["C"], base.demand_by_bus["C"] * 4.0)
        self.assertAlmostEqual(float(transformed.network.generators.at["Wind_A", "p_nom"]), 160.0)
        self.assertAlmostEqual(float(transformed.network.generators.at["Solar_C", "p_nom"]), 300.0)
        np.testing.assert_allclose(transformed.generator_availability["Wind_A"], wind_profile)
        np.testing.assert_allclose(transformed.generator_availability["Solar_C"], solar_profile)
        audit = scenario_audit(transformed)
        self.assertAlmostEqual(audit["installed_capacity_mw"]["onwind"], 160.0)
        self.assertAlmostEqual(audit["installed_capacity_mw"]["solar"], 300.0)

    def test_crossed_bootstrap_keeps_seed_and_window_units(self):
        def evaluation(seed_offset: float, policy_offset: float) -> dict:
            values = [100.0 + seed_offset + window + policy_offset for window in range(3)]
            return {
                "total_cost_mean": float(np.mean(values)),
                "episodes": [
                    {
                        "episode": window,
                        "start_index": window * 24,
                        "start_timestamp": str(pd.Timestamp("2023-01-01") + pd.Timedelta(days=window)),
                        "total_cost": value,
                    }
                    for window, value in enumerate(values)
                ],
            }

        results_a = [evaluation(0.0, -5.0), evaluation(2.0, -5.0)]
        results_b = [evaluation(0.0, 0.0), evaluation(2.0, 0.0)]
        comparison = compare_paired_evaluations(
            results_a,
            results_b,
            metrics=("total_cost_mean",),
            bootstrap_samples=200,
        )
        self.assertEqual(int(comparison.iloc[0]["n_training_seeds"]), 2)
        self.assertEqual(int(comparison.iloc[0]["n_shared_windows"]), 3)
        self.assertAlmostEqual(float(comparison.iloc[0]["mean_paired_difference"]), -5.0)

    def test_crossed_bootstrap_uses_episode_investment(self):
        def evaluation(investments: list[float]) -> dict:
            return {
                "action_stats": {"total_investment_mean": float(np.mean(investments))},
                "episodes": [
                    {
                        "episode": window,
                        "start_index": window * 24,
                        "start_timestamp": str(pd.Timestamp("2023-01-01") + pd.Timedelta(days=window)),
                        "total_investment": investment,
                    }
                    for window, investment in enumerate(investments)
                ],
            }

        results_a = [evaluation([10.0, 20.0]), evaluation([30.0, 40.0])]
        results_b = [evaluation([5.0, 15.0]), evaluation([25.0, 35.0])]
        comparison = compare_paired_evaluations(
            results_a,
            results_b,
            metrics=("total_investment_mean",),
            bootstrap_samples=100,
        )
        self.assertAlmostEqual(float(comparison.iloc[0]["mean_paired_difference"]), 5.0)

    def test_batched_output_shapley_recovers_linear_policy_contributions(self):
        class LinearPolicy:
            def policy_mean(self, states):
                states = np.asarray(states, dtype=np.float32)
                weights = np.asarray(
                    [
                        [1.0, 2.0, 0.0],
                        [-1.0, 0.0, 3.0],
                    ],
                    dtype=np.float32,
                )
                return states @ weights.T

        states = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        explainer = PolicyShapleyExplainer(
            agent=LinearPolicy(),
            feature_names=["f0", "f1", "f2"],
            n_samples=8,
            seed=7,
        )
        result = explainer.global_output_shapley(
            states,
            output_indices=[0, 1],
            output_names=["a0", "a1"],
            baseline_states=states,
        )
        a0 = result[result["output"] == "a0"].set_index("feature")["global_importance"]
        a1 = result[result["output"] == "a1"].set_index("feature")["global_importance"]
        np.testing.assert_allclose(a0.reindex(["f0", "f1", "f2"]), [1.0, 4.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(a1.reindex(["f0", "f1", "f2"]), [1.0, 0.0, 9.0], atol=1e-6)

    def test_capacity_preserving_corridor_ablation_keeps_total_and_caps(self):
        increments = pd.Series({"target": 30.0, "a": 20.0, "b": 10.0, "c": 0.0})
        remaining_cap = pd.Series({"target": 50.0, "a": 25.0, "b": 30.0, "c": 100.0})
        result = capacity_preserving_corridor_ablation(
            increments,
            remaining_cap,
            target_line="target",
        )
        self.assertAlmostEqual(float(result.loc["target"]), 0.0)
        self.assertAlmostEqual(float(result.sum()), float(increments.sum()))
        self.assertTrue((result >= 0.0).all())
        self.assertTrue((result <= remaining_cap + 1e-8).all())
        self.assertGreater(float(result.loc["a"] + result.loc["b"]), 30.0)

        grouped = capacity_preserving_corridor_ablation(
            increments,
            remaining_cap,
            target_line=["target", "b"],
        )
        self.assertAlmostEqual(float(grouped.loc[["target", "b"]].sum()), 0.0)
        self.assertAlmostEqual(float(grouped.sum()), float(increments.sum()))
        self.assertTrue((grouped <= remaining_cap + 1e-8).all())

        no_target = pd.Series({"target": 0.0, "a": 20.0, "b": 10.0, "c": 0.0})
        no_target_result = capacity_preserving_corridor_ablation(
            no_target,
            remaining_cap,
            target_line="target",
        )
        pd.testing.assert_series_equal(no_target_result, no_target)

        # The online wrapper must also preserve the original action object for a
        # true no-op.  This guards against a float32 decode/re-encode round trip
        # creating a numerical placebo effect in zero-target seeds.
        from scripts.run_robust_explainability import CapacityPreservingAblationAgent

        original_action = np.asarray([0.25, 0.75], dtype=np.float32)

        class _BaseAgent:
            env_reward_dim = 1

            @staticmethod
            def act(observation, deterministic=True):
                return original_action, 0.0, 0.0

        class _Environment:
            candidate_lines = pd.Index(["target", "a"])

            @staticmethod
            def _available_budget():
                return 10.0

            @staticmethod
            def _remaining_line_upgrade_cap():
                return pd.Series({"target": 10.0, "a": 10.0})

            @staticmethod
            def _decode_action(action, available_budget, remaining_line_cap):
                return pd.Series({"target": 0.0, "a": 7.5})

        wrapper = CapacityPreservingAblationAgent(_BaseAgent(), _Environment(), ["target"])
        wrapped_action, _, _ = wrapper.act(np.zeros(1), deterministic=True)
        self.assertIs(wrapped_action, original_action)

    def test_episode_grouped_split_has_no_episode_leakage(self):
        group_ids = np.repeat(np.arange(10), 4)
        train_idx, test_idx = episode_grouped_train_test_indices(
            group_ids,
            test_fraction=0.25,
            seed=14,
        )
        train_groups = set(group_ids[train_idx].tolist())
        test_groups = set(group_ids[test_idx].tolist())
        self.assertFalse(train_groups & test_groups)
        self.assertEqual(len(train_groups), 8)
        self.assertEqual(len(test_groups), 2)
        self.assertEqual(len(train_idx) + len(test_idx), len(group_ids))

    def test_corridor_importance_excludes_bus_features_with_line_like_ids(self):
        network = pypsa.Network()
        network.add("Bus", "shared-osm-id")
        network.add("Bus", "other-bus")
        network.add(
            "Line",
            "shared-osm-id",
            bus0="shared-osm-id",
            bus1="other-bus",
            x=0.1,
            r=0.01,
            s_nom=100.0,
        )
        importance = pd.DataFrame(
            {
                "feature": [
                    "renewable::shared-osm-id",
                    "line_loading_raw::shared-osm-id",
                    "candidate_loading_raw::shared-osm-id",
                    "upgrade::shared-osm-id",
                ],
                "global_importance": [100.0, 1.0, 2.0, 3.0],
            }
        )
        aggregated = aggregate_line_importance(importance, network)
        self.assertEqual(len(aggregated), 1)
        self.assertAlmostEqual(float(aggregated.iloc[0]["line_importance"]), 6.0)
        self.assertAlmostEqual(float(aggregated.iloc[0]["raw_loading_importance"]), 1.0)
        self.assertAlmostEqual(float(aggregated.iloc[0]["candidate_loading_importance"]), 2.0)
        self.assertAlmostEqual(float(aggregated.iloc[0]["upgrade_importance"]), 3.0)


if __name__ == "__main__":
    unittest.main()
