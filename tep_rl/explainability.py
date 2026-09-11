from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def collect_policy_states(env, agent, episodes: int = 5, deterministic: bool = True) -> np.ndarray:
    states = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=episode)
        terminated = False
        while not terminated:
            states.append(observation.copy())
            action, _, _ = agent.act(observation, deterministic=deterministic)
            observation, _, terminated, _, _ = env.step(action)
    return np.asarray(states, dtype=np.float32)


def permutation_feature_importance(
    agent,
    states: np.ndarray,
    feature_names: list[str],
    repeats: int = 5,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline_actions = agent.policy_mean(states)
    rows = []

    for feature_idx, feature_name in enumerate(feature_names):
        shifts = []
        for _ in range(repeats):
            permuted = states.copy()
            permuted[:, feature_idx] = rng.permutation(permuted[:, feature_idx])
            permuted_actions = agent.policy_mean(permuted)
            shifts.append(float(np.mean(np.abs(permuted_actions - baseline_actions))))
        rows.append({"feature": feature_name, "importance": float(np.mean(shifts))})

    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def policy_sensitivity(
    agent,
    states: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    observations = torch.tensor(states, dtype=torch.float32, device=agent.device, requires_grad=True)
    policy_mean = agent.network.policy_mean(observations)
    score = policy_mean.mean()
    score.backward()

    gradients = observations.grad.detach().abs().mean(dim=0).cpu().numpy()
    frame = pd.DataFrame({"feature": feature_names, "sensitivity": gradients})
    return frame.sort_values("sensitivity", ascending=False).reset_index(drop=True)
