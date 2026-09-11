"""Permutation-sampled Shapley policy attribution for the TEP-RL agent.

The implementation combines interventional permutation Shapley attribution
with a Shapley-guided multi-output ridge surrogate.

This module provides:

  * ``PolicyShapleyExplainer`` - main explainability class
  * ``local_shapley``         - per-step feature attribution (SHAP-like)
  * ``temporal_attribution``  - how feature importance shifts across the
                                decision stages of one episode
  * ``global_shapley``        - population-level importance aggregated over
                                many states (equivalent to global SHAP)
  * ``counterfactual_action`` - find the minimal feature perturbation that
                                changes the agent's preferred action

Design notes
------------
We use *permutation sampling* (a model-agnostic approximation of Shapley
values) rather than exact computation, because the final Austrian case has
459 observation features; exact Shapley would require evaluation over
``2**459`` coalitions.  Monte Carlo and observed-reference sensitivity are
therefore quantified through independent repeated runs (Strumbelj &
Kononenko, 2014).

Reference
---------
Beechey, D., Smith, T. M. S., & Simsek, O. (2023). Explaining Reinforcement
Learning with Shapley Values. Proceedings of ICML 2023.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level Shapley approximation
# ---------------------------------------------------------------------------

def _policy_output(agent, states: np.ndarray) -> np.ndarray:
    """Return the deterministic policy mean for a batch of states."""
    return agent.policy_mean(states)          # shape (n_states, action_dim)


def _permutation_shapley(
    agent,
    state: np.ndarray,
    baseline: np.ndarray,
    feature_indices: Sequence[int],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Approximate Shapley values for one state via permutation sampling.

    For each sampled permutation of features, compute the marginal
    contribution of adding feature i to the coalition that precedes it.

    Returns
    -------
    np.ndarray, shape (n_features, action_dim)
        Shapley value of each feature for each action dimension.
    """
    n_features = len(feature_indices)
    action_dim = _policy_output(agent, state[np.newaxis]).shape[1]
    shapley = np.zeros((n_features, action_dim), dtype=np.float64)

    for _ in range(n_samples):
        perm = rng.permutation(n_features)
        current = baseline.copy()
        prev_output = _policy_output(agent, current[np.newaxis])[0]

        for pos in range(n_features):
            feat_idx = feature_indices[perm[pos]]
            current[feat_idx] = state[feat_idx]
            new_output = _policy_output(agent, current[np.newaxis])[0]
            shapley[perm[pos]] += new_output - prev_output
            prev_output = new_output

    return shapley / n_samples


def _batched_permutation_shapley(
    agent,
    state: np.ndarray,
    baseline: np.ndarray,
    feature_indices: Sequence[int],
    n_samples: int,
    rng: np.random.Generator,
    output_indices: Sequence[int],
) -> np.ndarray:
    """Approximate selected policy outputs with batched permutation Shapley.

    One complete coalition path is evaluated as a batch.  This is numerically
    equivalent to the sequential permutation estimator above, but makes
    repeated seed/reference audits tractable for high-dimensional observations.

    Returns
    -------
    np.ndarray, shape (n_features, n_outputs)
        Signed Shapley contribution for each requested policy output.
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float32).reshape(-1)
    if state.shape != baseline.shape:
        raise ValueError("state and baseline must have identical shapes.")
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be positive.")

    feature_indices = list(feature_indices)
    output_indices = list(output_indices)
    if not feature_indices:
        return np.zeros((0, len(output_indices)), dtype=np.float64)
    if not output_indices:
        raise ValueError("At least one output index is required.")

    probe = _policy_output(agent, state[np.newaxis])
    action_dim = int(probe.shape[1])
    if min(output_indices) < 0 or max(output_indices) >= action_dim:
        raise IndexError(
            f"Requested output indices {output_indices} outside policy action dimension {action_dim}."
        )

    n_features = len(feature_indices)
    shapley = np.zeros((n_features, len(output_indices)), dtype=np.float64)
    coalition_states = np.empty((n_features + 1, state.size), dtype=np.float32)

    for _ in range(int(n_samples)):
        permutation = rng.permutation(n_features)
        current = baseline.copy()
        coalition_states[0] = current
        for position, local_feature_index in enumerate(permutation, start=1):
            state_feature_index = feature_indices[int(local_feature_index)]
            current[state_feature_index] = state[state_feature_index]
            coalition_states[position] = current

        outputs = _policy_output(agent, coalition_states)[:, output_indices]
        marginal = np.diff(outputs, axis=0)
        shapley[permutation] += marginal

    return shapley / float(n_samples)


# ---------------------------------------------------------------------------
# Main explainer class
# ---------------------------------------------------------------------------

class PolicyShapleyExplainer:
    """
    Interventional permutation-Shapley explainer for a PPO / MO-PPO policy.

    Parameters
    ----------
    agent:
        Trained ``PPOAgent`` or ``MOPPOAgent``.
    feature_names:
        List of feature names matching the observation dimension.
    n_samples:
        Number of permutation samples per Shapley estimate. Higher means more
        accurate but slower. 50-200 is usually sufficient.
    seed:
        Random seed for permutation sampling.
    """

    def __init__(
        self,
        agent,
        feature_names: list[str],
        n_samples: int = 100,
        seed: int = 7,
    ) -> None:
        self.agent = agent
        self.feature_names = feature_names
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self._feature_indices = list(range(len(feature_names)))

    # ------------------------------------------------------------------
    # Local explanation  (one step / one state)
    # ------------------------------------------------------------------

    def local_shapley(
        self,
        state: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute per-feature Shapley values for a single observation.

        Parameters
        ----------
        state:
            Observation vector, shape ``(obs_dim,)``.
        baseline:
            Reference point representing a "neutral" observation.  Defaults
            to the zero vector.  For TEP, consider using the mean of a
            representative state distribution.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``shapley_action_0``, ..., ``shapley_action_k``,
            ``shapley_mean`` (absolute mean across action dimensions).
        """
        if baseline is None:
            baseline = np.zeros_like(state)

        shapley = _permutation_shapley(
            self.agent,
            state,
            baseline,
            self._feature_indices,
            self.n_samples,
            self.rng,
        )
        action_dim = shapley.shape[1]
        records = []
        for i, name in enumerate(self.feature_names):
            row: dict = {"feature": name}
            for j in range(action_dim):
                row[f"shapley_action_{j}"] = float(shapley[i, j])
            row["shapley_mean"] = float(np.abs(shapley[i]).mean())
            records.append(row)

        return (
            pd.DataFrame(records)
            .sort_values("shapley_mean", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Temporal attribution  (one episode)
    # ------------------------------------------------------------------

    def temporal_attribution(
        self,
        episode_states: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        decision_steps: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        """
        Track how feature importance shifts across the decision horizon.

        Parameters
        ----------
        episode_states:
            Shape ``(episode_length, obs_dim)``.
        baseline:
            Reference observation (defaults to zero vector).
        decision_steps:
            Indices of decision steps within the episode.  If ``None``, all
            steps are included.

        Returns
        -------
        pd.DataFrame
            Long-format with columns: ``step``, ``feature``, ``shapley_mean``.
            Suitable for a seaborn lineplot or heatmap.
        """
        if baseline is None:
            baseline = np.zeros(episode_states.shape[1])

        steps = list(decision_steps) if decision_steps is not None else list(range(len(episode_states)))
        rows = []
        for step in steps:
            shapley = _permutation_shapley(
                self.agent,
                episode_states[step],
                baseline,
                self._feature_indices,
                self.n_samples,
                self.rng,
            )
            for i, name in enumerate(self.feature_names):
                rows.append({
                    "step": step,
                    "feature": name,
                    "shapley_mean": float(np.abs(shapley[i]).mean()),
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Global policy attribution (population level)
    # ------------------------------------------------------------------

    def global_shapley(
        self,
        states: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Aggregate local Shapley values over a state distribution.

        This is the global feature importance analogous to global SHAP: it
        shows which features drive the agent's behaviour *on average* across
        the observed state distribution.

        Parameters
        ----------
        states:
            Shape ``(n_states, obs_dim)``.
        baseline:
            Reference observation.  Defaults to the column-wise mean of
            ``states`` (a data-driven baseline).
        top_k:
            If set, return only the top-k features by mean |Shapley|.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``global_importance`` (mean |Shapley|),
            ``global_std`` (std of |Shapley| across states).
        """
        logger.info(
            "Computing global policy attribution over %d states with %d samples each ...",
            len(states), self.n_samples,
        )

        all_importance = []
        for state in states:
            # Use an actually observed state as the reference endpoint. This
            # avoids the former column-wise mean reference, which could be far
            # outside the joint observation manifold. Hybrid coalitions remain
            # interventional and are interpreted accordingly.
            state_baseline = baseline
            if state_baseline is None:
                state_baseline = states[int(self.rng.integers(0, len(states)))]
            shapley = _permutation_shapley(
                self.agent,
                state,
                state_baseline,
                self._feature_indices,
                self.n_samples,
                self.rng,
            )
            all_importance.append(np.abs(shapley).mean(axis=1))   # (n_features,)

        importance_matrix = np.vstack(all_importance)   # (n_states, n_features)
        rows = []
        for i, name in enumerate(self.feature_names):
            col = importance_matrix[:, i]
            rows.append({
                "feature": name,
                "global_importance": float(col.mean()),
                "global_std": float(col.std()),
            })

        df = (
            pd.DataFrame(rows)
            .sort_values("global_importance", ascending=False)
            .reset_index(drop=True)
        )
        if top_k is not None:
            df = df.head(top_k)
        return df

    def global_output_shapley(
        self,
        states: np.ndarray,
        output_indices: Sequence[int],
        output_names: Sequence[str] | None = None,
        baseline_states: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Global Shapley importance for explicitly selected action outputs.

        Unlike :meth:`global_shapley`, this method does not average over every
        action dimension.  It therefore distinguishes, for example, the spend
        fraction from allocation scores for specific transmission corridors.
        Each explained state is paired with a different observed reference
        state whenever at least two references are available.
        """
        references_are_explained_states = baseline_states is None or baseline_states is states
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or len(states) == 0:
            raise ValueError("states must have shape (n_states, n_features) with n_states > 0.")
        if states.shape[1] != len(self.feature_names):
            raise ValueError("State feature dimension does not match feature_names.")

        references = states if baseline_states is None else np.asarray(baseline_states, dtype=np.float32)
        if references.ndim != 2 or references.shape[1] != states.shape[1] or len(references) == 0:
            raise ValueError("baseline_states must contain observed states with the same feature dimension.")

        output_indices = list(output_indices)
        if output_names is None:
            output_names = [f"action_{index}" for index in output_indices]
        output_names = list(output_names)
        if len(output_names) != len(output_indices):
            raise ValueError("output_names must have one entry per output index.")

        per_state: list[np.ndarray] = []
        for state_index, state in enumerate(states):
            if len(references) == 1:
                reference_index = 0
            else:
                reference_index = int(self.rng.integers(0, len(references) - 1))
                if references_are_explained_states and reference_index >= state_index:
                    reference_index += 1
                reference_index %= len(references)
            shapley = _batched_permutation_shapley(
                self.agent,
                state,
                references[reference_index],
                self._feature_indices,
                self.n_samples,
                self.rng,
                output_indices,
            )
            per_state.append(shapley)

        contribution_cube = np.stack(per_state, axis=0)
        absolute_cube = np.abs(contribution_cube)
        rows: list[dict[str, float | int | str]] = []
        for output_position, (output_index, output_name) in enumerate(zip(output_indices, output_names)):
            for feature_index, feature_name in enumerate(self.feature_names):
                values = contribution_cube[:, feature_index, output_position]
                absolute_values = absolute_cube[:, feature_index, output_position]
                rows.append(
                    {
                        "output": output_name,
                        "output_index": int(output_index),
                        "feature": feature_name,
                        "global_importance": float(absolute_values.mean()),
                        "global_std": float(absolute_values.std()),
                        "signed_mean": float(values.mean()),
                        "n_states": int(len(states)),
                    }
                )
        return pd.DataFrame(rows).sort_values(
            ["output", "global_importance"], ascending=[True, False]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Counterfactual analysis
    # ------------------------------------------------------------------

    def counterfactual_action(
        self,
        state: np.ndarray,
        target_action: np.ndarray,
        candidate_features: Optional[Sequence[int]] = None,
        max_iter: int = 200,
        step_size: float = 0.02,
        tolerance: float = 0.05,
    ) -> dict:
        """
        Find the minimal feature perturbation that shifts the policy toward
        a desired target action.

        Uses gradient ascent on the action similarity loss in observation
        space.  Only the features listed in ``candidate_features`` are
        perturbed; all others are held fixed.

        Parameters
        ----------
        state:
            Current observation, shape ``(obs_dim,)``.
        target_action:
            Desired action, shape ``(action_dim,)``.
        candidate_features:
            Indices of features allowed to be perturbed.  Defaults to all
            features except the budget / progress scalars (last 7 dimensions).
        max_iter:
            Maximum gradient-ascent steps.
        step_size:
            Learning rate for the perturbation update.
        tolerance:
            L2 distance to target_action considered "close enough".

        Returns
        -------
        dict with keys:
            ``perturbed_state``, ``achieved_action``, ``delta``,
            ``n_iter``, ``converged``, ``feature_delta`` (per-feature change).
        """
        if candidate_features is None:
            # Exclude scalar global features (last 7 dims by convention).
            candidate_features = list(range(len(self.feature_names) - 7))

        device = self.agent.device
        perturbed = torch.tensor(
            state.copy(), dtype=torch.float32, device=device, requires_grad=False
        )
        target = torch.tensor(target_action, dtype=torch.float32, device=device)
        mask = torch.zeros_like(perturbed)
        for idx in candidate_features:
            mask[idx] = 1.0

        converged = False
        for iteration in range(max_iter):
            perturbed_req = perturbed.clone().requires_grad_(True)
            achieved = self.agent.network.policy_mean(perturbed_req.unsqueeze(0)).squeeze(0)
            loss = torch.nn.functional.mse_loss(achieved, target)

            if loss.item() < tolerance ** 2:
                converged = True
                break

            loss.backward()
            with torch.no_grad():
                grad = perturbed_req.grad * mask   # only allowed features
                perturbed = perturbed - step_size * grad
                perturbed = perturbed.detach()

        with torch.no_grad():
            achieved_action = (
                self.agent.network.policy_mean(perturbed.unsqueeze(0))
                .squeeze(0)
                .cpu()
                .numpy()
            )

        delta = (perturbed.cpu().numpy() - state)
        feature_delta = pd.Series(delta, index=self.feature_names)

        return {
            "perturbed_state": perturbed.cpu().numpy(),
            "achieved_action": achieved_action,
            "delta": delta,
            "n_iter": iteration + 1,
            "converged": converged,
            "feature_delta": feature_delta,
        }


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def collect_policy_states(
    env,
    agent,
    episodes: int = 5,
    deterministic: bool = True,
    return_episode_ids: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Roll out shared stratified windows and collect states and episode IDs."""
    from .evaluation import stratified_episode_start_indices

    states = []
    episode_ids: list[int] = []
    starts = stratified_episode_start_indices(env.dataset.snapshots, env.config.episode_length, episodes)
    for episode, start_index in enumerate(starts):
        observation, _ = env.reset(seed=episode, options={"start_index": int(start_index)})
        terminated = False
        while not terminated:
            states.append(observation.copy())
            episode_ids.append(episode)
            action, _, _ = agent.act(observation, deterministic=deterministic)
            observation, _, terminated, _, _ = env.step(action)
    state_array = np.asarray(states, dtype=np.float32)
    if return_episode_ids:
        return state_array, np.asarray(episode_ids, dtype=int)
    return state_array


def permutation_feature_importance(
    agent,
    states: np.ndarray,
    feature_names: list[str],
    repeats: int = 5,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Baseline permutation importance (faster than Shapley sampling, no coalitions).

    This is a quick sanity check alongside the full Shapley analysis.
    """
    rng = np.random.default_rng(seed)
    baseline_actions = agent.policy_mean(states)
    rows = []
    for feat_idx, feat_name in enumerate(feature_names):
        shifts = []
        for _ in range(repeats):
            permuted = states.copy()
            permuted[:, feat_idx] = rng.permutation(permuted[:, feat_idx])
            shifts.append(float(np.mean(np.abs(agent.policy_mean(permuted) - baseline_actions))))
        rows.append({"feature": feat_name, "importance": float(np.mean(shifts))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def policy_sensitivity(
    agent,
    states: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Gradient-based sensitivity (fast; linear approximation only)."""
    observations = torch.tensor(
        states, dtype=torch.float32, device=agent.device, requires_grad=True
    )
    policy_mean = agent.network.policy_mean(observations)
    policy_mean.mean().backward()
    gradients = observations.grad.detach().abs().mean(dim=0).cpu().numpy()
    return (
        pd.DataFrame({"feature": feature_names, "sensitivity": gradients})
        .sort_values("sensitivity", ascending=False)
        .reset_index(drop=True)
    )


def episode_grouped_train_test_indices(
    group_ids: np.ndarray,
    test_fraction: float = 0.25,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return state indices for a reproducible episode-grouped holdout split."""
    group_ids = np.asarray(group_ids)
    if group_ids.ndim != 1:
        raise ValueError("group_ids must be one-dimensional.")
    groups = np.unique(group_ids)
    if len(groups) < 2:
        raise ValueError("At least two episode groups are required for grouped holdout validation.")
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(groups)
    n_test_groups = min(max(1, int(round(len(groups) * float(test_fraction)))), len(groups) - 1)
    test_groups = set(shuffled_groups[:n_test_groups].tolist())
    test_mask = np.asarray([group in test_groups for group in group_ids], dtype=bool)
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def shapley_guided_ridge_surrogate(
    agent,
    states: np.ndarray,
    feature_names: list[str],
    global_importance: pd.DataFrame,
    action_names: list[str] | None = None,
    top_k: int = 12,
    ridge_alpha: float = 1e-3,
    test_fraction: float = 0.25,
    seed: int = 7,
    group_ids: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit a Shapley-guided explanation-regression surrogate policy.

    For the continuous-action policy, this implementation selects the most
    important state features from global permutation-Shapley attribution and
    fits a transparent multi-output ridge regression surrogate to the
    deterministic policy mean.  Callers reporting holdout fidelity must supply
    ``global_importance`` computed only from the training groups; the robust
    thesis analysis enforces that ordering.  The resulting fidelity metrics
    quantify how well this interpretable surrogate reproduces the neural policy.

    Returns
    -------
    metrics, coefficients, predictions:
        ``metrics`` contains per-action and aggregate fidelity statistics.
        ``coefficients`` contains feature coefficients for every action.
        ``predictions`` stores a compact holdout prediction sample for audit.
        Pass episode IDs as ``group_ids`` to prevent decision stages from the
        same episode appearing in both training and holdout data.
    """
    if states.ndim != 2:
        raise ValueError("states must have shape (n_states, obs_dim).")
    if len(states) < 4:
        raise ValueError("At least four states are required for explanation regression.")

    ranking = global_importance.copy()
    if "feature_raw" in ranking.columns:
        ranking_features = list(ranking["feature_raw"])
    else:
        ranking_features = list(ranking["feature"])
    selected_names = [name for name in ranking_features if name in feature_names][:top_k]
    if not selected_names:
        selected_names = feature_names[: min(top_k, len(feature_names))]
    selected_idx = [feature_names.index(name) for name in selected_names]

    x = states[:, selected_idx].astype(np.float64, copy=True)
    y = agent.policy_mean(states).astype(np.float64, copy=False)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if group_ids is not None:
        group_ids = np.asarray(group_ids)
        if group_ids.shape != (len(states),):
            raise ValueError("group_ids must have shape (n_states,).")
        train_idx, test_idx = episode_grouped_train_test_indices(
            group_ids,
            test_fraction=test_fraction,
            seed=seed,
        )
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(states))
        n_test = max(1, int(round(len(states) * float(test_fraction))))
        n_test = min(n_test, len(states) - 2)
        test_idx = order[:n_test]
        train_idx = order[n_test:]

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_train_z = (x_train - mean) / std
    x_test_z = (x_test - mean) / std

    x_train_aug = np.column_stack([np.ones(len(x_train_z)), x_train_z])
    x_test_aug = np.column_stack([np.ones(len(x_test_z)), x_test_z])

    penalty = np.eye(x_train_aug.shape[1]) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(x_train_aug.T @ x_train_aug + penalty) @ x_train_aug.T @ y_train
    y_pred = x_test_aug @ beta

    action_names = action_names or [f"action_{idx}" for idx in range(y.shape[1])]
    action_names = action_names[: y.shape[1]]

    metric_rows = []
    eps = 1e-12
    for action_idx, action_name in enumerate(action_names):
        truth = y_test[:, action_idx]
        pred = y_pred[:, action_idx]
        mse = float(np.mean((truth - pred) ** 2))
        mae = float(np.mean(np.abs(truth - pred)))
        variance = float(np.var(truth))
        r2 = float(1.0 - mse / max(variance, eps)) if variance > eps else float("nan")
        metric_rows.append(
            {
                "action": action_name,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "target_std": float(np.std(truth)),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.loc[len(metrics)] = {
        "action": "__mean__",
        "mse": float(metrics["mse"].mean()),
        "mae": float(metrics["mae"].mean()),
        "r2": float(metrics["r2"].dropna().mean()) if metrics["r2"].notna().any() else float("nan"),
        "target_std": float(metrics["target_std"].mean()),
    }
    centred = y_test - y_test.mean(axis=0, keepdims=True)
    total_sse = float(np.square(y_test - y_pred).sum())
    total_sst = float(np.square(centred).sum())
    metrics.loc[len(metrics)] = {
        "action": "__variance_weighted__",
        "mse": total_sse / max(float(y_test.size), 1.0),
        "mae": float(np.abs(y_test - y_pred).mean()),
        "r2": 1.0 - total_sse / total_sst if total_sst > eps else float("nan"),
        "target_std": float(np.std(y_test)),
    }

    coef_rows = []
    for action_idx, action_name in enumerate(action_names):
        coef_rows.append(
            {
                "action": action_name,
                "feature": "__intercept__",
                "coefficient": float(beta[0, action_idx]),
                "abs_coefficient": abs(float(beta[0, action_idx])),
            }
        )
        for feature_idx, feature_name in enumerate(selected_names, start=1):
            coefficient = float(beta[feature_idx, action_idx])
            coef_rows.append(
                {
                    "action": action_name,
                    "feature": feature_name,
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient),
                }
            )
    coefficients = pd.DataFrame(coef_rows)

    prediction_rows = []
    sample_count = min(100, len(test_idx))
    for row_pos in range(sample_count):
        for action_idx, action_name in enumerate(action_names):
            prediction_rows.append(
                {
                    "state_index": int(test_idx[row_pos]),
                    "episode_group": (
                        int(group_ids[test_idx[row_pos]]) if group_ids is not None else -1
                    ),
                    "action": action_name,
                    "policy_mean": float(y_test[row_pos, action_idx]),
                    "surrogate_prediction": float(y_pred[row_pos, action_idx]),
                    "absolute_error": float(abs(y_test[row_pos, action_idx] - y_pred[row_pos, action_idx])),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    return metrics, coefficients, predictions


def capacity_preserving_corridor_ablation(
    increments: pd.Series,
    remaining_line_cap: pd.Series,
    target_line: str | Sequence[str],
) -> pd.Series:
    """Remove target-corridor increments and redistribute the same MW elsewhere.

    Redistribution first follows the policy's remaining positive allocation
    pattern.  If those corridors saturate, residual capacity on all non-target
    candidates is used proportionally.  The function never reallocates to the
    ablated corridor and preserves total MW whenever sufficient alternative
    capacity exists.
    """
    base = increments.astype(float).copy()
    caps = remaining_line_cap.reindex(base.index).fillna(0.0).clip(lower=0.0).astype(float)
    targets = [target_line] if isinstance(target_line, str) else list(target_line)
    missing = [line for line in targets if line not in base.index]
    if missing:
        raise KeyError(f"Target corridor members {missing!r} are not present in the candidate action set.")
    if (base < -1e-10).any() or (base - caps > 1e-8).any():
        raise ValueError("increments must be non-negative and must not exceed remaining_line_cap.")

    released = float(base.loc[targets].clip(lower=0.0).sum())
    result = base.clip(lower=0.0)
    result.loc[targets] = 0.0
    if released <= 1e-12:
        return result

    available = (caps - result).clip(lower=0.0)
    available.loc[targets] = 0.0
    if float(available.sum()) + 1e-8 < released:
        raise ValueError("Insufficient non-target capacity for capacity-preserving ablation.")

    remaining = released
    preferred_weights = result.copy()
    preferred_weights.loc[targets] = 0.0
    for weights in (preferred_weights, available.copy()):
        while remaining > 1e-10:
            active = (available > 1e-10) & (weights > 0.0)
            if not active.any():
                break
            active_weights = weights.loc[active]
            proposal = active_weights / float(active_weights.sum()) * remaining
            allocated = np.minimum(
                proposal.to_numpy(dtype=float),
                available.loc[active].to_numpy(dtype=float),
            )
            allocated_series = pd.Series(allocated, index=active_weights.index, dtype=float)
            amount = float(allocated_series.sum())
            if amount <= 1e-12:
                break
            result.loc[active_weights.index] += allocated_series
            available.loc[active_weights.index] -= allocated_series
            remaining -= amount
        if remaining <= 1e-10:
            break

    if remaining > 1e-7:
        raise RuntimeError(f"Could not redistribute {remaining:.6g} MW during corridor ablation.")
    result.loc[targets] = 0.0
    return result.clip(lower=0.0)
