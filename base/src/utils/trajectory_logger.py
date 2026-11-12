"""Builds a logged (state, action, reward, next_state, done) dataset from a
mix of behavior policies, simulating "logged historical data" for offline RL
(Phase 3 spec item 2). Saved as a compressed .npz (transition arrays) plus a
companion per-episode CSV (which policy generated it, task, outcome) for
transparency/auditing.
"""

import os
from typing import Callable, List, NamedTuple, Tuple

import numpy as np

from src.envs.tasks import NUM_TASKS


class BehaviorPolicy(NamedTuple):
    name: str
    act_fn: Callable[[np.ndarray], int]
    reset_fn: Callable[[], None]  # no-op for stateless policies
    weight: float  # relative probability of being chosen per episode


def generate_logged_dataset(
    env,
    policies: List[BehaviorPolicy],
    n_episodes: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    weights = np.array([p.weight for p in policies], dtype=np.float64)
    weights /= weights.sum()

    states, actions, rewards, next_states, dones = [], [], [], [], []
    episode_records = []

    for ep in range(n_episodes):
        policy = policies[rng.choice(len(policies), p=weights)]
        policy.reset_fn()
        obs, info = env.reset(seed=seed + ep, options={"task_id": ep % NUM_TASKS})
        done = False
        ep_reward = 0.0
        while not done:
            action = policy.act_fn(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_obs)
            dones.append(float(done))

            obs = next_obs
            ep_reward += reward

        episode_records.append(
            {
                "episode": ep,
                "policy": policy.name,
                "task_id": info["task_id"],
                "task_name": info["task_name"],
                "reward": ep_reward,
                "success": bool(info["success"]),
                "cost": info["cost"],
                "latency": info["latency"],
                "steps": info["steps"],
            }
        )

    dataset = {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
    }
    return dataset, episode_records


def save_dataset(dataset: dict, episode_records: list, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "logged_transitions.npz"), **dataset)
    import pandas as pd

    pd.DataFrame(episode_records).to_csv(os.path.join(out_dir, "logged_episodes.csv"), index=False)


def load_dataset(out_dir: str) -> dict:
    data = np.load(os.path.join(out_dir, "logged_transitions.npz"))
    return {k: data[k] for k in data.files}
