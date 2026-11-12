"""Phase 3 training/evaluation entry point: PPO (online) + CQL (offline).

Steps:
  1. Train PPO online against the live environment.
  2. Train a deliberately under-trained ("partial") DQN -- a mediocre
     behavior policy.
  3. Generate a logged transition dataset from a 50/50 mix of the
     rule-based agent and the partial DQN (no PPO/full-DQN data -- this is
     meant to simulate historical logs from *weaker* policies than the ones
     we compare against).
  4. Train CQL purely on that fixed logged dataset (zero live-environment
     interaction during training).
  5. Evaluate the offline-trained CQL policy live in the environment (the
     standard offline-RL eval pattern) and compare it against online PPO
     and Phase 1/2 baselines.

Usage: python -m scripts.train_phase3 [--config configs/phase3.yaml]
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from src.envs.tool_orchestration_env import ToolOrchestrationEnv
from src.envs.oracle import OracleSolver
from src.envs.tasks import NUM_TASKS
from src.agents.ppo_agent import PPOAgent, RolloutBuffer
from src.agents.dqn_agent import DQNAgent, ReplayBuffer
from src.agents.cql_agent import CQLAgent
from src.agents.rule_based import RuleBasedAgent
from src.utils.trajectory_logger import BehaviorPolicy, generate_logged_dataset, save_dataset
from src.utils.logging_utils import EpisodeLogger, summarize, plot_learning_curve, plot_baseline_comparison

RESULTS_DIR = os.path.join("results", "phase3")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
DATASET_DIR = os.path.join(RESULTS_DIR, "logged_dataset")

PHASE1_EVAL_LOG = os.path.join("results", "phase1", "logs", "eval_log.csv")
PHASE2_EVAL_LOG = os.path.join("results", "phase2", "logs", "eval_log.csv")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# 1. PPO online training
# --------------------------------------------------------------------------
def run_ppo_training(env, agent: PPOAgent, cfg: dict, logger: EpisodeLogger, seed: int) -> int:
    ppo_cfg = cfg["ppo"]
    total_env_steps = 0
    episode_idx = 0
    episode_reward = 0.0
    obs, info = env.reset(seed=seed, options={"task_id": 0})

    while total_env_steps < ppo_cfg["total_train_steps"]:
        rollout = RolloutBuffer()
        for _ in range(ppo_cfg["n_steps_per_update"]):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rollout.add(obs, action, log_prob, value, reward, done)
            episode_reward += reward
            obs = next_obs
            total_env_steps += 1

            if done:
                logger.log(
                    agent="ppo",
                    episode=episode_idx,
                    task_id=info["task_id"],
                    task_name=info["task_name"],
                    reward=episode_reward,
                    success=bool(info["success"]),
                    cost=info["cost"],
                    latency=info["latency"],
                    steps=info["steps"],
                )
                episode_idx += 1
                episode_reward = 0.0
                obs, info = env.reset(seed=seed + episode_idx, options={"task_id": episode_idx % NUM_TASKS})

            if total_env_steps >= ppo_cfg["total_train_steps"]:
                break

        if rollout.dones[-1]:
            last_value = 0.0
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                _, last_value_t = agent.network(obs_t)
                last_value = float(last_value_t.item())
        agent.update(rollout, last_value)

    return episode_idx


def run_ppo_eval(env, agent: PPOAgent, n_episodes: int, logger: EpisodeLogger, seed: int):
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + 100000 + ep, options={"task_id": ep % NUM_TASKS})
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act_greedy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        logger.log(
            agent="ppo", episode=ep, task_id=info["task_id"], task_name=info["task_name"],
            reward=total_reward, success=bool(info["success"]), cost=info["cost"],
            latency=info["latency"], steps=info["steps"],
        )


# --------------------------------------------------------------------------
# 2. Partial (deliberately under-trained) DQN behavior policy
# --------------------------------------------------------------------------
def train_partial_dqn(env, cfg: dict) -> DQNAgent:
    pd_cfg = cfg["partial_dqn"]
    obs_dim = env.observation_space.shape[0]
    agent = DQNAgent(
        obs_dim=obs_dim, lr=pd_cfg["lr"], gamma=pd_cfg["gamma"],
        epsilon_start=pd_cfg["epsilon_start"], epsilon_end=pd_cfg["epsilon_end"],
        epsilon_decay_episodes=pd_cfg["epsilon_decay_episodes"],
        target_update_freq=pd_cfg["target_update_freq"], embed_dim=pd_cfg["embed_dim"],
        hidden_dim=pd_cfg["hidden_dim"], seed=pd_cfg["seed"],
    )
    buffer = ReplayBuffer(capacity=pd_cfg["buffer_capacity"], obs_dim=obs_dim, seed=pd_cfg["seed"])
    total_env_steps = 0
    for ep in range(pd_cfg["train_episodes"]):
        agent.start_episode()
        obs, info = env.reset(seed=pd_cfg["seed"] + ep, options={"task_id": ep % NUM_TASKS})
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            total_env_steps += 1
            if len(buffer) >= max(pd_cfg["batch_size"], pd_cfg["warmup_steps"]):
                agent.train_step(buffer.sample(pd_cfg["batch_size"]))
    return agent


# --------------------------------------------------------------------------
# 3-4. Logged dataset + offline CQL training
# --------------------------------------------------------------------------
def train_cql_offline(dataset: dict, cfg: dict) -> CQLAgent:
    cql_cfg = cfg["cql"]
    agent = CQLAgent(
        lr=cql_cfg["lr"], gamma=cql_cfg["gamma"], alpha=cql_cfg["alpha"],
        target_update_freq=cql_cfg["target_update_freq"], embed_dim=cql_cfg["embed_dim"],
        hidden_dim=cql_cfg["hidden_dim"], seed=cql_cfg["seed"],
    )
    rng = np.random.default_rng(cql_cfg["seed"])
    n = len(dataset["states"])
    losses = []
    for step in range(cql_cfg["train_steps"]):
        idx = rng.integers(0, n, size=cql_cfg["batch_size"])
        batch = (
            torch.as_tensor(dataset["states"][idx], dtype=torch.float32),
            torch.as_tensor(dataset["actions"][idx], dtype=torch.long),
            torch.as_tensor(dataset["rewards"][idx], dtype=torch.float32),
            torch.as_tensor(dataset["next_states"][idx], dtype=torch.float32),
            torch.as_tensor(dataset["dones"][idx], dtype=torch.float32),
        )
        stats = agent.train_step(batch)
        losses.append(stats["loss"])
    print(f"  CQL offline training: final-100-step avg loss = {np.mean(losses[-100:]):.4f}")
    return agent


def run_cql_eval(env, agent: CQLAgent, n_episodes: int, logger: EpisodeLogger, seed: int):
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + 100000 + ep, options={"task_id": ep % NUM_TASKS})
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act_greedy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        logger.log(
            agent="cql", episode=ep, task_id=info["task_id"], task_name=info["task_name"],
            reward=total_reward, success=bool(info["success"]), cost=info["cost"],
            latency=info["latency"], steps=info["steps"],
        )


def compute_regret(eval_df, oracle_plans) -> float:
    oracle_values = eval_df["task_id"].map(lambda tid: oracle_plans[tid].expected_value)
    return float((oracle_values - eval_df["reward"]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    env_cfg = cfg["env"]

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    env = ToolOrchestrationEnv(
        success_reward=env_cfg["success_reward"], lam=env_cfg["lam"], mu=env_cfg["mu"],
        failure_penalty=env_cfg["failure_penalty"], seed=env_cfg["seed"],
    )
    oracle = OracleSolver(success_reward=env_cfg["success_reward"], lam=env_cfg["lam"], mu=env_cfg["mu"])
    oracle_plans = oracle.solve_all()

    train_logger = EpisodeLogger()
    eval_logger = EpisodeLogger()

    print("=== [1/4] Training PPO online ===")
    ppo_agent = PPOAgent(
        hidden_dim=cfg["ppo"]["hidden_dim"], embed_dim=cfg["ppo"]["embed_dim"], lr=cfg["ppo"]["lr"],
        gamma=cfg["ppo"]["gamma"], gae_lambda=cfg["ppo"]["gae_lambda"], clip_eps=cfg["ppo"]["clip_eps"],
        vf_coef=cfg["ppo"]["vf_coef"], ent_coef=cfg["ppo"]["ent_coef"], update_epochs=cfg["ppo"]["update_epochs"],
        minibatch_size=cfg["ppo"]["minibatch_size"], seed=cfg["ppo"]["seed"],
    )
    n_ppo_episodes = run_ppo_training(env, ppo_agent, cfg, train_logger, seed=cfg["ppo"]["seed"])
    print(f"  PPO trained over {n_ppo_episodes} episodes ({cfg['ppo']['total_train_steps']} env steps)")
    run_ppo_eval(env, ppo_agent, cfg["ppo"]["eval_episodes"], eval_logger, seed=cfg["ppo"]["seed"])
    ppo_agent.save(os.path.join(RESULTS_DIR, "ppo_model.pt"))

    print("=== [2/4] Training partial (under-trained) DQN behavior policy ===")
    partial_dqn = train_partial_dqn(env, cfg)

    print("=== [3/4] Generating logged offline dataset (rule-based + partial DQN) ===")
    rule_agent = RuleBasedAgent(max_retries=2)
    policies = [
        BehaviorPolicy("rule_based", rule_agent.act, rule_agent.reset, cfg["trajectory_logging"]["rule_based_weight"]),
        BehaviorPolicy(
            "partial_dqn",
            lambda obs: partial_dqn.select_action(obs, greedy=False),
            lambda: None,
            cfg["trajectory_logging"]["partial_dqn_weight"],
        ),
    ]
    dataset, episode_records = generate_logged_dataset(
        env, policies, n_episodes=cfg["trajectory_logging"]["n_episodes"], seed=cfg["trajectory_logging"]["seed"]
    )
    save_dataset(dataset, episode_records, DATASET_DIR)
    ep_df = pd.DataFrame(episode_records)
    print(f"  Logged {len(dataset['states'])} transitions from {len(episode_records)} episodes")
    print(f"  Behavior policy success rates: {ep_df.groupby('policy')['success'].mean().to_dict()}")

    print("=== [4/4] Training CQL offline on the logged dataset (no live env access) ===")
    cql_agent = train_cql_offline(dataset, cfg)
    run_cql_eval(env, cql_agent, cfg["cql"]["eval_episodes"], eval_logger, seed=cfg["cql"]["seed"])
    cql_agent.save(os.path.join(RESULTS_DIR, "cql_model.pt"))

    # --- Save logs ---
    train_df = train_logger.to_dataframe()
    eval_df = eval_logger.to_dataframe()
    train_logger.save_csv(os.path.join(LOGS_DIR, "train_log.csv"))
    eval_logger.save_csv(os.path.join(LOGS_DIR, "eval_log.csv"))

    # --- Fold in Phase 1 + Phase 2 baselines ---
    combined_eval_df = eval_df.copy()
    for path in (PHASE1_EVAL_LOG, PHASE2_EVAL_LOG):
        if os.path.exists(path):
            prior_df = pd.read_csv(path)
            if path == PHASE2_EVAL_LOG:
                prior_df = prior_df[prior_df["agent"] == "dqn"]  # phase2 already includes phase1 rows
            combined_eval_df = pd.concat([combined_eval_df, prior_df], ignore_index=True)
        else:
            print(f"WARNING: {path} not found, comparison will be missing those agents.")

    summary_df = summarize(combined_eval_df)
    regret_rows = [
        {"agent": name, "regret_vs_oracle": compute_regret(sub, oracle_plans)}
        for name, sub in combined_eval_df.groupby("agent")
    ]
    summary_df = summary_df.merge(pd.DataFrame(regret_rows), on="agent")
    summary_df.to_csv(os.path.join(LOGS_DIR, "summary.csv"), index=False)

    plot_learning_curve(train_df[train_df["agent"] == "ppo"], os.path.join(PLOTS_DIR, "learning_curve_ppo.png"))
    plot_baseline_comparison(summary_df, os.path.join(PLOTS_DIR, "baseline_comparison.png"))

    print("\n=== Phase 3 evaluation summary (all agents so far) ===")
    print(summary_df.to_string(index=False))
    print(f"\nLogs saved to {LOGS_DIR}/, plots saved to {PLOTS_DIR}/, logged dataset in {DATASET_DIR}/")


if __name__ == "__main__":
    main()
