"""Phase 2 training/evaluation entry point: DQN with a learned task embedding.

Trains DQN on ToolOrchestrationEnv (same env, same reward formula as Phase 1),
evaluates it greedily, folds its results into the Phase 1 baseline comparison
(reloading results/phase1/logs/eval_log.csv), and writes an updated
comparison table/plot to results/phase2/.

Usage: python -m scripts.train_phase2 [--config configs/phase2.yaml]
"""

import argparse
import os

import pandas as pd
import yaml

from src.envs.tool_orchestration_env import ToolOrchestrationEnv
from src.envs.oracle import OracleSolver
from src.envs.tasks import NUM_TASKS
from src.agents.dqn_agent import DQNAgent, ReplayBuffer
from src.utils.logging_utils import EpisodeLogger, summarize, plot_learning_curve, plot_baseline_comparison

RESULTS_DIR = os.path.join("results", "phase2")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODEL_PATH = os.path.join(RESULTS_DIR, "dqn_model.pt")

PHASE1_EVAL_LOG = os.path.join("results", "phase1", "logs", "eval_log.csv")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_dqn_training(env, agent: DQNAgent, buffer: ReplayBuffer, cfg: dict, logger: EpisodeLogger, seed: int):
    total_env_steps = 0
    for ep in range(cfg["training"]["train_episodes"]):
        agent.start_episode()
        obs, info = env.reset(seed=seed + ep, options={"task_id": ep % NUM_TASKS})
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            total_reward += reward
            total_env_steps += 1

            if len(buffer) >= max(cfg["dqn"]["batch_size"], cfg["dqn"]["warmup_steps"]) and \
               total_env_steps % cfg["dqn"]["train_freq"] == 0:
                batch = buffer.sample(cfg["dqn"]["batch_size"])
                agent.train_step(batch)

        logger.log(
            agent="dqn",
            episode=ep,
            task_id=info["task_id"],
            task_name=info["task_name"],
            reward=total_reward,
            success=bool(info["success"]),
            cost=info["cost"],
            latency=info["latency"],
            steps=info["steps"],
            epsilon=agent.epsilon(),
        )


def run_dqn_eval(env, agent: DQNAgent, n_episodes: int, logger: EpisodeLogger, seed: int):
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + 100000 + ep, options={"task_id": ep % NUM_TASKS})
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        logger.log(
            agent="dqn",
            episode=ep,
            task_id=info["task_id"],
            task_name=info["task_name"],
            reward=total_reward,
            success=bool(info["success"]),
            cost=info["cost"],
            latency=info["latency"],
            steps=info["steps"],
        )


def compute_regret(eval_df, oracle_plans) -> float:
    oracle_values = eval_df["task_id"].map(lambda tid: oracle_plans[tid].expected_value)
    return float((oracle_values - eval_df["reward"]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    env = ToolOrchestrationEnv(
        success_reward=env_cfg["success_reward"],
        lam=env_cfg["lam"],
        mu=env_cfg["mu"],
        failure_penalty=env_cfg["failure_penalty"],
        seed=env_cfg["seed"],
    )
    obs_dim = env.observation_space.shape[0]

    oracle = OracleSolver(success_reward=env_cfg["success_reward"], lam=env_cfg["lam"], mu=env_cfg["mu"])
    oracle_plans = oracle.solve_all()

    dqn_cfg = cfg["dqn"]
    agent = DQNAgent(
        obs_dim=obs_dim,
        lr=dqn_cfg["lr"],
        gamma=dqn_cfg["gamma"],
        epsilon_start=dqn_cfg["epsilon_start"],
        epsilon_end=dqn_cfg["epsilon_end"],
        epsilon_decay_episodes=dqn_cfg["epsilon_decay_episodes"],
        target_update_freq=dqn_cfg["target_update_freq"],
        embed_dim=dqn_cfg["embed_dim"],
        hidden_dim=dqn_cfg["hidden_dim"],
        seed=cfg["training"]["seed"],
    )
    buffer = ReplayBuffer(capacity=dqn_cfg["buffer_capacity"], obs_dim=obs_dim, seed=cfg["training"]["seed"])

    train_logger = EpisodeLogger()
    eval_logger = EpisodeLogger()

    run_dqn_training(env, agent, buffer, cfg, train_logger, seed=cfg["training"]["seed"])
    run_dqn_eval(env, agent, cfg["training"]["eval_episodes"], eval_logger, seed=cfg["training"]["seed"])

    agent.save(MODEL_PATH)

    train_df = train_logger.to_dataframe()
    dqn_eval_df = eval_logger.to_dataframe()
    train_logger.save_csv(os.path.join(LOGS_DIR, "train_log.csv"))
    eval_logger.save_csv(os.path.join(LOGS_DIR, "eval_log.csv"))

    # --- Fold in Phase 1 baselines for a combined comparison ---
    if os.path.exists(PHASE1_EVAL_LOG):
        phase1_eval_df = pd.read_csv(PHASE1_EVAL_LOG)
        combined_eval_df = pd.concat([phase1_eval_df, dqn_eval_df], ignore_index=True)
    else:
        print(f"WARNING: {PHASE1_EVAL_LOG} not found, comparison will only include DQN.")
        combined_eval_df = dqn_eval_df

    summary_df = summarize(combined_eval_df)
    regret_rows = []
    for agent_name, sub in combined_eval_df.groupby("agent"):
        regret_rows.append({"agent": agent_name, "regret_vs_oracle": compute_regret(sub, oracle_plans)})
    regret_df = pd.DataFrame(regret_rows)
    summary_df = summary_df.merge(regret_df, on="agent")
    summary_df.to_csv(os.path.join(LOGS_DIR, "summary.csv"), index=False)

    plot_learning_curve(train_df, os.path.join(PLOTS_DIR, "learning_curve_dqn.png"))
    plot_baseline_comparison(summary_df, os.path.join(PLOTS_DIR, "baseline_comparison.png"))

    print("\n=== Phase 2 evaluation summary (rule_based/q_learning/sarsa from Phase 1 + dqn) ===")
    print(summary_df.to_string(index=False))
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Logs saved to {LOGS_DIR}/, plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
