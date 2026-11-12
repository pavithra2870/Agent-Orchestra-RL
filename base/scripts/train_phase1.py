"""Phase 1 training/evaluation entry point.

Runs:
  1. RuleBasedAgent  -- evaluated only (no learning).
  2. TabularQLearningAgent -- trained then evaluated greedily.
  3. SARSAAgent            -- trained then evaluated greedily.

Logs per-episode records to results/phase1/logs/, produces a learning-curve
plot and a baseline-comparison bar chart in results/phase1/plots/, and prints
a smoke-test summary (including regret vs. the OracleSolver) to stdout.

Usage: python -m scripts.train_phase1 [--config configs/phase1.yaml]
"""

import argparse
import os

import numpy as np
import yaml

from src.envs.tool_orchestration_env import ToolOrchestrationEnv
from src.envs.oracle import OracleSolver
from src.envs.tasks import NUM_TASKS
from src.agents.rule_based import RuleBasedAgent
from src.agents.tabular_agents import TabularQLearningAgent, SARSAAgent
from src.utils.discretization import discretize_state
from src.utils.logging_utils import EpisodeLogger, summarize, plot_learning_curve, plot_baseline_comparison

RESULTS_DIR = os.path.join("results", "phase1")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_rule_based_eval(env: ToolOrchestrationEnv, agent: RuleBasedAgent, n_episodes: int, logger: EpisodeLogger, seed: int):
    for ep in range(n_episodes):
        agent.reset()
        obs, info = env.reset(seed=seed + ep, options={"task_id": ep % NUM_TASKS})
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        logger.log(
            agent="rule_based",
            episode=ep,
            task_id=info["task_id"],
            task_name=info["task_name"],
            reward=total_reward,
            success=bool(info["success"]),
            cost=info["cost"],
            latency=info["latency"],
            steps=info["steps"],
        )


def run_tabular_training(env: ToolOrchestrationEnv, agent, n_episodes: int, logger: EpisodeLogger, seed: int, phase_name: str):
    on_policy = agent.name == "sarsa"
    for ep in range(n_episodes):
        agent.start_episode()
        obs, info = env.reset(seed=seed + ep, options={"task_id": ep % NUM_TASKS})
        state = discretize_state(obs)
        action = agent.select_action(state) if on_policy else None
        done = False
        total_reward = 0.0
        while not done:
            if not on_policy:
                action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_obs)
            done = terminated or truncated

            if on_policy:
                next_action = agent.select_action(next_state) if not done else None
                agent.update(state, action, reward, next_state, done, next_action=next_action)
                action = next_action
            else:
                agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        logger.log(
            agent=phase_name,
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


def run_tabular_eval(env: ToolOrchestrationEnv, agent, n_episodes: int, logger: EpisodeLogger, seed: int, agent_label: str):
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + 100000 + ep, options={"task_id": ep % NUM_TASKS})
        state = discretize_state(obs)
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            state = discretize_state(obs)
            total_reward += reward
            done = terminated or truncated
        logger.log(
            agent=agent_label,
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
    """Mean(oracle_expected_value - achieved_reward) over evaluation episodes,
    matched per-episode by the task actually drawn."""
    oracle_values = eval_df["task_id"].map(lambda tid: oracle_plans[tid].expected_value)
    return float((oracle_values - eval_df["reward"]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    tab_cfg = cfg["tabular"]
    train_cfg = cfg["training"]
    rb_cfg = cfg["rule_based"]

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    env = ToolOrchestrationEnv(
        success_reward=env_cfg["success_reward"],
        lam=env_cfg["lam"],
        mu=env_cfg["mu"],
        failure_penalty=env_cfg["failure_penalty"],
        seed=env_cfg["seed"],
    )

    oracle = OracleSolver(success_reward=env_cfg["success_reward"], lam=env_cfg["lam"], mu=env_cfg["mu"])
    oracle_plans = oracle.solve_all()

    print("=== Oracle best paths per task ===")
    for tid, plan in oracle_plans.items():
        print(
            f"  [{tid}] {plan.task_name}: {plan.best_path.sequence} "
            f"(quality={plan.best_path.quality}, expected_value={plan.expected_value:.3f})"
        )

    train_logger = EpisodeLogger()
    eval_logger = EpisodeLogger()

    # 1. Rule-based baseline (eval only).
    rule_agent = RuleBasedAgent(max_retries=rb_cfg["max_retries"])
    run_rule_based_eval(env, rule_agent, train_cfg["eval_episodes"], eval_logger, seed=train_cfg["seed"])

    # 2. Tabular Q-learning.
    q_agent = TabularQLearningAgent(
        alpha=tab_cfg["alpha"],
        gamma=tab_cfg["gamma"],
        epsilon_start=tab_cfg["epsilon_start"],
        epsilon_end=tab_cfg["epsilon_end"],
        epsilon_decay_episodes=tab_cfg["epsilon_decay_episodes"],
        seed=train_cfg["seed"],
    )
    run_tabular_training(env, q_agent, train_cfg["train_episodes"], train_logger, seed=train_cfg["seed"], phase_name="q_learning")
    run_tabular_eval(env, q_agent, train_cfg["eval_episodes"], eval_logger, seed=train_cfg["seed"], agent_label="q_learning")

    # 3. SARSA.
    sarsa_agent = SARSAAgent(
        alpha=tab_cfg["alpha"],
        gamma=tab_cfg["gamma"],
        epsilon_start=tab_cfg["epsilon_start"],
        epsilon_end=tab_cfg["epsilon_end"],
        epsilon_decay_episodes=tab_cfg["epsilon_decay_episodes"],
        seed=train_cfg["seed"] + 1,
    )
    run_tabular_training(env, sarsa_agent, train_cfg["train_episodes"], train_logger, seed=train_cfg["seed"] + 1, phase_name="sarsa")
    run_tabular_eval(env, sarsa_agent, train_cfg["eval_episodes"], eval_logger, seed=train_cfg["seed"] + 1, agent_label="sarsa")

    # --- Save trained models ---
    q_learning_model_path = os.path.join(RESULTS_DIR, "q_learning_model.pkl")
    sarsa_model_path = os.path.join(RESULTS_DIR, "sarsa_model.pkl")
    q_agent.save(q_learning_model_path)
    sarsa_agent.save(sarsa_model_path)
    print(f"\nSaved Q-Learning model to {q_learning_model_path}")
    print(f"Saved SARSA model to {sarsa_model_path}")

    # --- Save logs ---
    train_df = train_logger.to_dataframe()
    eval_df = eval_logger.to_dataframe()
    train_logger.save_csv(os.path.join(LOGS_DIR, "train_log.csv"))
    eval_logger.save_csv(os.path.join(LOGS_DIR, "eval_log.csv"))

    # --- Summary + regret ---
    summary_df = summarize(eval_df)
    regret_rows = []
    for agent_name, sub in eval_df.groupby("agent"):
        regret_rows.append({"agent": agent_name, "regret_vs_oracle": compute_regret(sub, oracle_plans)})
    import pandas as pd
    regret_df = pd.DataFrame(regret_rows)
    summary_df = summary_df.merge(regret_df, on="agent")
    summary_df.to_csv(os.path.join(LOGS_DIR, "summary.csv"), index=False)

    # --- Plots ---
    plot_learning_curve(train_df[train_df["agent"] == "q_learning"], os.path.join(PLOTS_DIR, "learning_curve_q_learning.png"))
    plot_learning_curve(train_df[train_df["agent"] == "sarsa"], os.path.join(PLOTS_DIR, "learning_curve_sarsa.png"))
    combined_curve_df = train_df[train_df["agent"].isin(["q_learning", "sarsa"])]
    plot_learning_curve(combined_curve_df, os.path.join(PLOTS_DIR, "learning_curve_comparison.png"))
    plot_baseline_comparison(summary_df, os.path.join(PLOTS_DIR, "baseline_comparison.png"))

    print("\n=== Phase 1 evaluation summary (evaluation episodes, greedy policy) ===")
    print(summary_df.to_string(index=False))
    print(f"\nLogs saved to {LOGS_DIR}/, plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
