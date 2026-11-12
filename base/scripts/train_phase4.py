"""Phase 4 training/evaluation entry point: independent PPO agents (triage +
execution) on MultiAgentToolOrchestrationEnv.

Training loop notes:
  - triage acts every environment step; its rollout is a normal continuing
    trajectory (GAE across the whole task episode, same as single-agent PPO
    in Phase 3).
  - execution only acts on turns triage delegates to it. Each such turn is
    modeled as a single-shot contextual-bandit decision for execution (state
    in, one heavy-tool choice out, reward out, `done=True` for that
    transition) -- there is no well-defined "next state" from execution's
    own point of view, since control immediately returns to triage's
    continuing episode. This makes GAE degenerate correctly to
    advantage = reward - value with no bootstrapping, which is the right
    behavior for a bandit-style decision.
  - The two agents update independently (different rollout-buffer sizes,
    different cadence) -- this is "independent PPO" (IPPO), the simple
    decentralized-training variant the spec explicitly allows as an
    alternative to a centralized critic.

Usage: python -m scripts.train_phase4 [--config configs/phase4.yaml]
"""

import argparse
import os

import torch
import yaml

from src.envs.multi_agent_env import MultiAgentToolOrchestrationEnv
from src.envs.oracle import OracleSolver
from src.envs.tasks import NUM_TASKS
from src.agents.ppo_agent import PPOAgent, RolloutBuffer
from src.utils.logging_utils import EpisodeLogger, summarize, plot_learning_curve, plot_baseline_comparison

RESULTS_DIR = os.path.join("results", "phase4")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_multi_agent_ppo_training(env, triage_agent, execution_agent, cfg, logger, seed):
    ppo_cfg = cfg["ppo"]
    total_env_steps = 0
    episode_idx = 0
    episode_reward = 0.0
    obs_dict, info = env.reset(seed=seed, options={"task_id": 0})
    triage_rollout = RolloutBuffer()
    execution_rollout = RolloutBuffer()

    while total_env_steps < ppo_cfg["total_train_steps"]:
        triage_obs = obs_dict["triage"]
        t_action, t_log_prob, t_value = triage_agent.act(triage_obs)

        action_dict = {"triage": t_action}
        execution_obs = e_action = e_log_prob = e_value = None
        if t_action == env.DELEGATE_ACTION:
            execution_obs = obs_dict["execution"]
            e_action, e_log_prob, e_value = execution_agent.act(execution_obs)
            action_dict["execution"] = e_action

        next_obs_dict, reward_dict, terminated, truncated, info = env.step(action_dict)
        done = terminated or truncated

        triage_rollout.add(triage_obs, t_action, t_log_prob, t_value, reward_dict["triage"], done)
        if t_action == env.DELEGATE_ACTION:
            execution_rollout.add(execution_obs, e_action, e_log_prob, e_value, reward_dict["execution"], True)

        episode_reward += reward_dict["triage"]
        obs_dict = next_obs_dict
        total_env_steps += 1

        if done:
            logger.log(
                agent="multi_agent", episode=episode_idx, task_id=info["task_id"], task_name=info["task_name"],
                reward=episode_reward, success=bool(info["success"]), cost=info["cost"],
                latency=info["latency"], steps=info["steps"],
            )
            episode_idx += 1
            episode_reward = 0.0
            obs_dict, info = env.reset(seed=seed + episode_idx, options={"task_id": episode_idx % NUM_TASKS})

        if len(triage_rollout) >= ppo_cfg["triage_n_steps_per_update"]:
            if triage_rollout.dones[-1]:
                last_value = 0.0
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs_dict["triage"], dtype=torch.float32).unsqueeze(0)
                    _, last_value_t = triage_agent.network(obs_t)
                    last_value = float(last_value_t.item())
            triage_agent.update(triage_rollout, last_value)
            triage_rollout = RolloutBuffer()

        if len(execution_rollout) >= ppo_cfg["execution_n_steps_per_update"]:
            execution_agent.update(execution_rollout, last_value=0.0)
            execution_rollout = RolloutBuffer()

    return episode_idx


def run_multi_agent_eval(env, triage_agent, execution_agent, n_episodes, logger, seed):
    n_delegations = 0
    for ep in range(n_episodes):
        obs_dict, info = env.reset(seed=seed + 100000 + ep, options={"task_id": ep % NUM_TASKS})
        done = False
        total_reward = 0.0
        while not done:
            t_action = triage_agent.act_greedy(obs_dict["triage"])
            action_dict = {"triage": t_action}
            if t_action == env.DELEGATE_ACTION:
                e_action = execution_agent.act_greedy(obs_dict["execution"])
                action_dict["execution"] = e_action
                n_delegations += 1
            obs_dict, reward_dict, terminated, truncated, info = env.step(action_dict)
            total_reward += reward_dict["triage"]
            done = terminated or truncated
        logger.log(
            agent="multi_agent", episode=ep, task_id=info["task_id"], task_name=info["task_name"],
            reward=total_reward, success=bool(info["success"]), cost=info["cost"],
            latency=info["latency"], steps=info["steps"],
        )
    return n_delegations


def compute_regret(eval_df, oracle_plans) -> float:
    oracle_values = eval_df["task_id"].map(lambda tid: oracle_plans[tid].expected_value)
    return float((oracle_values - eval_df["reward"]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    ppo_cfg = cfg["ppo"]

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    env = MultiAgentToolOrchestrationEnv(
        success_reward=env_cfg["success_reward"], lam=env_cfg["lam"], mu=env_cfg["mu"],
        failure_penalty=env_cfg["failure_penalty"], seed=env_cfg["seed"],
    )
    oracle = OracleSolver(success_reward=env_cfg["success_reward"], lam=env_cfg["lam"], mu=env_cfg["mu"])
    oracle_plans = oracle.solve_all_multi_agent()

    triage_agent = PPOAgent(
        num_actions=env.TRIAGE_NUM_ACTIONS, hidden_dim=ppo_cfg["hidden_dim"], embed_dim=ppo_cfg["embed_dim"],
        lr=ppo_cfg["lr"], gamma=ppo_cfg["gamma"], gae_lambda=ppo_cfg["gae_lambda"], clip_eps=ppo_cfg["clip_eps"],
        vf_coef=ppo_cfg["vf_coef"], ent_coef=ppo_cfg["ent_coef"], update_epochs=ppo_cfg["update_epochs"],
        minibatch_size=ppo_cfg["triage_minibatch_size"], seed=ppo_cfg["seed"],
    )
    execution_agent = PPOAgent(
        num_actions=env.EXECUTION_NUM_ACTIONS, hidden_dim=ppo_cfg["hidden_dim"], embed_dim=ppo_cfg["embed_dim"],
        lr=ppo_cfg["lr"], gamma=ppo_cfg["gamma"], gae_lambda=ppo_cfg["gae_lambda"], clip_eps=ppo_cfg["clip_eps"],
        vf_coef=ppo_cfg["vf_coef"], ent_coef=ppo_cfg["ent_coef"], update_epochs=ppo_cfg["update_epochs"],
        minibatch_size=ppo_cfg["execution_minibatch_size"], seed=ppo_cfg["seed"] + 1,
    )

    train_logger = EpisodeLogger()
    eval_logger = EpisodeLogger()

    print("=== Training independent PPO agents (triage + execution) ===")
    n_episodes = run_multi_agent_ppo_training(env, triage_agent, execution_agent, cfg, train_logger, seed=ppo_cfg["seed"])
    print(f"  Trained over {n_episodes} episodes ({ppo_cfg['total_train_steps']} env steps)")

    n_delegations = run_multi_agent_eval(env, triage_agent, execution_agent, ppo_cfg["eval_episodes"], eval_logger, seed=ppo_cfg["seed"])
    print(f"  Eval: {n_delegations} delegations across {ppo_cfg['eval_episodes']} episodes "
          f"({n_delegations / ppo_cfg['eval_episodes']:.2f} per episode)")

    triage_agent.save(os.path.join(RESULTS_DIR, "triage_ppo_model.pt"))
    execution_agent.save(os.path.join(RESULTS_DIR, "execution_ppo_model.pt"))

    train_df = train_logger.to_dataframe()
    eval_df = eval_logger.to_dataframe()
    train_logger.save_csv(os.path.join(LOGS_DIR, "train_log.csv"))
    eval_logger.save_csv(os.path.join(LOGS_DIR, "eval_log.csv"))

    summary_df = summarize(eval_df)
    regret_df = summary_df[["agent"]].copy()
    regret_df["regret_vs_oracle"] = [compute_regret(eval_df, oracle_plans)]
    summary_df = summary_df.merge(regret_df, on="agent")
    summary_df.to_csv(os.path.join(LOGS_DIR, "summary.csv"), index=False)

    plot_learning_curve(train_df, os.path.join(PLOTS_DIR, "learning_curve_multi_agent.png"))
    plot_baseline_comparison(summary_df, os.path.join(PLOTS_DIR, "multi_agent_summary.png"))

    print("\n=== Phase 4 multi-agent evaluation summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nLogs saved to {LOGS_DIR}/, plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
