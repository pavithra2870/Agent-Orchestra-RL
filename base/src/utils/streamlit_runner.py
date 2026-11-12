"""
Streamlit runner utility for live simulations.
Runs evaluation episodes for selected algorithms on specific task templates.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from src.envs.tool_orchestration_env import ToolOrchestrationEnv
from src.agents.rule_based import RuleBasedAgent
from src.agents.tabular_agents import TabularQLearningAgent, SARSAAgent
from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.cql_agent import CQLAgent
from src.utils.discretization import discretize_state


def load_trained_model(algorithm: str, base_dir: Path):
    """Load a trained model checkpoint - returns state_dict or model path."""
    results_dir = base_dir / "results"
    
    if algorithm == "dqn":
        model_path = results_dir / "phase2" / "dqn_model.pt"
        if model_path.exists():
            return torch.load(model_path, map_location='cpu')
    
    elif algorithm == "ppo":
        model_path = results_dir / "phase3" / "ppo_model.pt"
        if model_path.exists():
            return torch.load(model_path, map_location='cpu')
    
    elif algorithm == "cql":
        model_path = results_dir / "phase3" / "cql_model.pt"
        if model_path.exists():
            return torch.load(model_path, map_location='cpu')
    
    elif algorithm == "q_learning":
        model_path = results_dir / "phase1" / "q_learning_model.pkl"
        if model_path.exists():
            return str(model_path)
    
    elif algorithm == "sarsa":
        model_path = results_dir / "phase1" / "sarsa_model.pkl"
        if model_path.exists():
            return str(model_path)
    
    return None


def create_agent(algorithm: str, env, model_checkpoint=None, hyperparams: Optional[Dict] = None, include_metadata: bool = True):
    """Create an agent instance with optional hyperparameter overrides."""
    
    if hyperparams is None:
        hyperparams = {}
    
    if algorithm == "rule_based":
        return RuleBasedAgent(max_retries=hyperparams.get('max_retries', 2))
    
    elif algorithm == "q_learning":
        agent = TabularQLearningAgent(
            alpha=hyperparams.get('alpha', 0.1),
            gamma=hyperparams.get('gamma', 0.95),
            epsilon_start=0.0,  # Greedy for evaluation
            epsilon_end=0.0,
            epsilon_decay_episodes=1,
            seed=hyperparams.get('seed', 42)
        )
        # Load trained Q-table if available
        if model_checkpoint:
            agent.load(model_checkpoint)
        return agent
    
    elif algorithm == "sarsa":
        agent = SARSAAgent(
            alpha=hyperparams.get('alpha', 0.1),
            gamma=hyperparams.get('gamma', 0.95),
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_episodes=1,
            seed=hyperparams.get('seed', 42)
        )
        # Load trained Q-table if available
        if model_checkpoint:
            agent.load(model_checkpoint)
        return agent
    
    elif algorithm == "dqn":
        agent = DQNAgent(
            obs_dim=env.observation_space.shape[0],
            lr=hyperparams.get('lr', 0.001),
            gamma=hyperparams.get('gamma', 0.95),
            epsilon_start=0.0,  # Greedy for evaluation
            epsilon_end=0.0,
            epsilon_decay_episodes=1,
            embed_dim=hyperparams.get('embed_dim', 8),
            hidden_dim=hyperparams.get('hidden_dim', 64),
            include_metadata=include_metadata,  # NEW: support metadata
            seed=hyperparams.get('seed', 42)
        )
        if model_checkpoint:
            agent.q_network.load_state_dict(model_checkpoint)
            agent.target_network.load_state_dict(model_checkpoint)
            agent.q_network.eval()
            agent.target_network.eval()
        return agent
    
    elif algorithm == "ppo":
        agent = PPOAgent(
            num_tools=env.action_space.n - 1,
            num_actions=env.action_space.n,
            hidden_dim=hyperparams.get('hidden_dim', 64),
            embed_dim=hyperparams.get('embed_dim', 8),
            lr=hyperparams.get('lr', 0.0003),
            gamma=hyperparams.get('gamma', 0.99),
            clip_eps=hyperparams.get('clip_eps', 0.2),
            ent_coef=hyperparams.get('ent_coef', 0.01),
            include_metadata=include_metadata,  # NEW: support metadata
            seed=hyperparams.get('seed', 42)
        )
        if model_checkpoint:
            agent.network.load_state_dict(model_checkpoint)
            agent.network.eval()
        return agent
    
    elif algorithm == "cql":
        agent = CQLAgent(
            lr=hyperparams.get('lr', 0.001),
            gamma=hyperparams.get('gamma', 0.95),
            alpha=hyperparams.get('alpha', 1.0),
            hidden_dim=hyperparams.get('hidden_dim', 64),  # CQL was trained with 64
            embed_dim=hyperparams.get('embed_dim', 8),
            include_metadata=include_metadata,  # NEW: support metadata
            seed=hyperparams.get('seed', 42)
        )
        if model_checkpoint:
            agent.q_network.load_state_dict(model_checkpoint)
            agent.target_network.load_state_dict(model_checkpoint)
            agent.q_network.eval()
            agent.target_network.eval()
        return agent
    
    return None


def run_evaluation_simulation(
    algorithm: str,
    task_id: int,
    num_episodes: int,
    seed: int = 42,
    hyperparams: Optional[Dict] = None,
    include_metadata: bool = True  # NEW: support for task metadata features
) -> Dict[str, Any]:
    """
    Run evaluation episodes for a specific algorithm on a specific task.
    
    Args:
        algorithm: Algorithm name (e.g., 'dqn', 'ppo', 'rule_based')
        task_id: Task template ID (0-9)
        num_episodes: Number of evaluation episodes (recommended: 30-50 for quick demo, 100+ for robust stats)
        seed: Random seed
        hyperparams: Optional hyperparameter overrides (success_reward, lambda, mu, failure_penalty, etc.)
        include_metadata: Whether to include task metadata features in observations (default: True)
    
    Returns:
        Dictionary with metrics: success_rate, avg_reward, avg_cost, avg_latency, avg_steps
    """
    
    if hyperparams is None:
        hyperparams = {}
    
    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    base_dir = Path(__file__).parent.parent.parent
    
    # Get hyperparams with defaults
    success_reward = hyperparams.get('success_reward', 10.0)
    lam = hyperparams.get('lambda', 0.1)
    mu = hyperparams.get('mu', 0.1)
    failure_penalty = hyperparams.get('failure_penalty', 2.0)
    
    # Create environment (single-agent only for now - multi-agent has different interface)
    env = ToolOrchestrationEnv(
        success_reward=success_reward,
        lam=lam,
        mu=mu,
        failure_penalty=failure_penalty,
        seed=seed,
        include_task_metadata=include_metadata  # NEW: pass metadata flag
    )
    
    # Load model checkpoint
    model_checkpoint = load_trained_model(algorithm, base_dir)
    
    # Create agent
    agent = create_agent(algorithm, env, model_checkpoint, hyperparams, include_metadata)
    if agent is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Run evaluation episodes
    rewards = []
    successes = []
    costs = []
    latencies = []
    steps = []
    
    for ep in range(num_episodes):
        # Reset agent if needed
        if algorithm == "rule_based":
            agent.reset()
        
        # Reset to specific task
        obs, info = env.reset(options={"task_id": task_id})
        done = False
        ep_reward = 0
        ep_steps = 0
        
        while not done:
            # Select action based on algorithm type
            if algorithm in ["q_learning", "sarsa"]:
                # Tabular agents need discrete state (bins must match training, see scripts/train_phase1.py)
                discrete_state = discretize_state(obs, include_metadata=include_metadata)
                action = agent.select_action(discrete_state, greedy=True)
            
            elif algorithm == "rule_based":
                # Rule-based uses act() method
                action = agent.act(obs)
            
            elif algorithm == "dqn":
                # DQN uses select_action with greedy flag
                action = agent.select_action(obs, greedy=True)
            
            elif algorithm == "ppo":
                # PPO uses act_greedy method
                action = agent.act_greedy(obs)
            
            elif algorithm == "cql":
                # CQL uses act_greedy method
                action = agent.act_greedy(obs)
            
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
        
        # Record metrics
        rewards.append(ep_reward)
        successes.append(1.0 if info.get('success', False) else 0.0)
        costs.append(info.get('cost', 0.0))
        latencies.append(info.get('latency', 0.0))
        steps.append(ep_steps)
    
    # Compute statistics
    results = {
        'success_rate': np.mean(successes),
        'avg_reward': np.mean(rewards),
        'avg_cost': np.mean(costs),
        'avg_latency': np.mean(latencies),
        'avg_steps': np.mean(steps),
        'std_reward': np.std(rewards),
        'std_cost': np.std(costs),
        'std_latency': np.std(latencies),
        'num_episodes': num_episodes,
        'algorithm': algorithm,
        'task_id': task_id
    }
    
    return results
