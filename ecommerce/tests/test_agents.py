"""Agents: baselines meet their expected levels, checkpoints are safe to load, training code runs."""

import numpy as np
import pytest
import torch

from ecommerce.agents.baselines import RandomAgent, RuleBasedAgent
from ecommerce.agents.dqn import DQNAgent
from ecommerce.agents.nn import NEG_INF, mask_logits
from ecommerce.agents.tabular import QLearningAgent, discretize
from ecommerce.env.state import NUM_ACTIONS, OBS_DIM
from ecommerce.evaluation.runner import evaluate_agent
from ecommerce.evaluation.splits import get_splits
from ecommerce.sim.scenarios import generate_scenarios
from ecommerce.training import trainer


def test_baseline_levels_on_iid():
    sc = generate_scenarios(get_splits()["iid"], 100)
    rule = np.mean([r["success"] for r in evaluate_agent(RuleBasedAgent(), sc)])
    rand = np.mean([r["success"] for r in evaluate_agent(RandomAgent(seed=1), sc)])
    assert rule >= 0.98 and rand < 0.2


def test_mask_logits_makes_illegal_actions_unreachable():
    values = torch.tensor([[5.0, 9.0, 1.0]])
    mask = torch.tensor([[True, False, True]])
    masked = mask_logits(values, mask)
    assert masked.argmax().item() == 0 and masked[0, 1].item() == NEG_INF
    assert torch.distributions.Categorical(logits=masked).probs[0, 1].item() == pytest.approx(0.0, abs=1e-12)


def test_dqn_target_ignores_illegal_next_actions():
    agent = DQNAgent(seed=0, warmup=1, batch_size=4)
    s = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[0] = True
    with torch.no_grad():
        agent.target.eval()
        big = agent.target(torch.as_tensor(s).unsqueeze(0))
        best_legal = mask_logits(big, torch.as_tensor(mask).unsqueeze(0)).max().item()
    assert best_legal == pytest.approx(big[0, 0].item())


def test_dqn_checkpoint_roundtrip_and_layout_guard(tmp_path):
    a = DQNAgent(seed=1)
    path = tmp_path / "m.pt"
    a.save(str(path))
    b = DQNAgent(seed=2)
    b.load(str(path))
    obs = np.random.default_rng(0).random(OBS_DIM).astype(np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    assert a.act(obs, mask) == b.act(obs, mask)
    ck = torch.load(path, weights_only=False)
    ck["obs_dim"] = OBS_DIM + 1
    torch.save(ck, path)
    with pytest.raises(AssertionError, match="different state layout"):
        DQNAgent().load(str(path))


def test_tabular_agent_breaks_ties_randomly_and_saves(tmp_path):
    a = QLearningAgent(seed=0)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    picks = {a.act(obs, mask) for _ in range(60)}
    assert len(picks) > 3, "an untrained table must not always pick action 0"
    a.q[discretize(obs)][2] = 5.0
    a.save(str(tmp_path / "q.pkl"))
    b = QLearningAgent()
    b.load(str(tmp_path / "q.pkl"))
    assert b.act(obs, mask) == 2


@pytest.mark.parametrize("name,episodes", [("q_learning", 150), ("sarsa", 150), ("dqn", 150), ("ppo", 150)])
def test_online_training_runs_and_writes_artifacts(name, episodes, tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "TRAIN_DIR", tmp_path)
    s = trainer.train_online(name, seed=0, episodes=episodes, eval_every=75)
    d = tmp_path / f"{name}_seed0"
    for f in ("config.json", "train_log.csv", "val_log.csv", "summary.json"):
        assert (d / f).exists()
    assert (d / f"model_best.{'pkl' if name in ('q_learning', 'sarsa') else 'pt'}").exists()
    loaded = trainer.load_trained(name, 0)
    assert loaded.name == name and 0.0 <= s["final_val_success"] <= 1.0


def test_offline_cql_uses_only_the_log(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "TRAIN_DIR", tmp_path)
    monkeypatch.setattr(trainer, "DATA_DIR", tmp_path)
    meta = trainer.collect_logged_data(n_episodes=150, seed=0)
    assert meta["transitions"] > 150 and 0 < meta["behaviour_success_rate"] < 1
    s = trainer.train_cql(seed=0, steps=300, eval_every_steps=150, min_log_episodes=100)
    assert s["env_steps"] == 0, "CQL must not use live environment steps for training"
    assert (tmp_path / "cql_seed0" / "model_best.pt").exists()
