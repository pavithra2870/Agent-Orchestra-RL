"""Train q_learning / sarsa / dqn / ppo (online) and cql (offline) on the e-commerce environment.

Every run writes to ecommerce/results/training/<agent>_seed<k>/ :
  config.json     hyperparameters, seed, episodes, state layout version
  train_log.csv   one row per training episode
  val_log.csv     validation success/reward every `eval_every` episodes (the learning curve)
  summary.json    final numbers, training time, episodes needed to reach 90% validation success
  model_best.*    checkpoint with the best validation score (this is what gets evaluated)
  model_final.*   checkpoint at the end of training
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ecommerce.agents.baselines import RandomAgent, RuleBasedAgent
from ecommerce.agents.cql import CQLAgent, OfflineDataset
from ecommerce.agents.dqn import DQNAgent
from ecommerce.agents.ppo import PPOAgent
from ecommerce.agents.tabular import QLearningAgent, SarsaAgent
from ecommerce.config import RESULTS_DIR, Settings, get_settings
from ecommerce.env.env import EcomSupportEnv
from ecommerce.env.state import NUM_ACTIONS, OBS_DIM
from ecommerce.evaluation.runner import evaluate_agent
from ecommerce.evaluation.splits import get_splits
from ecommerce.sim.scenarios import TRAIN_COMBOS, SplitSpec, generate_scenarios

TRAIN_DIR = RESULTS_DIR / "training"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ONLINE_AGENTS = ("q_learning", "sarsa", "dqn", "ppo")
ALL_TRAINABLE = ONLINE_AGENTS + ("cql",)

DEFAULT_EPISODES = {"q_learning": 5000, "sarsa": 5000, "dqn": 4000, "ppo": 12000}
# chosen on the validation split by ecommerce/training/tune_ppo.py (see results/training/ppo_tuning.json).
# After Group B all five settings reach 100% validation success by episode 1,800, so the sweep no longer separates them and
# the tie-break picks the library defaults ({}). Before Group B the sweep chose lr=1e-3, rollout_steps=512, ent_coef=0.003, gamma=0.97.
PPO_TUNED_HP: dict = {}
POOL_SIZE = 4000
_POOL_CACHE: Dict[Any, List] = {}


def training_pool(settings: Settings, size: int = POOL_SIZE, mix: str = "standard"):
    """mix="robust" replaces 30% of the pool with stress scenarios (3x tool failures, sometimes a tool fully down),
    so the agent gets to practise "retry, then hand over to a human"."""
    key = (size, mix)
    if key not in _POOL_CACHE:
        if mix == "standard":
            _POOL_CACHE[key] = generate_scenarios(get_splits(settings)["train"], size, settings)
        else:
            sc = settings.scenarios
            n_stress = int(size * 0.3)
            stress = SplitSpec("train_robust", 1200, TRAIN_COMBOS, type_probs={"standard": 1.0},
                               failure_multiplier=sc.failure_multiplier_stress, down_prob=0.5)
            _POOL_CACHE[key] = (generate_scenarios(get_splits(settings)["train"], size - n_stress, settings)
                                + generate_scenarios(stress, n_stress, settings))
    return _POOL_CACHE[key]


def build_agent(name: str, seed: int, hp: Optional[Dict[str, Any]] = None, episodes: Optional[int] = None):
    hp = dict(hp or {})
    if name == "q_learning":
        hp.setdefault("eps_decay_episodes", int((episodes or 5000) * 0.5))
        return QLearningAgent(seed=seed, **hp)
    if name == "sarsa":
        hp.setdefault("eps_decay_episodes", int((episodes or 5000) * 0.5))
        return SarsaAgent(seed=seed, **hp)
    if name == "dqn":
        hp.setdefault("eps_decay_episodes", int((episodes or 4000) * 0.4))
        return DQNAgent(seed=seed, **hp)
    if name == "ppo":
        return PPOAgent(seed=seed, **{**PPO_TUNED_HP, **hp})
    if name == "cql":
        return CQLAgent(seed=seed, **hp)
    raise ValueError(f"unknown agent {name}")


def _ext(name: str) -> str:
    return "pkl" if name in ("q_learning", "sarsa") else "pt"


def run_dir(name: str, seed: int, tag: str = "") -> Path:
    return TRAIN_DIR / f"{name}{'_' + tag if tag else ''}_seed{seed}"


def model_path(name: str, seed: int, which: str = "best", tag: str = "") -> Path:
    return run_dir(name, seed, tag) / f"model_{which}.{_ext(name)}"


def load_trained(name: str, seed: int, which: str = "best", tag: str = ""):
    agent = build_agent(name, seed)
    agent.load(str(model_path(name, seed, which, tag)))
    agent.training = False
    if tag == "nomask":
        agent.uses_mask = False
    return agent


def _validate(agent, val_scenarios, settings) -> Dict[str, float]:
    was = agent.training
    agent.training = False
    res = evaluate_agent(agent, val_scenarios, k=1, settings=settings)
    agent.training = was
    return {"val_success": float(np.mean([r["success"] for r in res])), "val_reward": float(np.mean([r["reward"] for r in res]))}


class _Run:
    """Bookkeeping shared by all training loops: logs, validation, best checkpoint, progress events."""

    def __init__(self, name, seed, agent, settings, episodes, eval_every, progress, extra_cfg, tag=""):
        self.name, self.seed, self.agent, self.settings, self.tag = name, seed, agent, settings, tag
        self.episodes, self.eval_every, self.progress = episodes, eval_every, progress
        self.dir = run_dir(name, seed, tag)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.val_set = generate_scenarios(get_splits(settings)["validation"], 150, settings)
        self.train_rows: List[Dict[str, Any]] = []
        self.val_rows: List[Dict[str, Any]] = []
        self.best = (-1.0, -1e9)
        self.t0 = time.time()
        self.first_90: Optional[int] = None
        (self.dir / "config.json").write_text(json.dumps({
            "agent": name, "tag": tag, "seed": seed, "episodes": episodes, "obs_dim": OBS_DIM, "n_actions": NUM_ACTIONS,
            "hyperparameters": getattr(agent, "hp", {}), "reward": settings.reward.model_dump(), **extra_cfg}, indent=2))

    def log_episode(self, ep: int, env_steps: int, reward: float, info: Dict[str, Any], extra: Optional[Dict] = None):
        ev = info.get("eval", {})
        self.train_rows.append({"episode": ep, "env_steps": env_steps, "reward": reward,
                                "success": int(bool(info.get("success", False))), "steps": info.get("steps", 0),
                                "cost": info.get("cost", 0.0), "violations": len(ev.get("violations", [])),
                                **(extra or {})})
        if ep % self.eval_every == 0 or ep == self.episodes:
            self.validate(ep, env_steps)

    def validate(self, ep: int, env_steps: int):
        v = _validate(self.agent, self.val_set, self.settings)
        self.val_rows.append({"episode": ep, "env_steps": env_steps, **v, "seconds": time.time() - self.t0})
        if self.first_90 is None and v["val_success"] >= 0.9:
            self.first_90 = ep
        score = (v["val_success"], v["val_reward"])
        if score > self.best:
            self.best = score
            self.agent.save(str(model_path(self.name, self.seed, "best", self.tag)))
        if self.progress:
            self.progress({"agent": self.name, "seed": self.seed, "episode": ep, "episodes": self.episodes, **v})

    def finish(self) -> Dict[str, Any]:
        self.agent.save(str(model_path(self.name, self.seed, "final", self.tag)))
        pd.DataFrame(self.train_rows).to_csv(self.dir / "train_log.csv", index=False)
        pd.DataFrame(self.val_rows).to_csv(self.dir / "val_log.csv", index=False)
        last = self.val_rows[-1] if self.val_rows else {}
        summary = {"agent": self.name, "tag": self.tag, "seed": self.seed, "episodes": self.episodes,
                   "train_seconds": time.time() - self.t0, "final_val_success": last.get("val_success"),
                   "best_val_success": self.best[0], "episodes_to_90pct_val": self.first_90,
                   "env_steps": self.val_rows[-1]["env_steps"] if self.val_rows else 0}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def train_online(name: str, seed: int = 0, episodes: Optional[int] = None, hp: Optional[Dict[str, Any]] = None,
                 eval_every: int = 200, settings: Optional[Settings] = None,
                 progress: Optional[Callable[[Dict[str, Any]], None]] = None,
                 tag: str = "", masked: bool = True, mix: str = "standard") -> Dict[str, Any]:
    """tag="sparse" is used with settings whose goal_handled_bonus is 0 (outcome-only reward ablation);
    masked=False (tag="nomask") trains with no guardrail mask at all (safety ablation)."""
    settings = settings or get_settings()
    episodes = episodes or DEFAULT_EPISODES[name]
    agent = build_agent(name, seed, hp, episodes)
    agent.uses_mask = masked
    run = _Run(name, seed, agent, settings, episodes, eval_every, progress, {"online": True, "masked": masked}, tag)
    pool = training_pool(settings, mix=mix)
    rng = np.random.default_rng(seed)
    env = EcomSupportEnv(settings, enforce_mask=masked)
    agent.training = True
    env_steps = 0
    all_true = np.ones(NUM_ACTIONS, dtype=bool)
    mk = (lambda i: i["action_mask"]) if masked else (lambda i: all_true)

    if name == "ppo":
        return _train_ppo(run, agent, env, pool, rng, mk)

    for ep in range(1, episodes + 1):
        agent.episode = ep
        sc = pool[int(rng.integers(0, len(pool)))]
        obs, info = env.reset(options={"scenario": sc, "run_seed": ep})
        total, done = 0.0, False
        a = agent.act(obs, mk(info))
        while not done:
            nobs, r, term, trunc, info = env.step(a)
            done = term or trunc
            total += r
            env_steps += 1
            if name == "dqn":
                agent.remember(obs, a, r, nobs, term, mk(info))
                agent.train_step()
                na = agent.act(nobs, mk(info)) if not done else None
            else:
                na = agent.act(nobs, mk(info)) if not done else None
                agent.update(obs, a, r, nobs, term, mk(info), next_a=na)
            obs, a = nobs, na
        run.log_episode(ep, env_steps, total, info, {"epsilon": getattr(agent, "epsilon", lambda: 0.0)()})
    return run.finish()


def _train_ppo(run: _Run, agent: PPOAgent, env: EcomSupportEnv, pool, rng, mk) -> Dict[str, Any]:
    env_steps, ep, total = 0, 0, 0.0
    sc = pool[int(rng.integers(0, len(pool)))]
    obs, info = env.reset(options={"scenario": sc, "run_seed": 0})
    while ep < run.episodes:
        mask = mk(info)
        a, logp, val = agent.act_train(obs, mask)
        nobs, r, term, trunc, info = env.step(a)
        done = term or trunc
        agent.store(obs, a, logp, val, r, term, mask)
        total += r
        env_steps += 1
        obs = nobs
        if done:
            ep += 1
            run.log_episode(ep, env_steps, total, info)
            total = 0.0
            if ep < run.episodes:
                sc = pool[int(rng.integers(0, len(pool)))]
                obs, info = env.reset(options={"scenario": sc, "run_seed": ep})
        if agent.ready():
            agent.update(0.0 if done else agent.value_of(obs))
    return run.finish()


# ---------------------------------------------------------------------------
# Offline RL: log some "historical" conversations, then train CQL only on that log.
# ---------------------------------------------------------------------------
def collect_logged_data(n_episodes: int = 4000, seed: int = 0, settings: Optional[Settings] = None,
                        rule_share: float = 0.7, noise: float = 0.1) -> Dict[str, Any]:
    """Behaviour policy = 70% rule-based agent that makes a random legal move 10% of the time,
    30% fully random legal moves. This mimics a messy historical log: mostly sensible, sometimes wrong."""
    settings = settings or get_settings()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scenarios = generate_scenarios(SplitSpec("logged", 1500, TRAIN_COMBOS), n_episodes, settings)
    env = EcomSupportEnv(settings)
    rule, rows = RuleBasedAgent(), {k: [] for k in ("obs", "act", "rew", "next_obs", "done", "mask", "next_mask")}
    wins = 0
    for i, sc in enumerate(scenarios):
        use_rule = rng.random() < rule_share
        obs, info = env.reset(options={"scenario": sc, "run_seed": i})
        done = False
        while not done:
            mask = info["action_mask"]
            if use_rule and rng.random() >= noise:
                a = rule.act(obs, mask)
            else:
                a = int(rng.choice(np.flatnonzero(mask)))
            nobs, r, term, trunc, info = env.step(a)
            done = term or trunc
            for k, v in zip(rows, (obs, a, r, nobs, float(term), mask, info["action_mask"])):
                rows[k].append(v)
            obs = nobs
        wins += int(info["success"])
    np.savez_compressed(DATA_DIR / "logged_dataset.npz", **{k: np.array(v) for k, v in rows.items()})
    meta = {"episodes": n_episodes, "transitions": len(rows["act"]), "behaviour_success_rate": wins / n_episodes,
            "rule_share": rule_share, "noise": noise, "seed": seed}
    (DATA_DIR / "logged_dataset.json").write_text(json.dumps(meta, indent=2))
    return meta


LOGGED_EPISODES = 4000


def ensure_logged_data(settings: Settings, min_episodes: int = LOGGED_EPISODES) -> Dict[str, Any]:
    """The log CQL learns from must be at least `min_episodes` long; a smaller leftover file is regenerated."""
    meta_path = DATA_DIR / "logged_dataset.json"
    if (DATA_DIR / "logged_dataset.npz").exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["episodes"] >= min_episodes:
            return meta
    return collect_logged_data(n_episodes=max(min_episodes, 1), settings=settings)


def train_cql(seed: int = 0, steps: int = 20_000, hp: Optional[Dict[str, Any]] = None, eval_every_steps: int = 1000,
              settings: Optional[Settings] = None, progress: Optional[Callable[[Dict[str, Any]], None]] = None,
              tag: str = "", min_log_episodes: int = LOGGED_EPISODES) -> Dict[str, Any]:
    settings = settings or get_settings()
    log_meta = ensure_logged_data(settings, min_log_episodes)
    data = OfflineDataset.load(str(DATA_DIR / "logged_dataset.npz"))
    agent = build_agent("cql", seed, hp)
    run = _Run("cql", seed, agent, settings, steps, eval_every_steps, progress,
               {"offline": True, "steps_unit": "gradient steps", "logged_dataset": log_meta}, tag)
    for step in range(1, steps + 1):
        st = agent.train_step(data)
        if step % 100 == 0:
            run.train_rows.append({"episode": step, "env_steps": 0, "reward": 0.0, "success": 0, "steps": 0, "cost": 0.0,
                                   "violations": 0, **st})
        if step % eval_every_steps == 0 or step == steps:
            run.validate(step, 0)          # live evaluation only; no live data is used for training
    return run.finish()
