"""Small PPO hyperparameter sweep. Selection uses the VALIDATION split only (never the test splits).

  python -m ecommerce.training.tune_ppo
Writes ecommerce/results/training/ppo_tuning.json
"""

from __future__ import annotations

import json
import time

from ecommerce.training import trainer

EPISODES = 12000
GRID = [
    {"name": "default", "hp": {}},
    {"name": "lr1e-3", "hp": {"lr": 1e-3}},
    {"name": "lr1e-3_roll512", "hp": {"lr": 1e-3, "rollout_steps": 512}},
    {"name": "lr3e-4_roll512_ent.003", "hp": {"rollout_steps": 512, "ent_coef": 0.003}},
    {"name": "lr1e-3_roll512_ent.003_gamma.97", "hp": {"lr": 1e-3, "rollout_steps": 512, "ent_coef": 0.003, "gamma": 0.97}},
]


def main() -> None:
    results = []
    for i, g in enumerate(GRID):
        t = time.time()
        s = trainer.train_online("ppo", seed=0, episodes=EPISODES, hp=g["hp"], tag=f"tune{i}")
        results.append({**g, "best_val_success": s["best_val_success"], "final_val_success": s["final_val_success"],
                        "episodes_to_90pct_val": s["episodes_to_90pct_val"], "seconds": time.time() - t})
        print(g["name"], round(s["best_val_success"], 3), s["episodes_to_90pct_val"], flush=True)
    best = max(results, key=lambda r: (r["best_val_success"], -(r["episodes_to_90pct_val"] or 10**9)))
    out = {"episodes": EPISODES, "seed": 0, "selection": "highest best_val_success on the validation split, ties broken by fewest episodes to 90%", "grid": results,
           "chosen": {"name": best["name"], "hp": best["hp"]}}
    (trainer.TRAIN_DIR / "ppo_tuning.json").write_text(json.dumps(out, indent=2))
    print("chosen:", out["chosen"])


if __name__ == "__main__":
    main()
