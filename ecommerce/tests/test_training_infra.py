"""Training infrastructure: data guards, robust pool, and background jobs (real subprocess)."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from ecommerce.backend.main import create_app
from ecommerce.backend.services import Services
from ecommerce.config import get_settings
from ecommerce.training import trainer


def test_robust_pool_contains_scenarios_with_a_tool_down():
    settings = get_settings()
    std = trainer.training_pool(settings, size=300, mix="standard")
    robust = trainer.training_pool(settings, size=300, mix="robust")
    assert not any(s.down_tools for s in std)
    stress = [s for s in robust if s.down_tools]
    assert len(stress) > 20 and all(s.failure_multiplier > 1 for s in robust if s.down_tools)
    assert {s.scenario_id for s in std}.isdisjoint({s.scenario_id for s in robust if s.split == "train_robust"})


def test_a_too_small_log_is_regenerated_not_silently_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "DATA_DIR", tmp_path)
    trainer.collect_logged_data(n_episodes=60, seed=0)
    assert json.loads((tmp_path / "logged_dataset.json").read_text())["episodes"] == 60
    meta = trainer.ensure_logged_data(get_settings(), min_episodes=120)
    assert meta["episodes"] >= 120
    again = trainer.ensure_logged_data(get_settings(), min_episodes=120)
    assert again == meta, "a log that is large enough must be reused, not regenerated"


def test_dashboard_training_job_runs_end_to_end_and_leaves_official_models_alone(tmp_path):
    official = trainer.model_path("dqn", 0, "best")
    before = official.stat().st_mtime if official.exists() else None
    client = TestClient(create_app(Services(db_path=tmp_path / "t.sqlite", results_dir=tmp_path / "res"), warm_up=False))
    job = client.post("/api/jobs/train", json={"agent": "dqn", "seeds": [0], "episodes": 100, "eval_iid": 20}).json()
    assert job["status"] == "running" and job["params_json"]["run_tag"].startswith("u")
    deadline = time.time() + 240
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{job['job_id']}").json()
        if j["status"] != "running":
            break
        time.sleep(2)
    assert j["status"] == "succeeded", j.get("error")
    assert j["progress"]["done"] and j["progress"]["evaluated"], "the job must report a test result when it finishes"
    tag = job["params_json"]["run_tag"]
    run = trainer.run_dir("dqn", 0, tag)
    assert (run / "model_best.pt").exists() and (run / "config.json").exists()
    assert (official.stat().st_mtime if official.exists() else None) == before, "official model must be untouched"
    import shutil
    shutil.rmtree(run)
    assert any(x["job_id"] == job["job_id"] for x in client.get("/api/jobs").json())
