"""Background training / evaluation jobs.

Jobs run as separate processes (so the API stays responsive and a crash cannot take the server down).
Safety rules:
  - arguments are validated against a whitelist and passed as a list (never through a shell);
  - dashboard training always uses its own run tag, so it can never overwrite the official models.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ecommerce.backend.db import new_id
from ecommerce.backend.services import Services

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_AGENTS = ("q_learning", "sarsa", "dqn", "ppo", "cql")
HP_WHITELIST = {
    "q_learning": {"alpha", "gamma"},
    "sarsa": {"alpha", "gamma"},
    "dqn": {"lr", "gamma", "hidden", "batch_size", "target_update"},
    "ppo": {"lr", "gamma", "clip_eps", "ent_coef", "hidden", "rollout_steps"},
    "cql": {"lr", "gamma", "alpha", "hidden", "batch_size"},
}


class JobError(ValueError):
    pass


def _validate_training(p: Dict[str, Any]) -> Dict[str, Any]:
    agent = p.get("agent")
    if agent not in ALLOWED_AGENTS:
        raise JobError(f"agent must be one of {ALLOWED_AGENTS}")
    seeds = [int(x) for x in p.get("seeds", [0])]
    if not seeds or len(seeds) > 5 or any(s < 0 or s > 99 for s in seeds):
        raise JobError("seeds: 1 to 5 integers between 0 and 99")
    episodes = p.get("episodes")
    if episodes is not None:
        episodes = int(episodes)
        if not 100 <= episodes <= 50_000:
            raise JobError("episodes must be between 100 and 50000")
    variant = p.get("variant") or None
    if variant not in (None, "sparse", "nomask"):
        raise JobError("variant must be empty, 'sparse' or 'nomask'")
    hp = {}
    for k, v in (p.get("hp") or {}).items():
        if k not in HP_WHITELIST[agent]:
            raise JobError(f"hyperparameter '{k}' is not supported for {agent}")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise JobError(f"hyperparameter '{k}' must be a number")
        hp[k] = v
    return {"agent": agent, "seeds": seeds, "episodes": episodes, "variant": variant, "hp": hp,
            "eval_iid": int(min(max(int(p.get("eval_iid", 100)), 0), 300))}


class JobManager:
    def __init__(self, svc: Services):
        self.svc = svc
        self.procs: Dict[str, subprocess.Popen] = {}
        self.dir = svc.results_dir / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _spawn(self, kind: str, cmd: List[str], params: Dict[str, Any], job_id: str, progress_file: Path) -> Dict[str, Any]:
        log = self.dir / f"{job_id}.log"
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        self.procs[job_id] = proc
        job = {"job_id": job_id, "kind": kind, "status": "running", "params_json": params, "created_at": time.time(),
               "started_at": time.time(), "finished_at": None, "progress_file": str(progress_file), "log_file": str(log),
               "result_json": None, "error": None}
        self.svc.db.save_job(job)
        return job

    def submit_training(self, params: Dict[str, Any]) -> Dict[str, Any]:
        p = _validate_training(params)
        job_id = new_id("train")
        tag = "u" + job_id.split("_")[-1]
        progress = self.dir / f"{job_id}.jsonl"
        cmd = [sys.executable, "-m", "ecommerce.training.run_training", "--agents", p["agent"], "--seeds",
               *map(str, p["seeds"]), "--tag", tag, "--progress-file", str(progress), "--eval-iid", str(p["eval_iid"])]
        if p["episodes"] and p["agent"] != "cql":
            cmd += ["--episodes", str(p["episodes"])]
        if p["variant"]:
            cmd += ["--variant", p["variant"]]
        if p["hp"]:
            cmd += ["--hp", json.dumps(p["hp"])]
        return self._spawn("train", cmd, {**p, "run_tag": tag}, job_id, progress)

    def submit_evaluation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        llm = int(params.get("llm", 0))
        if not 0 <= llm <= 200:
            raise JobError("llm sample size must be between 0 and 200")
        job_id = new_id("eval")
        progress = self.dir / f"{job_id}.jsonl"
        cmd = [sys.executable, "-m", "ecommerce.evaluation.run_eval"] + (["--llm", str(llm)] if llm else [])
        return self._spawn("evaluate", cmd, {"llm": llm}, job_id, progress)

    # ---- status ---------------------------------------------------------------
    def refresh(self, job: Dict[str, Any]) -> Dict[str, Any]:
        proc = self.procs.get(job["job_id"])
        if job["status"] == "running" and proc is not None:
            code = proc.poll()
            if code is not None:
                job["status"] = "succeeded" if code == 0 else "failed"
                job["finished_at"] = time.time()
                if code != 0:
                    job["error"] = self._tail(job["log_file"], 8)
                self.svc.db.save_job(job)
        job["progress"] = self._progress(job)
        return job

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.svc.db.get_job(job_id)
        return self.refresh(job) if job else None

    def list(self) -> List[Dict[str, Any]]:
        return [self.refresh(j) for j in self.svc.db.list_jobs()]

    def cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.svc.db.get_job(job_id)
        proc = self.procs.get(job_id)
        if job and proc and proc.poll() is None:
            proc.terminate()
            job["status"], job["finished_at"] = "cancelled", time.time()
            self.svc.db.save_job(job)
        return self.refresh(job) if job else None

    @staticmethod
    def _tail(path: str, n: int) -> str:
        try:
            return "\n".join(Path(path).read_text(errors="replace").strip().splitlines()[-n:])
        except OSError:
            return ""

    def _progress(self, job: Dict[str, Any]) -> Dict[str, Any]:
        events: List[Dict[str, Any]] = []
        pf = Path(job["progress_file"]) if job.get("progress_file") else None
        if pf and pf.exists():
            for line in pf.read_text().splitlines()[-400:]:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        val = [e for e in events if "val_success" in e]
        last = val[-1] if val else None
        return {"last": last, "curve": [{"episode": e["episode"], "val_success": e["val_success"], "seed": e["seed"]} for e in val],
                "evaluated": [e for e in events if e.get("event") == "evaluated"],
                "done": [e for e in events if e.get("event") == "done"],
                "percent": (round(100 * last["episode"] / last["episodes"]) if last else 0)}
