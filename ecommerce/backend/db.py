"""SQLite run tracker. Structured, queryable records here; big raw outputs stay as files under results/.

Tables
  runs   one row per chat / replay run (who, what, agent, seed, outcome, reply, evaluator result)
  steps  one row per event of a run (parse, decision, tool, state, reply ...) so any run can be replayed
  jobs   background training / evaluation jobs and their progress
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at REAL, kind TEXT, name TEXT, agent TEXT, language_mode TEXT, seed INTEGER,
    customer_id TEXT, message TEXT, status TEXT, success INTEGER, reward REAL, steps INTEGER, cost REAL, latency REAL,
    terminal TEXT, reply TEXT, parsed_json TEXT, eval_json TEXT, config_json TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT, idx INTEGER, type TEXT, payload_json TEXT, PRIMARY KEY (run_id, idx)
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, kind TEXT, status TEXT, params_json TEXT, created_at REAL, started_at REAL,
    finished_at REAL, progress_file TEXT, log_file TEXT, result_json TEXT, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            c = sqlite3.connect(self.path, timeout=30)
            c.row_factory = sqlite3.Row
            try:
                yield c
                c.commit()
            except Exception:
                c.rollback()
                raise
            finally:
                c.close()

    # ---- runs ---------------------------------------------------------------
    def save_run(self, run: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
        cols = ["run_id", "created_at", "kind", "name", "agent", "language_mode", "seed", "customer_id", "message",
                "status", "success", "reward", "steps", "cost", "latency", "terminal", "reply", "parsed_json",
                "eval_json", "config_json", "notes"]
        row = {c: run.get(c) for c in cols}
        for k in ("parsed_json", "eval_json", "config_json"):
            if row[k] is not None and not isinstance(row[k], str):
                row[k] = json.dumps(row[k])
        with self.conn() as c:
            c.execute(f"INSERT OR REPLACE INTO runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                      [row[k] for k in cols])
            c.execute("DELETE FROM steps WHERE run_id=?", (run["run_id"],))
            c.executemany("INSERT INTO steps (run_id, idx, type, payload_json) VALUES (?,?,?,?)",
                          [(run["run_id"], i, e.get("type", ""), json.dumps(e)) for i, e in enumerate(events)])

    def list_runs(self, limit: int = 50, offset: int = 0, agent: Optional[str] = None, status: Optional[str] = None,
                  q: Optional[str] = None, kind: Optional[str] = None) -> Dict[str, Any]:
        where, args = [], []
        for col, val in (("agent", agent), ("status", status), ("kind", kind)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        if q:
            where.append("(message LIKE ? OR run_id LIKE ? OR name LIKE ?)")
            args += [f"%{q}%"] * 3
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with self.conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM runs {clause}", args).fetchone()[0]
            rows = c.execute(f"SELECT run_id, created_at, kind, name, agent, language_mode, seed, customer_id, message, "
                             f"status, success, reward, steps, cost, latency, terminal FROM runs {clause} "
                             f"ORDER BY created_at DESC LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()
        return {"total": total, "runs": [dict(r) for r in rows]}

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self.conn() as c:
            r = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if r is None:
                return None
            steps = c.execute("SELECT payload_json FROM steps WHERE run_id=? ORDER BY idx", (run_id,)).fetchall()
        out = dict(r)
        for k in ("parsed_json", "eval_json", "config_json"):
            out[k] = json.loads(out[k]) if out[k] else None
        out["events"] = [json.loads(s[0]) for s in steps]
        return out

    # ---- jobs ---------------------------------------------------------------
    def save_job(self, job: Dict[str, Any]) -> None:
        cols = ["job_id", "kind", "status", "params_json", "created_at", "started_at", "finished_at", "progress_file",
                "log_file", "result_json", "error"]
        row = {c: job.get(c) for c in cols}
        for k in ("params_json", "result_json"):
            if row[k] is not None and not isinstance(row[k], str):
                row[k] = json.dumps(row[k])
        with self.conn() as c:
            c.execute(f"INSERT OR REPLACE INTO jobs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                      [row[k] for k in cols])

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._job(r) if r else None

    def list_jobs(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._job(r) for r in rows]

    @staticmethod
    def _job(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        for k in ("params_json", "result_json"):
            d[k] = json.loads(d[k]) if d[k] else None
        return d

    def mark_interrupted_jobs(self) -> int:
        """Jobs left 'running' by a previous server process cannot still be running."""
        with self.conn() as c:
            cur = c.execute("UPDATE jobs SET status='interrupted', finished_at=? WHERE status IN ('queued','running')",
                            (time.time(),))
            return cur.rowcount
