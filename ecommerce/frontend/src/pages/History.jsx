import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { Empty, ErrorBox, PageHead, Pill } from "../components/ui.jsx";
import { RunView } from "../components/RunView.jsx";

const when = (t) => new Date(t * 1000).toLocaleString();
const tone = (s) => (s === "ok" ? "good" : s === "failed" ? "bad" : "warn");

export default function History() {
  const [list, setList] = useState({ total: 0, runs: [] });
  const [filters, setFilters] = useState({ agent: "", status: "", kind: "", q: "" });
  const [offset, setOffset] = useState(0);
  const [picked, setPicked] = useState([]);
  const [open, setOpen] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => { api.runs({ ...filters, limit: 15, offset }).then(setList).catch((e) => setError(e.message)); }, [filters, offset]);
  const set = (k, v) => { setOffset(0); setFilters((f) => ({ ...f, [k]: v })); };
  const toggle = (id) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p.slice(-1), id]));
  const view = async (id) => { try { setOpen(await api.run(id)); } catch (e) { setError(e.message); } };
  const cmp = picked.map((id) => list.runs.find((r) => r.run_id === id)).filter(Boolean);

  return (
    <>
      <PageHead title="Run history">Every conversation and replay is saved with its settings and the full step-by-step record, so it can be searched, replayed and compared.</PageHead>
      <ErrorBox error={error} />
      {open && (
        <div className="card" style={{ marginBottom: 14 }}>
          <div className="card-head"><h3>{open.kind} · <span className="mono">{open.run_id}</span></h3><button className="btn ghost small" onClick={() => setOpen(null)}>Close</button></div>
          <p className="small"><b>Message:</b> "{open.message}" · agent {open.agent} · language {open.language_mode}</p>
          <RunView events={open.events} running={false} />
        </div>
      )}
      <div className="card">
        <div className="card-head" style={{ flexWrap: "wrap" }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <input type="text" placeholder="search message or id" value={filters.q} onChange={(e) => set("q", e.target.value)} style={{ width: 220 }} />
            <select value={filters.agent} onChange={(e) => set("agent", e.target.value)} style={{ width: 150 }}><option value="">all agents</option>{["dqn", "ppo", "cql", "q_learning", "sarsa", "rule_based", "random"].map((a) => <option key={a}>{a}</option>)}</select>
            <select value={filters.status} onChange={(e) => set("status", e.target.value)} style={{ width: 130 }}><option value="">any result</option><option value="ok">ok</option><option value="failed">failed</option><option value="waiting">waiting</option></select>
            <select value={filters.kind} onChange={(e) => set("kind", e.target.value)} style={{ width: 130 }}><option value="">chat + replay</option><option value="chat">chat</option><option value="replay">replay</option></select>
          </div>
          <span className="muted">{list.total} runs</span>
        </div>
        {list.runs.length === 0 ? <Empty>No runs yet. Send a message on the Live conversation page.</Empty> : (
          <table className="t">
            <thead><tr><th /><th>When</th><th>Agent</th><th>Message</th><th>Result</th><th className="r">Steps</th><th className="r">Reward</th><th /></tr></thead>
            <tbody>{list.runs.map((r) => (
              <tr key={r.run_id} className={picked.includes(r.run_id) ? "sel" : ""}>
                <td><input type="checkbox" checked={picked.includes(r.run_id)} onChange={() => toggle(r.run_id)} title="select two runs to compare" /></td>
                <td className="small muted">{when(r.created_at)}</td><td>{r.agent}<div className="muted small">{r.language_mode} · {r.kind}</div></td>
                <td style={{ maxWidth: 380 }} className="small">{r.message}</td>
                <td><Pill tone={tone(r.status)}>{r.status}</Pill> <span className="muted small">{r.terminal}</span></td>
                <td className="r num">{r.steps}</td><td className="r num">{r.reward != null ? r.reward.toFixed(2) : "-"}</td>
                <td><button className="btn ghost small" onClick={() => view(r.run_id)}>Open</button></td></tr>))}</tbody>
          </table>
        )}
        <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
          <button className="btn ghost small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 15))}>Previous</button>
          <button className="btn ghost small" disabled={offset + 15 >= list.total} onClick={() => setOffset(offset + 15)}>Next</button>
        </div>
      </div>

      {cmp.length === 2 && (
        <div className="card" style={{ marginTop: 14 }}>
          <div className="card-head"><h3>Side by side</h3></div>
          <table className="t">
            <thead><tr><th /><th>{cmp[0].agent}</th><th>{cmp[1].agent}</th></tr></thead>
            <tbody>
              {[["Message", "message"], ["Result", "status"], ["Ending", "terminal"], ["Steps", "steps"], ["Reward", "reward"], ["Cost", "cost"], ["Latency", "latency"]].map(([l, k]) => (
                <tr key={k}><td className="muted">{l}</td>{cmp.map((r) => <td key={r.run_id} className="small">{typeof r[k] === "number" ? r[k].toFixed(2) : String(r[k])}</td>)}</tr>))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
