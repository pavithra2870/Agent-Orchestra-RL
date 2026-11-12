import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { Empty, ErrorBox, PageHead, Pill } from "../components/ui.jsx";
import { reasonText, RunView } from "../components/RunView.jsx";

const SPLITS = ["", "iid", "heldout_combos", "tool_failures", "missing_info", "out_of_scope", "returns", "policy"];
const REPLAYABLE = ["dqn", "ppo", "cql", "q_learning", "sarsa", "rule_based", "random"];

export default function Failures() {
  const [agents, setAgents] = useState([]);
  const [agent, setAgent] = useState("dqn");
  const [split, setSplit] = useState("");
  const [reason, setReason] = useState("");
  const [seed, setSeed] = useState(0);
  const [data, setData] = useState(null);
  const [replay, setReplay] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => { api.summary().then((s) => setAgents(Object.entries(s.agents).map(([k, a]) => [k, a.label]))).catch((e) => setError(e.message)); }, []);
  useEffect(() => { setReplay(null); api.failures({ agent, split, reason, seed, limit: 40 }).then(setData).catch((e) => setError(e.message)); }, [agent, split, reason, seed]);

  async function open(row) {
    setError("");
    try { setReplay(await api.replay({ scenario_id: row.scenario_id, agent: REPLAYABLE.includes(agent) ? agent : "rule_based" })); window.scrollTo({ top: 0, behavior: "smooth" }); }
    catch (e) { setError(e.message); }
  }

  return (
    <>
      <PageHead title="Failures">
        Why do agents fail? Pick an agent and test set, filter by reason, and open any failed scenario to replay it step by step. Scenarios are
        reproducible from their id, so the replay is the same situation the agent faced in the evaluation.
      </PageHead>
      <ErrorBox error={error} />
      {replay && (
        <div className="card" style={{ marginBottom: 14 }}>
          <div className="card-head"><h3>Replay · <span className="mono">{replay.scenario.scenario_id}</span> <span className="muted small">({replay.scenario.type}; true goals: {replay.scenario.true_goals.join(", ") || "none"})</span></h3>
            <button className="btn ghost small" onClick={() => setReplay(null)}>Close</button></div>
          <p className="small"><b>Customer said:</b> "{replay.scenario.text}"</p>
          <RunView events={replay.events} running={false} />
        </div>
      )}
      <div className="card">
        <div className="card-head">
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <select value={agent} onChange={(e) => { setAgent(e.target.value); setReason(""); }} style={{ width: 260 }}>{agents.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>
            <select value={split} onChange={(e) => setSplit(e.target.value)} style={{ width: 180 }}>{SPLITS.map((s) => <option key={s} value={s}>{s || "all test sets"}</option>)}</select>
            <select value={seed} onChange={(e) => setSeed(+e.target.value)} style={{ width: 150 }} title="Each learning agent was trained 3 times with different seeds"><option value={0}>training seed 0</option><option value={1}>training seed 1</option><option value={2}>training seed 2</option></select>
          </div>
          {data && <span className="muted">{data.total} failed episodes (first run of each scenario, one training seed)</span>}
        </div>
        {!REPLAYABLE.includes(agent) && <div className="note" style={{ marginBottom: 10 }}>This variant has no replay checkpoint in the demo; replays will use the rule-based agent for comparison.</div>}
        {data && (
          <>
            <div className="chips" style={{ marginBottom: 12 }}>
              <button className={`chip ${reason === "" ? "pill accent" : ""}`} onClick={() => setReason("")}>all reasons</button>
              {Object.entries(data.by_reason).sort((a, b) => b[1] - a[1]).map(([r, n]) => (
                <button key={r} className="chip" style={reason === r ? { borderColor: "var(--accent)", color: "var(--ink)" } : null} onClick={() => setReason(r)}>{reasonText(r)} · {n}</button>))}
            </div>
            {data.rows.length === 0 ? <Empty>No failures for this selection.</Empty> : (
              <table className="t"><thead><tr><th>Scenario</th><th>Test set</th><th>Type</th><th>Goals</th><th>Why it failed</th><th className="r">Steps</th><th /></tr></thead>
                <tbody>{data.rows.map((r) => (
                  <tr key={r.scenario_id + r.seed} className="click" onClick={() => open(r)}>
                    <td className="mono">{r.scenario_id}</td><td>{r.split}</td><td>{r.type.replace("_", " ")}</td><td className="small">{r.combo.replaceAll("_", " ").replaceAll("+", " + ")}</td>
                    <td>{String(r.failure_reasons).split(";").filter(Boolean).map((f) => <Pill key={f} tone="bad">{reasonText(f)}</Pill>)}</td>
                    <td className="r num">{r.steps}</td><td><button className="btn ghost small">Replay</button></td></tr>))}</tbody></table>
            )}
          </>
        )}
      </div>
    </>
  );
}
