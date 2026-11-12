import React, { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts";
import { api, fmt } from "../api.js";
import { Empty, ErrorBox, PageHead, Pill, Progress, Tip } from "../components/ui.jsx";

const HP = {
  q_learning: [["alpha", "Learning rate", 0.2, "How far each update moves the table. Higher learns faster but is noisier."], ["gamma", "Discount", 0.97, "How much future rewards count. Closer to 1 means the agent plans further ahead."]],
  sarsa: [["alpha", "Learning rate", 0.2, "How far each update moves the table."], ["gamma", "Discount", 0.97, "How much future rewards count."]],
  dqn: [["lr", "Learning rate", 0.001, "Step size of the neural network."], ["gamma", "Discount", 0.97, "How much future rewards count."], ["hidden", "Hidden units", 128, "Width of the network (more = more capacity)."], ["batch_size", "Batch size", 64, "Experiences used per update."], ["target_update", "Target sync", 250, "Updates between copying the network to its stable target copy."]],
  ppo: [["lr", "Learning rate", 0.001, "Step size of the neural network."], ["gamma", "Discount", 0.97, "How much future rewards count."], ["clip_eps", "Clip", 0.2, "Limits how much the policy may change per update (stability)."], ["ent_coef", "Entropy bonus", 0.003, "Encourages exploring; too high keeps the agent random."], ["rollout_steps", "Rollout length", 512, "Steps collected before each update."]],
  cql: [["lr", "Learning rate", 0.0005, "Step size of the neural network."], ["gamma", "Discount", 0.97, "How much future rewards count."], ["alpha", "Conservatism", 1.0, "Weight of the penalty on actions the log never shows. 0 turns CQL into plain offline Q-learning."], ["batch_size", "Batch size", 128, "Log samples per update."]],
};
const defaultsFor = (a) => Object.fromEntries(HP[a].map(([k, , v]) => [k, v]));

function MiniCurve({ points }) {
  const data = points.map((p) => ({ x: p.episode, y: +(100 * p.val_success).toFixed(1), s: p.seed }));
  if (!data.length) return null;
  return (<div style={{ height: 120 }}><ResponsiveContainer><LineChart data={data}><CartesianGrid stroke="#e6e1d6" /><XAxis dataKey="x" tick={{ fontSize: 10 }} /><YAxis domain={[0, 100]} tick={{ fontSize: 10 }} width={30} /><Line dataKey="y" stroke="#c8502a" dot={false} strokeWidth={2} /></LineChart></ResponsiveContainer></div>);
}

export default function RunSetup() {
  const [agent, setAgent] = useState("dqn");
  const [hp, setHp] = useState(defaultsFor("dqn"));
  const [seeds, setSeeds] = useState("0");
  const [episodes, setEpisodes] = useState("");
  const [variant, setVariant] = useState("");
  const [jobs, setJobs] = useState([]);
  const [error, setError] = useState("");
  const [llm, setLlm] = useState(0);

  const load = () => api.jobs().then(setJobs).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);
  useEffect(() => {
    if (!jobs.some((j) => j.status === "running")) return undefined;
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [jobs]);

  const pick = (a) => { setAgent(a); setHp(defaultsFor(a)); };

  async function start() {
    setError("");
    try {
      const changed = Object.fromEntries(Object.entries(hp).filter(([k, v]) => Number(v) !== defaultsFor(agent)[k]).map(([k, v]) => [k, Number(v)]));
      await api.train({ agent, seeds: seeds.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => !Number.isNaN(n)), episodes: episodes ? parseInt(episodes, 10) : null, variant: variant || null, hp: changed, eval_iid: 100 });
      load();
    } catch (e) { setError(e.message); }
  }

  return (
    <>
      <PageHead title="Run setup">
        Start a training run from here. It runs as a separate background process under its own name, so it can never overwrite the official models
        used in the comparison. When it finishes, the best checkpoint is tested on 100 fresh scenarios.
      </PageHead>
      <ErrorBox error={error} />
      <div className="grid g2" style={{ alignItems: "start" }}>
        <div className="card">
          <div className="card-head"><h3>New training run</h3></div>
          <label className="f"><span>Method</span>
            <select value={agent} onChange={(e) => pick(e.target.value)}>
              <option value="dqn">DQN (online)</option><option value="ppo">PPO (online)</option><option value="cql">CQL (offline)</option>
              <option value="q_learning">Q-learning (tabular)</option><option value="sarsa">SARSA (tabular)</option>
            </select></label>
          <div className="grid g2">
            <label className="f"><span>Seeds <Tip text="Comma-separated. Each seed is an independent training run (different randomness). At most 5." /></span><input type="text" value={seeds} onChange={(e) => setSeeds(e.target.value)} /></label>
            <label className="f"><span>Episodes <Tip text={agent === "cql" ? "CQL trains for a fixed number of gradient steps on the log; this field is ignored." : "Training conversations. Leave empty for the default used in the report."} /></span>
              <input type="number" value={episodes} placeholder="default" disabled={agent === "cql"} onChange={(e) => setEpisodes(e.target.value)} /></label>
          </div>
          {agent !== "cql" && (
            <label className="f"><span>Ablation <Tip text="Optional. 'Outcome-only reward' removes the bonus for each handled goal. 'No guardrail' trains without the action mask (shows why the mask matters)." /></span>
              <select value={variant} onChange={(e) => setVariant(e.target.value)}><option value="">None (normal training)</option><option value="sparse">Outcome-only reward</option><option value="nomask">No guardrail</option></select></label>
          )}
          <h3 style={{ margin: "6px 0 8px" }}>Hyperparameters</h3>
          <div className="grid g2">
            {HP[agent].map(([k, label, , tip]) => (
              <label className="f" key={k}><span>{label} <Tip text={tip} /></span>
                <input type="number" step="any" value={hp[k]} onChange={(e) => setHp((h) => ({ ...h, [k]: e.target.value }))} /></label>))}
          </div>
          <button className="btn accent" onClick={start}>Start training</button>
        </div>

        <div>
          <div className="card">
            <div className="card-head"><h3>Jobs</h3><button className="btn ghost small" onClick={load}>Refresh</button></div>
            {jobs.length === 0 && <Empty>No jobs yet.</Empty>}
            {jobs.map((j) => <Job key={j.job_id} job={j} onChange={load} />)}
          </div>
          <div className="card">
            <div className="card-head"><h3>Re-run the full evaluation</h3></div>
            <p className="small muted">Recomputes the comparison tables from the saved official models (about 10 minutes). Optionally benchmark the LLM agent on a sample (slow).</p>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <select value={llm} onChange={(e) => setLlm(+e.target.value)} style={{ width: 200 }}><option value={0}>No LLM benchmark</option><option value={30}>LLM benchmark: 30 scenarios</option><option value={60}>LLM benchmark: 60 scenarios</option></select>
              <button className="btn" onClick={() => api.evaluate({ llm }).then(load).catch((e) => setError(e.message))}>Start evaluation</button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function Job({ job, onChange }) {
  const p = job.progress || {};
  const tone = { running: "accent", succeeded: "good", failed: "bad", cancelled: "warn", interrupted: "warn" }[job.status] || "";
  const prm = job.params_json || {};
  return (
    <div className="step">
      <div className="step-h">
        <Pill tone={tone}>{job.status}</Pill><b>{job.kind === "train" ? `${prm.agent}${prm.variant ? " · " + prm.variant : ""} · seeds ${prm.seeds?.join(",")}` : `evaluation${prm.llm ? ` + LLM(${prm.llm})` : ""}`}</b>
        <span className="muted small mono">{job.job_id}</span>
        {job.status === "running" && <button className="btn ghost small" onClick={() => api.cancel(job.job_id).then(onChange)}>Cancel</button>}
      </div>
      {job.kind === "train" && (<>
        <Progress value={job.status === "succeeded" ? 100 : p.percent || 0} />
        {p.last && <div className="small muted num" style={{ marginTop: 4 }}>{p.last.agent} seed {p.last.seed}: episode {p.last.episode}/{p.last.episodes} · validation success {fmt.pct(p.last.val_success)}</div>}
        <MiniCurve points={(p.curve || []).filter((c) => c.seed === p.last?.seed)} />
        {p.evaluated?.map((e, i) => <div key={i} className="chips" style={{ marginTop: 6 }}><Pill tone="good">test iid: {fmt.pct(e.success_rate)} success</Pill><Pill>regret {fmt.num(e.regret)}</Pill><Pill tone={e.violation_rate ? "bad" : "good"}>{fmt.pct(e.violation_rate)} violations</Pill><span className="small muted">seed {e.seed}</span></div>)}
      </>)}
      {job.error && <pre className="json" style={{ marginTop: 6 }}>{job.error}</pre>}
    </div>
  );
}
