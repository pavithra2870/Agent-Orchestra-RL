import React, { useContext, useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AGENT_COLORS, SHORT_NAMES, api, fmt } from "../api.js";
import { Empty, ErrorBox, HelpContext, PageHead, Pill, Stat, Tip } from "../components/ui.jsx";

const SPLITS = [["overall", "Overall"], ["iid", "iid"], ["heldout_combos", "held-out combos"], ["tool_failures", "tool failures"],
  ["missing_info", "missing info"], ["out_of_scope", "out of scope"], ["returns", "returns and damage"], ["policy", "policy questions"]];
const COLS = [
  ["success_rate", "Success", "pct"], ["pass_k", "pass^4", "pct"], ["violation_rate", "Violations", "pct"], ["regret", "Regret", "num"],
  ["avg_steps", "Steps", "num1"], ["avg_cost", "Cost", "num1"], ["avg_latency", "Latency", "num1"],
  ["clarify_recall", "Asks when needed", "pct"], ["escalate_recall", "Hands over when needed", "pct"],
];
const LOWER_BETTER = new Set(["violation_rate", "regret", "avg_steps", "avg_cost", "avg_latency"]);
const KIND_ORDER = [["online", "Learning agents (online)"], ["offline", "Learning agents (offline)"],
  ["improvement", "Improvement tried: training with tool-failure scenarios"], ["baseline", "Baselines"],
  ["ablation", "Ablations (what happens if we remove something)"]];
const NOT_A_CANDIDATE = new Set(["random", "random_no_guardrail"]);

function metricOf(agent, split, key) {
  return split === "overall" ? agent.overall?.[key] : agent.splits?.[split]?.[key];
}
function show(m, kind) {
  if (!m) return "-";
  if (kind === "pct") return `${(100 * m.mean).toFixed(1)}%` + (m.std > 0.0005 ? ` ±${(100 * m.std).toFixed(1)}` : "");
  const d = kind === "num1" ? 1 : 2;
  return m.mean.toFixed(d) + (m.std > 0.005 ? ` ±${m.std.toFixed(d)}` : "");
}

export default function Compare() {
  const help = useContext(HelpContext);
  const [summary, setSummary] = useState(null);
  const [breakdowns, setBreakdowns] = useState(null);
  const [split, setSplit] = useState("overall");
  const [sort, setSort] = useState({ key: "success_rate", dir: -1 });
  const [error, setError] = useState("");

  useEffect(() => { Promise.all([api.summary(), api.breakdowns()]).then(([s, b]) => { setSummary(s); setBreakdowns(b); }).catch((e) => setError(e.message)); }, []);

  const rows = useMemo(() => {
    if (!summary) return [];
    return Object.entries(summary.agents).map(([key, a]) => ({ key, ...a }));
  }, [summary]);

  if (error) return <><PageHead title="Compare agents" /><ErrorBox error={error} /><Empty>No evaluation results yet. Run: python -m ecommerce.evaluation.run_eval</Empty></>;
  if (!summary) return <PageHead title="Compare agents">Loading...</PageHead>;

  const best = {};
  for (const [k] of COLS) {
    const vals = rows.filter((r) => r.kind !== "ablation" && !NOT_A_CANDIDATE.has(r.key) && metricOf(r, split, k)).map((r) => metricOf(r, split, k).mean);
    best[k] = vals.length ? (LOWER_BETTER.has(k) ? Math.min(...vals) : Math.max(...vals)) : null;
  }
  const sortVal = (r) => metricOf(r, split, sort.key)?.mean ?? -1;
  const chart = rows.filter((r) => r.kind !== "ablation").sort((a, b) => sortVal(b) - sortVal(a)).map((r) => {
    const m = metricOf(r, split, "success_rate");
    return { name: SHORT_NAMES[r.key] || r.label, key: r.key, value: +(100 * m.mean).toFixed(1), err: +(100 * m.std).toFixed(1) };
  });
  const get = (k) => summary.agents[k];
  const stat = (k, s = "overall") => metricOf(get(k) || {}, s, "success_rate")?.mean;
  const top = rows.filter((r) => ["online", "offline", "improvement"].includes(r.kind)).sort((a, b) => sortVal(b) - sortVal(a))[0];
  const meta = summary.meta;
  const viol = (k) => metricOf(get(k) || {}, "overall", "violation_rate")?.mean;

  return (
    <>
      <PageHead title="Compare agents">
        Every agent is tested on the same scenarios, {meta.k} runs each (tool failures differ between runs), and the learning agents are trained with
        {" "}{meta.seeds.length} different seeds (shown as ± spread). Success is judged from the database, not from what the agent says.
      </PageHead>

      <div className="grid g4" style={{ marginBottom: 14 }}>
        <Stat label="Best learning agent" value={top?.label.replace(" (offline)", "")} sub={`${fmt.pct(metricOf(top, split, "success_rate")?.mean)} success`} k="success_rate" />
        <Stat label="Rule-based playbook" value={fmt.pct(stat("rule_based", split === "overall" ? "overall" : split))} sub="hand-written baseline" />
        <Stat label="Random (legal actions)" value={fmt.pct(stat("random", split))} sub="lower bound" />
        <Stat label="Evaluation size" value={Object.values(meta.sizes).reduce((a, b) => a + b, 0)} sub={`scenarios × ${meta.k} runs, 5 test sets`} />
      </div>

      <div className="card">
        <div className="card-head">
          <h3>Results by test set</h3>
          <div className="seg">{SPLITS.map(([k, l]) => <button key={k} className={split === k ? "on" : ""} onClick={() => setSplit(k)}>{l}</button>)}</div>
        </div>
        {split !== "overall" && <p className="muted small"><Tip k={`split_${split}`} /> {help[`split_${split}`]?.text}</p>}
        <table className="t">
          <thead><tr><th>Agent</th>{COLS.map(([k, l]) => (
            <th key={k} className={`r sort`} onClick={() => setSort((s) => ({ key: k, dir: s.key === k ? -s.dir : (LOWER_BETTER.has(k) ? 1 : -1) }))}>
              {l} <Tip k={k} />{sort.key === k ? (sort.dir < 0 ? " ↓" : " ↑") : ""}</th>))}</tr></thead>
          <tbody>
            {KIND_ORDER.map(([kind, title]) => {
              const group = rows.filter((r) => r.kind === kind).sort((a, b) => sort.dir * ((metricOf(a, split, sort.key)?.mean ?? 0) - (metricOf(b, split, sort.key)?.mean ?? 0)));
              if (!group.length) return null;
              return (
                <React.Fragment key={kind}>
                  <tr className="grp"><td colSpan={COLS.length + 1}>{title}</td></tr>
                  {group.map((r) => (
                    <tr key={r.key}>
                      <td><span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: AGENT_COLORS[r.key] || "#999", marginRight: 8 }} />{r.label}
                        <span className="muted small"> · {r.seeds.length} seed{r.seeds.length > 1 ? "s" : ""}</span></td>
                      {COLS.map(([k, , kind2]) => {
                        const m = metricOf(r, split, k);
                        const isBest = r.kind !== "ablation" && !NOT_A_CANDIDATE.has(r.key) && m && best[k] != null && Math.abs(m.mean - best[k]) < 1e-9;
                        const bad = k === "violation_rate" && m && m.mean > 0.001;
                        return <td key={k} className={`r num ${isBest ? "best" : ""} ${bad ? "worst" : ""}`}>{show(m, kind2)}</td>;
                      })}
                    </tr>
                  ))}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
        <p className="muted small" style={{ marginTop: 8 }}>Green = best among the real agents (random and ablations excluded; random "wins" steps and cost only by giving up). Red violations = rules were broken. "±" is the spread across training seeds.</p>
      </div>

      <div className="grid g2" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="card-head"><h3>Success rate ({SPLITS.find((s) => s[0] === split)[1]})</h3></div>
          <div style={{ height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={chart} margin={{ top: 8, right: 12, bottom: 34, left: 0 }}>
                <CartesianGrid stroke="#e6e1d6" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-25} textAnchor="end" height={62} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} unit="%" />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {chart.map((c) => <Cell key={c.key} fill={AGENT_COLORS[c.key] || "#999"} />)}
                  <ErrorBar dataKey="err" width={4} stroke="#1d1c1a" strokeWidth={1.2} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <Ablations summary={summary} />
      </div>

      <Breakdown breakdowns={breakdowns} summary={summary} />
    </>
  );
}

function Ablations({ summary }) {
  const a = summary.agents;
  const m = (k, split = "overall", metric = "success_rate") => (split === "overall" ? a[k]?.overall?.[metric] : a[k]?.splits?.[split]?.[metric])?.mean;
  const items = [
    ["No guardrail (DQN)", "dqn", "dqn_nomask"], ["No guardrail (PPO)", "ppo", "ppo_nomask"], ["No guardrail (random)", "random", "random_no_guardrail"],
    ["Outcome-only reward (DQN)", "dqn", "dqn_sparse"], ["Outcome-only reward (PPO)", "ppo", "ppo_sparse"],
    ["CQL without conservative penalty", "cql", "cql_alpha0"], ["PPO with a small untuned budget", "ppo", "ppo_default"],
    ["Train with tool-failure scenarios (DQN), tool-failure test set", "dqn", "dqn_robust", "tool_failures"],
    ["Train with tool-failure scenarios (PPO), tool-failure test set", "ppo", "ppo_robust", "tool_failures"],
  ].filter(([, base, v]) => a[base] && a[v]);
  const tone = (d) => (d > 0.01 ? "best" : d < -0.01 ? "worst" : "");
  const g = (k, metric) => m(k, "overall", metric);
  return (
    <div className="card">
      <div className="card-head"><h3>What each design choice changes</h3></div>
      <table className="t">
        <thead><tr><th>Change</th><th className="r">Before</th><th className="r">After</th><th className="r">Rule breaks after</th></tr></thead>
        <tbody>{items.map(([label, base, v, split = "overall"]) => (
          <tr key={v}><td>{label}</td><td className="r num">{fmt.pct(m(base, split))}</td>
            <td className={`r num ${tone(m(v, split) - m(base, split))}`}>{fmt.pct(m(v, split))}</td>
            <td className="r num">{fmt.pct(g(v, "violation_rate"))}</td></tr>))}</tbody>
      </table>
      <p className="muted small" style={{ marginTop: 8 }}>
        {a.random_no_guardrail && a.dqn_nomask && a.ppo_nomask && (<>
          <b>Guardrail:</b> without it a random agent breaks a rule in {fmt.pct(g("random_no_guardrail", "violation_rate"), 0)} of runs. The learning agents mostly learn to
          avoid unsafe actions (DQN {fmt.pct(g("dqn_nomask", "violation_rate"))}, PPO {fmt.pct(g("ppo_nomask", "violation_rate"))} rule breaks) but can no longer learn the task:
          success falls to {fmt.pct(g("dqn_nomask"), 0)} and {fmt.pct(g("ppo_nomask"), 0)}, and PPO ends most conversations after {g("ppo_nomask", "avg_steps")?.toFixed(1)} step.
          The mask is what makes the task learnable and the safety a guarantee instead of a hope.{" "}</>)}
        {a.dqn_sparse && a.ppo_sparse && (<><b>Outcome-only reward</b> (no bonus when a goal is handled) changed little once PPO had enough training: DQN {fmt.pct(g("dqn"))} to {fmt.pct(g("dqn_sparse"))}, PPO {fmt.pct(g("ppo"))} to {fmt.pct(g("ppo_sparse"))}.</>)}
      </p>
    </div>
  );
}

function Breakdown({ breakdowns, summary }) {
  if (!breakdowns) return null;
  const keys = Object.keys(breakdowns).filter((k) => summary.agents[k]?.kind !== "ablation");
  const types = [...new Set(keys.flatMap((k) => Object.keys(breakdowns[k].by_type)))];
  const combos = [...new Set(keys.flatMap((k) => Object.keys(breakdowns[k].by_combo)))];
  const cell = (v) => <td className="c" style={{ background: v == null ? "#fff" : `rgba(${v > 0.9 ? "31,111,107" : v > 0.7 ? "168,112,15" : "179,38,30"},${0.08 + 0.3 * Math.abs(v - 0.5)})` }}>{v == null ? "-" : (100 * v).toFixed(0) + "%"}</td>;
  return (
    <div className="card" style={{ marginTop: 14 }}>
      <div className="card-head"><h3>Never trust only the overall number: success by scenario type and goal combination <Tip text="Pooled over all test sets and seeds. Goal combinations come from the iid and held-out sets; held-out combinations were never seen in training." /></h3></div>
      <div style={{ overflowX: "auto" }}>
        <table className="t heat">
          <thead><tr><th>Agent</th>{types.map((t) => <th key={t} className="r">{t.replace("_", " ")}</th>)}{combos.map((c) => <th key={c} className="r" title={c}>{c.split("+").map((x) => x.split("_")[0]).join("+")}</th>)}</tr></thead>
          <tbody>{keys.map((k) => (
            <tr key={k}><td>{summary.agents[k].label}</td>
              {types.map((t) => cell(breakdowns[k].by_type[t]?.success_rate))}{combos.map((c) => cell(breakdowns[k].by_combo[c]?.success_rate))}</tr>))}</tbody>
        </table>
      </div>
    </div>
  );
}
