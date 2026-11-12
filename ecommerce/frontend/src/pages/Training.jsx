import React, { useEffect, useMemo, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AGENT_COLORS, api, fmt } from "../api.js";
import { Empty, ErrorBox, PageHead, Stat, Tip } from "../components/ui.jsx";

const MAIN = ["q_learning", "sarsa", "dqn", "ppo"];
const ABL = ["dqn_sparse", "ppo_sparse", "dqn_nomask", "ppo_nomask", "ppo_default"];

function Curves({ curves, keys, xLabel, bands }) {
  const data = useMemo(() => {
    const xs = new Set();
    keys.forEach((k) => curves[k]?.x.forEach((x) => xs.add(x)));
    return [...xs].sort((a, b) => a - b).map((x) => {
      const row = { x };
      keys.forEach((k) => {
        const c = curves[k]; const i = c?.x.indexOf(x);
        if (c && i >= 0) { row[k] = +(100 * c.mean[i]).toFixed(1); row[k + "_b"] = [Math.max(0, 100 * (c.mean[i] - c.std[i])), Math.min(100, 100 * (c.mean[i] + c.std[i]))]; }
      });
      return row;
    });
  }, [curves, keys]);
  return (
    <div style={{ height: 320 }}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 4, right: 16, bottom: 22, left: 0 }}>
          <CartesianGrid stroke="#e6e1d6" />
          <XAxis dataKey="x" type="number" domain={["dataMin", "dataMax"]} tick={{ fontSize: 11 }} label={{ value: xLabel, position: "insideBottom", offset: -8, fontSize: 12 }} />
          <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
          <Tooltip formatter={(v, n) => (Array.isArray(v) ? null : [`${v}%`, curves[n]?.label || n])} />
          <Legend verticalAlign="top" height={28} formatter={(v) => curves[v]?.label || v} wrapperStyle={{ fontSize: 12 }} />
          {bands && keys.map((k) => <Area key={k + "b"} dataKey={k + "_b"} stroke="none" fill={AGENT_COLORS[k]} fillOpacity={0.12} legendType="none" tooltipType="none" connectNulls />)}
          {keys.map((k) => <Line key={k} dataKey={k} stroke={AGENT_COLORS[k]} strokeWidth={2} dot={false} connectNulls strokeDasharray={ABL.includes(k) ? "5 4" : undefined} />)}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Training() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [showAbl, setShowAbl] = useState(false);
  useEffect(() => { api.training().then(setData).catch((e) => setError(e.message)); }, []);
  if (error) return <><PageHead title="Training" /><ErrorBox error={error} /><Empty>No training results yet.</Empty></>;
  if (!data) return <PageHead title="Training">Loading...</PageHead>;
  const { summary, curves, ppo_tuning: tuning, logged_dataset: logged } = data;
  const onlineKeys = [...MAIN, ...(showAbl ? ABL : [])].filter((k) => curves[k]);
  const rows = Object.entries(summary);
  return (
    <>
      <PageHead title="Training">
        How quickly each method learns. The curve is the success rate on a separate validation set (150 scenarios never used for training),
        averaged over {summary.dqn?.seeds || 3} training seeds, with the spread shaded.
      </PageHead>
      <div className="grid g4" style={{ marginBottom: 14 }}>
        <Stat label="Fastest to 90%" k="episodes_to_90" value={(() => { const r = rows.filter(([, s]) => s.kind === "online" && s.to_90pct.mean).sort((a, b) => a[1].to_90pct.mean - b[1].to_90pct.mean)[0]; return r ? r[1].label : "-"; })()} sub="online methods, by episodes" />
        <Stat label="CQL live episodes" value="0" sub={logged ? `learned from a log of ${logged.episodes} past conversations` : "offline"} />
        <Stat label="Log quality" value={logged ? fmt.pct(logged.behaviour_success_rate, 0) : "-"} sub="the logged behaviour policy's own success" />
        <Stat label="Seeds per method" value={summary.dqn?.seeds || "-"} sub="independent training runs" />
      </div>

      <div className="card">
        <div className="card-head"><h3>Learning curves: online methods</h3>
          <label className="small"><input type="checkbox" checked={showAbl} onChange={(e) => setShowAbl(e.target.checked)} /> show ablations (dashed)</label></div>
        <Curves curves={curves} keys={onlineKeys} xLabel="training episodes" bands />
        <p className="muted small">An episode is one customer conversation. Q-learning and SARSA need thousands of conversations; DQN learns in the hundreds.</p>
      </div>

      <div className="grid g2" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="card-head"><h3>CQL: offline learning <Tip text="CQL never talks to the simulator while learning. It trains on a fixed log of past conversations (a rule-based agent that sometimes makes random moves, plus some fully random runs) and is only tested live afterwards." /></h3></div>
          {curves.cql ? <Curves curves={curves} keys={["cql", "cql_alpha0"].filter((k) => curves[k])} xLabel="gradient steps (no live data used)" bands /> : <Empty>CQL has not been trained.</Empty>}
          <p className="muted small">The dashed line removes CQL's conservative penalty. Compare how stable each is.</p>
        </div>
        <div className="card">
          <div className="card-head"><h3>PPO tuning (validation split only)</h3></div>
          {tuning ? (
            <table className="t"><thead><tr><th>Setting</th><th className="r">Best val</th><th className="r">Episodes to 90%</th></tr></thead>
              <tbody>{tuning.grid.map((g) => (<tr key={g.name} style={tuning.chosen.name === g.name ? { background: "#f4f0e6" } : null}>
                <td className="mono small">{g.name}{tuning.chosen.name === g.name ? "  ← chosen" : ""}</td><td className="r num">{fmt.pct(g.best_val_success)}</td><td className="r num">{g.episodes_to_90pct_val ?? "never"}</td></tr>))}</tbody></table>
          ) : <Empty>No tuning run found.</Empty>}
          <p className="muted small" style={{ marginTop: 8 }}>The first version of PPO looked weak (72.7% validation success) only because its budget was too small. With 12,000 episodes every setting reaches 100%; the chosen one is simply fastest.</p>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="card-head"><h3>Sample efficiency and final scores</h3></div>
        <table className="t">
          <thead><tr><th>Method</th><th className="r">Budget</th><th className="r">To reach 90% <Tip k="episodes_to_90" /></th><th className="r">Best validation</th><th className="r">Final validation</th><th className="r">Train time</th></tr></thead>
          <tbody>{rows.map(([k, s]) => (
            <tr key={k}><td><span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: AGENT_COLORS[k] || "#999", marginRight: 8 }} />{s.label}</td>
              <td className="r num">{s.total_budget.toLocaleString()} {s.unit}</td>
              <td className="r num">{s.to_90pct.mean ? `${Math.round(s.to_90pct.mean).toLocaleString()} (${s.to_90pct.reached_in})` : `never (${s.to_90pct.reached_in})`}</td>
              <td className="r num">{fmt.pm(s.best_val_success)}</td><td className="r num">{fmt.pm(s.final_val_success)}</td>
              <td className="r num">{s.train_seconds.mean.toFixed(0)} s</td></tr>))}</tbody>
        </table>
        <p className="muted small" style={{ marginTop: 8 }}>The checkpoint with the best validation score is the one evaluated on the test sets. "Final" shows whether training stayed stable.</p>
      </div>
    </>
  );
}
