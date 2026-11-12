const J = { "Content-Type": "application/json" };

async function req(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { const b = await r.json(); msg = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail ?? b); } catch { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  health: () => req("/api/health"),
  agents: () => req("/api/agents"),
  help: () => req("/api/metrics/help"),
  customers: () => req("/api/customers"),
  examples: () => req("/api/examples"),
  demoChanges: () => req("/api/demo/changes"),
  restoreDemo: () => req("/api/demo/restore", { method: "POST" }),
  demoReturns: () => req("/api/demo/returns"),
  demoClaims: () => req("/api/demo/claims"),
  inspect: (order_id, result) => req("/api/demo/inspect", { method: "POST", headers: J, body: JSON.stringify({ order_id, result }) }),
  summary: () => req("/api/results/summary"),
  breakdowns: () => req("/api/results/breakdowns"),
  training: () => req("/api/results/training"),
  language: () => req("/api/results/language"),
  llmBenchmark: () => req("/api/results/llm_benchmark"),
  failures: (p) => req("/api/results/failures?" + new URLSearchParams(Object.entries(p).filter(([, v]) => v !== "" && v != null))),
  replay: (body) => req("/api/replay", { method: "POST", headers: J, body: JSON.stringify(body) }),
  runs: (p) => req("/api/runs?" + new URLSearchParams(Object.entries(p).filter(([, v]) => v !== "" && v != null))),
  run: (id) => req(`/api/runs/${id}`),
  jobs: () => req("/api/jobs"),
  job: (id) => req(`/api/jobs/${id}`),
  train: (body) => req("/api/jobs/train", { method: "POST", headers: J, body: JSON.stringify(body) }),
  evaluate: (body) => req("/api/jobs/evaluate", { method: "POST", headers: J, body: JSON.stringify(body) }),
  cancel: (id) => req(`/api/jobs/${id}/cancel`, { method: "POST" }),
};

/** Stream one conversation as server-sent events. Returns a function that stops it. */
export function streamChat(params, onEvent, onEnd, onError) {
  const q = new URLSearchParams({ ...params, context: params.context ? JSON.stringify(params.context) : "",
    history: params.history && params.history.length ? JSON.stringify(params.history) : "" });
  if (!params.context) q.delete("context");
  if (!q.get("history")) q.delete("history");
  const es = new EventSource("/api/chat/stream?" + q.toString());
  let finished = false;
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    onEvent(ev);
    if (ev.type === "done" || ev.type === "error") { finished = true; es.close(); onEnd(ev); }
  };
  es.onerror = () => { es.close(); if (!finished) onError("The connection to the API was lost. Is the backend running on port 8000?"); };
  return () => es.close();
}

export const AGENT_COLORS = {
  rule_based: "#57534e", random: "#a8a29e", random_no_guardrail: "#c9c4ba", q_learning: "#8b6f47", sarsa: "#c08a2b",
  dqn: "#1f6f6b", ppo: "#c8502a", cql: "#34568b", dqn_sparse: "#1f6f6b", ppo_sparse: "#c8502a", dqn_nomask: "#1f6f6b",
  ppo_nomask: "#c8502a", cql_alpha0: "#34568b", ppo_default: "#c8502a", dqn_robust: "#1f6f6b", ppo_robust: "#c8502a",
};

export const fmt = {
  pct: (x, d = 1) => (x == null || Number.isNaN(x) ? "-" : (100 * x).toFixed(d) + "%"),
  num: (x, d = 2) => (x == null || Number.isNaN(x) ? "-" : Number(x).toFixed(d)),
  pm: (m, d = 1, pct = true) => (m == null ? "-" : pct ? `${(100 * m.mean).toFixed(d)}% ±${(100 * m.std).toFixed(d)}` : `${m.mean.toFixed(d)} ±${m.std.toFixed(d)}`),
};

export const SHORT_NAMES = {
  rule_based: "Rule-based", random: "Random", dqn: "DQN", ppo: "PPO", cql: "CQL", q_learning: "Q-learning", sarsa: "SARSA",
  dqn_robust: "DQN + failure training", ppo_robust: "PPO + failure training",
};
