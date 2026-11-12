import React from "react";
import { Pill } from "./ui.jsx";

const STAGES = [
  ["Message", "customer writes"], ["Understand", "router / parser"], ["State", "what is known"],
  ["Policy", "RL picks a tool"], ["Tool", "runs on the database"], ["Reply", "from verified facts"], ["Check", "independent evaluator"],
];
const STAGE_OF = { start: 0, parse: 1, state: 2, decision: 3, tool: 4, reply: 5, evaluation: 6, done: 7 };

export function Workflow({ events, running }) {
  const stages = STAGES;
  const stageOf = STAGE_OF;
  const last = events.length ? events[events.length - 1] : null;
  const idx = last ? stageOf[last.type] ?? 0 : -1;
  const finished = idx === stages.length;
  const steps = events.filter((e) => e.type === "decision").length;
  return (
    <div className="flow">
      {stages.map(([lb, sb], i) => {
        const done = idx > i || (finished && !running);
        const act = running && idx === i;
        return (
          <React.Fragment key={lb}>
            {i > 0 && <div className={`ln ${idx >= i ? "done" : ""}`} />}
            <div className={`st ${done ? "done" : ""} ${act ? "act" : ""}`}>
              <div className="nd">{done ? "✓" : i + 1}</div>
              <div className="lb">{lb}{(i === 3 || i === 4) && steps > 0 ? ` ×${steps}` : ""}</div>
              <div className="sb">{sb}</div>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );
}

const GOAL_LABEL = { delivery_status: "delivery status", payment_change: "payment change", refund: "refund", cancel_order: "cancel order", return_item: "return / exchange / damage", policy_info: "policy question" };
const tone = (s) => (s === "handled" ? "good" : s === "needs_human" ? "warn" : "");

export function StateCard({ state }) {
  if (!state) return null;
  const f = state.facts || {};
  return (
    <div className="card rise">
      <div className="card-head"><h3>What the agent knows</h3><span className="muted small">step {state.steps}/{state.max_steps}</span></div>
      <dl className="kv">
        <dt>Goals</dt>
        <dd className="chips">
          {Object.keys(state.goals).length === 0 && <span className="muted">none found</span>}
          {Object.entries(state.goals).map(([g, s]) => <Pill key={g} tone={tone(s)}>{GOAL_LABEL[g] || g}: {s.replace("_", " ")}</Pill>)}
        </dd>
        <dt>Verified</dt>
        <dd className="chips">
          <Pill tone={state.customer_verified ? "good" : ""}>customer {state.customer_verified ? "✓" : "–"}</Pill>
          <Pill tone={state.order_verified ? "good" : state.order_unclear ? "warn" : ""}>
            order {state.order_id || "?"} {state.order_verified ? "✓" : state.order_unclear ? "unclear" : "–"}
          </Pill>
        </dd>
        <dt>Facts</dt>
        <dd>
          {[["shipment", f.shipment_status], ["payment change", f.payment_change_check], ["refund", f.refund_check], ["cancel", f.cancel_check],
            ["return", f.return_check], ["policy", f.policy_looked_up ? "looked up" : null]]
            .filter(([, v]) => v).map(([k, v]) => <Pill key={k}>{k}: {v}</Pill>)}
          {!Object.values(f).some(Boolean) && <span className="muted">nothing retrieved yet</span>}
        </dd>
        <dt>Parser</dt><dd className="num">confidence {state.parse_conf} · scope {state.scope_score}</dd>
        <dt>Spent</dt><dd className="num">cost {state.cost} · latency {state.latency}</dd>
        {state.locked_tools.length > 0 && (<><dt>Locked</dt><dd><Pill tone="bad">{state.locked_tools.join(", ")} (retry limit)</Pill></dd></>)}
      </dl>
    </div>
  );
}

export function ParseCard({ parse }) {
  if (!parse) return null;
  return (
    <div className="card rise">
      <div className="card-head"><h3>1 · Understanding the message</h3><Pill tone={parse.parse_conf >= 0.6 ? "good" : "warn"}>confidence {Number(parse.parse_conf).toFixed(2)}</Pill></div>
      <dl className="kv">
        <dt>Goals found</dt>
        <dd className="chips">{parse.goals.length ? parse.goals.map((g) => <Pill key={g} tone="accent">{GOAL_LABEL[g] || g}</Pill>) : <span className="muted">none (out of scope?)</span>}</dd>
        <dt>Order id</dt><dd className="mono">{parse.order_id || "not mentioned"}</dd>
        {parse.resolution && (<><dt>Which order</dt><dd>{parse.resolution.order_id ? <><span className="mono">{parse.resolution.order_id}</span> <span className="muted small">({parse.resolution.how.replace("_", " ")}: {parse.resolution.reason}{parse.resolution.assumed ? ", assumed" : ""})</span></> : <span className="muted">{parse.resolution.how === "none_qualifies" ? parse.resolution.reason : parse.resolution.how === "ambiguous" ? `several fit: ${parse.resolution.candidates.join(", ")}` : "not decided"}</span>}</dd></>)}
        <dt>New payment method</dt><dd>{parse.target_method || "not mentioned"}</dd>
        <dt>Scope score</dt><dd className="num">{Number(parse.scope_score).toFixed(2)} <span className="muted small">(similarity to known requests; low means unrelated)</span></dd>
        <dt>Read by</dt><dd>{parse.source}{parse.llm_goals ? ` · LLM said ${JSON.stringify(parse.llm_goals)}` : ""}</dd>
      </dl>
      {parse.notes?.map((n, i) => <div key={i} className="note" style={{ marginTop: 8 }}>{n}</div>)}
      {parse.router?.clauses?.length > 0 && (
        <details style={{ marginTop: 8 }}><summary className="small muted" style={{ cursor: "pointer" }}>How each part of the message was matched</summary>
          <table className="t"><thead><tr><th>Part of the message</th><th>Closest request type</th><th className="r">Similarity</th></tr></thead>
            <tbody>{parse.router.clauses.map((c, i) => (<tr key={i}><td>"{c.text}"</td><td>{c.label}</td><td className="r num">{c.sim.toFixed(2)}</td></tr>))}</tbody></table>
        </details>
      )}
    </div>
  );
}

function ScoreBars({ scores, chosen }) {
  if (!scores?.top?.length) return <div className="muted small">This policy does not produce action scores.</div>;
  const vals = scores.top.map((t) => t.value);
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
  return (
    <div className="bars">
      <div className="muted small" style={{ marginBottom: 2 }}>{scores.kind === "probability" ? "Probability of each legal action" : "Estimated value (Q) of each legal action"}</div>
      {scores.top.map((t) => (
        <div key={t.action} className={`bar ${t.action === chosen ? "pick" : ""}`}>
          <span className="mono">{t.action}</span>
          <div className="tr"><div className="fi" style={{ width: `${scores.kind === "probability" ? t.value * 100 : 12 + 88 * ((t.value - lo) / span)}%` }} /></div>
          <span className="num">{t.value.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

function ToolResult({ t }) {
  if (t.blocked) return <div><Pill tone="bad">blocked by guardrail</Pill> <span className="small">{t.note}</span></div>;
  const keys = Object.entries(t.data || {}).filter(([k]) => !["tracking", "order_ids", "items", "orders", "answer"].includes(k));
  return (
    <div>
      {t.ok ? <Pill tone="good">ok</Pill> : <Pill tone="bad">{t.error}</Pill>}{" "}
      {t.mutated && <Pill tone="accent">changed the database</Pill>}{" "}
      {t.note && <span className="small muted">{t.note}</span>}
      {t.data?.answer && <div className="json" style={{ fontFamily: "var(--sans)", fontSize: 13, marginTop: 6 }}>{t.data.answer}</div>}
      {keys.length > 0 && <div className="small" style={{ marginTop: 4 }}>{keys.map(([k, v]) => <span key={k} style={{ marginRight: 12 }}><span className="muted">{k}:</span> {String(v)}</span>)}</div>}
      <div className="small muted num" style={{ marginTop: 2 }}>cost {t.cost?.toFixed(2)} · latency {t.latency?.toFixed(2)} · reward {t.reward}</div>
    </div>
  );
}

export function Timeline({ events }) {
  const steps = [];
  for (const e of events) {
    if (e.type === "decision") steps.push({ d: e });
    else if (e.type === "tool" && steps.length) steps[steps.length - 1].t = e;
  }
  if (!steps.length) return null;
  return (
    <div>
      <h3 style={{ margin: "4px 0 8px" }}>2 · Decisions, step by step</h3>
      {steps.map(({ d, t }) => (
        <div key={d.step} className={`step rise ${t?.blocked ? "blocked" : ""}`}>
          <div className="step-h">
            <span className="n">{d.step}</span><b className="mono">{d.action}</b>
            <span className="muted small">chosen by {d.policy} from {d.legal.length} legal actions</span>
          </div>
          <ScoreBars scores={d.scores} chosen={d.action} />
          {t && <ToolResult t={t} />}
        </div>
      ))}
    </div>
  );
}

const REASON = {
  step_limit: "ran out of steps", unnecessary_escalation: "handed over although it could finish", missing_escalation: "should have handed over to a human",
  premature_finish: "finished before every goal was handled", should_have_asked: "should have asked the customer", tool_failure_unrecovered: "a tool failure was not recovered",
};
const CATEGORY = { missed_goal: "missed a goal", missed_escalation: "a goal needed a human hand-over", violation: "broke a safety rule" };
export const reasonText = (r) => REASON[r] || CATEGORY[r] || (r.startsWith("missed_goal") ? `missed goal: ${r.split(":")[1]}` : r.startsWith("missed_escalation") ? `should have handed over (${r.split(":")[1]})` : r.startsWith("violation") ? `rule broken: ${r.split(":")[1]}` : r);

export function EvalCard({ ev, expected, judged }) {
  if (!ev) return null;
  const r = ev.result;
  return (
    <div className="card rise">
      <div className="card-head"><h3>4 · Independent check</h3><Pill tone={r.success ? "good" : "bad"}>{r.success ? "success" : "failed"}</Pill></div>
      <p className="small muted">Judged from the database and the trace, not from what the agent said. Compared with: {judged}.</p>
      <table className="t">
        <thead><tr><th>Goal</th><th>Should end as</th><th>Result</th></tr></thead>
        <tbody>
          {Object.entries(r.goal_results).map(([g, x]) => (
            <tr key={g}><td>{GOAL_LABEL[g] || g}</td><td>{x.expected}{x.reason && x.reason !== "info" ? <span className="muted small"> ({x.reason.replaceAll("_", " ")})</span> : null}</td>
              <td><Pill tone={x.passed ? "good" : "bad"}>{x.passed ? "pass" : "fail"}</Pill> <span className="small muted">{x.detail}</span></td></tr>
          ))}
          <tr><td>Ending</td><td>{r.expected_terminal}</td><td><Pill tone={r.terminal === r.expected_terminal ? "good" : "bad"}>{r.terminal}</Pill></td></tr>
        </tbody>
      </table>
      <div className="chips" style={{ marginTop: 8 }}>
        <Pill tone={r.violations.length ? "bad" : "good"}>{r.violations.length} safety violations</Pill>
        <Pill>{r.blocked_attempts} blocked attempts</Pill>
        <Pill>{r.clarifications_asked} questions asked{r.clarification_needed ? " (needed)" : ""}</Pill>
        {r.failure_reasons.map((f) => <Pill key={f} tone="bad">{reasonText(f)}</Pill>)}
      </div>
    </div>
  );
}

export function DbChanges({ changes }) {
  return (
    <div className="card rise">
      <div className="card-head"><h3>Database changes</h3></div>
      {(!changes || !changes.length) ? <div className="muted small">No changes were made to the customer's orders.</div> :
        changes.map((c, i) => (<div className="diff" key={i}><span className="mono">{c.order_id}</span><span>{c.field.replace("_", " ")}</span>
          <Pill>{String(c.before)}</Pill><span>→</span><Pill tone="accent">{String(c.after)}</Pill></div>))}
    </div>
  );
}

/** Everything about one run, built from its event list. */
export function RunView({ events, running }) {
  const parse = events.find((e) => e.type === "parse");
  const reply = events.find((e) => e.type === "reply");
  const evaluation = events.find((e) => e.type === "evaluation");
  const done = events.find((e) => e.type === "done");
  const states = events.filter((e) => e.type === "state");
  const state = states.length ? states[states.length - 1].state : null;
  return (
    <div>
      <Workflow events={events} running={running} />
      <div className="grid g2" style={{ alignItems: "start" }}>
        <div>
          <ParseCard parse={parse} />
          <div style={{ marginTop: 14 }}><Timeline events={events} /></div>
        </div>
        <div>
          <StateCard state={state} />
          {reply && (
            <div className="card rise" style={{ marginTop: 14 }}>
              <div className="card-head"><h3>3 · Reply to the customer</h3><Pill tone={reply.mode === "llm" ? "accent" : ""}>{reply.mode}</Pill></div>
              <div className="bubble a" style={{ maxWidth: "100%", margin: 0 }}>{reply.text}</div>
              {reply.problems?.length > 0 && (<div className="note" style={{ marginTop: 8 }}><b>The language model's draft was rejected</b> ({reply.problems.join("; ")}). The verified template was sent instead.</div>)}
            </div>
          )}
          {evaluation && <div style={{ marginTop: 14 }}><EvalCard ev={evaluation} judged={evaluation.judged_against} /></div>}
          {done && <div style={{ marginTop: 14 }}><DbChanges changes={done.database_changes} /></div>}
          {done && (
            <div className="grid g4" style={{ marginTop: 14 }}>
              <div className="stat"><div className="k">Steps</div><div className="v num">{done.steps}</div></div>
              <div className="stat"><div className="k">Cost</div><div className="v num">{done.cost}</div></div>
              <div className="stat"><div className="k">Latency</div><div className="v num">{done.latency}</div></div>
              <div className="stat"><div className="k">Reward</div><div className="v num">{done.reward}</div></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
