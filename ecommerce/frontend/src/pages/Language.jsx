import React, { useEffect, useState } from "react";
import { AGENT_COLORS, api, fmt } from "../api.js";
import { Empty, ErrorBox, PageHead, Pill, Stat, Tip } from "../components/ui.jsx";

const LAYERS = [
  ["1 · Input checks", "Is the message about something we handle? Is the order clear? Is the parser sure?", "Embedding router (similarity to example requests), scope score, calibrated confidence. Low confidence makes the agent ask first."],
  ["2 · Action rules (in code)", "May this action happen now?", "Action masking: a state-changing tool needs a verified order, a passed eligibility check, and a customer request. Retries are capped, then a human takes over."],
  ["3 · Output checks", "Is the reply true?", "The reply is built only from verified tool results. An LLM draft is checked for invented dates, numbers, ids and false 'done' claims, and replaced by the template if anything is unsupported."],
  ["4 · Escalation", "When must a human decide?", "Out-of-scope requests, large refunds, repeated tool failures, and low confidence that a question cannot fix."],
];

export default function Language() {
  const [data, setData] = useState(null);
  const [bench, setBench] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => { api.language().then(setData).catch((e) => setError(e.message)); api.llmBenchmark().then(setBench).catch(() => {}); }, []);

  const r = data?.router;
  const A = r?.examples_A_only;
  const rows = A && [
    ["Wording the router has seen (set A)", A.seen_wording_A], ["Tuning wording (set C)", A.tuning_wording_C],
    ["Held-out wording (set B), final test", A.held_out_wording_B], ["Held-out wording (set B), with more example sentences", r.examples_A_plus_C.held_out_wording_B],
  ];
  const llm = r?.llm_parser;
  const pipe = data?.pipeline;
  const kn = data?.knowledge;

  return (
    <>
      <PageHead title="Language and guardrails">
        The RL agent never reads text. A language layer turns the message into goals first, and checks what is said back. These numbers show how reliable
        that layer is, and how much its mistakes cost the agents.
      </PageHead>
      <ErrorBox error={error} />
      <div className="grid g2" style={{ marginBottom: 14 }}>
        {LAYERS.map(([t, q, d]) => (<div className="card" key={t}><h3>{t}</h3><p className="muted" style={{ margin: "2px 0 6px" }}>{q}</p><p className="small" style={{ margin: 0 }}>{d}</p></div>))}
      </div>

      {rows ? (
        <div className="card">
          <div className="card-head"><h3>Router: does it find the right goals? <Tip text="Three separate sets of example wordings are used. Set A is what the router compares against. Set C only tunes the threshold and confidence. Set B is the final test and is never used for anything else." /></h3><Pill>{r.embedder}</Pill></div>
          <table className="t">
            <thead><tr><th>Messages</th><th className="r">Goal set exactly right <Tip k="exact_match" /></th><th className="r">Out-of-scope caught <Tip k="oos_recall" /></th><th className="r">Mean similarity to examples</th><th className="r">Low confidence on errors</th></tr></thead>
            <tbody>{rows.map(([l, x], i) => (<tr key={l} style={i === 2 ? { background: "#f4f0e6" } : null}><td>{l}</td><td className="r num">{fmt.pct(x.exact_match)}</td><td className="r num">{fmt.pct(x.oos_recall, 0)}</td>
              <td className="r num">{fmt.num(x.mean_best_similarity)}</td><td className="r num">{x.low_conf_catches_errors == null ? "-" : fmt.pct(x.low_conf_catches_errors, 0)}</td></tr>))}</tbody>
          </table>
          <p className="muted small" style={{ marginTop: 8 }}>
            Unseen wording is harder: similarity to the examples drops and accuracy falls. The confidence score flags only part of the wrong parses,
            which is why a second reader (the LLM) and the agent's own "ask first" behaviour matter.
          </p>
          {r.errors_on_B?.length > 0 && (
            <details><summary className="small muted" style={{ cursor: "pointer" }}>Examples of router mistakes on held-out wording</summary>
              <table className="t"><thead><tr><th>Message</th><th>Truth</th><th>Router said</th></tr></thead>
                <tbody>{r.errors_on_B.slice(0, 10).map((e, i) => (<tr key={i}><td className="small">{e.text}</td><td className="small">{e.truth.join(", ") || "none"}</td><td className="small">{e.predicted.join(", ") || "none"}</td></tr>))}</tbody></table></details>
          )}
        </div>
      ) : !error && <Empty>No language evaluation yet.</Empty>}

      {llm?.available && (
        <div className="card" style={{ marginTop: 14 }}>
          <div className="card-head"><h3>Reading messages: router vs LLM vs both <Tip text="Same held-out messages (set B). 'Invented goal' means the parser reported a goal the customer did not ask for, which could trigger an unrequested action." /></h3><Pill>{llm.model} · {llm.messages} messages</Pill></div>
          <table className="t">
            <thead><tr><th>Method</th><th className="r">Exactly right</th><th className="r">Invented a goal</th><th className="r">Missed a goal</th><th className="r">Low confidence on errors</th></tr></thead>
            <tbody>{[["Router only", llm.router_only], ["LLM only (raw)", llm.llm_only_raw], ["LLM + router cross-check", llm.llm_plus_router]].map(([l, x]) => (
              <tr key={l}><td>{l}</td><td className="r num">{fmt.pct(x.exact_match)}</td><td className="r num">{fmt.pct(x.invented_goal_rate)}</td><td className="r num">{fmt.pct(x.missed_goal_rate)}</td>
                <td className="r num">{x.low_conf_catches_errors == null ? "-" : fmt.pct(x.low_conf_catches_errors, 0)}</td></tr>))}</tbody>
          </table>
          <p className="muted small" style={{ marginTop: 8 }}>Average {fmt.num(llm.avg_llm_seconds, 1)} s per message for the LLM on this machine. {llm.goals_dropped_by_guard} goal(s) were dropped by the cross-check because the router found no support for them.</p>
        </div>
      )}

      {kn && (
        <div className="card" style={{ marginTop: 14 }}>
          <div className="card-head"><h3>Policy questions: finding the topic <Tip text="A policy question is a goal like any other. The router recognises it, a topic is found by comparing the question with example questions, and the RL agent learns to call the lookup_policy tool (no order needed, nothing changes). Held-out questions are phrase set B." /></h3>
            <Pill>{kn.topics.length} topics</Pill></div>
          <table className="t">
            <thead><tr><th>Held-out test (set B)</th><th className="r">Count</th><th className="r">Result</th></tr></thead>
            <tbody>
              <tr><td>Router recognises a policy question</td><td className="r num">{kn.test.policy_questions}</td><td className="r num">{fmt.pct(kn.test.router_recognises_policy_question)}</td></tr>
              <tr><td>Right topic found</td><td className="r num">{kn.test.policy_questions}</td><td className="r num">{fmt.pct(kn.test.topic_accuracy)}</td></tr>
              <tr><td>Both right</td><td className="r num">{kn.test.policy_questions}</td><td className="r num">{fmt.pct(kn.test.both_right)}</td></tr>
              <tr><td>Ordinary requests wrongly called policy questions</td><td className="r num">{kn.test.ordinary_requests}</td><td className="r num">{fmt.pct(kn.test.ordinary_requests_called_policy)}</td></tr>
            </tbody>
          </table>
          <p className="muted small" style={{ marginTop: 8 }}>
            Topics: {kn.topics.map((t) => t.replace("_", " ")).join(", ")}. Answers use the same numbers as the tools (for example the 7-day return window).
            {kn.misses?.length ? ` Missed: "${kn.misses[0].text}".` : ""}
          </p>
        </div>
      )}

      {pipe && (
        <div className="card" style={{ marginTop: 14 }}>
          <div className="card-head"><h3>End to end: what do language mistakes cost the agents? <Tip text="Same agents, same held-out wording. 'Oracle parser' hands the agent the true goals. 'Router parser' uses the real parser output. The gap is the cost of understanding errors." /></h3>
            <Pill>router read the right goals in {fmt.pct(pipe.goal_set_exact_match)} of {pipe.messages} messages</Pill></div>
          <table className="t">
            <thead><tr><th>Agent</th><th className="r">Oracle parser</th><th className="r">Real router parser</th><th className="r">Cost of language errors</th><th className="r">Rule breaks</th><th>Main failure reasons</th></tr></thead>
            <tbody>{Object.entries(pipe.agents).map(([k, a]) => (
              <tr key={k}><td><span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: AGENT_COLORS[k], marginRight: 8 }} />{k}</td>
                <td className="r num">{fmt.pm(a.oracle_parser.success_rate)}</td><td className="r num">{fmt.pm(a.router_parser.success_rate)}</td>
                <td className="r num worst">−{(100 * a.cost_of_language_errors).toFixed(1)} pts</td><td className="r num">{fmt.pct(a.router_parser.violation_rate.mean)}</td>
                <td className="small">{Object.entries(a.failure_reasons_with_parser).slice(0, 3).map(([f, v]) => `${f.replaceAll("_", " ")} ${(100 * v).toFixed(0)}%`).join(", ")}</td></tr>))}</tbody>
          </table>
          <div className="grid g3" style={{ marginTop: 12 }}>
            <Stat label="Template replies flagged" value={`${pipe.reply_template_flagged.flagged_by_checker} / ${pipe.reply_template_flagged.replies}`} sub="faithfulness checker on templated replies" />
            {pipe.reply_llm && <Stat label="LLM drafts rejected" value={`${pipe.reply_llm.llm_drafts_flagged} / ${pipe.reply_llm.replies}`} sub="replaced by the verified template" />}
            {pipe.reply_llm && <Stat label="Unsupported claims shipped" value={`${pipe.reply_llm.shipped_flagged} / ${pipe.reply_llm.replies}`} sub={`${fmt.num(pipe.reply_llm.seconds_per_reply, 1)} s per LLM reply`} />}
          </div>
        </div>
      )}

      {bench?.available && (
        <div className="card" style={{ marginTop: 14 }}>
          <div className="card-head"><h3>Benchmark: an LLM choosing the tools instead <Tip text="phi4-mini is shown the same state and legal actions as the RL agents and picks the next action. It is a benchmark, not part of the product, and is run on a small sample because it is slow." /></h3><Pill>{bench.scenarios} scenarios · {bench.model}</Pill></div>
          <table className="t">
            <thead><tr><th>Agent</th><th className="r">Success</th><th className="r">Rule breaks</th><th className="r">Blocked attempts / run</th><th className="r">Regret</th><th className="r">Seconds per conversation</th></tr></thead>
            <tbody>
              <tr><td><b>LLM (phi4-mini)</b></td><td className="r num">{fmt.pct(bench.llm.success_rate)}</td><td className="r num">{fmt.pct(bench.llm.violation_rate)}</td><td className="r num">{fmt.num(bench.llm.blocked_per_episode)}</td><td className="r num">{fmt.num(bench.llm.regret)}</td><td className="r num">{fmt.num(bench.llm.wall_seconds / bench.scenarios, 1)}</td></tr>
              {Object.entries(bench.others).map(([k, o]) => (<tr key={k}><td>{k}</td><td className="r num">{fmt.pct(o.success_rate)}</td><td className="r num">{fmt.pct(o.violation_rate)}</td><td className="r num">{fmt.num(o.blocked_per_episode)}</td><td className="r num">{fmt.num(o.regret)}</td><td className="r num">&lt; 0.1</td></tr>))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
