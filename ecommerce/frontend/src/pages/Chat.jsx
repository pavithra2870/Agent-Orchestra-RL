import React, { useEffect, useRef, useState } from "react";
import { api, streamChat } from "../api.js";
import { ErrorBox, PageHead, Pill, Tip } from "../components/ui.jsx";
import { RunView } from "../components/RunView.jsx";

const SPEEDS = { Instant: 0, Normal: 400, Slow: 900 };

export default function Chat({ health }) {
  const [customers, setCustomers] = useState([]);
  const [examples, setExamples] = useState([]);
  const [agents, setAgents] = useState([]);
  const [form, setForm] = useState({ customer_id: "", message: "", agent: "dqn", language_mode: "auto", speed: "Normal" });
  const [events, setEvents] = useState([]);
  const [convo, setConvo] = useState([]);
  const [running, setRunning] = useState(false);
  const [context, setContext] = useState(null);
  const [error, setError] = useState("");
  const [changed, setChanged] = useState(0);
  const [waiting, setWaiting] = useState([]);
  const [notice, setNotice] = useState("");
  const [claims, setClaims] = useState([]);
  const stop = useRef(null);

  const refreshData = () => Promise.all([api.customers(), api.demoChanges(), api.demoReturns(), api.demoClaims()])
    .then(([c, d, r, cl]) => { setCustomers(c); setChanged(d.changed_orders.length); setWaiting(r.waiting); setClaims(cl.claims); }).catch(() => {});
  async function inspect(order_id, result) {
    try { const r = await api.inspect(order_id, result); setNotice(`${order_id}: ${r.message}`); await refreshData(); } catch (e) { setError(e.message); }
  }

  useEffect(() => {
    refreshData();
    Promise.all([api.customers(), api.examples(), api.agents()]).then(([c, e, a]) => {
      setCustomers(c); setExamples(e); setAgents(a);
      setForm((f) => ({ ...f, customer_id: e[0]?.customer_id || c[0]?.customer_id || "", message: e[0]?.message || "" }));
    }).catch((e) => setError(e.message));
    return () => stop.current && stop.current();
  }, []);

  const customer = customers.find((c) => c.customer_id === form.customer_id);
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  function send(override) {
    const text = typeof override === "string" ? override : form.message;
    if (!text.trim() || running) return;
    setError(""); setEvents([]); setRunning(true);
    const history = convo.slice(-3).map((m) => ({ role: m.role === "u" ? "user" : "agent", text: m.text }));
    setConvo((c) => [...c, { role: "u", text }]);
    const sent = text;
    stop.current = streamChat(
      { message: sent, customer_id: form.customer_id, agent: form.agent, language_mode: "auto", delay_ms: SPEEDS[form.speed], context, history },
      (ev) => {
        setEvents((p) => [...p, ev]);
        if (ev.type === "reply") setConvo((c) => [...c, { role: "a", text: ev.text, meta: ev.mode, options: ev.options || [] }]);
        if (ev.type === "error") setError(ev.message);
      },
      (end) => { refreshData(); setRunning(false); setContext(end.type === "done" && end.terminal === "awaiting_customer" ? end.context : null); setForm((f) => ({ ...f, message: "" })); },
      (msg) => { setRunning(false); setError(msg); },
    );
  }

  async function restore() {
    try { await api.restoreDemo(); await refreshData(); reset(); }
    catch (e) { setError(e.message); }
  }
  function reset() { if (stop.current) stop.current(); setEvents([]); setConvo([]); setContext(null); setRunning(false); setError(""); }
  const useExample = (ex) => { reset(); setForm((f) => ({ ...f, customer_id: ex.customer_id, message: ex.message })); };
  const agentInfo = agents.find((a) => a.key === form.agent);

  return (
    <>
      <PageHead title="Live conversation">
        Type a customer message and watch the agent work: the message is understood, the trained policy chooses each tool, the tool runs on a
        simulated database, and an independent check judges the result. Everything runs on fake data.
      </PageHead>
      <ErrorBox error={error} />

      <div className="card">
        <div className="card-head"><h3>Customer and message</h3>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {context && <Pill tone="warn">the agent asked a question, answer it below</Pill>}
            <span className="muted small">{changed ? `${changed} order${changed > 1 ? "s" : ""} changed by earlier chats` : "demo data is original"}</span>
            <button className="btn ghost" onClick={restore} disabled={running || !changed} title="Put every order back to its original values (status, payment method, payment state, refund id)">Restore original data</button>
          </div></div>
        <div className="grid g-top">
          <div>
            <label className="f"><span>Customer (logged in)</span>
              <select value={form.customer_id} onChange={(e) => { set("customer_id", e.target.value); reset(); }}>
                {customers.map((c) => <option key={c.customer_id} value={c.customer_id}>{c.name} · {c.customer_id}</option>)}
              </select>
            </label>
            {customer && (
              <table className="t">
                <thead><tr><th>Order</th><th>Status</th><th>Payment</th></tr></thead>
                <tbody>{customer.orders.map((o) => (
                  <tr key={o.order_id}><td className="mono">{o.order_id}<div className="muted small">{o.items.join(", ")}</div></td>
                    <td><Pill tone={o.status === "delayed" || o.status === "lost" ? "warn" : ""}>{o.status}</Pill>{o.return_state && <div className="muted small">return: {o.return_state.replace("_", " ")}</div>}</td>
                    <td className="small">{o.payment_method}<div className="muted">{o.payment_state}</div></td></tr>))}</tbody>
              </table>
            )}
          </div>

          <div>
            <label className="f"><span>{context ? "Your answer to the agent's question" : "Customer message"}</span>
              <textarea value={form.message} placeholder={context ? "For example: ORD-1234, or UPI" : "Type a request, for example: where is my order and can I change the payment to UPI?"}
                onChange={(e) => set("message", e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) send(); }} />
            </label>
            <div className="muted small" style={{ marginBottom: 6 }}>Examples</div>
            <div className="chips">
              {examples.map((ex) => <button key={ex.label} className="chip" onClick={() => useExample(ex)} title={ex.message}>{ex.label}</button>)}
            </div>
          </div>

          <div>
            <label className="f"><span>Decision-making agent <Tip text="Which trained policy chooses the tools. The language parts (reading and writing) are the same for all of them." /></span>
              <select value={form.agent} onChange={(e) => set("agent", e.target.value)}>
                {agents.map((a) => <option key={a.key} value={a.key} disabled={!a.available}>{a.label}{a.available ? "" : " (not trained)"}</option>)}
              </select>
              {agentInfo && <div className="muted small" style={{ marginTop: 4 }}>{agentInfo.about}</div>}
            </label>
            <div className="grid g2" style={{ gap: 10 }}>
              <div className="f"><span>Language <Tip text="The local model (phi4-mini) reads the message and words the reply when it is running. Its output is checked and replaced by a fixed template if it adds anything unverified. If the model is offline, templates are used." /></span>
                <div className="muted small">{health?.llm_available ? "LLM with template fallback" : "Templates (LLM offline)"}</div>
              </div>
              <label className="f"><span>Animation</span>
                <div className="seg">{Object.keys(SPEEDS).map((s) => <button key={s} className={form.speed === s ? "on" : ""} onClick={() => set("speed", s)}>{s}</button>)}</div>
              </label>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn accent" onClick={() => send()} disabled={running || !form.message.trim()}>{running ? "Working..." : context ? "Send answer" : "Send"}</button>
              <button className="btn ghost" onClick={reset}>New conversation</button>
            </div>
            <p className="muted small" style={{ marginTop: 8 }}>Ctrl+Enter sends. Changes the agent makes are kept in the demo database until you restore the original data.</p>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="card-head"><h3>Conversation</h3></div>
        {convo.length === 0 && <div className="muted">Pick an example or write your own message above, then press Send.</div>}
        {convo.map((m, i) => (
          <div key={i} className={`bubble ${m.role} rise`}><small>{m.role === "u" ? (customer?.name || "Customer") : `Support agent${m.meta === "llm" ? " (LLM wording)" : m.meta === "knowledge base" ? " (knowledge base)" : ""}`}</small>{m.text}
            {m.options?.length > 0 && i === convo.length - 1 && !running && context && (
              <div className="chips" style={{ marginTop: 8 }}>{m.options.map((o) => <button key={o.label} className="chip" onClick={() => send(o.text)}>{o.label}</button>)}</div>
            )}</div>
        ))}
      </div>

      {events.length > 0 && <div style={{ marginTop: 18 }}><h2 style={{ marginBottom: 8 }}>How the agent handled it</h2><RunView events={events} running={running} /></div>}
      <details key={waiting.length} open={waiting.length > 0} className="small" style={{ marginTop: 28, opacity: 0.9 }}>
        <summary className="muted" style={{ cursor: "pointer" }}>Staff side (demo): returns waiting for inspection{waiting.length ? ` (${waiting.length})` : ""} and damage claims</summary>
        <div style={{ marginTop: 8 }}>
          <div className="muted" style={{ marginBottom: 4 }}>After the agent opens a return, a human team inspects the item. These buttons play that team: passed releases the refund or replacement, failed means no refund and the item goes back.</div>
          {notice && <div className="note" style={{ marginBottom: 6 }}>{notice}</div>}
          {waiting.length === 0 ? <div className="muted">No returns waiting.</div> : waiting.map((r) => (
            <div key={r.order_id} className="diff" style={{ flexWrap: "wrap", gap: 8 }}>
              <span className="mono">{r.order_id}</span><span>{r.customer} · {r.items.join(", ")} · {r.kind === "exchange" ? "exchange" : "return and refund"} ({r.reason === "damaged" ? "damaged" : "changed mind"})</span>
              <button className="btn" onClick={() => inspect(r.order_id, "passed")}>Inspection passed</button>
              <button className="btn ghost" onClick={() => inspect(r.order_id, "failed")}>Inspection failed</button>
            </div>))}
          <div className="muted" style={{ margin: "8px 0 2px" }}>Latest damage claims (staff review instant refunds afterwards)</div>
          {claims.length === 0 ? <div className="muted">No claims.</div> : (
            <table className="t"><tbody>{claims.slice(-4).map((c, i) => <tr key={i}><td>{c.customer}</td><td>{c.date}</td><td className="r num">{c.amount} rupees</td><td>{c.mode === "instant_refund" ? "instant refund" : c.mode}</td></tr>)}</tbody></table>)}
        </div>
      </details>
    </>
  );
}
