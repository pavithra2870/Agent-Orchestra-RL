import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import { HelpContext } from "./components/ui.jsx";
import Chat from "./pages/Chat.jsx";
import Compare from "./pages/Compare.jsx";
import Failures from "./pages/Failures.jsx";
import History from "./pages/History.jsx";
import Language from "./pages/Language.jsx";
import RunSetup from "./pages/RunSetup.jsx";
import Training from "./pages/Training.jsx";

const PAGES = [
  ["chat", "Live conversation", "watch an agent work", Chat],
  ["compare", "Compare agents", "tables and charts", Compare],
  ["training", "Training", "learning curves", Training],
  ["failures", "Failures", "why and replay", Failures],
  ["language", "Language & guardrails", "reading and checking", Language],
  ["history", "Run history", "search and replay", History],
  ["setup", "Run setup", "train, tune, evaluate", RunSetup],
];

export default function App() {
  const initial = (window.location.hash || "#chat").slice(1);
  const [page, setPage] = useState(PAGES.some((p) => p[0] === initial) ? initial : "chat");
  const [health, setHealth] = useState(null);
  const [help, setHelp] = useState({});
  const [showHelp, setShowHelp] = useState(false);

  useEffect(() => {
    const check = () => api.health().then(setHealth).catch(() => setHealth({ ok: false }));
    check();
    api.help().then(setHelp).catch(() => {});
    const t = setInterval(check, 20000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => { window.location.hash = page; }, [page]);

  const Page = PAGES.find((p) => p[0] === page)[3];
  return (
    <HelpContext.Provider value={help}>
      <div className="shell">
        <aside className="rail">
          <div className="brand"><b>Support Agent Lab</b><span>RL tool-orchestration case study</span></div>
          <nav className="nav">
            {PAGES.map(([k, label, sub]) => (<button key={k} className={page === k ? "on" : ""} onClick={() => setPage(k)}>{label}<small>{sub}</small></button>))}
          </nav>
          <div className="rail-foot">
            <p><span className={`dot ${health == null ? "" : health.ok ? "ok" : "off"}`} />API {health == null ? "checking..." : health.ok ? "connected" : "not reachable"}</p>
            <p><span className={`dot ${health?.llm_available ? "ok" : ""}`} />LLM {health?.llm_available ? health.llm_model : "offline"}</p>
            <button className="btn ghost small" onClick={() => setShowHelp(true)}>Metric glossary</button>
            <p style={{ marginTop: 10 }}>Simulated shop. No real customer data.</p>
          </div>
        </aside>
        <main className="main"><Page health={health} /></main>
        {showHelp && (
          <div className="help-panel">
            <div className="card-head"><h2>What the numbers mean</h2><button className="btn ghost small" onClick={() => setShowHelp(false)}>Close</button></div>
            <dl>{Object.entries(help).map(([k, h]) => (<React.Fragment key={k}><dt>{h.label}{h.better !== "info" ? <span className="muted small"> · {h.better} is better</span> : null}</dt><dd>{h.text}</dd></React.Fragment>))}</dl>
          </div>
        )}
      </div>
    </HelpContext.Provider>
  );
}
