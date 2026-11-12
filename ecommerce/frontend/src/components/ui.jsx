import React, { createContext, useContext } from "react";

export const HelpContext = createContext({});

/** Small "?" that explains a metric in plain words. */
export function Tip({ k, text, title }) {
  const help = useContext(HelpContext);
  const h = k ? help[k] : null;
  const body = text || (h && h.text);
  if (!body) return null;
  const head = title || (h && `${h.label}${h.better && h.better !== "info" ? ` (${h.better} is better)` : ""}`);
  return (
    <span className="tip" tabIndex={0}>
      <i>?</i>
      <span className="bub">{head && <b>{head}</b>}{body}</span>
    </span>
  );
}

export function Stat({ label, value, sub, k }) {
  return (
    <div className="stat">
      <div className="k">{label}{k && <Tip k={k} />}</div>
      <div className="v num">{value}</div>
      {sub && <div className="s">{sub}</div>}
    </div>
  );
}

export const Pill = ({ tone = "", children }) => <span className={`pill ${tone}`}>{children}</span>;
export const Empty = ({ children }) => <div className="empty">{children}</div>;
export const ErrorBox = ({ error }) => (error ? <div className="err">{String(error)}</div> : null);

export function PageHead({ title, children }) {
  return (
    <div className="page-head">
      <div><h1>{title}</h1>{children && <p>{children}</p>}</div>
    </div>
  );
}

export function Progress({ value }) {
  return <div className="progress"><i style={{ width: `${Math.min(100, Math.max(0, value))}%` }} /></div>;
}

export function Json({ data }) {
  return <pre className="json">{JSON.stringify(data, null, 2)}</pre>;
}
