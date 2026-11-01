import { useCallback, useEffect, useState } from "react";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type Status = "open" | "reviewed";
type Verdict = "fraud" | "legit";

interface Flagged {
  txn_id: string;
  card_id: string;
  ts: number;
  amount: number;
  merchant_id: string;
  category: string;
  score: number;
  action: "review" | "decline";
  rules: string[];
  features: Record<string, number>;
  verdict: Verdict | null;
  analyst: string | null;
}

interface Stats {
  total: number;
  approved: number;
  in_review: number;
  declined: number;
  latency_p50_ms: number;
  latency_p99_ms: number;
  reviewed: number;
  confirmed_fraud: number;
  review_precision: number | null;
}

const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

const EVIDENCE: [string, string, (value: number) => string][] = [
  ["count_1h", "Uses in the last hour", (v) => v.toFixed(0)],
  ["amount_24h", "Spent in the last 24 hours", (v) => money.format(v)],
  ["amount_ratio_7d", "Compared with card's average", (v) => `${v.toFixed(1)}x`],
  ["km_from_last", "Distance from last purchase", (v) => (v < 0 ? "n/a" : `${Math.round(v).toLocaleString()} km`)],
  ["speed_kmh", "Travel speed that implies", (v) => (v === 0 ? "n/a" : `${Math.round(v).toLocaleString()} km/h`)],
];

function ruleLabel(rule: string) {
  return rule.replaceAll("_", " ");
}

export default function App() {
  const [status, setStatus] = useState<Status>("open");
  const [rows, setRows] = useState<Flagged[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    try {
      const [flaggedResponse, statsResponse] = await Promise.all([
        fetch(`${API}/flagged?status=${status}&limit=100`),
        fetch(`${API}/stats`),
      ]);
      if (!flaggedResponse.ok || !statsResponse.ok) throw new Error("request failed");
      const nextRows: Flagged[] = await flaggedResponse.json();
      setRows(nextRows);
      setStats(await statsResponse.json());
      setSelectedId((current) => (nextRows.some((row) => row.txn_id === current) ? current : nextRows[0]?.txn_id ?? null));
      setError(null);
    } catch {
      setError("Can't reach the API. Is it running?");
    }
  }, [status]);

  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), 5000);
    return () => clearInterval(timer);
  }, [load]);

  const selected = rows.find((row) => row.txn_id === selectedId) ?? null;

  const submit = async (verdict: Verdict) => {
    if (!selected || saving) return;
    setSaving(true);
    try {
      const response = await fetch(`${API}/flagged/${encodeURIComponent(selected.txn_id)}/review`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ verdict, analyst: "analyst" }),
      });
      if (!response.ok) throw new Error("request failed");
      await load();
    } catch {
      setError("Couldn't save that review. Try again.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <main>
      <header>
        <h1>Fraud Review</h1>
        {stats && (
          <dl className="stats">
            <div><dt>Scored</dt><dd>{stats.total.toLocaleString()}</dd></div>
            <div><dt>Flagged for review</dt><dd>{stats.in_review.toLocaleString()}</dd></div>
            <div><dt>Declined</dt><dd>{stats.declined.toLocaleString()}</dd></div>
            <div><dt>Analyst-confirmed fraud</dt><dd>{stats.review_precision === null ? "n/a" : `${Math.round(stats.review_precision * 100)}%`}</dd></div>
            <div><dt>Median scoring time</dt><dd>{stats.latency_p50_ms.toFixed(1)} ms</dd></div>
          </dl>
        )}
      </header>

      {error && <p className="error" role="alert">{error}</p>}

      <div className="tabs" role="tablist">
        {(["open", "reviewed"] as Status[]).map((tab) => (
          <button key={tab} role="tab" aria-selected={status === tab} onClick={() => setStatus(tab)}>
            {tab === "open" ? "To review" : "Reviewed"}
          </button>
        ))}
      </div>

      <div className="layout">
        <section className="queue" aria-label="Flagged transactions">
          {rows.length === 0 && !error && <p className="empty">Nothing here.</p>}
          {rows.map((row) => (
            <button key={row.txn_id} className={row.txn_id === selectedId ? "row selected" : "row"} onClick={() => setSelectedId(row.txn_id)}>
              <span className="amount">{money.format(row.amount)}</span>
              <span className="meta">{row.category.replaceAll("_", " ")} · {row.card_id}</span>
              <span className={`badge ${row.action}`}>{row.action}</span>
              <span className="score">{Math.round(row.score * 100)}%</span>
            </button>
          ))}
        </section>

        {selected && (
          <section className="detail" aria-label="Transaction detail">
            <h2>{money.format(selected.amount)} at {selected.merchant_id}</h2>
            <p className="meta">{selected.txn_id} · {selected.card_id} · {new Date(selected.ts * 1000).toLocaleString()}</p>
            <p>Model score: <strong>{Math.round(selected.score * 100)}%</strong></p>
            <h3>Why it was flagged</h3>
            {selected.rules.length ? (
              <ul className="rules">{selected.rules.map((rule) => <li key={rule}>{ruleLabel(rule)}</li>)}</ul>
            ) : (
              <p className="meta">No rule fired. Flagged by the model score.</p>
            )}
            <h3>Evidence</h3>
            <dl className="evidence">
              {EVIDENCE.map(([key, label, format]) => (
                <div key={key}><dt>{label}</dt><dd>{format(selected.features[key] ?? 0)}</dd></div>
              ))}
            </dl>
            {selected.verdict && <p className="verdict">Marked as <strong>{selected.verdict}</strong> by {selected.analyst}.</p>}
            <div className="actions">
              <button className="fraud" disabled={saving} onClick={() => void submit("fraud")}>Confirm fraud</button>
              <button className="legit" disabled={saving} onClick={() => void submit("legit")}>Mark legitimate</button>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
