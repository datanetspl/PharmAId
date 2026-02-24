import { useState } from "react";

const API_BASE = "http://localhost:8000";

const STATUS_COLORS = {
  active: { bg: "#d1fae5", text: "#065f46", dot: "#10b981" },
  inactive: { bg: "#fee2e2", text: "#991b1b", dot: "#ef4444" },
  completed: { bg: "#dbeafe", text: "#1e40af", dot: "#3b82f6" },
  stopped: { bg: "#fef9c3", text: "#854d0e", dot: "#eab308" },
  pending: { bg: "#fef9c3", text: "#854d0e", dot: "#eab308" },
};

function parseFHIR(bundle) {
  const res = bundle?.entry?.[0]?.resource;
  if (!res) return null;
  const dosage = res.dosageInstruction?.[0];
  const dispense = res.dispenseRequest;
  const timing = dosage?.timing?.repeat;
  const ordered = dosage?.doseAndRate?.find(d => d.type?.text === "ordered")?.doseQuantity;
  return {
    patient: res.subject?.display ?? "—",
    status: res.status ?? "—",
    intent: res.intent ?? "—",
    authoredOn: res.authoredOn ?? "—",
    medication: res.medicationReference?.display ?? "—",
    category: res.category?.[0]?.text ?? "—",
    courseOfTherapy: res.courseOfTherapyType?.text ?? "—",
    requester: res.requester?.display ?? "—",
    encounter: res.encounter?.display ?? "—",
    instruction: dosage?.patientInstruction ?? dosage?.text ?? "—",
    route: dosage?.route?.text ?? "—",
    dose: ordered ? `${ordered.value} ${ordered.unit}` : "—",
    timeOfDay: timing?.timeOfDay?.[0] ?? "—",
    daysCount: timing?.count ?? "—",
    startDate: timing?.boundsPeriod?.start ?? dispense?.validityPeriod?.start ?? "—",
    quantity: dispense?.quantity ? `${dispense.quantity.value} ${dispense.quantity.unit}` : "—",
    supplyDuration: dispense?.expectedSupplyDuration
      ? `${dispense.expectedSupplyDuration.value} ${dispense.expectedSupplyDuration.unit}s`
      : "—",
    refills: dispense?.numberOfRepeatsAllowed ?? "—",
    reasons: (res.reasonCode ?? []).map(r => ({
      text: r.text,
      icd10: r.coding?.find(c => c.system?.includes("icd-10"))?.code,
      snomed: r.coding?.find(c => c.system?.includes("snomed"))?.display,
    })),
  };
}

const fmtDate = s => {
  if (!s || s === "—") return "—";
  try { return new Date(s).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" }); }
  catch { return s; }
};
const fmtTime = s => {
  if (!s || s === "—") return "—";
  try {
    const [h, m] = s.split(":");
    const d = new Date(); d.setHours(+h, +m);
    return d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
  } catch { return s; }
};

// ── Icons ──────────────────────────────────────────────────────
const PillIcon = () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m10.5 20.5-7-7a4.95 4.95 0 1 1 7-7l7 7a4.95 4.95 0 0 1-7 7Z" /><path d="m8.5 8.5 7 7" /></svg>;
const ClockIcon = () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>;
const BoxIcon = () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" /><polyline points="3.29 7 12 12 20.71 7" /><line x1="12" y1="22" x2="12" y2="12" /></svg>;
const HeartIcon = () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>;
const UserIcon = () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" /></svg>;
const CodeIcon = () => <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" /></svg>;
const SparkIcon = () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3v1m0 16v1M4.22 4.22l.7.7m12.16 12.16.7.7M3 12h1m16 0h1M4.22 19.78l.7-.7M18.36 5.64l.7-.7" /><circle cx="12" cy="12" r="4" /></svg>;
const WarnIcon = () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" /><line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" /></svg>;
const CheckIcon = () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>;
const InfoIcon = () => <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></svg>;

// ── Shared sub-components ──────────────────────────────────────
function Field({ label, value }) {
  const empty = !value || value === "—";
  return (
    <div className="field">
      <div className="field-label">{label}</div>
      <div className={`field-value${empty ? " muted" : ""}`}>{empty ? "Not specified" : value}</div>
    </div>
  );
}

function Section({ icon, title, children }) {
  return (
    <div className="section">
      <div className="section-header">
        <span className="section-icon">{icon}</span>
        <span className="section-title">{title}</span>
      </div>
      <div className="section-body">{children}</div>
    </div>
  );
}

// ── Page 1: MedCard ────────────────────────────────────────────
function MedCard({ raw, animate }) {
  const [showRaw, setShowRaw] = useState(false);
  const p = parseFHIR(raw);
  if (!p) return null;
  const st = STATUS_COLORS[p.status] ?? { bg: "#f3f4f6", text: "#374151", dot: "#9ca3af" };
  const initials = p.patient.split(/[,\s]+/).filter(Boolean).map(w => w[0]).join("").toUpperCase().slice(0, 2) || "?";

  return (
    <div className={`result-card${animate ? " visible" : ""}`}>
      <div className="rx-header">
        <div className="avatar">{initials}</div>
        <div className="header-info">
          <div className="patient-name">{p.patient}</div>
          <div className="meta-row">
            <span className="meta-chip">
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" /><line x1="16" y1="2" x2="16" y2="6" /><line x1="8" y1="2" x2="8" y2="6" /><line x1="3" y1="10" x2="21" y2="10" /></svg>
              {fmtDate(p.authoredOn)}
            </span>
            <span className="meta-chip">
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><polyline points="9 22 9 12 15 12 15 22" /></svg>
              {p.encounter}
            </span>
          </div>
        </div>
        <div className="status-badge" style={{ background: st.bg, color: st.text }}>
          <div className="status-dot" style={{ background: st.dot }} />
          {p.status}
        </div>
      </div>
      <div className="rx-body">
        <Section icon={<PillIcon />} title="Medication">
          <div className="med-name">{p.medication}</div>
          <div className="instruction-box">📋 {p.instruction}</div>
          <div className="field-row">
            <Field label="Route" value={p.route} />
            <Field label="Dose" value={p.dose} />
            <Field label="Category" value={p.category} />
            <Field label="Course" value={p.courseOfTherapy} />
          </div>
        </Section>
        <div className="divider" />
        <Section icon={<ClockIcon />} title="Schedule">
          <div className="field-row">
            <Field label="Start Date" value={fmtDate(p.startDate)} />
            <Field label="Time of Day" value={fmtTime(p.timeOfDay)} />
            <Field label="Total Doses" value={p.daysCount !== "—" ? `${p.daysCount} doses` : "—"} />
            <Field label="Intent" value={p.intent} />
          </div>
        </Section>
        <div className="divider" />
        <Section icon={<BoxIcon />} title="Dispense">
          <div className="field-row">
            <Field label="Quantity" value={p.quantity} />
            <Field label="Supply Duration" value={p.supplyDuration} />
            <Field label="Refills Allowed" value={p.refills !== "—" ? `${p.refills} refills` : "—"} />
          </div>
        </Section>
        <div className="divider" />
        <Section icon={<HeartIcon />} title="Indication">
          {p.reasons.length === 0
            ? <div className="field-value muted">No reason recorded</div>
            : p.reasons.map((r, i) => (
              <div className="reason-row" key={i}>
                <div className="reason-text">{r.text}</div>
                <div className="reason-codes">
                  {r.icd10 && <span className="code-chip icd">ICD-10: {r.icd10}</span>}
                  {r.snomed && <span className="code-chip snomed">SNOMED: {r.snomed}</span>}
                </div>
              </div>
            ))
          }
        </Section>
        <div className="divider" />
        <Section icon={<UserIcon />} title="Prescriber">
          <div className="field-row">
            <Field label="Requested By" value={p.requester} />
            <Field label="Recorded By" value={p.requester} />
          </div>
        </Section>
        <button className="raw-toggle" onClick={() => setShowRaw(v => !v)}>
          <CodeIcon /> {showRaw ? "Hide" : "Show"} raw FHIR JSON
        </button>
        {showRaw && (
          <div className="raw-block">
            <pre>{JSON.stringify(raw, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Page 2: ProcessResult renderer ────────────────────────────
function ProcessResult({ result, animate }) {
  const [showRaw, setShowRaw] = useState(false);

  // Try to render smartly based on what the backend returns
  const renderValue = (val) => {
    if (val === null || val === undefined) return <span className="muted">—</span>;
    if (typeof val === "boolean") return (
      <span className={`bool-chip ${val ? "bool-yes" : "bool-no"}`}>
        {val ? <><CheckIcon /> Yes</> : <>✕ No</>}
      </span>
    );
    if (Array.isArray(val)) {
      if (val.length === 0) return <span className="muted">None</span>;
      return (
        <ul className="result-list">
          {val.map((item, i) => (
            <li key={i}>{typeof item === "object" ? JSON.stringify(item) : String(item)}</li>
          ))}
        </ul>
      );
    }
    if (typeof val === "object") {
      return (
        <div className="nested-fields">
          {Object.entries(val).map(([k, v]) => (
            <div className="nested-row" key={k}>
              <span className="nested-key">{k.replace(/_/g, " ")}:</span>
              <span className="nested-val">{typeof v === "object" ? JSON.stringify(v) : String(v)}</span>
            </div>
          ))}
        </div>
      );
    }
    return <span>{String(val)}</span>;
  };

  // Detect severity/alert keys
  const severityMap = {
    interaction: "warn",
    warning: "warn",
    alert: "warn",
    contraindication: "warn",
    recommendation: "ok",
    approved: "ok",
    safe: "ok",
    error: "err",
    denied: "err",
  };

  const getSeverity = (key) => {
    const lk = key.toLowerCase();
    for (const [kw, sev] of Object.entries(severityMap)) {
      if (lk.includes(kw)) return sev;
    }
    return null;
  };

  const entries = typeof result === "object" && !Array.isArray(result)
    ? Object.entries(result)
    : null;

  return (
    <div className={`result-card process-result${animate ? " visible" : ""}`}>
      <div className="rx-header process-header">
        <div className="process-icon-wrap">
          <SparkIcon />
        </div>
        <div className="header-info">
          <div className="patient-name">Analysis Complete</div>
          <div className="meta-row">
            <span className="meta-chip">AI-powered review</span>
            <span className="meta-chip">{new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      </div>

      <div className="rx-body">
        {entries ? (
          entries.map(([key, val]) => {
            const sev = getSeverity(key);
            return (
              <div key={key}>
                <div className={`process-section${sev ? ` sev-${sev}` : ""}`}>
                  <div className="process-key">
                    {sev === "warn" && <span className="sev-icon warn"><WarnIcon /></span>}
                    {sev === "ok" && <span className="sev-icon ok"><CheckIcon /></span>}
                    {sev === "err" && <span className="sev-icon err"><InfoIcon /></span>}
                    {key.replace(/_/g, " ")}
                  </div>
                  <div className="process-val">{renderValue(val)}</div>
                </div>
                <div className="divider" />
              </div>
            );
          })
        ) : (
          <div className="process-section">
            <div className="process-val">{renderValue(result)}</div>
          </div>
        )}

        <button className="raw-toggle" onClick={() => setShowRaw(v => !v)}>
          <CodeIcon /> {showRaw ? "Hide" : "Show"} raw response
        </button>
        {showRaw && (
          <div className="raw-block">
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Page 2: Process ────────────────────────────────────────────
function ProcessPage({ patientData, onBack }) {
  const [medication, setMedication] = useState("");
  const [symptoms, setSymptoms] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [animate, setAnimate] = useState(false);

  const parsed = parseFHIR(patientData);

  const handleProcess = async () => {
    if (!medication.trim() && !symptoms.trim()) return;
    setLoading(true); setError(null); setResult(null); setAnimate(false);
    try {
      const res = await fetch(`${API_BASE}/processpatient`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientData?.entry?.[0]?.resource?.subject?.reference ?? "",
          patient_name: parsed?.patient ?? "",
          existing_medication: parsed?.medication ?? "",
          new_medication: medication.trim(),
          new_symptoms: symptoms.trim(),
          fhir_bundle: patientData,
        }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}: ${res.statusText}`);
      const json = await res.json();
      setResult(json);
      setTimeout(() => setAnimate(true), 50);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Step nav */}
      <div className="step-nav">
        <button className="back-btn" onClick={onBack}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6" /></svg>
          Back to Record
        </button>
        <div className="step-pills">
          <div className="step-pill done">
            <span className="step-num">✓</span> Patient Record
          </div>
          <div className="step-arrow">→</div>
          <div className="step-pill active">
            <span className="step-num">2</span> Process
          </div>
        </div>
      </div>

      {/* Context banner */}
      {parsed && (
        <div className="context-banner">
          <div className="context-avatar">
            {parsed.patient.split(/[,\s]+/).filter(Boolean).map(w => w[0]).join("").toUpperCase().slice(0, 2) || "?"}
          </div>
          <div className="context-info">
            <div className="context-name">{parsed.patient}</div>
            <div className="context-med">Current: {parsed.medication}</div>
          </div>
        </div>
      )}

      {/* Input card */}
      <div className="process-input-card">
        <div className="process-input-header">
          <div className="process-input-title">New Medication & Symptoms</div>
          <div className="process-input-sub">Enter the patient's new prescription and any reported symptoms for AI analysis</div>
        </div>

        <div className="input-group">
          <label className="input-label">
            <PillIcon /> New Medication
          </label>
          <input
            className="text-input"
            type="text"
            placeholder="e.g. Metformin 500mg twice daily"
            value={medication}
            onChange={e => setMedication(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleProcess()}
          />
        </div>

        <div className="input-group">
          <label className="input-label">
            <HeartIcon /> New Symptoms
          </label>
          <textarea
            className="text-input textarea"
            placeholder="e.g. Nausea, dizziness, fatigue after meals, mild headache..."
            value={symptoms}
            onChange={e => setSymptoms(e.target.value)}
            rows={3}
          />
        </div>

        <button
          className="process-btn"
          onClick={handleProcess}
          disabled={loading || (!medication.trim() && !symptoms.trim())}
        >
          {loading ? (
            <><div className="btn-spinner" /> Analysing…</>
          ) : (
            <><SparkIcon /> Process with AI</>
          )}
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div className="loading-state">
          <div className="spinner" />
          <div className="loading-text">Running AI analysis…</div>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="error-card">
          <div className="error-icon">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <div>
            <div className="error-title">Processing Failed</div>
            <div className="error-msg">{error}</div>
          </div>
        </div>
      )}

      {/* Result */}
      {result && !loading && <ProcessResult result={result} animate={animate} />}
    </>
  );
}

// ── Root App ───────────────────────────────────────────────────
export default function PatientLookup() {
  const [page, setPage] = useState(1);    // 1 = lookup, 2 = process
  const [inputId, setInputId] = useState("");
  const [rawData, setRawData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [animate, setAnimate] = useState(false);

  const fetchDetails = async (id) => {
    if (!id.trim()) return;
    setLoading(true); setError(null); setRawData(null); setAnimate(false);
    try {
      const res = await fetch(`${API_BASE}/patientdetails?pid=${id.trim()}`);
      if (!res.ok) throw new Error(`No record found for ID: ${id}`);
      const json = await res.json();
      setRawData(json);
      setTimeout(() => setAnimate(true), 50);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body, #root { width: 100%; min-height: 100vh; }
        body { background: #eef2f7; font-family: 'DM Sans', sans-serif; }

        .page {
          min-height: 100vh; width: 100%; background: #eef2f7;
          display: flex; flex-direction: column; align-items: center;
          padding: 40px 20px 80px; position: relative; overflow-x: hidden;
        }
        .bg-grid { position: fixed; inset: 0; background-image: linear-gradient(rgba(6,95,70,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(6,95,70,.04) 1px,transparent 1px); background-size: 40px 40px; pointer-events: none; z-index: 0; }
        .bg-blob  { position: fixed; width: 600px; height: 600px; border-radius: 50%; background: radial-gradient(circle,rgba(16,185,129,.10) 0%,transparent 70%); top: -100px; right: -100px; pointer-events: none; z-index: 0; }
        .bg-blob2 { position: fixed; width: 400px; height: 400px; border-radius: 50%; background: radial-gradient(circle,rgba(6,95,70,.07) 0%,transparent 70%); bottom: -80px; left: -80px; pointer-events: none; z-index: 0; }

        .content { position: relative; z-index: 1; width: 100%; max-width: 640px; }

        /* Header */
        .header { margin-bottom: 40px; text-align: center; }
        .logo-row { display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 6px; }
        .logo-icon { width: 44px; height: 44px; background: linear-gradient(135deg,#065f46,#10b981); border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 16px rgba(16,185,129,.3); }
        .logo-icon svg { color: white; }
        .app-name { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 800; color: #0f2419; letter-spacing: -0.5px; }
        .app-name span { color: #10b981; }
        .tagline { font-size: .78rem; color: #6b7280; font-weight: 300; letter-spacing: .07em; text-transform: uppercase; }

        /* Search card */
        .search-card { background: white; border-radius: 18px; padding: 26px; box-shadow: 0 1px 3px rgba(0,0,0,.06),0 8px 32px rgba(0,0,0,.06); margin-bottom: 20px; }
        .search-label { font-family: 'Syne', sans-serif; font-size: .68rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #374151; margin-bottom: 10px; display: block; }
        .input-row { display: flex; gap: 10px; }
        .id-input { flex: 1; height: 50px; border: 2px solid #e5e7eb; border-radius: 12px; padding: 0 16px; font-family: 'DM Sans', sans-serif; font-size: .95rem; color: #111827; background: #f9fafb; outline: none; transition: border-color .2s,box-shadow .2s,background .2s; }
        .id-input:focus { border-color: #10b981; background: white; box-shadow: 0 0 0 4px rgba(16,185,129,.1); }
        .id-input::placeholder { color: #9ca3af; }
        .search-btn { height: 50px; padding: 0 22px; background: linear-gradient(135deg,#065f46,#10b981); color: white; border: none; border-radius: 12px; font-family: 'Syne', sans-serif; font-weight: 700; font-size: .875rem; letter-spacing: .05em; cursor: pointer; transition: opacity .2s,transform .1s,box-shadow .2s; box-shadow: 0 4px 12px rgba(16,185,129,.3); white-space: nowrap; }
        .search-btn:hover:not(:disabled) { opacity: .9; transform: translateY(-1px); box-shadow: 0 6px 18px rgba(16,185,129,.35); }
        .search-btn:active:not(:disabled) { transform: translateY(0); }
        .search-btn:disabled { opacity: .6; cursor: not-allowed; }

        /* Next button */
        .next-btn-wrap { display: flex; justify-content: flex-end; margin-top: 20px; }
        .next-btn { display: inline-flex; align-items: center; gap: 8px; height: 48px; padding: 0 26px; background: linear-gradient(135deg,#065f46,#10b981); color: white; border: none; border-radius: 12px; font-family: 'Syne', sans-serif; font-weight: 700; font-size: .9rem; letter-spacing: .04em; cursor: pointer; transition: opacity .2s, transform .1s, box-shadow .2s; box-shadow: 0 4px 14px rgba(16,185,129,.35); }
        .next-btn:hover { opacity: .9; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(16,185,129,.4); }
        .next-btn:active { transform: translateY(0); }

        /* States */
        .loading-state { display: flex; flex-direction: column; align-items: center; padding: 48px 0; gap: 14px; }
        .spinner { width: 38px; height: 38px; border: 3px solid #e5e7eb; border-top-color: #10b981; border-radius: 50%; animation: spin .7s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-text { font-size: .875rem; color: #6b7280; font-style: italic; }
        .error-card { background: #fff1f2; border: 1px solid #fecdd3; border-radius: 14px; padding: 18px 22px; display: flex; align-items: flex-start; gap: 14px; animation: slideUp .3s ease; }
        .error-icon { width: 34px; height: 34px; background: #fee2e2; border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-shrink: 0; color: #ef4444; }
        .error-title { font-family: 'Syne', sans-serif; font-weight: 700; color: #991b1b; font-size: .875rem; margin-bottom: 2px; }
        .error-msg { font-size: .82rem; color: #b91c1c; }
        .empty-state { text-align: center; padding: 48px 20px; color: #9ca3af; }
        .empty-icon { width: 60px; height: 60px; margin: 0 auto 14px; background: #f3f4f6; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
        .empty-text { font-size: .875rem; }

        /* Result card (shared) */
        .result-card { background: white; border-radius: 20px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.06),0 8px 32px rgba(0,0,0,.07); opacity: 0; transform: translateY(16px); transition: opacity .4s ease,transform .4s ease; }
        .result-card.visible { opacity: 1; transform: translateY(0); }
        .rx-header { background: linear-gradient(135deg,#065f46 0%,#059669 100%); padding: 26px 30px; display: flex; align-items: center; gap: 16px; }
        .avatar { width: 54px; height: 54px; border-radius: 50%; background: rgba(255,255,255,.2); border: 2px solid rgba(255,255,255,.4); display: flex; align-items: center; justify-content: center; font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 800; color: white; flex-shrink: 0; }
        .header-info { flex: 1; min-width: 0; }
        .patient-name { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.1rem; color: white; margin-bottom: 6px; }
        .meta-row { display: flex; flex-wrap: wrap; gap: 7px; }
        .meta-chip { display: inline-flex; align-items: center; gap: 5px; background: rgba(255,255,255,.15); border-radius: 6px; padding: 3px 9px; font-size: .72rem; color: rgba(255,255,255,.88); font-weight: 500; }
        .status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 13px; border-radius: 999px; font-size: .7rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; flex-shrink: 0; }
        .status-dot { width: 7px; height: 7px; border-radius: 50%; }
        .rx-body { padding: 0 0 20px; }
        .section { padding: 22px 30px 10px; }
        .section-header { display: flex; align-items: center; gap: 9px; margin-bottom: 14px; }
        .section-icon { width: 28px; height: 28px; background: #f0fdf4; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #059669; flex-shrink: 0; }
        .section-title { font-family: 'Syne', sans-serif; font-size: .67rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #374151; }
        .med-name { font-family: 'Syne', sans-serif; font-weight: 700; font-size: .97rem; color: #0f2419; margin-bottom: 12px; line-height: 1.55; }
        .instruction-box { background: #f0fdf4; border-left: 3px solid #10b981; border-radius: 0 10px 10px 0; padding: 11px 14px; font-size: .85rem; color: #065f46; font-weight: 500; margin-bottom: 14px; line-height: 1.65; }
        .field-row { display: flex; flex-wrap: wrap; gap: 10px; }
        .field { background: #f9fafb; border-radius: 10px; padding: 11px 14px; border: 1px solid #f3f4f6; min-width: 110px; flex: 1; }
        .field-label { font-size: .64rem; color: #9ca3af; font-weight: 500; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 3px; }
        .field-value { font-size: .85rem; color: #111827; font-weight: 500; }
        .field-value.muted { color: #9ca3af; font-style: italic; }
        .reason-row { margin-bottom: 12px; }
        .reason-text { font-size: .88rem; color: #111827; font-weight: 500; margin-bottom: 6px; }
        .reason-codes { display: flex; flex-wrap: wrap; gap: 6px; }
        .code-chip { font-size: .67rem; font-weight: 700; padding: 3px 8px; border-radius: 5px; letter-spacing: .04em; }
        .code-chip.icd    { background: #eff6ff; color: #1d4ed8; }
        .code-chip.snomed { background: #f5f3ff; color: #6d28d9; }
        .divider { height: 1px; background: #f3f4f6; margin: 4px 30px 0; }
        .raw-toggle { display: flex; align-items: center; gap: 7px; margin: 16px 30px 10px; font-size: .74rem; color: #6b7280; cursor: pointer; border: none; background: none; padding: 0; font-family: 'DM Sans', sans-serif; transition: color .2s; }
        .raw-toggle:hover { color: #065f46; }
        .raw-block { margin: 0 30px 8px; background: #0f2419; border-radius: 12px; padding: 16px 18px; overflow-x: auto; }
        .raw-block pre { font-size: .7rem; color: #6ee7b7; line-height: 1.7; font-family: 'Fira Code','Courier New',monospace; }

        /* ── Page 2 specific ── */

        /* Step nav */
        .step-nav { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
        .back-btn { display: inline-flex; align-items: center; gap: 6px; background: white; border: 1.5px solid #e5e7eb; border-radius: 10px; padding: 8px 14px; font-size: .8rem; font-weight: 500; color: #374151; cursor: pointer; font-family: 'DM Sans', sans-serif; transition: border-color .2s, color .2s, box-shadow .2s; }
        .back-btn:hover { border-color: #10b981; color: #065f46; box-shadow: 0 2px 8px rgba(16,185,129,.12); }
        .step-pills { display: flex; align-items: center; gap: 8px; }
        .step-pill { display: flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 999px; font-size: .72rem; font-weight: 600; letter-spacing: .04em; }
        .step-pill.done { background: #d1fae5; color: #065f46; }
        .step-pill.active { background: #065f46; color: white; }
        .step-num { font-family: 'Syne', sans-serif; font-weight: 800; }
        .step-arrow { color: #9ca3af; font-size: .85rem; }

        /* Context banner */
        .context-banner { background: white; border-radius: 14px; padding: 16px 20px; display: flex; align-items: center; gap: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.05),0 4px 16px rgba(0,0,0,.05); margin-bottom: 16px; border-left: 3px solid #10b981; }
        .context-avatar { width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg,#065f46,#10b981); display: flex; align-items: center; justify-content: center; font-family: 'Syne', sans-serif; font-weight: 800; font-size: .95rem; color: white; flex-shrink: 0; }
        .context-name { font-family: 'Syne', sans-serif; font-weight: 700; font-size: .95rem; color: #0f2419; margin-bottom: 2px; }
        .context-med { font-size: .78rem; color: #6b7280; }

        /* Process input card */
        .process-input-card { background: white; border-radius: 18px; padding: 28px; box-shadow: 0 1px 3px rgba(0,0,0,.06),0 8px 32px rgba(0,0,0,.06); margin-bottom: 20px; }
        .process-input-header { margin-bottom: 22px; }
        .process-input-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.1rem; color: #0f2419; margin-bottom: 4px; }
        .process-input-sub { font-size: .82rem; color: #6b7280; line-height: 1.5; }
        .input-group { margin-bottom: 18px; }
        .input-label { display: flex; align-items: center; gap: 6px; font-family: 'Syne', sans-serif; font-size: .68rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #374151; margin-bottom: 8px; }
        .text-input { width: 100%; border: 2px solid #e5e7eb; border-radius: 12px; padding: 12px 16px; font-family: 'DM Sans', sans-serif; font-size: .95rem; color: #111827; background: #f9fafb; outline: none; transition: border-color .2s,box-shadow .2s,background .2s; resize: vertical; }
        .text-input:focus { border-color: #10b981; background: white; box-shadow: 0 0 0 4px rgba(16,185,129,.1); }
        .text-input::placeholder { color: #9ca3af; }
        .text-input.textarea { line-height: 1.6; min-height: 80px; }
        .process-btn { width: 100%; height: 52px; display: flex; align-items: center; justify-content: center; gap: 8px; background: linear-gradient(135deg,#065f46,#10b981); color: white; border: none; border-radius: 12px; font-family: 'Syne', sans-serif; font-weight: 700; font-size: .95rem; letter-spacing: .04em; cursor: pointer; transition: opacity .2s,transform .1s,box-shadow .2s; box-shadow: 0 4px 14px rgba(16,185,129,.3); margin-top: 4px; }
        .process-btn:hover:not(:disabled) { opacity: .9; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(16,185,129,.4); }
        .process-btn:active:not(:disabled) { transform: translateY(0); }
        .process-btn:disabled { opacity: .55; cursor: not-allowed; }
        .btn-spinner { width: 18px; height: 18px; border: 2.5px solid rgba(255,255,255,.35); border-top-color: white; border-radius: 50%; animation: spin .7s linear infinite; }

        /* Process result card */
        .process-header { background: linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%); }
        .process-icon-wrap { width: 54px; height: 54px; border-radius: 50%; background: rgba(255,255,255,.2); border: 2px solid rgba(255,255,255,.4); display: flex; align-items: center; justify-content: center; color: white; flex-shrink: 0; }
        .process-section { padding: 18px 30px 10px; }
        .process-key { font-family: 'Syne', sans-serif; font-size: .7rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #374151; margin-bottom: 8px; display: flex; align-items: center; gap: 7px; }
        .process-val { font-size: .88rem; color: #111827; line-height: 1.65; }
        .sev-icon { width: 22px; height: 22px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; }
        .sev-icon.warn { background: #fef9c3; color: #b45309; }
        .sev-icon.ok   { background: #d1fae5; color: #065f46; }
        .sev-icon.err  { background: #fee2e2; color: #b91c1c; }
        .sev-warn .process-key { color: #b45309; }
        .sev-ok   .process-key { color: #065f46; }
        .sev-err  .process-key { color: #b91c1c; }
        .result-list { padding-left: 20px; margin-top: 4px; }
        .result-list li { margin-bottom: 6px; font-size: .87rem; color: #374151; }
        .bool-chip { display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 999px; font-size: .78rem; font-weight: 700; }
        .bool-yes { background: #d1fae5; color: #065f46; }
        .bool-no  { background: #fee2e2; color: #991b1b; }
        .nested-fields { display: flex; flex-direction: column; gap: 6px; margin-top: 4px; }
        .nested-row { display: flex; gap: 8px; font-size: .85rem; }
        .nested-key { color: #6b7280; font-weight: 500; text-transform: capitalize; min-width: 120px; flex-shrink: 0; }
        .nested-val { color: #111827; }
        .muted { color: #9ca3af; font-style: italic; }

        @keyframes slideUp { from { opacity:0; transform: translateY(8px); } to { opacity:1; transform: translateY(0); } }

        @media (max-width: 480px) {
          .rx-header,.process-header { flex-wrap: wrap; }
          .status-badge { margin-top: 8px; }
          .section,.process-section { padding: 18px 20px 8px; }
          .divider { margin: 4px 20px 0; }
          .raw-toggle { margin: 14px 20px 8px; }
          .raw-block { margin: 0 20px 8px; }
          .step-nav { flex-direction: column; align-items: flex-start; gap: 12px; }
        }
      `}</style>

      <div className="page">
        <div className="bg-grid" />
        <div className="bg-blob" />
        <div className="bg-blob2" />

        <div className="content">
          {/* Logo — always shown */}
          <div className="header">
            <div className="logo-row">
              <div className="logo-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2L2 7l10 5 10-5-10-5z" /><path d="M2 17l10 5 10-5" /><path d="M2 12l10 5 10-5" />
                </svg>
              </div>
              <div className="app-name">Pharm<span>AI</span>d</div>
            </div>
            <div className="tagline">Pharmacy Intelligence System</div>
          </div>

          {/* ── PAGE 1 ── */}
          {page === 1 && (
            <>
              <div className="search-card">
                <label className="search-label">Patient ID</label>
                <div className="input-row">
                  <input
                    className="id-input"
                    type="text"
                    placeholder="e.g. erXuFYUfucBZaryVksYEcMg3"
                    value={inputId}
                    onChange={e => setInputId(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && fetchDetails(inputId)}
                  />
                  <button
                    className="search-btn"
                    onClick={() => fetchDetails(inputId)}
                    disabled={loading || !inputId.trim()}
                  >
                    {loading ? "Searching…" : "Look Up →"}
                  </button>
                </div>
              </div>

              {loading && (
                <div className="loading-state">
                  <div className="spinner" />
                  <div className="loading-text">Fetching FHIR record…</div>
                </div>
              )}

              {error && !loading && (
                <div className="error-card">
                  <div className="error-icon">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
                    </svg>
                  </div>
                  <div>
                    <div className="error-title">Record Not Found</div>
                    <div className="error-msg">{error}</div>
                  </div>
                </div>
              )}

              {!loading && !error && !rawData && (
                <div className="empty-state">
                  <div className="empty-icon">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#d1d5db" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
                    </svg>
                  </div>
                  <div className="empty-text">Enter a patient ID to retrieve their EHR medication record</div>
                </div>
              )}

              {rawData && !loading && (
                <>
                  <MedCard raw={rawData} animate={animate} />
                  <div className="next-btn-wrap">
                    <button className="next-btn" onClick={() => setPage(2)}>
                      Process Patient
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6" /></svg>
                    </button>
                  </div>
                </>
              )}
            </>
          )}

          {/* ── PAGE 2 ── */}
          {page === 2 && (
            <ProcessPage
              patientData={rawData}
              onBack={() => setPage(1)}
            />
          )}
        </div>
      </div>
    </>
  );
}
