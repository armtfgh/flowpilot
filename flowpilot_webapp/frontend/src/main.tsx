import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity, AlertTriangle, Archive, ArrowRight, Atom, Beaker, Bot,
  Box, Check, CheckCircle2, ChevronDown, ChevronRight, CircleGauge,
  ClipboardCheck, Clock3, Database, Download, ExternalLink, FileJson,
  FileText, FlaskConical, Gauge, GitBranch, History, Image, Info,
  Layers3, Library, ListChecks, Menu, PackageCheck, Play, Plus,
  RefreshCw, Save, Send, Settings2, ShieldCheck, SlidersHorizontal,
  Sparkles, TestTube2, Thermometer, Trash2, Upload, User, Waypoints,
  X, Zap
} from "lucide-react";
import "./styles.css";

type Json = Record<string, any>;
type View = "design" | "inventory" | "runs" | "system";
type ResultTab = "overview" | "topology" | "engineering" | "recipe" | "inventory" | "council" | "cycles" | "json";

type Health = {
  status: string;
  version: string;
  corpus_records: number;
  inventory_profiles: number;
  models: Record<string, string>;
};

type Question = {
  question_id: string;
  section: string;
  question: string;
  expected_format: string;
  required: boolean;
  why_needed?: string;
};

type Job = {
  job_id: string;
  archive_run_id?: string;
  status: string;
  progress: number;
  phase: string;
  messages: { time: string; level: string; message: string }[];
  result?: Json;
  error?: string | null;
  autosave_dir?: string | null;
  created_at?: string;
};

type InventoryProfile = {
  profile_id: string;
  name: string;
  laboratory?: string;
  version?: number;
  status?: string;
  lab_inventory: Json;
  operating_constraints?: Json;
  validation?: { valid: boolean; errors: string[]; warnings: string[]; unresolved_fields: string[] };
  [key: string]: any;
};

type ModelRoute = {
  route_id: string;
  label: string;
  provider: string;
  model: string;
  base_url?: string | null;
  local: boolean;
  available: boolean;
  availability: string;
  reason?: string;
};

type ModelCatalog = {
  models: ModelRoute[];
  defaults: { upstream: string; downstream: string };
};

const UNAVAILABLE_QUESTION_IDS = new Set([
  "Q-HIST-001", "Q-INV-001", "Q-CONSTR-001", "Q-HYP-001", "Q-PREF-001"
]);

const DEMO_PROTOCOL = `Fmoc-L-methionine (3.7 mmol, 1.0 equiv) and a flavin photocatalyst (10 mol%) were dissolved in acetonitrile (0.10 M). Air was supplied as the oxidant. The mixture was irradiated with a 420 nm LED at 21 °C for 5 min and afforded the product in 99% yield.`;

const DEMO_RESULT: Json = {
  confidence: "HIGH",
  recommended_disposition: "SCREEN",
  final_design: {
    schema_version: "flowpilot_final_design_v2.0",
    status: "executable",
    parameters: {
      concentration_M: 0.10, flow_rate_mL_min: 0.124,
      residence_time_min: 8.05, residence_time_inlet_min: 8.05,
      residence_time_in_channel_min: 5.21, reactor_volume_mL: 1.0,
      tubing_ID_mm: 0.8, reactor_type: "PFA photocoil",
      material: "PFA", temperature_C: 21, BPR_bar: 3,
      wavelength_nm: 420, inventory_selection: "KHU photochemical system"
    },
    streams: [
      { stream_label: "A", phase: "liquid", flow_rate_mL_min: 0.124, contents: ["substrate", "photocatalyst in MeCN"] },
      { stream_label: "B", phase: "gas", gas_flow_sccm: 1.857, gas_flow_actual_mL_min: 0.505, contents: "air" }
    ],
    stages: [{ stage_number: "ST-01", stage_name: "photochemical aerobic oxidation", reactor_volume_mL: 1.0, residence_time_min: 8.05 }],
    consistency: { passed: true, issues: [] }
  },
  process_topology: {
    unit_operations: [
      { op_id: "pump_a", label: "Liquid feed", op_type: "pump", inventory_item_id: "pump_a" },
      { op_id: "mfc_b", label: "Air MFC", op_type: "mfc", inventory_item_id: "mfc_b" },
      { op_id: "mix_1", label: "T-mixer", op_type: "mixer", inventory_item_id: "mix_1" },
      { op_id: "rx_1", label: "420 nm photocoil", op_type: "photoreactor", inventory_item_id: "rx_1" },
      { op_id: "bpr_1", label: "3 bar BPR", op_type: "bpr", inventory_item_id: "bpr_1" },
      { op_id: "sep_1", label: "G-L separator", op_type: "separator", inventory_item_id: "sep_1" },
      { op_id: "col_1", label: "Amber collector", op_type: "collector", inventory_item_id: "col_1" }
    ]
  },
  chemistry_plan: {
    reaction_name: "Photocatalytic aerobic oxidation of Fmoc-L-methionine",
    mechanism_type: "flavin-mediated aerobic photooxidation",
    key_risks: ["oxygen transfer", "photon attenuation", "gas-liquid stability"],
    stream_logic: ["Substrate and photocatalyst share the liquid feed", "Air is metered independently before mixing"]
  },
  instrument_manifest: [
    { equipment_id: "pump_hplc_01", name: "HPLC liquid pump", role: "liquid feed" },
    { equipment_id: "mfc_air_01", name: "Air mass-flow controller", role: "gas feed" },
    { equipment_id: "photo_1ml_420", name: "1.0 mL PFA photocoil", role: "reaction" },
    { equipment_id: "bpr_3bar", name: "3 bar BPR", role: "pressure control" }
  ],
  council_rounds: 2,
  council_messages: [
    { agent: "Chemistry", content: "Transformation and oxygen demand are consistent with an aerobic photooxidation screen." },
    { agent: "Fluidics", content: "Liquid residence time and pressure-corrected gas flow close against the 1.0 mL coil." },
    { agent: "Safety", content: "Use a vented separator and shield the illuminated pressurized coil." }
  ]
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { ...(init?.headers as Record<string, string> || {}) };
  if (!(init?.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch { /* retain status */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function formatValue(value: any, digits = 3): string {
  if (value === null || value === undefined || value === "") return "--";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
  return String(value);
}

function shortTime(value?: string): string {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function operationId(op: Json, fallback = "unassigned"): string {
  return op.equipment_id || op.inventory_item_id || op.parameters?.inventory_item_id ||
    op.parameters?.inventory_equipment_id || op.operation_id || op.op_id || fallback;
}

function operationTitle(op: Json): string {
  const label = String(op.label || op.name || op.op_type || "operation");
  return label.split("|")[0].split("—")[0].trim() || label;
}

function streamContents(stream: Json): string {
  const value = stream.composition ?? stream.contents ?? stream.role;
  return Array.isArray(value) ? value.join("; ") : value || "--";
}

function Kpi({ label, value, note, icon, tone = "teal" }: { label: string; value: string; note: string; icon: React.ReactNode; tone?: string }) {
  return <div className={`kpi tone-${tone}`}>
    <div className="kpiHead"><span>{icon}{label}</span></div>
    <strong>{value}</strong><small>{note}</small>
  </div>;
}

function App() {
  const demo = new URLSearchParams(window.location.search).get("demo") === "1";
  const [view, setView] = useState<View>("design");
  const [mobileNav, setMobileNav] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [profiles, setProfiles] = useState<Json[]>([]);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalog | null>(null);
  const [upstreamModelId, setUpstreamModelId] = useState("");
  const [downstreamModelId, setDownstreamModelId] = useState("");
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [selectedProfile, setSelectedProfile] = useState<InventoryProfile | null>(null);
  const [protocol, setProtocol] = useState(demo ? DEMO_PROTOCOL : "");
  const [intakePackage, setIntakePackage] = useState<Json | null>(demo ? {
    raw_protocol: DEMO_PROTOCOL, objective: "Produce one conservative inventory-constrained first flow screen.",
    ready_for_design: true, missing_question_ids: [], historical_data: null,
    hypotheses: ["oxygen transfer may control the practical rate"], operating_limits: { max_pressure_bar: 3 },
    extracted_batch_fields: { solvent: "acetonitrile", concentration_M: 0.1, reaction_time_min: 5, temperature_C: 21 }
  } : null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [unavailable, setUnavailable] = useState<Record<string, boolean>>({});
  const [useLlm, setUseLlm] = useState(true);
  const [job, setJob] = useState<Job | null>(demo ? { job_id: "demo", status: "completed", progress: 1, phase: "Design complete", messages: [], result: DEMO_RESULT } : null);
  const [result, setResult] = useState<Json | null>(demo ? DEMO_RESULT : null);
  const [resultTab, setResultTab] = useState<ResultTab>("overview");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [runs, setRuns] = useState<Json[]>([]);

  const refreshSystem = async () => {
    try {
      const [h, p] = await Promise.all([
        api<Health>("/api/health"),
        api<{ profiles: Json[] }>("/api/inventory/profiles")
      ]);
      setHealth(h); setProfiles(p.profiles);
      const m = await api<ModelCatalog>("/api/models").catch(() => null);
      if (!m) {
        setModelCatalog(null);
        setError("The running backend is older than this GUI. Restart FlowPilot to enable model routing.");
        return;
      }
      setModelCatalog(m);
      setUpstreamModelId((current) => current || m.defaults.upstream);
      setDownstreamModelId((current) => current || m.defaults.downstream);
    } catch (e) { setError(String((e as Error).message || e)); }
  };

  useEffect(() => { if (!demo) refreshSystem(); }, []);
  useEffect(() => {
    if (!selectedProfileId) { setSelectedProfile(null); return; }
    api<InventoryProfile>(`/api/inventory/profiles/${selectedProfileId}`).then(setSelectedProfile).catch((e) => setError(String(e.message || e)));
  }, [selectedProfileId]);
  useEffect(() => {
    if (!job || !job.job_id || job.status === "completed" || job.status === "failed" || job.job_id === "demo") return;
    const timer = window.setInterval(async () => {
      try {
        const next = await api<Job>(`/api/design/jobs/${job.job_id}`);
        setJob(next);
        if (next.status === "completed" && next.result) { setResult(next.result); setResultTab("overview"); }
        if (next.status === "failed") setError(next.error || "Design failed");
      } catch (e) { setError(String((e as Error).message || e)); }
    }, 1200);
    return () => window.clearInterval(timer);
  }, [job?.job_id, job?.status]);

  const analyze = async (newAnswers: Json[] = []) => {
    if (!protocol.trim()) return;
    setBusy(true); setError("");
    try {
      const response = await api<{ package: Json; pending_questions: Question[] }>("/api/intake/analyze", {
        method: "POST",
        body: JSON.stringify({
          raw_protocol: protocol,
          existing_package: intakePackage,
          answers: newAnswers,
          use_llm: newAnswers.length ? false : useLlm,
          inventory_profile: selectedProfile,
          upstream_model_id: upstreamModelId || undefined
        })
      });
      setIntakePackage(response.package); setQuestions(response.pending_questions);
      setAnswers({}); setUnavailable({});
    } catch (e) { setError(String((e as Error).message || e)); }
    finally { setBusy(false); }
  };

  const submitAnswers = async () => {
    const payload = questions.filter((q) => unavailable[q.question_id] || answers[q.question_id]?.trim()).map((q) => ({
      question_id: q.question_id,
      status: unavailable[q.question_id] ? "unavailable" : "answered",
      answer: unavailable[q.question_id] ? null : answers[q.question_id],
      source: "webapp"
    }));
    await analyze(payload);
  };

  const runDesign = async () => {
    if (!intakePackage?.ready_for_design) return;
    setBusy(true); setError(""); setResult(null);
    try {
      const next = await api<Job>("/api/design/jobs", {
        method: "POST",
        body: JSON.stringify({
          batch_input: protocol,
          intake_package: intakePackage,
          inventory_profile: selectedProfile,
          upstream_model_id: upstreamModelId || undefined,
          downstream_model_id: downstreamModelId || undefined
        })
      });
      setJob(next);
    } catch (e) { setError(String((e as Error).message || e)); }
    finally { setBusy(false); }
  };

  const loadRuns = async () => {
    try { setRuns((await api<{ runs: Json[] }>("/api/runs")).runs); }
    catch (e) { setError(String((e as Error).message || e)); }
  };
  const openRun = async (runId: string) => {
    setBusy(true); setError("");
    try {
      const archived = await api<Json>(`/api/runs/${runId}`);
      setResult(archived);
      const archivedIntake = archived.intake_package || archived.intake_context || null;
      setIntakePackage(archivedIntake);
      if (archivedIntake?.raw_protocol) setProtocol(archivedIntake.raw_protocol);
      setJob({ job_id: `archive-${runId}`, archive_run_id: runId, status: "completed", progress: 1, phase: "Archived design", messages: [], result: archived });
      setResultTab("overview"); setView("design");
    } catch (e) { setError(String((e as Error).message || e)); }
    finally { setBusy(false); }
  };
  useEffect(() => { if (view === "runs") loadRuns(); }, [view]);

  const nav = (next: View) => { setView(next); setMobileNav(false); };
  return <div className="appShell">
    <aside className={`sidebar ${mobileNav ? "open" : ""}`}>
      <div className="brand"><div className="brandMark"><Waypoints size={19}/></div><div><h1>FlowPilot</h1><p>Batch-to-flow design</p></div></div>
      <nav className="nav">
        <button className={view === "design" ? "active" : ""} onClick={() => nav("design")}><FlaskConical size={17}/>Design studio<ChevronRight size={14}/></button>
        <button className={view === "inventory" ? "active" : ""} onClick={() => nav("inventory")}><Archive size={17}/>Inventory<ChevronRight size={14}/></button>
        <button className={view === "runs" ? "active" : ""} onClick={() => nav("runs")}><History size={17}/>Saved runs<ChevronRight size={14}/></button>
        <div className="navDivider"><span>System</span></div>
        <button className={view === "system" ? "active" : ""} onClick={() => nav("system")}><CircleGauge size={17}/>Status<ChevronRight size={14}/></button>
      </nav>
      <div className="sideStatus">
        <div><span className={`liveDot ${health || demo ? "ok" : ""}`}/><b>{health || demo ? "Core online" : "Connecting"}</b></div>
        <p>{health?.models?.chemistry || "Validated pipeline"}</p>
      </div>
      <div className="sideFooter"><ShieldCheck size={14}/>Deterministic gates authoritative</div>
    </aside>

    <main className="workspace">
      <header className="topbar">
        <button className="iconButton mobileMenu" onClick={() => setMobileNav(!mobileNav)} aria-label="Menu"><Menu size={19}/></button>
        <div><h1>{view === "design" ? "Design studio" : view === "inventory" ? "Laboratory inventory" : view === "runs" ? "Saved designs" : "System status"}</h1>
          <p>{view === "design" ? "Standardized intake, constrained design, and experimental refinement" : view === "inventory" ? "Convert laboratory documents into enforceable equipment profiles" : view === "runs" ? "Immutable design records and provenance" : "Models, corpus, and local services"}</p></div>
        <div className="topbarRight"><span className="version">v{health?.version || "2.0"}</span><button className="iconButton" onClick={refreshSystem} title="Refresh"><RefreshCw size={16}/></button></div>
      </header>

      {error && <div className="errorBanner"><AlertTriangle size={17}/><span>{error}</span><button onClick={() => setError("")}><X size={15}/></button></div>}
      {view === "design" && <DesignStudio
        protocol={protocol} setProtocol={setProtocol} intakePackage={intakePackage}
        questions={questions} answers={answers} setAnswers={setAnswers}
        unavailable={unavailable} setUnavailable={setUnavailable} useLlm={useLlm}
        setUseLlm={setUseLlm} profiles={profiles} selectedProfileId={selectedProfileId}
        setSelectedProfileId={setSelectedProfileId} selectedProfile={selectedProfile}
        modelCatalog={modelCatalog} upstreamModelId={upstreamModelId}
        setUpstreamModelId={setUpstreamModelId} downstreamModelId={downstreamModelId}
        setDownstreamModelId={setDownstreamModelId}
        analyze={() => analyze()} submitAnswers={submitAnswers} runDesign={runDesign}
        busy={busy} job={job} result={result} resultTab={resultTab} setResultTab={setResultTab}
      />}
      {view === "inventory" && <InventoryWorkspace profiles={profiles} refresh={refreshSystem} onUse={(profile) => { setSelectedProfile(profile); setSelectedProfileId(profile.profile_id); setView("design"); }}/>} 
      {view === "runs" && <RunsView runs={runs} onOpen={openRun}/>} 
      {view === "system" && <SystemView health={health}/>} 
    </main>
  </div>;
}

function DesignStudio(props: any) {
  const { protocol, setProtocol, intakePackage, questions, answers, setAnswers, unavailable, setUnavailable,
    useLlm, setUseLlm, profiles, selectedProfileId, setSelectedProfileId, selectedProfile,
    modelCatalog, upstreamModelId, setUpstreamModelId, downstreamModelId, setDownstreamModelId,
    analyze, submitAnswers, runDesign, busy, job, result, resultTab, setResultTab } = props;
  const upstreamRoute = modelCatalog?.models.find((model: ModelRoute) => model.route_id === upstreamModelId);
  const downstreamRoute = modelCatalog?.models.find((model: ModelRoute) => model.route_id === downstreamModelId);
  const modelsReady = Boolean(upstreamRoute?.available && downstreamRoute?.available);
  const ready = Boolean(intakePackage?.ready_for_design && modelsReady);
  const answered = intakePackage?.answers?.length || 0;
  const hasEvidence = Boolean(intakePackage?.historical_data && String(intakePackage.historical_data).trim());
  const inventoryBound = selectedProfile || intakePackage?.inventory_profile_snapshot || intakePackage?.inventory_constraints;
  return <div className="viewStack">
    <section className="kpiGrid">
      <Kpi label="Intake" value={ready ? "READY" : intakePackage ? "OPEN" : "NEW"} note={ready ? "package frozen" : `${questions.length || "--"} questions pending`} icon={<ClipboardCheck size={14}/>} tone={ready ? "green" : "amber"}/>
      <Kpi label="Evidence" value={hasEvidence ? "MEASURED" : "NONE"} note="highest authority" icon={<TestTube2 size={14}/>} tone={hasEvidence ? "blue" : "gray"}/>
      <Kpi label="Inventory" value={inventoryBound ? "BOUND" : "OPEN"} note={selectedProfile?.name || intakePackage?.inventory_profile_snapshot?.name || (inventoryBound ? "frozen intake constraints" : "select a laboratory")} icon={<PackageCheck size={14}/>} tone={inventoryBound ? "green" : "amber"}/>
      <Kpi label="Design" value={job?.status?.toUpperCase() || "IDLE"} note={job?.phase || "no active run"} icon={<Activity size={14}/>} tone={job?.status === "completed" ? "green" : job?.status === "failed" ? "red" : "teal"}/>
    </section>

    {!result && <section className="designGrid">
      <div className="panel intakePanel">
        <div className="panelHead"><div><span className="eyebrow">01 / protocol</span><h2><Bot size={17}/>Standardized intake</h2></div><span className="tag">{answered} answers stored</span></div>
        <div className="messageList compact">
          <div className="message agent"><div className="messageMeta"><Bot size={13}/>FlowPilot intake</div><p>Provide the batch protocol. I will extract protocol facts and ask only the fixed questions required to freeze the design input.</p></div>
          {protocol && intakePackage && <div className="message user"><div className="messageMeta"><User size={13}/>Chemist</div><p>{protocol}</p></div>}
        </div>
        <label className="field"><span>Initial batch protocol</span><textarea className="protocolInput" value={protocol} onChange={(e) => setProtocol(e.target.value)} placeholder="Paste the complete batch protocol…"/></label>
        <div className="inlineControls"><label className="toggle"><input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)}/><i/><span>LLM-assisted extraction</span></label><button className="primary" disabled={busy || !protocol.trim()} onClick={analyze}><Sparkles size={15}/>{intakePackage ? "Re-analyze" : "Analyze intake"}</button></div>
      </div>

      <div className="panel questionsPanel">
        <div className="panelHead"><div><span className="eyebrow">02 / clarify</span><h2><ListChecks size={17}/>Fixed questions</h2></div><span className={`statusPill ${ready ? "success" : "warning"}`}>{ready ? <Check size={13}/> : <Clock3 size={13}/>} {ready ? "complete" : `${questions.length} pending`}</span></div>
        {!intakePackage && <Empty icon={<Send size={22}/>} title="Analyze the protocol first" text="The reproducible question bank will appear here."/>}
        {intakePackage && !questions.length && <div className="successState"><CheckCircle2 size={30}/><h3>Design input is frozen</h3><p>All mandatory questions are answered or explicitly unavailable.</p></div>}
        {!!questions.length && <div className="questionScroll">{questions.map((q: Question) => <div className="question" key={q.question_id}>
          <div className="questionHead"><code>{q.question_id}</code><span>{q.section.replaceAll("_", " ")}</span></div><p>{q.question}</p>
          <textarea disabled={unavailable[q.question_id]} value={answers[q.question_id] || ""} onChange={(e) => setAnswers({...answers, [q.question_id]: e.target.value})} placeholder={q.expected_format}/>
          {UNAVAILABLE_QUESTION_IDS.has(q.question_id) && <label className="check"><input type="checkbox" checked={Boolean(unavailable[q.question_id])} onChange={(e) => setUnavailable({...unavailable, [q.question_id]: e.target.checked})}/><span>Explicitly unavailable</span></label>}
        </div>)}</div>}
        {!!questions.length && <button className="primary full" disabled={busy || !questions.some((q: Question) => unavailable[q.question_id] || answers[q.question_id]?.trim())} onClick={submitAnswers}><CheckCircle2 size={15}/>Save answers</button>}
      </div>

      <div className="panel launchPanel">
        <div className="panelHead"><div><span className="eyebrow">03 / constrain</span><h2><Archive size={17}/>Laboratory context</h2></div></div>
        <div className="modelSelectors">
          <div className="modelSelectorsHead"><Bot size={14}/><span>Model routing</span></div>
          <label className="field"><span>Upstream chemistry</span><select aria-label="Upstream chemistry model" value={upstreamModelId} onChange={(e) => setUpstreamModelId(e.target.value)} disabled={!modelCatalog?.models.length}>{modelCatalog?.models.map((model: ModelRoute) => <option disabled={!model.available} value={model.route_id} key={`up-${model.route_id}`}>{model.label}{model.available ? "" : " · unavailable"}</option>)}</select><small>{upstreamRoute?.available ? "Protocol interpretation and chemistry plan" : upstreamRoute?.reason || "Select an available model"}</small></label>
          <label className="field"><span>Downstream and council</span><select aria-label="Downstream and council model" value={downstreamModelId} onChange={(e) => setDownstreamModelId(e.target.value)} disabled={!modelCatalog?.models.length}>{modelCatalog?.models.map((model: ModelRoute) => <option disabled={!model.available} value={model.route_id} key={`down-${model.route_id}`}>{model.label}{model.available ? "" : " · unavailable"}</option>)}</select><small>{downstreamRoute?.available ? "Flow proposal, council agents, revision, and selection" : downstreamRoute?.reason || "Select an available model"}</small></label>
        </div>
        <label className="field"><span>Inventory profile</span><select value={selectedProfileId} onChange={(e) => setSelectedProfileId(e.target.value)}><option value="">No saved profile selected</option>{profiles.map((p: Json) => <option value={p.profile_id} key={p.profile_id}>{p.name} · v{p.version}</option>)}</select></label>
        {selectedProfile ? <InventorySnapshot profile={selectedProfile}/> : <Empty icon={<Archive size={22}/>} title="Inventory not bound" text="Select a validated profile or mark inventory unavailable during intake."/>}
        <div className="authorityOrder"><span>Design authority</span><ol><li>Measured evidence</li><li>Hard constraints</li><li>Protocol facts</li><li>Chemist hypotheses</li><li>Model inference</li></ol></div>
        <button className="runButton" disabled={busy || !ready} onClick={runDesign}><Play size={17}/><span><b>Run FlowPilot design</b><small>{!intakePackage?.ready_for_design ? "Complete intake to unlock" : !modelsReady ? "Select available upstream and downstream models" : "Start constrained pipeline"}</small></span><ArrowRight size={16}/></button>
      </div>
    </section>}

    {job && job.status !== "completed" && !result && <JobProgress job={job}/>} 
    {result && <ResultWorkspace result={result} job={job} tab={resultTab} setTab={setResultTab}/>} 
  </div>;
}

function JobProgress({ job }: { job: Job }) {
  return <section className="panel jobPanel"><div className="jobTitle"><div className="spinner"/><div><h2>{job.phase}</h2><p>Job {job.job_id} · FlowPilot is preserving every stage transition.</p></div><strong>{Math.round((job.progress || 0) * 100)}%</strong></div>
    <div className="progress"><i style={{width: `${Math.max(2, job.progress * 100)}%`}}/></div>
    <div className="logConsole">{job.messages?.slice(-8).map((m, i) => <div key={i}><span>{shortTime(m.time)}</span>{m.message}</div>)}</div></section>;
}

function ResultWorkspace({ result, job, tab, setTab }: { result: Json; job: Job | null; tab: ResultTab; setTab: (tab: ResultTab) => void }) {
  const final = result.final_design || {};
  const params = final.parameters || {};
  const executable = final.status === "executable";
  const tabs: [ResultTab, string, React.ReactNode][] = [
    ["overview", "Overview", <Gauge size={14}/>], ["topology", "Process", <GitBranch size={14}/>],
    ["engineering", "Engineering", <SlidersHorizontal size={14}/>], ["recipe", "Chemistry", <Beaker size={14}/>],
    ["inventory", "Equipment", <Archive size={14}/>], ["council", "Council", <Layers3 size={14}/>],
    ["cycles", "Experiment loop", <RefreshCw size={14}/>], ["json", "JSON", <FileJson size={14}/>]
  ];
  const download = () => {
    if (job?.archive_run_id) window.open(`/api/runs/${job.archive_run_id}/download`, "_blank");
    else if (job?.job_id && job.job_id !== "demo") window.open(`/api/design/jobs/${job.job_id}/download`, "_blank");
  };
  return <section className="resultWorkspace">
    <div className={`resultBanner ${executable ? "executable" : "blocked"}`}><div>{executable ? <CheckCircle2 size={22}/> : <AlertTriangle size={22}/>}<span><b>{executable ? "Executable screening design" : "Design withheld"}</b><small>{result.disposition_rationale || (executable ? "Canonical contract closed against inventory and engineering gates." : "Inspect blocking reasons before execution.")}</small></span></div><div className="bannerActions"><span className="confidence">{result.confidence || "--"} confidence</span><button onClick={download} disabled={job?.job_id === "demo"}><Download size={15}/>JSON</button></div></div>
    <div className="resultTabs">{tabs.map(([id, label, icon]) => <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>{icon}{label}</button>)}</div>
    <div className="resultBody">
      {tab === "overview" && <Overview result={result} params={params} executable={executable}/>} 
      {tab === "topology" && <Topology result={result} job={job}/>} 
      {tab === "engineering" && <Engineering final={final} result={result}/>} 
      {tab === "recipe" && <Chemistry result={result} final={final}/>} 
      {tab === "inventory" && <Equipment result={result}/>} 
      {tab === "council" && <Council result={result}/>} 
      {tab === "cycles" && <ExperimentLoop job={job} result={result}/>} 
      {tab === "json" && <JsonViewer value={result}/>} 
    </div>
  </section>;
}

function Overview({ result, params, executable }: { result: Json; params: Json; executable: boolean }) {
  const gas = (result.final_design?.streams || []).find((s: Json) => s.phase === "gas") || {};
  return <div className="overviewGrid">
    <div className="metricBand">
      <Kpi label="Residence time" value={`${formatValue(params.residence_time_min)} min`} note={`inlet ${formatValue(params.residence_time_inlet_min)} · channel ${formatValue(params.residence_time_in_channel_min)}`} icon={<Clock3 size={14}/>} tone="teal"/>
      <Kpi label="Liquid flow" value={`${formatValue(params.flow_rate_mL_min)} mL/min`} note={`${formatValue(params.concentration_M)} M feed`} icon={<Activity size={14}/>} tone="blue"/>
      <Kpi label="Reactor" value={`${formatValue(params.reactor_volume_mL)} mL`} note={`${params.reactor_type || params.material || "--"} · ${formatValue(params.tubing_ID_mm)} mm ID`} icon={<Box size={14}/>} tone="amber"/>
      <Kpi label="Gas at inlet" value={`${formatValue(gas.gas_flow_sccm)} sccm`} note={`${formatValue(gas.gas_flow_actual_mL_min)} mL/min in-channel`} icon={<Zap size={14}/>} tone="green"/>
    </div>
    <div className="panel unframed"><div className="panelHead"><h2><ClipboardCheck size={17}/>Run conditions</h2><span className={`statusPill ${executable ? "success" : "danger"}`}>{executable ? "authoritative" : "diagnostic"}</span></div><ConditionTable params={params}/></div>
    <div className="panel unframed"><div className="panelHead"><h2><ShieldCheck size={17}/>Closure</h2></div><Closure result={result}/></div>
  </div>;
}

function ConditionTable({ params }: { params: Json }) {
  const rows = [
    ["Concentration", formatValue(params.concentration_M), "M"], ["Temperature", formatValue(params.temperature_C), "°C"],
    ["Pressure", formatValue(params.BPR_bar), "bar"], ["Wavelength", formatValue(params.wavelength_nm), "nm"],
    ["Liquid flow", formatValue(params.flow_rate_mL_min), "mL/min"], ["Reactor volume", formatValue(params.reactor_volume_mL), "mL"],
    ["Tubing ID", formatValue(params.tubing_ID_mm), "mm"], ["Material", formatValue(params.material || params.tubing_material || params.reactor_type), ""]
  ];
  return <div className="conditionTable">{rows.map(([label, value, unit]) => <div key={label}><span>{label}</span><b>{value} <small>{unit}</small></b></div>)}</div>;
}

function Closure({ result }: { result: Json }) {
  const issues = result.final_design?.consistency?.issues || [];
  const checks = [
    ["Final schema", Boolean(result.final_design?.schema_version)],
    ["Numerical closure", Boolean(result.final_design?.consistency?.passed)],
    ["Inventory assignment", Boolean(result.inventory_allocation?.complete ?? result.instrument_manifest?.length)],
    ["Process topology", Boolean(result.process_topology?.unit_operations?.length)],
    ["Autosave", Boolean(result.autosave_dir)]
  ];
  return <div className="checkList">{checks.map(([label, pass]) => <div key={String(label)} className={pass ? "pass" : "fail"}>{pass ? <CheckCircle2 size={16}/> : <AlertTriangle size={16}/>}<span>{label}</span><b>{pass ? "closed" : "review"}</b></div>)}{issues.slice(0, 4).map((issue: Json, i: number) => <div className="issue" key={i}><Info size={16}/><span>{issue.message || String(issue)}</span></div>)}</div>;
}

function Topology({ result, job }: { result: Json; job: Job | null }) {
  const executable = result.final_design?.status === "executable";
  const topology = executable ? result.process_topology : result.diagnostic_topology || result.process_requirements_topology;
  const ops = topology?.unit_operations || [];
  const artifact = job?.archive_run_id
    ? `/api/runs/${job.archive_run_id}/artifacts/${executable ? "process-svg" : "diagnostic-svg"}`
    : job?.job_id && job.job_id !== "demo" ? `/api/design/jobs/${job.job_id}/artifacts/${executable ? "process-svg" : "diagnostic-svg"}` : "";
  const [artifactFailed, setArtifactFailed] = useState(false);
  useEffect(() => setArtifactFailed(false), [artifact]);
  return <div className="topologyLayout"><div className="topologyCanvas">
    <div className="topologyHead"><div><h2>{executable ? "Executable process topology" : "Requirements topology"}</h2><p>{ops.length} declared unit operations · inventory identifiers preserved</p></div>{artifact && <a href={artifact} target="_blank"><ExternalLink size={15}/>Open image</a>}</div>
    {artifact && !artifactFailed ? <img className="processImage" src={artifact} alt="FlowPilot process topology" onError={() => setArtifactFailed(true)}/>: null}
    {(!artifact || artifactFailed) && <ProcessChain operations={ops}/>} 
  </div><div className="operationList"><h3>Unit operations</h3>{ops.map((op: Json, i: number) => <div key={op.operation_id || op.op_id || i}><span>{String(i + 1).padStart(2, "0")}</span><div><b>{op.instrument_name || operationTitle(op)}</b><small>{operationId(op)}</small></div></div>)}</div></div>;
}

function ProcessChain({ operations }: { operations: Json[] }) {
  if (!operations?.length) return <Empty icon={<GitBranch size={22}/>} title="No topology published" text="The deterministic process graph is unavailable for this result."/>;
  return <div className="processChain">{operations.map((op, i) => <React.Fragment key={op.operation_id || op.op_id || i}><div className={`processNode type-${op.op_type}`}><OperationIcon type={op.op_type}/><span>{operationTitle(op)}</span><small>{op.parameters?.volume_mL ? `${op.parameters.volume_mL} mL · ${operationId(op)}` : operationId(op)}</small></div>{i < operations.length - 1 && <ArrowRight size={17} className="chainArrow"/>}</React.Fragment>)}</div>;
}

function OperationIcon({ type }: { type: string }) {
  if (type?.includes("reactor") || type === "photoreactor") return <Atom size={19}/>;
  if (type === "pump" || type === "mfc") return <Activity size={19}/>;
  if (type === "separator") return <GitBranch size={19}/>;
  if (type === "collector") return <TestTube2 size={19}/>;
  if (type === "bpr") return <Gauge size={19}/>;
  return <Waypoints size={19}/>;
}

function Engineering({ final, result }: { final: Json; result: Json }) {
  const stages = final.stages || [];
  const streams = final.streams || [];
  return <div className="splitView"><section><div className="sectionTitle"><h2>Canonical stages</h2><span>{stages.length} stages</span></div>{stages.length ? <table className="dataTable"><thead><tr><th>Stage</th><th>Operation</th><th>Volume</th><th>Residence (inlet / channel)</th><th>Temperature</th></tr></thead><tbody>{stages.map((s: Json, i: number) => <tr key={i}><td>{s.stage_id || s.stage_number || i + 1}</td><td>{s.operation || s.stage_name || s.reaction_name || "reaction"}</td><td>{formatValue(s.reactor_volume_mL ?? s.V_R_mL)} mL</td><td>{formatValue(s.residence_time_inlet_min ?? s.residence_time_min)} / {formatValue(s.residence_time_in_channel_min ?? s.residence_time_min)} min</td><td>{formatValue(s.temperature_C ?? final.parameters?.temperature_C)} °C</td></tr>)}</tbody></table> : <ConditionTable params={final.parameters || {}}/>}</section><section><div className="sectionTitle"><h2>Stream assignments</h2><span>{streams.length} streams</span></div><table className="dataTable"><thead><tr><th>Stream</th><th>Phase</th><th>Composition</th><th>Flow</th></tr></thead><tbody>{streams.map((s: Json, i: number) => <tr key={i}><td>{s.stream_label || i + 1}</td><td><span className={`phase ${s.phase}`}>{s.phase || "liquid"}</span></td><td>{streamContents(s)}</td><td>{s.phase === "gas" ? `${formatValue(s.gas_flow_sccm)} sccm` : `${formatValue(s.flow_rate_mL_min)} mL/min`}</td></tr>)}</tbody></table></section></div>;
}

function Chemistry({ result, final }: { result: Json; final: Json }) {
  const plan = result.chemistry_plan || {};
  return <div className="splitView"><section><div className="sectionTitle"><h2>Chemistry plan</h2><span>upstream analysis</span></div><dl className="detailList"><dt>Reaction</dt><dd>{plan.reaction_name || result.batch_record?.reaction_description || "--"}</dd><dt>Mechanism</dt><dd>{plan.mechanism_type || "--"}</dd><dt>Solvent</dt><dd>{plan.solvent_rationale || result.batch_record?.solvent || "--"}</dd></dl><TagList values={plan.key_risks || plan.safety_flags || []}/></section><section><div className="sectionTitle"><h2>Operating procedure</h2><span>compiled from final design</span></div><ol className="procedure"><li>Prepare the declared feeds at {formatValue(final.parameters?.concentration_M)} M using the assigned inventory.</li><li>Prime liquid and gas paths independently and verify pressure control.</li><li>Set liquid flow to {formatValue(final.parameters?.flow_rate_mL_min)} mL/min and stabilize the reactor at {formatValue(final.parameters?.temperature_C)} °C.</li><li>Start the reaction stage only after the final stream and topology checks are closed.</li><li>Collect after the declared startup volume and document actual run conditions.</li></ol></section></div>;
}

function Equipment({ result }: { result: Json }) {
  const items = result.instrument_manifest || [];
  const unresolved = result.inventory_allocation?.unresolved_requirements || [];
  return <div><div className="sectionTitle"><h2>Assigned equipment</h2><span>{items.length} instruments</span></div>{items.length ? <div className="equipmentGrid">{items.map((item: Json, i: number) => <div className="equipmentItem" key={item.equipment_id || i}><div className="equipmentIcon"><Box size={18}/></div><div><b>{item.name || item.label || item.equipment_id}</b><small>{item.role || item.category || "process equipment"}</small><code>{item.equipment_id || "unassigned"}</code></div></div>)}</div> : <Empty icon={<Archive size={22}/>} title="No final allocation" text="Resolve inventory requirements before execution."/>}{unresolved.map((item: Json, i: number) => <div className="warningRow" key={i}><AlertTriangle size={16}/><span><b>{item.operation_id}</b>{item.reason}</span></div>)}</div>;
}

function Council({ result }: { result: Json }) {
  const messages = result.council_messages || [];
  return <div><div className="sectionTitle"><h2>Council deliberation</h2><span>{result.council_rounds || 0} rounds · {messages.length} records</span></div><div className="councilTimeline">{messages.length ? messages.map((m: Json, i: number) => <div key={i}><div className="agentAvatar">{String(m.agent || m.role || "A").slice(0, 1)}</div><div><b>{m.agent || m.role || `Agent ${i + 1}`}</b><p>{m.content || m.message || m.reasoning || JSON.stringify(m)}</p></div></div>) : <Empty icon={<Layers3 size={22}/>} title="No council transcript" text="No deliberation messages were stored with this result."/>}</div></div>;
}

function ExperimentLoop({ job, result }: { job: Job | null; result: Json }) {
  const params = result.final_design?.parameters || {};
  const [experiments, setExperiments] = useState<Json[]>([]);
  const [form, setForm] = useState<Json>({ residence_time_min: params.residence_time_min || "", flow_rate_mL_min: params.flow_rate_mL_min || "", temperature_C: params.temperature_C || "", reactor_volume_mL: params.reactor_volume_mL || "", yield_pct: "", conversion_pct: "", notes: "" });
  const [refinement, setRefinement] = useState<Json | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const add = () => {
    if (form.yield_pct === "" && form.conversion_pct === "") return;
    const next = { run_id: `web_${experiments.length + 1}`, design_version: experiments.length + 1, actual_conditions: {
      residence_time_min: Number(form.residence_time_min), flow_rate_mL_min: Number(form.flow_rate_mL_min), temperature_C: Number(form.temperature_C), reactor_volume_mL: Number(form.reactor_volume_mL)
    }, outcomes: { yield_pct: form.yield_pct === "" ? null : Number(form.yield_pct), conversion_pct: form.conversion_pct === "" ? null : Number(form.conversion_pct), notes: form.notes }, free_text_observations: form.notes };
    setExperiments([...experiments, next]); setForm({...form, yield_pct: "", conversion_pct: "", notes: ""});
  };
  const refine = async () => {
    setBusy(true); setError("");
    try { setRefinement(await api<Json>("/api/refinement", { method: "POST", body: JSON.stringify({ job_id: job?.job_id === "demo" ? null : job?.job_id, current_result: job?.job_id === "demo" ? result : null, experiments }) })); }
    catch (e) { setError(String((e as Error).message || e)); } finally { setBusy(false); }
  };
  const fields = [["Residence time", "residence_time_min", "min"], ["Liquid flow", "flow_rate_mL_min", "mL/min"], ["Temperature", "temperature_C", "°C"], ["Reactor volume", "reactor_volume_mL", "mL"], ["Yield", "yield_pct", "%"], ["Conversion", "conversion_pct", "%"]];
  return <div className="cycleLayout"><section><div className="sectionTitle"><h2>Add experimental result</h2><span>actual wet-lab values</span></div><div className="formGrid">{fields.map(([label, key, unit]) => <label className="field" key={key}><span>{label} <small>{unit}</small></span><input type="number" value={form[key]} onChange={(e) => setForm({...form, [key]: e.target.value})}/></label>)}</div><label className="field"><span>Observations</span><textarea value={form.notes} onChange={(e) => setForm({...form, notes: e.target.value})} placeholder="Pressure drift, precipitation, gas-liquid stability, impurities…"/></label><div className="actions"><button onClick={add}><Plus size={15}/>Add cycle</button><button className="primary" disabled={!experiments.length || busy} onClick={refine}><Sparkles size={15}/>Refine from {experiments.length} cycle{experiments.length === 1 ? "" : "s"}</button></div>{error && <div className="inlineError">{error}</div>}</section><section><div className="sectionTitle"><h2>Campaign history</h2><span>{experiments.length} cycles</span></div>{experiments.length ? <div className="cycleList">{experiments.map((e, i) => <div key={i}><span>v{i + 1}</span><div><b>{formatValue(e.outcomes.yield_pct ?? e.outcomes.conversion_pct)}%</b><small>{formatValue(e.actual_conditions.residence_time_min)} min · {formatValue(e.actual_conditions.flow_rate_mL_min)} mL/min</small></div><button onClick={() => setExperiments(experiments.filter((_, j) => i !== j))}><Trash2 size={14}/></button></div>)}</div> : <Empty icon={<TestTube2 size={22}/>} title="No wet-lab cycles" text="Add the actual conditions and analytical result from the first run."/>}{refinement && <div className="refinement"><span className="eyebrow">Next design</span><h3>{refinement.decision?.status?.replaceAll("_", " ")}</h3><p>{refinement.decision?.diagnosis}</p><TagList values={refinement.decision?.recommended_actions || []}/><JsonViewer value={refinement.decision?.next_experiment || {}} compact/></div>}</section></div>;
}

function InventoryWorkspace({ profiles, refresh, onUse }: { profiles: Json[]; refresh: () => void; onUse: (profile: InventoryProfile) => void }) {
  const [name, setName] = useState("New laboratory inventory");
  const [laboratory, setLaboratory] = useState("");
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [useLlm, setUseLlm] = useState(true);
  const [profile, setProfile] = useState<InventoryProfile | null>(null);
  const [jsonText, setJsonText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const extract = async () => {
    setBusy(true); setError("");
    const body = new FormData(); body.append("name", name); body.append("laboratory", laboratory); body.append("source_text", text); body.append("use_llm", String(useLlm)); files.forEach((f) => body.append("files", f));
    try { const p = await api<InventoryProfile>("/api/inventory/extract", { method: "POST", body }); setProfile(p); setJsonText(JSON.stringify(p, null, 2)); }
    catch (e) { setError(String((e as Error).message || e)); } finally { setBusy(false); }
  };
  const applyJson = async () => { try { const p = await api<InventoryProfile>("/api/inventory/import", { method: "POST", body: JSON.stringify(JSON.parse(jsonText)) }); setProfile(p); setJsonText(JSON.stringify(p, null, 2)); } catch (e) { setError(String((e as Error).message || e)); } };
  const save = async () => { if (!profile) return; setBusy(true); try { const r = await api<{ profile: InventoryProfile }>("/api/inventory/save", { method: "POST", body: JSON.stringify({ profile }) }); setProfile(r.profile); setJsonText(JSON.stringify(r.profile, null, 2)); refresh(); } catch (e) { setError(String((e as Error).message || e)); } finally { setBusy(false); } };
  const counts = profile ? equipmentCounts(profile.lab_inventory) : [];
  return <div className="viewStack"><section className="inventoryBuild"><div className="panel sourcePanel"><div className="panelHead"><div><span className="eyebrow">Source</span><h2><Upload size={17}/>Import laboratory inventory</h2></div></div><div className="twoFields"><label className="field"><span>Profile name</span><input value={name} onChange={(e) => setName(e.target.value)}/></label><label className="field"><span>Laboratory</span><input value={laboratory} onChange={(e) => setLaboratory(e.target.value)}/></label></div><button className="dropZone" onClick={() => inputRef.current?.click()}><Upload size={24}/><b>Choose PDF, Word, PowerPoint, spreadsheet, or JSON</b><span>{files.length ? files.map((f) => f.name).join(", ") : "Multiple source documents are supported"}</span></button><input ref={inputRef} hidden type="file" multiple accept=".pdf,.docx,.pptx,.xlsx,.json,.txt,.md,.csv" onChange={(e) => setFiles(Array.from(e.target.files || []))}/><label className="field"><span>Additional equipment and constraints</span><textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Example: No inline degasser. Minimum syringe-pump flow is 0.010 mL/min…"/></label><div className="inlineControls"><label className="toggle"><input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)}/><i/><span>LLM-assisted extraction</span></label><button className="primary" disabled={busy || (!files.length && !text.trim())} onClick={extract}><Sparkles size={15}/>Extract inventory</button></div>{error && <div className="inlineError">{error}</div>}</div>
    <div className="panel inventoryPreview"><div className="panelHead"><div><span className="eyebrow">Normalized profile</span><h2><FileJson size={17}/>Schema preview</h2></div>{profile && <span className={`statusPill ${profile.validation?.valid ? "success" : "warning"}`}>{profile.validation?.valid ? "valid" : "review"}</span>}</div>{profile ? <><div className="inventoryCounts">{counts.map(([label, value]) => <div key={label}><strong>{value}</strong><span>{label}</span></div>)}</div><InventorySnapshot profile={profile}/><div className="validationList">{profile.validation?.errors?.map((v, i) => <div className="bad" key={`e${i}`}><AlertTriangle size={14}/>{v}</div>)}{profile.validation?.warnings?.map((v, i) => <div key={`w${i}`}><Info size={14}/>{v}</div>)}</div><div className="actions"><button onClick={applyJson}><RefreshCw size={15}/>Validate edits</button><button className="primary" onClick={save} disabled={busy}><Save size={15}/>Save profile</button><button onClick={() => onUse(profile)}><ArrowRight size={15}/>Use in design</button></div></> : <Empty icon={<FileJson size={24}/>} title="No extracted profile" text="Upload equipment documentation or enter the laboratory constraints."/>}</div></section>{profile && <section className="panel jsonEditor"><div className="panelHead"><h2><FileJson size={17}/>Editable inventory JSON</h2><span className="tag">FlowPilot inventory profile v3</span></div><textarea value={jsonText} onChange={(e) => setJsonText(e.target.value)} spellCheck={false}/></section>}<section className="panel savedProfiles"><div className="panelHead"><h2><Archive size={17}/>Saved profiles</h2><span className="tag">{profiles.length} profiles</span></div><div className="profileRows">{profiles.map((p) => <div key={p.profile_id}><div><b>{p.name}</b><small>{p.laboratory || "Laboratory not specified"} · version {p.version}</small></div><span>{p.reactor_count} reactors · {p.pump_count} pumps</span><button onClick={async () => onUse(await api<InventoryProfile>(`/api/inventory/profiles/${p.profile_id}`))}>Use <ArrowRight size={14}/></button></div>)}</div></section></div>;
}

function InventorySnapshot({ profile }: { profile: InventoryProfile }) {
  const inv = profile.lab_inventory || {};
  return <div className="snapshot"><div><span>Reactors</span><b>{inv.reactors?.map((r: Json) => `${r.name || r.system} (${r.volume_mL} mL)`).join(", ") || "none"}</b></div><div><span>Pumps</span><b>{inv.pumps?.map((p: Json) => p.name || p.type).join(", ") || "none"}</b></div><div><span>Pressure</span><b>{inv.pressure_controllers?.map((p: Json) => p.name).join(", ") || inv.BPR_available?.join(", ") || "none"}</b></div><div><span>Constraints</span><b>{Object.keys(profile.operating_constraints || {}).length} declared</b></div></div>;
}

function equipmentCounts(inv: Json): [string, number][] {
  return [["reactors", inv.reactors?.length || 0], ["pumps", inv.pumps?.length || 0], ["tubing", inv.tubing?.length || 0], ["lights", inv.light_sources?.length || 0], ["gas devices", inv.gas_hardware?.length || 0], ["accessories", ["mixers","separators","connectors","collectors"].reduce((n, k) => n + (inv[k]?.length || 0), 0)]];
}

function RunsView({ runs, onOpen }: { runs: Json[]; onOpen: (runId: string) => void }) {
  return <section className="panel runsPanel"><div className="panelHead"><div><span className="eyebrow">Provenance</span><h2><History size={17}/>Autosaved designs</h2></div><span className="tag">{runs.length} recent runs</span></div>{runs.length ? <table className="dataTable"><thead><tr><th>Run</th><th>Reaction</th><th>Status</th><th>Residence</th><th>Flow</th><th>Reactor</th><th>Open</th></tr></thead><tbody>{runs.map((r) => <tr key={r.run_id}><td><b>{r.run_id}</b><small>{r.source}</small></td><td>{r.reaction || "--"}</td><td><span className={`statusPill ${r.final_design_status === "executable" ? "success" : "danger"}`}>{r.final_design_status}</span></td><td>{formatValue(r.residence_time_min)} min</td><td>{formatValue(r.flow_rate_mL_min)} mL/min</td><td>{formatValue(r.reactor_volume_mL)} mL</td><td><button className="openRun" onClick={() => onOpen(r.run_id)} title="Open saved design"><ExternalLink size={14}/></button></td></tr>)}</tbody></table> : <Empty icon={<History size={24}/>} title="No autosaved runs" text="Completed web and Streamlit designs appear here automatically."/>}</section>;
}

function SystemView({ health }: { health: Health | null }) {
  const models = health?.models || {};
  return <div className="systemGrid"><section className="panel"><div className="panelHead"><h2><Activity size={17}/>Core services</h2><span className="statusPill success">online</span></div><div className="systemRows"><div><span>API</span><b>FastAPI · local</b><CheckCircle2 size={16}/></div><div><span>Corpus</span><b>{health?.corpus_records || 0} records</b><Database size={16}/></div><div><span>Inventory</span><b>{health?.inventory_profiles || 0} profiles</b><Archive size={16}/></div></div></section><section className="panel"><div className="panelHead"><h2><Bot size={17}/>Model routing</h2></div><div className="modelRoute"><div><span>Intake</span><b>{models.intake || "--"}</b></div><ChevronDown size={16}/><div><span>Chemistry</span><b>{models.chemistry || "--"}</b></div><ChevronDown size={16}/><div><span>Translation</span><b>{models.translation || "--"}</b></div><ChevronDown size={16}/><div><span>Council</span><b>{models.council_model || "--"}</b></div></div></section><section className="panel fullSpan"><div className="panelHead"><h2><ShieldCheck size={17}/>Authority boundaries</h2></div><div className="authorityDiagram"><div><TestTube2 size={20}/><b>Measured evidence</b><span>Overrides model assumptions</span></div><ArrowRight size={18}/><div><Archive size={20}/><b>Hard constraints</b><span>Validated deterministically</span></div><ArrowRight size={18}/><div><FileText size={20}/><b>Protocol facts</b><span>Frozen during intake</span></div><ArrowRight size={18}/><div><Bot size={20}/><b>Model inference</b><span>Advisory until closure</span></div></div></section></div>;
}

function JsonViewer({ value, compact = false }: { value: any; compact?: boolean }) {
  const [copied, setCopied] = useState(false);
  const text = JSON.stringify(value, null, 2);
  const copy = async () => { await navigator.clipboard.writeText(text); setCopied(true); window.setTimeout(() => setCopied(false), 1200); };
  return <div className={`jsonViewer ${compact ? "compact" : ""}`}><button onClick={copy}>{copied ? <Check size={14}/> : <FileText size={14}/>} {copied ? "Copied" : "Copy"}</button><pre>{text}</pre></div>;
}

function TagList({ values }: { values: any[] }) { return <div className="tagList">{values.map((v, i) => <span key={i}>{typeof v === "string" ? v : JSON.stringify(v)}</span>)}</div>; }
function Empty({ icon, title, text }: { icon: React.ReactNode; title: string; text: string }) { return <div className="emptyState">{icon}<b>{title}</b><span>{text}</span></div>; }

createRoot(document.getElementById("root")!).render(<App/>);
