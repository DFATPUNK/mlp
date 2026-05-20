import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { persistSelectDatasetStep } from "../lib/builder/builderPersistence";
import { profileDataset } from "../lib/builder/datasetProfiling";
import { fetchHuggingFaceDatasetRows } from "../lib/builder/huggingFaceDatasets";
import { getMockConnections, getProviderDatasets } from "../lib/builder/mockDatasetProviders";
import type { BuilderPipeType } from "../types/pipe";
import type { DatasetProvider } from "../types/builder";

const steps = ["Select dataset", "Clean data", "Split data", "Choose target", "Train models", "Review results", "Test prediction", "Publish pipe"];
const providerLabels: Record<DatasetProvider, string> = { huggingface: "Hugging Face Datasets", airtable: "Airtable", google_sheets: "Google Sheets" };

export function PipeBuilderPage() {
  const { user } = useAuth();
  const { pipeType } = useParams();
  const normalizedPipeType: BuilderPipeType = pipeType === "tabular_regression" ? "tabular_regression" : "tabular_classification";
  const [provider, setProvider] = useState<DatasetProvider>("huggingface");
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [loadingRows, setLoadingRows] = useState(false);
  const [rowsError, setRowsError] = useState<string | null>(null);
  const [loadedRows, setLoadedRows] = useState<Record<string, unknown>[]>([]);
  const [acceptedArtifactId, setAcceptedArtifactId] = useState<string | null>(null);
  const [pipeId, setPipeId] = useState<string | null>(() => sessionStorage.getItem(`builder_pipe_id_${normalizedPipeType}`));

  const connections = getMockConnections(provider);
  const datasets = getProviderDatasets(provider);
  const selectedDataset = datasets.find((dataset) => dataset.externalId === selectedDatasetId) ?? datasets[0];

  useEffect(() => {
    let cancelled = false;
    async function loadRows() {
      setRowsError(null);
      setLoadingRows(true);
      try {
        if (provider === "huggingface") {
          const rows = await fetchHuggingFaceDatasetRows(selectedDataset);
          if (!cancelled) setLoadedRows(rows);
        } else {
          setLoadedRows(selectedDataset.rows);
        }
      } catch {
        if (!cancelled) setRowsError("We could not load this Hugging Face dataset right now.");
      } finally {
        if (!cancelled) setLoadingRows(false);
      }
    }
    void loadRows();
    return () => {
      cancelled = true;
    };
  }, [provider, selectedDataset]);

  const profile = useMemo(() => profileDataset(loadedRows), [loadedRows]);

  async function handleUseDataset() {
    if (!user || !profile.eligibility.eligible) return;
    setSaving(true);
    try {
      const result = await persistSelectDatasetStep({ userId: user.id, pipeType: normalizedPipeType, pipeId, provider, connection: connections[0], dataset: { ...selectedDataset, rows: loadedRows }, profile });
      setAcceptedArtifactId(result.artifactId);
      setPipeId(result.pipeId);
      if (result.pipeId) sessionStorage.setItem(`builder_pipe_id_${normalizedPipeType}`, result.pipeId);
    } finally {
      setSaving(false);
    }
  }

  return <div className="grid gap-8 lg:grid-cols-[280px_1fr]">{/* unchanged layout */}
    <aside className="rounded-3xl border border-black/10 bg-white/60 p-5"><p className="text-xs uppercase tracking-[0.2em] text-black/40">Builder steps</p><ol className="mt-4 space-y-3">{steps.map((step, idx) => <li key={step} className={`rounded-2xl px-3 py-2 text-sm ${idx===0?"bg-black text-white":"text-black/60"}`}>{idx + 1}. {step}</li>)}</ol></aside>
    <div><Link to="/app/pipes/new" className="text-sm text-black/50 hover:text-black">← Back to pipe types</Link><h1 className="mt-4 text-4xl font-semibold tracking-tight">Step 1: Select dataset</h1><p className="mt-2 text-black/60">Connect data, preview it, and confirm it is eligible for the MVP before continuing.</p>
      <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="font-semibold">1) Connect data</h2><div className="mt-4 grid gap-3 md:grid-cols-3">{(["huggingface","airtable","google_sheets"] as DatasetProvider[]).map((item)=><button key={item} onClick={()=>{setProvider(item);setSelectedDatasetId("");}} className={`rounded-2xl border px-4 py-3 text-left ${provider===item?"border-black bg-white":"border-black/10"}`}><p className="font-medium">{providerLabels[item]}</p><p className="text-xs text-black/50">{item==="huggingface"?"Live read-only":"Alpha mock, OAuth coming soon"}</p></button>)}</div><div className="mt-4 rounded-2xl border border-black/10 p-4"><p className="text-sm text-black/50">Connected account</p><p className="mt-1 font-medium">{connections[0].providerAccountLabel}</p></div></section>
      <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="font-semibold">2) Choose a dataset source</h2><div className="mt-4 grid gap-3">{datasets.map((dataset)=><button key={dataset.externalId} onClick={()=>setSelectedDatasetId(dataset.externalId)} className={`rounded-2xl border p-4 text-left ${selectedDataset.externalId===dataset.externalId?"border-black bg-white":"border-black/10"}`}><p className="font-medium">{dataset.name}</p><p className="text-sm text-black/50">{dataset.sourceLabel}</p></button>)}</div></section>
      {rowsError ? <p className="mt-4 rounded-2xl bg-red-500/10 px-4 py-3 text-sm text-red-700">{rowsError}</p> : null}
      <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="font-semibold">3) Preview data</h2>{loadingRows ? <p className="mt-3 text-sm text-black/50">Loading rows…</p> : <div className="mt-4 overflow-x-auto"><table className="min-w-full text-sm"><thead><tr>{Object.keys(profile.preview[0] ?? {}).map((column)=><th key={column} className="border-b border-black/10 px-3 py-2 text-left font-medium">{column}</th>)}</tr></thead><tbody>{profile.preview.slice(0,5).map((row,idx)=><tr key={idx}>{Object.entries(row).map(([k,v])=><td key={k} className="border-b border-black/5 px-3 py-2 text-black/70">{String(v)}</td>)}</tr>)}</tbody></table></div>}</section>
      <section className="mt-6 grid gap-6 lg:grid-cols-2"><div className="rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="font-semibold">4) Dataset profile</h2><p className="mt-3 text-sm text-black/60">{profile.rowCount} rows • {profile.columnCount} columns</p></div><div className="rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="font-semibold">5) Eligibility</h2><p className={`mt-2 text-sm font-medium ${profile.eligibility.eligible?"text-emerald-700":"text-red-700"}`}>{profile.eligibility.eligible ? "This dataset is eligible." : "This dataset is not eligible yet."}</p><ul className="mt-3 list-disc space-y-1 pl-4 text-sm text-red-700">{profile.eligibility.blocking_issues.map((issue)=><li key={`${issue.code}-${issue.column ?? ""}`}>{issue.message}</li>)}</ul><button onClick={handleUseDataset} disabled={!profile.eligibility.eligible || saving || loadingRows} className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-black/20">{saving ? "Saving…" : "Use this dataset"}</button>{acceptedArtifactId ? <div className="mt-4 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800">Dataset saved. Continue to Clean data.<p className="mt-2 text-xs">dataset_artifact_id: {acceptedArtifactId}</p></div> : null}</div></section>
    </div></div>;
}
