import { useEffect, useMemo, useState } from "react";
import { supabase } from "../../lib/supabaseClient";
import type { PublishPipeStepOutput, ReviewResultsStepOutput, SelectDatasetStepOutput, TestPredictionStepOutput, TrainModelsStepOutput } from "../../lib/pipes";

type PublishPipeStepProps = {
  pipeId: string;
  selectDatasetOutput: SelectDatasetStepOutput | null;
  trainModelsOutput: TrainModelsStepOutput | null;
  reviewResultsOutput: ReviewResultsStepOutput | null;
  testPredictionOutput: TestPredictionStepOutput | null;
  initialPublishOutput: PublishPipeStepOutput | null;
  onCompleted: (output: PublishPipeStepOutput) => void;
  onBackToReviewResults: () => void;
  onBackToTestPrediction: () => void;
};

type PublicationStatus = {
  status: "draft" | "live" | "unpublished";
  publication: null | {
    status: "live" | "unpublished";
    public_id: string;
    version: number;
    endpoint_url: string;
    api_key_prefix: string;
    published_at?: string;
    unpublished_at?: string;
    last_key_rotated_at?: string;
    model_snapshot?: Record<string, unknown>;
    active_deployment_id?: string;
  };
};

type PublishResponse = {
  publication: PublishPipeStepOutput;
  api_key: string | null;
  api_key_is_new: boolean;
  request_example: { input: Record<string, unknown> };
  response_example: Record<string, unknown>;
};

function displayValue(value: unknown) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : "—";
  return String(value);
}

function metricLabel(metricName: string | null | undefined) {
  if (!metricName) return "Validation metric";
  if (metricName === "mae") return "Average error";
  if (metricName === "rmse") return "RMSE";
  if (metricName === "r2") return "R²";
  if (metricName === "f1_macro" || metricName === "f1_weighted") return "F1 score";
  return metricName.replaceAll("_", " ");
}

export function PublishPipeStep({ pipeId, selectDatasetOutput, trainModelsOutput, reviewResultsOutput, testPredictionOutput, initialPublishOutput, onCompleted, onBackToReviewResults, onBackToTestPrediction }: PublishPipeStepProps) {
  const [status, setStatus] = useState<PublicationStatus | null>(null);
  const [publishOutput, setPublishOutput] = useState<PublishPipeStepOutput | null>(initialPublishOutput);
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [requestExample, setRequestExample] = useState<Record<string, unknown>>({ input: {} });
  const [responseExample, setResponseExample] = useState<Record<string, unknown>>({});
  const [activeTab, setActiveTab] = useState<"endpoint" | "request" | "response">("endpoint");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const serviceUrl = import.meta.env.VITE_ML_TRAINING_API_URL as string | undefined;
  const livePublication = status?.publication ?? (publishOutput?.publication_status === "live" ? {
    status: "live" as const,
    public_id: publishOutput.public_id,
    version: publishOutput.active_version,
    endpoint_url: publishOutput.endpoint_url,
    api_key_prefix: publishOutput.api_key_prefix,
    published_at: publishOutput.published_at,
    active_deployment_id: publishOutput.active_deployment_id,
  } : null);
  const requestJson = useMemo(() => JSON.stringify(requestExample, null, 2), [requestExample]);
  const responseJson = useMemo(() => JSON.stringify(responseExample, null, 2), [responseExample]);
  const curlExample = livePublication ? `curl -X POST ${livePublication.endpoint_url} \\\n  -H "Content-Type: application/json" \\\n  -H "X-MLP-API-Key: ${apiKey ?? livePublication.api_key_prefix}" \\\n  -d '${JSON.stringify(requestExample)}'` : "";

  useEffect(() => { void loadStatus(); }, []);

  async function serviceFetch(path: string, body: Record<string, unknown>) {
    if (!serviceUrl) throw new Error("Prediction service is not configured.");
    const { data } = await supabase.auth.getSession();
    const token = data.session?.access_token;
    if (!token) throw new Error("Please sign in again before publishing.");
    const response = await fetch(`${serviceUrl.replace(/\/$/, "")}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify(body),
    });
    const payload = await response.json() as { detail?: string; error?: string };
    if (!response.ok) throw new Error(payload.detail ?? payload.error ?? "Publication request failed.");
    return payload;
  }

  async function loadStatus() {
    setError(null);
    try {
      const payload = await serviceFetch("/published-pipe-status", { pipe_id: pipeId }) as PublicationStatus;
      setStatus(payload);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to load publication status.");
    }
  }

  async function publish() {
    if (!reviewResultsOutput || !testPredictionOutput) return;
    setLoading(true);
    setError(null);
    try {
      const payload = await serviceFetch("/publish-pipe", { pipe_id: pipeId, review_results_artifact_id: reviewResultsOutput.review_results_artifact_id, test_prediction_artifact_id: testPredictionOutput.test_prediction_artifact_id }) as PublishResponse;
      setPublishOutput(payload.publication);
      setApiKey(payload.api_key);
      setRequestExample(payload.request_example);
      setResponseExample(payload.response_example);
      onCompleted(payload.publication);
      await loadStatus();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to publish this pipe.");
    } finally {
      setLoading(false);
    }
  }

  async function rotateKey() {
    if (!window.confirm("Rotate the API key? The old key will stop working immediately.")) return;
    setLoading(true);
    setError(null);
    try {
      const payload = await serviceFetch("/rotate-published-pipe-key", { pipe_id: pipeId }) as PublicationStatus & { api_key: string };
      setApiKey(payload.api_key);
      setStatus(payload);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to rotate the API key.");
    } finally {
      setLoading(false);
    }
  }

  async function unpublish() {
    if (!window.confirm("Unpublish this pipe? External workflow calls will stop working.")) return;
    setLoading(true);
    setError(null);
    try {
      const payload = await serviceFetch("/unpublish-pipe", { pipe_id: pipeId }) as PublicationStatus;
      setStatus(payload);
      if (publishOutput) onCompleted({ ...publishOutput, publication_status: "unpublished", unpublished_at: new Date().toISOString() });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to unpublish this pipe.");
    } finally {
      setLoading(false);
    }
  }

  if (!reviewResultsOutput || reviewResultsOutput.review_acknowledged === false) {
    return <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Review results first.</h2><p className="mt-2 text-sm text-black/60">You need to acknowledge the recommended model before publishing.</p><button type="button" onClick={onBackToReviewResults} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Back to Review results</button></section>;
  }

  if (!testPredictionOutput) {
    return <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Test prediction first.</h2><p className="mt-2 text-sm text-black/60">Run at least one real prediction before publishing this pipe.</p><button type="button" onClick={onBackToTestPrediction} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Back to Test prediction</button></section>;
  }

  return <div className="mt-6 space-y-6">
    <section><h2 className="text-2xl font-semibold tracking-tight">Publish pipe</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-black/60">Publish this tested model as a secure prediction endpoint for your workflows.</p></section>
    <section className="rounded-3xl border border-black/10 bg-white/60 p-5"><div className="flex items-center justify-between"><h3 className="font-semibold">Readiness summary</h3><span className={`rounded-full px-3 py-1 text-xs font-medium ${livePublication?.status === "live" ? "bg-emerald-500/10 text-emerald-700" : "bg-black/5 text-black/60"}`}>{livePublication?.status === "live" ? "Live" : "Draft"}</span></div><dl className="mt-4 grid gap-3 text-sm md:grid-cols-2"><div><dt className="text-black/45">Dataset</dt><dd className="font-medium">{selectDatasetOutput?.source_label ?? "Unavailable"}</dd></div><div><dt className="text-black/45">Target</dt><dd className="font-medium">{trainModelsOutput?.target_column ?? "Unavailable"}</dd></div><div><dt className="text-black/45">Task</dt><dd className="font-medium">{trainModelsOutput?.task_type?.replaceAll("_", " ") ?? "Unavailable"}</dd></div><div><dt className="text-black/45">Selected model</dt><dd className="font-medium">{trainModelsOutput?.recommended_model_name ?? "Unavailable"}</dd></div><div><dt className="text-black/45">Validation metric</dt><dd className="font-medium">{metricLabel(trainModelsOutput?.primary_metric_name)}: {displayValue(trainModelsOutput?.primary_metric_value)}</dd></div><div><dt className="text-black/45">Test prediction</dt><dd className="font-medium">Completed</dd></div></dl>{!livePublication ? <p className="mt-4 rounded-2xl bg-black/5 p-3 text-sm text-black/60">Draft — this pipe is not available to external workflows yet.</p> : <p className="mt-4 rounded-2xl bg-emerald-500/10 p-3 text-sm text-emerald-800">Your pipe is ready to receive prediction requests.</p>}{error ? <p className="mt-4 rounded-2xl bg-red-500/10 p-3 text-sm text-red-700">{error}</p> : null}<button type="button" onClick={publish} disabled={loading} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white disabled:bg-black/30">{loading ? "Publishing pipe…" : livePublication ? "Republish new version" : "Publish pipe"}</button></section>

    {apiKey ? <section className="rounded-3xl border border-amber-200 bg-amber-50 p-5"><h3 className="font-semibold text-amber-900">Copy this API key now. For security, it will not be shown again.</h3><div className="mt-3 flex flex-wrap gap-2"><code className="rounded-2xl bg-white px-3 py-2 text-sm">{showKey ? apiKey : "•".repeat(24)}</code><button type="button" onClick={() => setShowKey((value) => !value)} className="rounded-full border border-amber-300 px-3 py-2 text-xs font-medium">{showKey ? "Hide" : "Show"}</button><button type="button" onClick={() => void navigator.clipboard.writeText(apiKey)} className="rounded-full bg-black px-3 py-2 text-xs font-medium text-white">Copy key</button></div></section> : null}

    {livePublication ? <section className="rounded-3xl border border-black/10 bg-white/60 p-5"><div className="flex flex-wrap gap-2">{(["endpoint", "request", "response"] as const).map((tab) => <button key={tab} type="button" onClick={() => setActiveTab(tab)} className={`rounded-full px-3 py-1.5 text-sm ${activeTab === tab ? "bg-black text-white" : "bg-white text-black/60"}`}>{tab === "endpoint" ? "Endpoint" : tab === "request" ? "Request" : "Response"}</button>)}</div>{activeTab === "endpoint" ? <div className="mt-5 space-y-3 text-sm"><p><span className="font-medium">Method:</span> POST</p><p><span className="font-medium">URL:</span> <code className="break-all">{livePublication.endpoint_url}</code></p><p><span className="font-medium">Authentication:</span> X-MLP-API-Key: {livePublication.api_key_prefix}</p><p><span className="font-medium">Version:</span> {livePublication.version}</p></div> : null}{activeTab === "request" ? <div className="mt-5"><h3 className="font-semibold">Send a request</h3><pre className="mt-3 overflow-x-auto rounded-2xl bg-black p-4 text-xs text-white">{curlExample}</pre><div className="mt-4 rounded-2xl bg-black/5 p-4 text-sm"><p className="font-medium">n8n HTTP Request setup</p><ul className="mt-2 list-disc space-y-1 pl-5 text-black/65"><li>Method: POST</li><li>URL: {livePublication.endpoint_url}</li><li>Header: X-MLP-API-Key</li><li>Body Content Type: JSON</li><li>Body: {`{ "input": { ... } }`}</li></ul></div></div> : null}{activeTab === "response" ? <div className="mt-5"><h3 className="font-semibold">What your workflow receives</h3><pre className="mt-3 overflow-x-auto rounded-2xl bg-black p-4 text-xs text-white">{responseJson || requestJson}</pre><button type="button" onClick={() => void navigator.clipboard.writeText(responseJson)} className="mt-3 rounded-full border border-black/10 px-3 py-1.5 text-xs font-medium">Copy response</button></div> : null}</section> : null}

    {livePublication ? <section className="rounded-3xl border border-black/10 bg-white/60 p-5"><h3 className="font-semibold">Manage publication</h3><div className="mt-4 flex flex-wrap gap-3"><button type="button" onClick={rotateKey} disabled={loading} className="rounded-full border border-black/10 px-4 py-2 text-sm font-medium">Rotate API key</button><button type="button" onClick={unpublish} disabled={loading} className="rounded-full border border-red-500/20 bg-red-500/5 px-4 py-2 text-sm font-medium text-red-700">Unpublish pipe</button></div></section> : null}
  </div>;
}
