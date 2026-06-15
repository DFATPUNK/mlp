import { useEffect, useMemo, useRef, useState } from "react";
import { getArtifactById, type ReviewResultsStepOutput, type TrainModelsStepOutput } from "../../lib/pipes";
import { supabase } from "../../lib/supabaseClient";

type ReviewResultsStepProps = {
  pipeId: string;
  trainModelsOutput: TrainModelsStepOutput | null;
  initialReviewResultsOutput: ReviewResultsStepOutput | null;
  onCompleted: (output: ReviewResultsStepOutput) => void;
  onBackToTrainModels: () => void;
};

type ReviewContent = {
  previous_trained_models_artifact_id: string;
  task_type: "tabular_classification" | "tabular_regression";
  target_column: string;
  recommended_model: {
    model_id: string;
    model_name: string;
    primary_metric_name: string;
    primary_metric_value: number;
    metrics: Record<string, unknown>;
    explanation: string;
    pros: string[];
    cons: string[];
    warnings: string[];
  };
  model_comparison: Array<{
    model_id: string;
    model_name: string;
    primary_metric_name: string;
    primary_metric_value: number | null;
    metrics: Record<string, unknown>;
    training_time_ms: number;
    status: "completed" | "failed";
  }>;
  plain_english_summary: string;
  validation_summary: { rows_evaluated: number; notes: string[] };
  charts: Array<{ chart_key: string; title: string; kind: string; image_format: "png_base64"; image_base64: string; description: string }>;
  prediction_examples: { columns: string[]; rows: Record<string, unknown>[] };
};

type ReviewResponse = ReviewContent & { review_results_artifact_id: string };

function metricLabel(metricName: string) {
  if (metricName === "mae") return "Average error";
  if (metricName === "rmse") return "RMSE";
  if (metricName === "r2") return "R²";
  if (metricName === "f1_macro" || metricName === "f1_weighted") return "F1 score";
  return metricName.replaceAll("_", " ");
}

function formatMetric(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—";
}

function displayValue(value: unknown) {
  if (value === null) return "null";
  if (value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(3) : "—";
  return String(value);
}

function isReviewContent(value: unknown): value is ReviewContent {
  return !!value && typeof value === "object" && "recommended_model" in value && "charts" in value;
}

export function ReviewResultsStep({ pipeId, trainModelsOutput, initialReviewResultsOutput, onCompleted, onBackToTrainModels }: ReviewResultsStepProps) {
  const [reviewOutput, setReviewOutput] = useState<ReviewResultsStepOutput | null>(initialReviewResultsOutput);
  const [content, setContent] = useState<ReviewContent | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const startedRef = useRef(false);
  const output = reviewOutput ?? initialReviewResultsOutput;

  useEffect(() => {
    if (!initialReviewResultsOutput || content) return;
    let mounted = true;
    void getArtifactById(initialReviewResultsOutput.review_results_artifact_id)
      .then((artifact) => {
        if (!mounted) return;
        if (isReviewContent(artifact?.content)) setContent(artifact.content);
      })
      .catch(() => {
        if (mounted) setError("We could not load the saved review results artifact.");
      });
    return () => {
      mounted = false;
    };
  }, [initialReviewResultsOutput, content]);

  useEffect(() => {
    if (!trainModelsOutput || output || startedRef.current) return;
    startedRef.current = true;
    void generateReview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainModelsOutput, output]);

  const keyMetrics = useMemo(() => {
    const metrics = content?.recommended_model.metrics ?? {};
    const wanted = content?.task_type === "tabular_regression" ? ["mae", "rmse", "r2"] : ["accuracy", "f1_macro", "f1_weighted"];
    return wanted.filter((key) => key in metrics).map((key) => ({ key, value: metrics[key] }));
  }, [content]);

  async function generateReview() {
    if (!trainModelsOutput) return;
    setLoading(true);
    setError(null);
    try {
      const trainingApiUrl = import.meta.env.VITE_ML_TRAINING_API_URL as string | undefined;
      if (!trainingApiUrl) throw new Error("Training service is not configured.");
      const { data } = await supabase.auth.getSession();
      const token = data.session?.access_token;
      if (!token) throw new Error("Please sign in again before reviewing results.");
      const response = await fetch(`${trainingApiUrl.replace(/\/$/, "")}/review-results`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ pipe_id: pipeId, trained_models_artifact_id: trainModelsOutput.trained_models_artifact_id }),
      });
      const payload = await response.json() as ReviewResponse & { detail?: string; error?: string };
      if (!response.ok) throw new Error(payload.detail ?? payload.error ?? "Unable to generate review results.");
      const nextOutput: ReviewResultsStepOutput = {
        step_key: "review_results",
        status: "completed",
        review_results_artifact_id: payload.review_results_artifact_id,
        previous_trained_models_artifact_id: trainModelsOutput.trained_models_artifact_id,
        recommended_model_name: payload.recommended_model.model_name,
        primary_metric_name: payload.recommended_model.primary_metric_name,
        primary_metric_value: payload.recommended_model.primary_metric_value,
        storage: { format: "json", uri: `artifact:${payload.review_results_artifact_id}` },
      };
      setContent(payload);
      setReviewOutput(nextOutput);
      onCompleted(nextOutput);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to generate review results.");
    } finally {
      setLoading(false);
    }
  }

  if (!trainModelsOutput) {
    return <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Train models before reviewing results.</h2><button type="button" onClick={onBackToTrainModels} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Back to Train models</button></section>;
  }

  if (loading && !content) {
    return <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Generating your results review…</h2><p className="mt-2 text-sm text-black/60">MLP is reading the trained model artifact, checking validation predictions, and creating charts. No models are being retrained.</p></section>;
  }

  if (!content) {
    return <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Review results</h2><p className="mt-2 text-sm text-black/60">MLP can generate a plain-English review from the trained models artifact.</p>{error ? <p className="mt-4 rounded-2xl bg-red-500/10 px-4 py-3 text-sm text-red-700">{error}</p> : null}<button type="button" onClick={generateReview} className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Generate review</button></section>;
  }

  const warnings = [...(content.recommended_model.warnings ?? []), ...(content.validation_summary.notes ?? [])];
  return <div>
    <section className="mt-6 rounded-3xl border border-emerald-200 bg-emerald-50 p-6"><h2 className="text-lg font-semibold text-emerald-900">Review generated.</h2><p className="mt-2 text-sm text-emerald-800">MLP reviewed the real validation results from the trained models artifact.</p><dl className="mt-4 grid gap-2 text-sm text-emerald-900 md:grid-cols-2"><div><dt className="font-medium">Recommended model</dt><dd>{content.recommended_model.model_name}</dd></div><div><dt className="font-medium">Primary metric</dt><dd>{metricLabel(content.recommended_model.primary_metric_name)}: {formatMetric(content.recommended_model.primary_metric_value)}</dd></div><div><dt className="font-medium">Rows reviewed</dt><dd>{content.validation_summary.rows_evaluated}</dd></div>{output ? <div><dt className="font-medium">Review results artifact ID</dt><dd className="font-mono text-xs">{output.review_results_artifact_id}</dd></div> : null}</dl></section>

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Recommended model</h2><p className="mt-3 text-sm text-black/70">{content.plain_english_summary}</p><p className="mt-3 text-sm text-black/60"><span className="font-medium text-black">Why this model?</span> {content.recommended_model.explanation}</p></section>

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Key metrics</h2><div className="mt-4 grid gap-3 md:grid-cols-3">{keyMetrics.map((metric) => <div key={metric.key} className="rounded-2xl border border-black/10 bg-white/70 p-4"><p className="text-xs uppercase tracking-[0.16em] text-black/40">{metricLabel(metric.key)}</p><p className="mt-2 text-2xl font-semibold">{formatMetric(metric.value)}</p></div>)}</div></section>

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Model comparison</h2><div className="mt-4 overflow-x-auto"><table className="min-w-full text-sm"><thead><tr><th className="border-b border-black/10 px-3 py-2 text-left font-medium">Model</th><th className="border-b border-black/10 px-3 py-2 text-left font-medium">Status</th><th className="border-b border-black/10 px-3 py-2 text-left font-medium">Metric</th><th className="border-b border-black/10 px-3 py-2 text-left font-medium">Training time</th></tr></thead><tbody>{content.model_comparison.map((model) => <tr key={model.model_id} className={model.model_name === content.recommended_model.model_name ? "bg-emerald-500/10" : undefined}><td className="border-b border-black/5 px-3 py-2">{model.model_name}{model.model_name === content.recommended_model.model_name ? " — recommended" : ""}</td><td className="border-b border-black/5 px-3 py-2">{model.status}</td><td className="border-b border-black/5 px-3 py-2">{metricLabel(model.primary_metric_name)}: {formatMetric(model.primary_metric_value)}</td><td className="border-b border-black/5 px-3 py-2">{model.training_time_ms} ms</td></tr>)}</tbody></table></div></section>

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Charts</h2><div className="mt-4 grid gap-4 lg:grid-cols-2">{content.charts.map((chart) => <figure key={chart.chart_key} className="rounded-2xl border border-black/10 bg-white p-4"><figcaption className="font-medium">{chart.title}</figcaption><img className="mt-3 w-full rounded-xl border border-black/5" src={`data:image/png;base64,${chart.image_base64}`} alt={chart.title} /><p className="mt-3 text-sm text-black/60">{chart.description}</p></figure>)}</div></section>

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Prediction examples</h2><div className="mt-4 overflow-x-auto"><table className="min-w-full text-sm"><thead><tr>{content.prediction_examples.columns.map((column) => <th key={column} className="border-b border-black/10 px-3 py-2 text-left font-medium">{column}</th>)}</tr></thead><tbody>{content.prediction_examples.rows.map((row, idx) => <tr key={idx}>{content.prediction_examples.columns.map((column) => <td key={column} className="border-b border-black/5 px-3 py-2 text-black/70">{displayValue(row[column])}</td>)}</tr>)}</tbody></table></div></section>

    {warnings.length ? <section className="mt-6 rounded-3xl border border-amber-200 bg-amber-50 p-6"><h2 className="text-lg font-semibold text-amber-900">Warnings and notes</h2><ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-amber-900">{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></section> : null}

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><button type="button" className="rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Continue to Test prediction</button></section>
  </div>;
}
