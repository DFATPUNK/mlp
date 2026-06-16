import { useEffect, useMemo, useRef, useState } from "react";
import { supabase } from "../../lib/supabaseClient";
import type { ReviewResultsStepOutput, TestPredictionStepOutput } from "../../lib/pipes";

type TestPredictionStepProps = {
  pipeId: string;
  reviewResultsOutput: ReviewResultsStepOutput | null;
  initialTestPredictionOutput: TestPredictionStepOutput | null;
  onCompleted: (output: TestPredictionStepOutput) => void;
  onBackToReviewResults: () => void;
};

type InputField = {
  name: string;
  label: string;
  type: "number" | "text" | "boolean";
  required: boolean;
  example: string | number | boolean | null;
  helper_text: string;
};

type PredictionSchemaResponse = {
  task_type: "tabular_classification" | "tabular_regression";
  target_column: string;
  model: { model_id: string; model_name: string };
  input_schema: { fields: InputField[] };
};

type TestPredictionResponse = PredictionSchemaResponse & {
  test_prediction_artifact_id: string;
  input: Record<string, unknown>;
  prediction: {
    value: string | number | boolean;
    label: string;
    confidence: number | null;
    class_probabilities: Record<string, number> | null;
  };
  plain_english_result: string;
  mappable_output: Record<string, unknown>;
};

function displayValue(value: unknown) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : "—";
  return String(value);
}

function confidenceLabel(confidence: number | null) {
  return typeof confidence === "number" ? `${Math.round(confidence * 100)}%` : "Not available";
}

export function TestPredictionStep({ pipeId, reviewResultsOutput, initialTestPredictionOutput, onCompleted, onBackToReviewResults }: TestPredictionStepProps) {
  const [schema, setSchema] = useState<PredictionSchemaResponse | null>(null);
  const [formValues, setFormValues] = useState<Record<string, string | boolean>>({});
  const [loadingSchema, setLoadingSchema] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TestPredictionResponse | null>(null);
  const [latestOutput, setLatestOutput] = useState<TestPredictionStepOutput | null>(initialTestPredictionOutput);
  const loadedSchemaRef = useRef(false);
  const output = latestOutput ?? initialTestPredictionOutput;

  const fields = schema?.input_schema.fields ?? [];

  useEffect(() => {
    if (!reviewResultsOutput || loadedSchemaRef.current) return;
    loadedSchemaRef.current = true;
    void loadSchema();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reviewResultsOutput]);

  const mappableJson = useMemo(() => JSON.stringify(result?.mappable_output ?? (output ? { prediction: output.prediction, confidence: output.confidence, model_name: output.model_name, pipe_id: pipeId, pipe_version: "draft" } : {}), null, 2), [result, output, pipeId]);

  async function serviceFetch(path: string, body: Record<string, unknown>) {
    const serviceUrl = import.meta.env.VITE_ML_TRAINING_API_URL as string | undefined;
    if (!serviceUrl) throw new Error("Prediction service is not configured.");
    const { data } = await supabase.auth.getSession();
    const token = data.session?.access_token;
    if (!token) throw new Error("Please sign in again before testing predictions.");
    const response = await fetch(`${serviceUrl.replace(/\/$/, "")}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify(body),
    });
    const payload = await response.json() as { detail?: string; error?: string };
    if (!response.ok) throw new Error(payload.detail ?? payload.error ?? "Prediction request failed.");
    return payload;
  }

  async function loadSchema() {
    if (!reviewResultsOutput) return;
    setLoadingSchema(true);
    setError(null);
    try {
      const payload = await serviceFetch("/test-prediction-schema", { pipe_id: pipeId, review_results_artifact_id: reviewResultsOutput.review_results_artifact_id }) as PredictionSchemaResponse;
      setSchema(payload);
      const defaults: Record<string, string | boolean> = {};
      for (const field of payload.input_schema.fields) {
        if (field.type === "boolean") defaults[field.name] = Boolean(field.example ?? false);
        else defaults[field.name] = field.example === null || field.example === undefined ? "" : String(field.example);
      }
      setFormValues(defaults);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to load prediction fields.");
    } finally {
      setLoadingSchema(false);
    }
  }

  async function runPrediction() {
    if (!reviewResultsOutput) return;
    setRunning(true);
    setError(null);
    try {
      const payload = await serviceFetch("/test-prediction", { pipe_id: pipeId, review_results_artifact_id: reviewResultsOutput.review_results_artifact_id, input: formValues }) as TestPredictionResponse;
      setResult(payload);
      const nextOutput: TestPredictionStepOutput = {
        step_key: "test_prediction",
        status: "completed",
        test_prediction_artifact_id: payload.test_prediction_artifact_id,
        previous_review_results_artifact_id: reviewResultsOutput.review_results_artifact_id,
        prediction: payload.prediction.value,
        confidence: payload.prediction.confidence,
        model_name: payload.model.model_name,
        storage: { format: "json", uri: `artifact:${payload.test_prediction_artifact_id}` },
      };
      setLatestOutput(nextOutput);
      onCompleted(nextOutput);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to run prediction.");
    } finally {
      setRunning(false);
    }
  }

  if (!reviewResultsOutput) {
    return <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Review results before testing a prediction.</h2><button type="button" onClick={onBackToReviewResults} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Back to Review results</button></section>;
  }

  return <div>
    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Test one concrete example</h2><p className="mt-2 text-sm leading-6 text-black/60">Use one example to see what this pipe would predict. This is different from model evaluation: it helps you test whether the pipe feels useful on a real case.</p>{output ? <div className="mt-4 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900"><p className="font-medium">Last test saved.</p><p className="mt-1">Prediction: {displayValue(output.prediction)}{output.confidence !== null ? ` • Confidence: ${confidenceLabel(output.confidence)}` : ""}</p><p className="mt-1 text-xs">test_prediction_artifact_id: {output.test_prediction_artifact_id}</p></div> : null}</section>

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between"><div><h2 className="text-lg font-semibold">Input form</h2><p className="mt-2 text-sm text-black/60">These fields are the original feature columns used by the trained model.</p></div>{loadingSchema ? <span className="text-sm text-black/50">Loading fields…</span> : null}</div>{error ? <p className="mt-4 rounded-2xl bg-red-500/10 px-4 py-3 text-sm text-red-700">{error}</p> : null}{fields.length ? <div className="mt-5 grid gap-4 md:grid-cols-2">{fields.map((field) => <label key={field.name} className="rounded-2xl border border-black/10 bg-white/70 p-4"><span className="text-sm font-medium">{field.label}</span><span className="mt-1 block text-xs text-black/50">{field.helper_text}{field.example !== null && field.example !== undefined ? ` Example: ${displayValue(field.example)}` : ""}</span>{field.type === "boolean" ? <select value={String(formValues[field.name] ?? false)} onChange={(event) => setFormValues((current) => ({ ...current, [field.name]: event.target.value === "true" }))} className="mt-3 w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-sm"><option value="true">true</option><option value="false">false</option></select> : <input type={field.type === "number" ? "number" : "text"} value={String(formValues[field.name] ?? "")} onChange={(event) => setFormValues((current) => ({ ...current, [field.name]: event.target.value }))} className="mt-3 w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-sm outline-none focus:border-black" />}</label>)}</div> : !loadingSchema ? <p className="mt-4 text-sm text-black/60">No usable input fields were found for this trained model.</p> : null}<button type="button" onClick={runPrediction} disabled={running || loadingSchema || !fields.length} className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-black/30">{running ? "Running prediction…" : "Run prediction"}</button></section>

    {result ? <section className="mt-6 rounded-3xl border border-emerald-200 bg-emerald-50 p-6"><h2 className="text-lg font-semibold text-emerald-900">Prediction result</h2><p className="mt-3 text-lg font-medium text-emerald-950">{result.plain_english_result}</p><dl className="mt-4 grid gap-3 text-sm text-emerald-900 md:grid-cols-2"><div><dt className="font-medium">Model</dt><dd>{result.model.model_name}</dd></div><div><dt className="font-medium">Predicted {result.target_column}</dt><dd>{displayValue(result.prediction.value)}</dd></div><div><dt className="font-medium">Confidence</dt><dd>{result.task_type === "tabular_classification" ? confidenceLabel(result.prediction.confidence) : "Regression models return a numeric prediction. Confidence is not available in this MVP."}</dd></div></dl>{result.prediction.class_probabilities ? <div className="mt-4"><h3 className="text-sm font-semibold text-emerald-900">Class probabilities</h3><div className="mt-2 grid gap-2 md:grid-cols-2">{Object.entries(result.prediction.class_probabilities).map(([label, probability]) => <div key={label} className="rounded-xl bg-white/70 px-3 py-2 text-sm"><span className="font-medium">{label}</span>: {confidenceLabel(probability)}</div>)}</div></div> : null}</section> : null}

    {(result || output) ? <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Mappable workflow output</h2><p className="mt-2 text-sm text-black/60">These fields are what you will be able to map into workflows later.</p><pre className="mt-4 overflow-x-auto rounded-2xl bg-black p-4 text-xs text-white">{mappableJson}</pre><button type="button" className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Continue to Publish pipe</button></section> : null}
  </div>;
}
