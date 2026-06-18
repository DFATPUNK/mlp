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

type SplitCounts = { training: number; validation: number; test: number; total: number };
type SplitRatios = { training: number; validation: number; test: number };

type SampleContext = {
  kind: "validation_row";
  validation_row_index: number;
  validation_row_number: number;
  validation_rows_total: number;
  target_is_available_after_prediction: boolean;
  split_counts: SplitCounts | null;
  split_ratios: SplitRatios | null;
};

type Provenance = {
  kind: "validation_row" | "custom_input";
  validation_row_number: number | null;
  validation_rows_total: number | null;
  split_counts: SplitCounts | null;
  split_ratios: SplitRatios | null;
  message: string;
};

type GroundTruth = {
  available: boolean;
  target_column: string;
  actual_value: string | number | boolean | null;
  matches_prediction: boolean | null;
  absolute_error: number | null;
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
  provenance?: Provenance;
  ground_truth?: GroundTruth;
  plain_english_result: string;
  mappable_output: Record<string, unknown>;
};

type TestPredictionSampleResponse = {
  input: Record<string, unknown>;
  sample_context: SampleContext;
  source: { kind: "validation_row"; description: string };
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

function percentLabel(value: number | null | undefined) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "—";
}

function booleanInputValue(value: unknown) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return ["true", "1", "yes", "y", "on"].includes(value.trim().toLowerCase());
  return Boolean(value ?? false);
}

function splitSummary(counts: SplitCounts | null | undefined, ratios: SplitRatios | null | undefined) {
  if (!counts) return null;
  return [
    { label: "Training", count: counts.training, ratio: ratios?.training },
    { label: "Validation", count: counts.validation, ratio: ratios?.validation },
    { label: "Test", count: counts.test, ratio: ratios?.test },
  ];
}

export function TestPredictionStep({ pipeId, reviewResultsOutput, initialTestPredictionOutput, onCompleted, onBackToReviewResults }: TestPredictionStepProps) {
  const [schema, setSchema] = useState<PredictionSchemaResponse | null>(null);
  const [formValues, setFormValues] = useState<Record<string, string | boolean>>({});
  const [loadingSchema, setLoadingSchema] = useState(false);
  const [running, setRunning] = useState(false);
  const [sampling, setSampling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TestPredictionResponse | null>(null);
  const [sampleContext, setSampleContext] = useState<SampleContext | null>(null);
  const [seenValidationRowIndices, setSeenValidationRowIndices] = useState<number[]>([]);
  const [exampleNotice, setExampleNotice] = useState<string | null>(null);
  const [hasEditedSinceSample, setHasEditedSinceSample] = useState(false);
  const [showInputs, setShowInputs] = useState(true);
  const [latestOutput, setLatestOutput] = useState<TestPredictionStepOutput | null>(initialTestPredictionOutput);
  const loadedSchemaRef = useRef(false);
  const output = latestOutput ?? initialTestPredictionOutput;

  const fields = schema?.input_schema.fields ?? [];
  const provenance = result?.provenance;
  const groundTruth = result?.ground_truth;
  const activeCounts = provenance?.split_counts ?? sampleContext?.split_counts ?? null;
  const activeRatios = provenance?.split_ratios ?? sampleContext?.split_ratios ?? null;
  const activeSplitSummary = splitSummary(activeCounts, activeRatios);
  const isValidationExample = !hasEditedSinceSample && (provenance?.kind === "validation_row" || (!result && sampleContext));
  const aboutTitle = isValidationExample ? "About this validation example" : "About this custom input";

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

  async function runPrediction(inputOverride?: Record<string, string | boolean>, contextOverride?: SampleContext | null) {
    if (!reviewResultsOutput) return;
    const contextForRun = contextOverride === undefined ? sampleContext : contextOverride;
    setRunning(true);
    setError(null);
    try {
      const payload = await serviceFetch("/test-prediction", {
        pipe_id: pipeId,
        review_results_artifact_id: reviewResultsOutput.review_results_artifact_id,
        input: inputOverride ?? formValues,
        sample_context: contextForRun ? { kind: "validation_row", validation_row_index: contextForRun.validation_row_index } : null,
      }) as TestPredictionResponse;
      setResult(payload);
      if (payload.provenance?.kind === "validation_row") setHasEditedSinceSample(false);
      if (payload.provenance?.kind === "custom_input") setHasEditedSinceSample(true);
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

  async function loadAnotherValidationExampleAndPredict() {
    if (!reviewResultsOutput) return;
    setSampling(true);
    setError(null);
    try {
      const totalSeen = sampleContext?.validation_rows_total;
      const excludeIndices = totalSeen && seenValidationRowIndices.length >= totalSeen ? [] : seenValidationRowIndices;
      if (totalSeen && seenValidationRowIndices.length >= totalSeen) {
        setExampleNotice("You have seen all available validation examples. Starting again from the validation pool.");
      } else {
        setExampleNotice(null);
      }
      const sample = await serviceFetch("/test-prediction-sample", {
        pipe_id: pipeId,
        review_results_artifact_id: reviewResultsOutput.review_results_artifact_id,
        exclude_validation_row_indices: excludeIndices,
      }) as TestPredictionSampleResponse;
      const nextValues: Record<string, string | boolean> = {};
      for (const field of fields) {
        const value = sample.input[field.name];
        if (field.type === "boolean") nextValues[field.name] = booleanInputValue(value);
        else nextValues[field.name] = value === null || value === undefined ? "" : String(value);
      }
      setFormValues(nextValues);
      setSampleContext(sample.sample_context);
      setHasEditedSinceSample(false);
      setSeenValidationRowIndices((current) => {
        const shouldReset = sample.sample_context.validation_rows_total && current.length >= sample.sample_context.validation_rows_total;
        const base = shouldReset ? [] : current;
        return base.includes(sample.sample_context.validation_row_index) ? base : [...base, sample.sample_context.validation_row_index];
      });
      await runPrediction(nextValues, sample.sample_context);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to load another validation example.");
    } finally {
      setSampling(false);
    }
  }

  function updateField(field: InputField, value: string | boolean) {
    setFormValues((current) => ({ ...current, [field.name]: value }));
    if (sampleContext) setHasEditedSinceSample(true);
  }

  if (!reviewResultsOutput) {
    return <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Review results before testing a prediction.</h2><button type="button" onClick={onBackToReviewResults} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Back to Review results</button></section>;
  }

  return <div className="mt-6 space-y-6">
    <section className="rounded-3xl border border-black/10 bg-white/70 p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-xl font-semibold">Test prediction</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-black/60">Try a prediction on a concrete example. Validation examples were not used to train the model, so they help you see how the pipe behaves on unseen data.</p>
          {output ? <p className="mt-3 text-xs text-black/50">Latest saved test prediction: {displayValue(output.prediction)} · artifact {output.test_prediction_artifact_id}</p> : null}
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={loadAnotherValidationExampleAndPredict} disabled={running || sampling || loadingSchema || !fields.length} className="rounded-full bg-black px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-black/30">{sampling ? "Loading validation example…" : "Try another validation example"}</button>
          <button type="button" onClick={() => runPrediction()} disabled={running || sampling || loadingSchema || !fields.length} className="rounded-full border border-black/10 bg-white px-4 py-2 text-sm font-medium text-black disabled:cursor-not-allowed disabled:text-black/30">{running ? "Running prediction…" : "Run prediction"}</button>
          <button type="button" onClick={() => setShowInputs((current) => !current)} className="rounded-full border border-black/10 bg-white px-4 py-2 text-sm font-medium text-black">{showInputs ? "Hide input values" : `Edit input values (${fields.length})`}</button>
        </div>
      </div>
      <p className="mt-3 text-xs text-black/50">Use a validation row to test instantly, or edit the values to make a custom input.</p>
      {exampleNotice ? <p className="mt-3 rounded-2xl bg-amber-500/10 px-4 py-3 text-sm text-amber-800">{exampleNotice}</p> : null}
      {error ? <p className="mt-3 rounded-2xl bg-red-500/10 px-4 py-3 text-sm text-red-700">{error}</p> : null}
    </section>

    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
      <section className="rounded-3xl border border-black/10 bg-white/60 p-5">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="font-semibold">Input values</h3>
            <p className="text-xs text-black/50">{fields.length} model input fields. All fields remain editable.</p>
          </div>
          {loadingSchema ? <span className="text-sm text-black/50">Loading fields…</span> : null}
        </div>
        {showInputs ? fields.length ? <div className="mt-4 grid gap-3 md:grid-cols-2">
          {fields.map((field) => <label key={field.name} className="rounded-2xl border border-black/10 bg-white/70 p-3">
            <span className="text-sm font-medium">{field.label}</span>
            <span className="mt-1 block text-[11px] leading-4 text-black/45">{field.type}{field.example !== null && field.example !== undefined ? ` · example ${displayValue(field.example)}` : ""}</span>
            {field.type === "boolean" ? <select value={String(formValues[field.name] ?? false)} onChange={(event) => updateField(field, event.target.value === "true")} className="mt-2 w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-sm"><option value="true">true</option><option value="false">false</option></select> : <input type={field.type === "number" ? "number" : "text"} value={String(formValues[field.name] ?? "")} onChange={(event) => updateField(field, event.target.value)} className="mt-2 w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-sm outline-none focus:border-black" />}
          </label>)}
        </div> : !loadingSchema ? <p className="mt-4 text-sm text-black/60">No usable input fields were found for this trained model.</p> : null : <div className="mt-4 rounded-2xl border border-black/10 bg-white/70 p-4 text-sm text-black/60">Input values are hidden. Choose “Edit input values” to reveal and modify all {fields.length} model fields.</div>}
      </section>

      <aside className="space-y-6">
        <section className="rounded-3xl border border-black/10 bg-white/60 p-5">
          <h3 className="font-semibold">{aboutTitle}</h3>
          {isValidationExample && sampleContext ? <div className="mt-3 space-y-3 text-sm text-black/65">
            <p>This is validation example {sampleContext.validation_row_number} of {sampleContext.validation_rows_total}.</p>
            {activeSplitSummary ? <div><p className="font-medium text-black/80">Your dataset was split into:</p><ul className="mt-2 space-y-1">{activeSplitSummary.map((item) => <li key={item.label}>{item.label}: {item.count.toLocaleString()} rows ({percentLabel(item.ratio)})</li>)}</ul></div> : null}
            <p>Training rows were used to teach the model patterns.</p>
            <p>Validation rows were not used to fit the model. They were used to compare models and are used here to demonstrate how the selected model behaves on unseen examples.</p>
            <p>Test rows are kept separate for a future final evaluation. This interactive test does not use test rows.</p>
          </div> : <div className="mt-3 space-y-3 text-sm text-black/65"><p>{provenance?.message ?? (sampleContext ? "This started from a validation example, but you changed one or more values." : "This is a custom input.")}</p><p>The model can still make a real prediction, but there is no known target value to compare against.</p></div>}
          <details className="mt-4 rounded-2xl border border-black/10 bg-white/70 p-3 text-sm text-black/60">
            <summary className="cursor-pointer font-medium text-black/80">How the dataset split works</summary>
            <div className="mt-3 space-y-2 leading-6">
              <p><strong>Training split:</strong> rows used to teach the model patterns.</p>
              <p><strong>Validation split:</strong> rows held out from fitting and used to compare model options.</p>
              <p><strong>Test split:</strong> rows kept separate for a future final evaluation.</p>
              <p>Using more validation data can make validation estimates more stable, but it leaves less data available for training. There is no universally best split; the right balance depends on dataset size and the goal of the project.</p>
            </div>
          </details>
        </section>

        {result ? <section className={`rounded-3xl border p-5 ${groundTruth?.available && groundTruth.matches_prediction === false ? "border-red-200 bg-red-50" : "border-emerald-200 bg-emerald-50"}`}>
          <h3 className="font-semibold">Prediction result</h3>
          <p className="mt-3 text-lg font-semibold">Predicted {result.target_column}: {displayValue(result.prediction.value)}</p>
          {result.task_type === "tabular_classification" ? <p className="mt-1 text-sm">Model confidence: {confidenceLabel(result.prediction.confidence)}</p> : <p className="mt-1 text-sm">Regression models return a numeric prediction. Confidence is not available in this MVP.</p>}
          {groundTruth?.available ? <div className="mt-4 rounded-2xl bg-white/70 p-4 text-sm">
            <p className="font-medium">Known validation value: {displayValue(groundTruth.actual_value)}</p>
            {result.task_type === "tabular_classification" ? <p className={`mt-2 font-semibold ${groundTruth.matches_prediction ? "text-emerald-800" : "text-red-700"}`}>{groundTruth.matches_prediction ? "Correct prediction" : "Incorrect prediction"}</p> : <p className="mt-2 font-semibold">Absolute error: {displayValue(groundTruth.absolute_error)}</p>}
            {result.task_type === "tabular_regression" ? <p className="mt-1 text-xs text-black/50">Absolute error is the difference between the predicted value and the real validation value.</p> : null}
          </div> : <p className="mt-4 rounded-2xl bg-white/70 p-4 text-sm">This is a custom input, so there is no known answer to compare against.</p>}
          {result.prediction.class_probabilities ? <div className="mt-4"><h4 className="text-sm font-semibold">Class probabilities</h4><div className="mt-2 grid gap-2">{Object.entries(result.prediction.class_probabilities).map(([label, probability]) => <div key={label} className="rounded-xl bg-white/70 px-3 py-2 text-sm"><span className="font-medium">{label}</span>: {confidenceLabel(probability)}</div>)}</div></div> : null}
        </section> : null}
      </aside>
    </div>

    {(result || output) ? <section className="rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Mappable workflow output</h2><p className="mt-2 text-sm text-black/60">These fields are what you will be able to map into workflows later.</p><pre className="mt-4 overflow-x-auto rounded-2xl bg-black p-4 text-xs text-white">{mappableJson}</pre><button type="button" className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Continue to Publish pipe</button></section> : null}
  </div>;
}
