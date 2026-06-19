import { useEffect, useMemo, useRef, useState } from "react";
import { supabase } from "../../lib/supabaseClient";
import type { ReviewResultsStepOutput, TestPredictionStepOutput } from "../../lib/pipes";

type TestPredictionStepProps = {
  pipeId: string;
  reviewResultsOutput: ReviewResultsStepOutput | null;
  initialTestPredictionOutput: TestPredictionStepOutput | null;
  onCompleted: (output: TestPredictionStepOutput) => void;
  onContinueToPublish: () => void;
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


type TreeVote = {
  tree_index: number;
  tree_prediction: string | number | boolean;
  agrees_with_final_prediction: boolean;
  vote_strength: number | null;
  role: string;
};

type RepresentativeTree = {
  tree_index: number;
  role: string;
  tree_prediction: string | number | boolean;
  agrees_with_final_prediction: boolean;
  confidence: number | null;
  total_decision_count?: number;
  visible_decision_count?: number;
  hidden_decision_count?: number;
  leaf_summary: string;
  decision_path: Array<{
    step: number;
    feature: string;
    operator: "<=" | ">";
    threshold: number | string;
    input_value: string | number | boolean | null;
    outcome: boolean;
    display_text: string;
  }>;
};

type PredictionExplanation = {
  task_type: "tabular_classification" | "tabular_regression";
  prediction: TestPredictionResponse["prediction"];
  model_explanation: {
    supported: boolean;
    model_type: string;
    model_name: string;
    headline: string;
    plain_english_summary: string;
    caveats: string[];
    forest_vote: null | {
      tree_count: number;
      final_prediction: string | number | boolean;
      agreement_count: number;
      disagreement_count: number;
      vote_summary: string;
      trees: TreeVote[];
    };
    regression_summary?: null | {
      tree_count: number;
      forest_average: number;
      min_tree_prediction: number;
      max_tree_prediction: number;
      tree_prediction_std: number;
      within_reasonable_band_count: number;
      tree_estimates: Array<{ tree_index: number; tree_prediction: number; role?: string }>;
    };
    representative_trees: RepresentativeTree[];
    features_consulted: Array<{ feature: string; trees_used: number; tree_count: number; share_of_trees: number }>;
    linear_contributions?: null | {
      intercept: number | null;
      positive: Array<{ feature: string; contribution: number; coefficient: number; input_value: number }>;
      negative: Array<{ feature: string; contribution: number; coefficient: number; input_value: number }>;
    };
    dummy_explanation?: null | { strategy: string; prediction: string | number | boolean };
  };
};

type TreeDetail = RepresentativeTree & { hidden_step_count?: number };

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

function fieldLabel(name: string) {
  return name
    .replace(/[._-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function fieldTypeLabel(type: InputField["type"]) {
  return type === "text" ? "Text" : type === "number" ? "Number" : "Boolean";
}

function stableInputKey(input: Record<string, unknown>) {
  return JSON.stringify(Object.keys(input).sort().map((key) => [key, input[key]]));
}

export function TestPredictionStep({ pipeId, reviewResultsOutput, onCompleted, onContinueToPublish, onBackToReviewResults }: TestPredictionStepProps) {
  const [schema, setSchema] = useState<PredictionSchemaResponse | null>(null);
  const [formValues, setFormValues] = useState<Record<string, string | boolean>>({});
  const [loadingSchema, setLoadingSchema] = useState(false);
  const [running, setRunning] = useState(false);
  const [sampling, setSampling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TestPredictionResponse | null>(null);
  const [activeTab, setActiveTab] = useState<"run" | "understand">("run");
  const [explanation, setExplanation] = useState<PredictionExplanation | null>(null);
  const [explanationInputKey, setExplanationInputKey] = useState<string | null>(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);
  const [explanationError, setExplanationError] = useState<string | null>(null);
  const [selectedTreeIndex, setSelectedTreeIndex] = useState<number | null>(null);
  const [selectedTreeDetail, setSelectedTreeDetail] = useState<TreeDetail | null>(null);
  const [showFullTreePath, setShowFullTreePath] = useState(false);
  const [loadingTreeDetail, setLoadingTreeDetail] = useState(false);
  const [treeDetailError, setTreeDetailError] = useState<string | null>(null);
  const [sampleContext, setSampleContext] = useState<SampleContext | null>(null);
  const [seenValidationRowIndices, setSeenValidationRowIndices] = useState<number[]>([]);
  const [exampleNotice, setExampleNotice] = useState<string | null>(null);
  const [hasEditedSinceSample, setHasEditedSinceSample] = useState(false);
  const loadedSchemaRef = useRef(false);

  const fields = schema?.input_schema.fields ?? [];
  const provenance = result?.provenance;
  const groundTruth = result?.ground_truth;
  const activeCounts = provenance?.split_counts ?? sampleContext?.split_counts ?? null;
  const activeRatios = provenance?.split_ratios ?? sampleContext?.split_ratios ?? null;
  const activeSplitSummary = splitSummary(activeCounts, activeRatios);
  const isValidationExample = !hasEditedSinceSample && (provenance?.kind === "validation_row" || (!result && sampleContext));
  const aboutTitle = isValidationExample ? "About this validation example" : "About this custom input";
  const showGroundTruth = Boolean(groundTruth?.available && !hasEditedSinceSample && provenance?.kind === "validation_row");
  const currentInputKey = useMemo(() => stableInputKey(formValues), [formValues]);
  const explanationIsStale = Boolean(explanation && explanationInputKey !== currentInputKey);

  useEffect(() => {
    if (!reviewResultsOutput || loadedSchemaRef.current) return;
    loadedSchemaRef.current = true;
    void loadSchema();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reviewResultsOutput]);

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


  async function loadExplanation(inputForExplanation: Record<string, string | boolean>, inputKey: string) {
    if (!reviewResultsOutput) return;
    setLoadingExplanation(true);
    setExplanationError(null);
    try {
      const payload = await serviceFetch("/explain-prediction", {
        pipe_id: pipeId,
        review_results_artifact_id: reviewResultsOutput.review_results_artifact_id,
        input: inputForExplanation,
      }) as PredictionExplanation;
      setExplanation(payload);
      setExplanationInputKey(inputKey);
      const firstTree = payload.model_explanation.forest_vote?.trees[0]?.tree_index ?? payload.model_explanation.regression_summary?.tree_estimates[0]?.tree_index ?? null;
      setSelectedTreeIndex(firstTree);
      setSelectedTreeDetail(null);
      if (firstTree !== null && firstTree !== undefined) void loadTreeDetail(firstTree, inputForExplanation);
    } catch (caught) {
      setExplanationError(caught instanceof Error ? caught.message : "Unable to explain this prediction.");
      setExplanation(null);
      setExplanationInputKey(null);
      setSelectedTreeDetail(null);
    } finally {
      setLoadingExplanation(false);
    }
  }

  async function loadTreeDetail(treeIndex: number, inputForTree: Record<string, string | boolean> = formValues) {
    if (!reviewResultsOutput) return;
    setSelectedTreeIndex(treeIndex);
    setShowFullTreePath(false);
    setLoadingTreeDetail(true);
    setTreeDetailError(null);
    try {
      const payload = await serviceFetch("/explain-prediction-tree", {
        pipe_id: pipeId,
        review_results_artifact_id: reviewResultsOutput.review_results_artifact_id,
        input: inputForTree,
        tree_index: treeIndex,
      }) as TreeDetail;
      setSelectedTreeDetail(payload);
    } catch (caught) {
      setTreeDetailError(caught instanceof Error ? caught.message : "Unable to load this tree's decision path.");
      setSelectedTreeDetail(null);
    } finally {
      setLoadingTreeDetail(false);
    }
  }

  async function runPrediction(inputOverride?: Record<string, string | boolean>, contextOverride?: SampleContext | null) {
    if (!reviewResultsOutput) return;
    const contextForRun = contextOverride === undefined ? sampleContext : contextOverride;
    setRunning(true);
    setError(null);
    try {
      const inputForRun = inputOverride ?? formValues;
      const inputKey = stableInputKey(inputForRun);
      const payload = await serviceFetch("/test-prediction", {
        pipe_id: pipeId,
        review_results_artifact_id: reviewResultsOutput.review_results_artifact_id,
        input: inputForRun,
        sample_context: contextForRun ? { kind: "validation_row", validation_row_index: contextForRun.validation_row_index } : null,
      }) as TestPredictionResponse;
      setResult(payload);
      void loadExplanation(inputForRun, inputKey);
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
      onCompleted(nextOutput);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to run prediction.");
    } finally {
      setRunning(false);
    }
  }

  async function loadValidationExample() {
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
      setResult(null);
      setExplanation(null);
      setExplanationInputKey(null);
      setExplanationError(null);
      setActiveTab("run");
      setSeenValidationRowIndices((current) => {
        const shouldReset = sample.sample_context.validation_rows_total && current.length >= sample.sample_context.validation_rows_total;
        const base = shouldReset ? [] : current;
        return base.includes(sample.sample_context.validation_row_index) ? base : [...base, sample.sample_context.validation_row_index];
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to load another validation example.");
    } finally {
      setSampling(false);
    }
  }

  function updateField(field: InputField, value: string | boolean) {
    setFormValues((current) => ({ ...current, [field.name]: value }));
    if (sampleContext) setHasEditedSinceSample(true);
    setSelectedTreeDetail(null);
  }


  function renderExplanation() {
    if (!result) {
      return <section className="rounded-3xl border border-black/10 bg-white/60 p-6 text-sm text-black/60">Run a prediction first to see how the selected model reached its result.</section>;
    }
    if (explanationIsStale) {
      return <section className="rounded-3xl border border-amber-200 bg-amber-50 p-6 text-sm text-amber-800">Run prediction again to refresh this explanation.</section>;
    }
    if (loadingExplanation) {
      return <section className="rounded-3xl border border-black/10 bg-white/60 p-6 text-sm text-black/60">Explaining the latest prediction…</section>;
    }
    if (explanationError) {
      return <section className="rounded-3xl border border-red-200 bg-red-50 p-6"><p className="text-sm text-red-700">{explanationError}</p><button type="button" onClick={() => void loadExplanation(formValues, currentInputKey)} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Retry explanation</button></section>;
    }
    if (!explanation) {
      return <section className="rounded-3xl border border-black/10 bg-white/60 p-6 text-sm text-black/60">Run a prediction first to see how the selected model reached its result.</section>;
    }
    const modelExplanation = explanation.model_explanation;
    const selectedTree = selectedTreeDetail;
    return <section className="space-y-5">
      <div className="rounded-3xl border border-black/10 bg-white/60 p-5">
        <h3 className="text-lg font-semibold">Your model: {modelExplanation.model_name}</h3>
        <p className="mt-2 text-sm leading-6 text-black/65">{modelExplanation.plain_english_summary}</p>
        <p className="mt-2 text-xs text-black/50">This explains the latest prediction input only. It does not prove why the outcome happens in the real world.</p>
        {!modelExplanation.supported ? <div className="mt-4 rounded-2xl bg-white/70 p-4 text-sm text-black/65"><p className="font-medium">{modelExplanation.headline}</p><p className="mt-2">Prediction: {displayValue(explanation.prediction.value)}{explanation.prediction.confidence !== null ? ` · Model confidence: ${confidenceLabel(explanation.prediction.confidence)}` : ""}</p></div> : null}
      </div>
      {modelExplanation.supported && modelExplanation.forest_vote ? <div className="rounded-3xl border border-black/10 bg-white/60 p-5">
        <h3 className="font-semibold">Forest decision summary</h3>
        <p className="mt-2 text-sm text-black/65">{modelExplanation.forest_vote.vote_summary}</p>
        <p className="mt-1 text-sm text-black/65">Final prediction: {displayValue(modelExplanation.forest_vote.final_prediction)} · Model confidence: {confidenceLabel(explanation.prediction.confidence)}</p>
        <p className="mt-1 text-xs text-black/50">Tiles show each tree's top predicted class. The forest's final confidence is based on averaged tree probabilities.</p>
        <div className="mt-3 flex gap-4 text-xs"><span><span className="mr-1 inline-block h-3 w-3 rounded bg-emerald-500" />Green: agrees with final forest prediction</span><span><span className="mr-1 inline-block h-3 w-3 rounded bg-red-400" />Red: predicts a different class</span></div>
        <div className="mt-4 grid grid-cols-10 gap-1 sm:grid-cols-12 md:grid-cols-16">
          {modelExplanation.forest_vote.trees.map((tree) => <button key={tree.tree_index} type="button" aria-label={`Tree ${tree.tree_index + 1}. Predicts ${displayValue(tree.tree_prediction)}. ${tree.agrees_with_final_prediction ? "Agrees with forest" : "Disagrees with forest"}.`} onClick={() => void loadTreeDetail(tree.tree_index)} className={`h-5 rounded ${tree.agrees_with_final_prediction ? "bg-emerald-500" : "bg-red-400"} ${selectedTreeIndex === tree.tree_index ? "ring-2 ring-black ring-offset-2" : ""}`} />)}
        </div>
        {modelExplanation.forest_vote.disagreement_count ? <p className="mt-4 text-xs text-black/50">{modelExplanation.forest_vote.disagreement_count} trees predicted a different class. Click any red tile to inspect that exact tree.</p> : <p className="mt-4 text-xs text-black/50">All trees agreed with the final prediction.</p>}
      </div> : null}
      {modelExplanation.supported && modelExplanation.regression_summary ? <div className="rounded-3xl border border-black/10 bg-white/60 p-5">
        <h3 className="font-semibold">Tree estimates</h3>
        <dl className="mt-3 grid gap-3 text-sm md:grid-cols-2"><div><dt className="text-black/50">Forest average</dt><dd>{displayValue(modelExplanation.regression_summary.forest_average)}</dd></div><div><dt className="text-black/50">Tree estimate range</dt><dd>{displayValue(modelExplanation.regression_summary.min_tree_prediction)} – {displayValue(modelExplanation.regression_summary.max_tree_prediction)}</dd></div><div><dt className="text-black/50">Standard deviation</dt><dd>{displayValue(modelExplanation.regression_summary.tree_prediction_std)}</dd></div><div><dt className="text-black/50">Trees near average</dt><dd>{modelExplanation.regression_summary.within_reasonable_band_count} of {modelExplanation.regression_summary.tree_count}</dd></div></dl>
        <div className="mt-4 grid grid-cols-10 gap-1 sm:grid-cols-12 md:grid-cols-16">{modelExplanation.regression_summary.tree_estimates.map((tree) => <button key={tree.tree_index} type="button" aria-label={`Tree ${tree.tree_index + 1}. Estimate ${displayValue(tree.tree_prediction)}.`} onClick={() => void loadTreeDetail(tree.tree_index)} className={`h-5 rounded bg-sky-500 ${selectedTreeIndex === tree.tree_index ? "ring-2 ring-black ring-offset-2" : ""}`} />)}</div>
      </div> : null}
      {(modelExplanation.forest_vote || modelExplanation.regression_summary) ? <div className="rounded-3xl border border-black/10 bg-white/60 p-5">
        <h3 className="font-semibold">Selected tree detail</h3>
        {loadingTreeDetail ? <p className="mt-2 text-sm text-black/60">Loading this tree's decision path…</p> : null}
        {treeDetailError ? <p className="mt-2 rounded-2xl bg-red-500/10 px-4 py-3 text-sm text-red-700">{treeDetailError}</p> : null}
        {selectedTree ? <><p className="mt-2 text-sm text-black/65">Tree {selectedTree.tree_index + 1} — {selectedTree.agrees_with_final_prediction ? "agrees with the forest" : "predicts a different class than the forest"}</p><p className="mt-1 text-sm text-black/65">This tree predicts: {displayValue(selectedTree.tree_prediction)}{selectedTree.confidence !== null ? ` · This tree’s leaf vote: ${confidenceLabel(selectedTree.confidence)} for ${displayValue(selectedTree.tree_prediction)}` : ""}</p>{selectedTree.hidden_decision_count ? <p className="mt-3 rounded-2xl bg-black/5 p-3 text-xs leading-5 text-black/60">This tree made {selectedTree.total_decision_count} successive yes/no checks before reaching its prediction. To keep this view readable, we show the first {selectedTree.visible_decision_count} checks. The {selectedTree.hidden_decision_count} remaining checks are intermediate steps in the same path — they are not separate predictions.</p> : null}<ol className="mt-4 space-y-2 text-sm">{(showFullTreePath ? selectedTree.decision_path : selectedTree.decision_path.slice(0, selectedTree.visible_decision_count ?? 8)).map((step) => <li key={step.step} className="rounded-2xl bg-white/70 px-3 py-2"><span className="font-medium">{step.step}.</span> {step.display_text}<span className="ml-2 text-black/45">Your value: {displayValue(step.input_value)}</span></li>)}</ol>{selectedTree.hidden_decision_count && !showFullTreePath ? <button type="button" onClick={() => setShowFullTreePath(true)} className="mt-3 text-xs font-medium underline">Show all {selectedTree.total_decision_count} checks</button> : null}</> : !loadingTreeDetail ? <p className="mt-2 text-sm text-black/60">Select a tree tile to inspect its real decision path.</p> : null}
      </div> : null}
      {modelExplanation.linear_contributions ? <div className="rounded-3xl border border-black/10 bg-white/60 p-5">
        <h3 className="font-semibold">Signals behind this prediction</h3>
        <p className="mt-2 text-sm text-black/60">{modelExplanation.headline}. Intercept/baseline: {displayValue(modelExplanation.linear_contributions.intercept)}</p>
        <div className="mt-4 grid gap-4 md:grid-cols-2"><div><h4 className="text-sm font-semibold">Strongest positive signals</h4><div className="mt-2 space-y-2">{modelExplanation.linear_contributions.positive.map((item) => <div key={`pos-${item.feature}-${item.contribution}`} className="rounded-xl bg-emerald-500/10 px-3 py-2 text-sm"><span className="font-medium">{item.feature}</span><span className="float-right">{displayValue(item.contribution)}</span></div>)}</div></div><div><h4 className="text-sm font-semibold">Strongest negative signals</h4><div className="mt-2 space-y-2">{modelExplanation.linear_contributions.negative.map((item) => <div key={`neg-${item.feature}-${item.contribution}`} className="rounded-xl bg-red-500/10 px-3 py-2 text-sm"><span className="font-medium">{item.feature}</span><span className="float-right">{displayValue(item.contribution)}</span></div>)}</div></div></div>
      </div> : null}
      {modelExplanation.dummy_explanation ? <div className="rounded-3xl border border-black/10 bg-white/60 p-5"><h3 className="font-semibold">Baseline strategy</h3><p className="mt-2 text-sm text-black/65">This baseline model does not make a feature-by-feature decision. It uses the real strategy: {modelExplanation.dummy_explanation.strategy}.</p></div> : null}
      {modelExplanation.features_consulted.length ? <div className="rounded-3xl border border-black/10 bg-white/60 p-5">
        <h3 className="font-semibold">Features consulted for this prediction</h3>
        <p className="mt-2 text-sm text-black/60">These are the features that appeared in the decision paths used by the forest for this specific prediction.</p>
        <div className="mt-4 space-y-3">{modelExplanation.features_consulted.map((feature) => <div key={feature.feature}><div className="flex justify-between text-sm"><span className="font-medium">{feature.feature}</span><span>{feature.trees_used} of {feature.tree_count} trees</span></div><div className="mt-1 h-2 rounded-full bg-black/10"><div className="h-2 rounded-full bg-black" style={{ width: `${Math.round(feature.share_of_trees * 100)}%` }} /></div></div>)}</div>
        <p className="mt-4 text-xs text-black/50">Step 6 showed what mattered across validation data. This view shows which features were consulted for this exact input.</p>
      </div> : null}
      {modelExplanation.caveats.length ? <div className="rounded-3xl border border-amber-200 bg-amber-50 p-5"><h3 className="font-semibold text-amber-900">Caveats</h3><ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-amber-900">{modelExplanation.caveats.map((caveat) => <li key={caveat}>{caveat}</li>)}</ul></div> : null}
    </section>;
  }

  if (!reviewResultsOutput) {
    return <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Review results before testing a prediction.</h2><button type="button" onClick={onBackToReviewResults} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Back to Review results</button></section>;
  }

  const explanationContent = renderExplanation();

  return <div className="mt-6 space-y-5">
    <div>
      <h2 className="text-2xl font-semibold tracking-tight">Test prediction</h2>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-black/60">Try a prediction on a concrete example. Validation examples were not used to train the model, so they help you see how the pipe behaves on unseen data.</p>
      <div className="mt-4 inline-flex rounded-full border border-black/10 bg-white/70 p-1 text-sm">
        <button type="button" onClick={() => setActiveTab("run")} className={`rounded-full px-4 py-2 ${activeTab === "run" ? "bg-black text-white" : "text-black/60"}`}>Run prediction</button>
        <button type="button" onClick={() => setActiveTab("understand")} className={`rounded-full px-4 py-2 ${activeTab === "understand" ? "bg-black text-white" : "text-black/60"}`}>Understand this prediction</button>
      </div>
      {exampleNotice ? <p className="mt-3 rounded-2xl bg-amber-500/10 px-4 py-3 text-sm text-amber-800">{exampleNotice}</p> : null}
      {error ? <p className="mt-3 rounded-2xl bg-red-500/10 px-4 py-3 text-sm text-red-700">{error}</p> : null}
    </div>

    {activeTab === "run" ? <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(360px,0.85fr)]">
      <section className="rounded-3xl border border-black/10 bg-white/60 p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="font-semibold">Input values</h3>
            <p className="text-xs text-black/50">{fields.length} model input fields. All fields remain editable.</p>
          </div>
          <div className="group relative">
            <button type="button" aria-label="Load a new validation row" title="Load a new validation row" onClick={loadValidationExample} disabled={running || sampling || loadingSchema || !fields.length} className="flex h-9 w-9 items-center justify-center rounded-full border border-black/10 bg-white text-lg font-semibold text-black transition hover:border-black disabled:cursor-not-allowed disabled:text-black/30">{sampling ? "…" : "↻"}</button>
            <span className="pointer-events-none absolute right-0 top-11 z-10 hidden whitespace-nowrap rounded-lg bg-black px-2 py-1 text-xs text-white group-hover:block group-focus-within:block">Load a new validation row</span>
          </div>
        </div>
        {sampleContext && !hasEditedSinceSample ? <p className="mt-3 text-xs font-medium text-emerald-700">Validation example {sampleContext.validation_row_number} of {sampleContext.validation_rows_total}</p> : <p className="mt-3 text-xs font-medium text-black/50">Custom input</p>}
        {fields.length ? <div className="mt-4 overflow-hidden rounded-2xl border border-black/10 bg-white/70">
          {fields.map((field) => <div key={field.name} className="grid gap-2 border-b border-black/10 px-3 py-2 last:border-b-0 md:grid-cols-[minmax(0,1fr)_220px] md:items-center">
            <label htmlFor={`test-input-${field.name}`} className="min-w-0">
              <span className="block truncate text-sm font-medium">{fieldLabel(field.name)}</span>
              <span className="block text-[11px] leading-4 text-black/45">{fieldTypeLabel(field.type)}{field.example !== null && field.example !== undefined ? ` · Example: ${displayValue(field.example)}` : ""}</span>
            </label>
            {field.type === "boolean" ? <select id={`test-input-${field.name}`} value={String(formValues[field.name] ?? false)} onChange={(event) => updateField(field, event.target.value === "true")} className="w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-sm"><option value="true">true</option><option value="false">false</option></select> : <input id={`test-input-${field.name}`} type="text" inputMode={field.type === "number" ? "decimal" : undefined} value={String(formValues[field.name] ?? "")} onChange={(event) => updateField(field, event.target.value)} className="w-full rounded-xl border border-black/10 bg-white px-3 py-2 text-sm outline-none focus:border-black" />}
          </div>)}
        </div> : !loadingSchema ? <p className="mt-4 text-sm text-black/60">No usable input fields were found for this trained model.</p> : null}
        <button type="button" onClick={() => runPrediction()} disabled={running || sampling || loadingSchema || !fields.length} className="mt-4 w-full rounded-full bg-black px-4 py-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-black/30">{running ? "Running prediction…" : "Run prediction"}</button>
      </section>

      <aside className="space-y-6">
        <section className="rounded-3xl border border-black/10 bg-white/60 p-5">
          <h3 className="font-semibold">{aboutTitle}</h3>
          {isValidationExample && sampleContext ? <div className="mt-3 space-y-3 text-sm text-black/65">
            <p>This is validation example {sampleContext.validation_row_number} of {sampleContext.validation_rows_total}.</p>
            {activeSplitSummary ? <div><p className="font-medium text-black/80">Your dataset was split into:</p><ul className="mt-2 space-y-1">{activeSplitSummary.map((item) => <li key={item.label}>{item.label}: {item.count.toLocaleString()} rows ({percentLabel(item.ratio)})</li>)}</ul></div> : null}
            <p>Training rows were used to teach the model patterns.</p>
            <p>Validation rows were held out from training. They are used to compare models and, here, to demonstrate how the selected model behaves on unseen examples.</p>
            <p>Test rows are kept separate for a future final evaluation. This interactive prediction does not use test rows.</p>
          </div> : <div className="mt-3 space-y-3 text-sm text-black/65"><p>{provenance?.message ?? (sampleContext ? "This started from a validation example, but you changed one or more values." : "This is a custom input.")}</p><p>The model can still make a real prediction, but there is no known target value to compare against.</p></div>}
          <details open className="mt-4 rounded-2xl border border-black/10 bg-white/70 p-3 text-sm text-black/60">
            <summary className="cursor-pointer font-medium text-black/80">How the dataset split works</summary>
            <div className="mt-3 space-y-2 leading-6">
              <p><strong>Training split:</strong> rows used to teach the model patterns.</p>
              <p><strong>Validation split:</strong> rows held out from fitting and used to compare model options.</p>
              <p><strong>Test split:</strong> rows kept separate for a future final evaluation.</p>
              <p>Using more validation data can make validation estimates more stable, but it leaves less data available for training. There is no universally best split; the right balance depends on dataset size and the goal of the project.</p>
            </div>
          </details>
        </section>

        {result ? <section className={`rounded-3xl border p-5 ${showGroundTruth && groundTruth?.matches_prediction === false ? "border-red-200 bg-red-50" : "border-emerald-200 bg-emerald-50"}`}>
          <h3 className="font-semibold">Prediction result</h3>
          <p className="mt-3 text-lg font-semibold">Predicted {result.target_column}: {displayValue(result.prediction.value)}</p>
          {result.task_type === "tabular_classification" ? <p className="mt-1 text-sm">Model confidence: {confidenceLabel(result.prediction.confidence)}</p> : <p className="mt-1 text-sm">Regression models return a numeric prediction. Confidence is not available in this MVP.</p>}
          {showGroundTruth ? <div className="mt-4 rounded-2xl bg-white/70 p-4 text-sm">
            <p className="font-medium">Known validation value: {displayValue(groundTruth!.actual_value)}</p>
            {result.task_type === "tabular_classification" ? <p className={`mt-2 font-semibold ${groundTruth!.matches_prediction ? "text-emerald-800" : "text-red-700"}`}>{groundTruth!.matches_prediction ? "Correct prediction" : "Incorrect prediction"}</p> : <p className="mt-2 font-semibold">Absolute error: {displayValue(groundTruth!.absolute_error)}</p>}
            {result.task_type === "tabular_regression" ? <p className="mt-1 text-xs text-black/50">Absolute error is the difference between the predicted value and the real validation value.</p> : null}
          </div> : <p className="mt-4 rounded-2xl bg-white/70 p-4 text-sm">This is a custom input, so there is no known answer to compare against.</p>}
          {result.prediction.class_probabilities ? <div className="mt-4"><h4 className="text-sm font-semibold">Class probabilities</h4><div className="mt-2 grid gap-2">{Object.entries(result.prediction.class_probabilities).map(([label, probability]) => <div key={label} className="rounded-xl bg-white/70 px-3 py-2 text-sm"><span className="font-medium">{label}</span>: {confidenceLabel(probability)}</div>)}</div></div> : null}
        </section> : <section className="rounded-3xl border border-black/10 bg-white/60 p-5"><h3 className="font-semibold">Prediction result</h3><p className="mt-2 text-sm text-black/60">Ready to predict. The known target remains hidden until you run the model on an unchanged validation example.</p></section>}
      </aside>
    </div> : explanationContent}
    {result ? <div className="flex justify-end"><button type="button" onClick={onContinueToPublish} className="rounded-full bg-black px-5 py-3 text-sm font-medium text-white">Continue to publish pipe →</button></div> : null}
  </div>;
}
