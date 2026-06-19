import { supabase } from "./supabaseClient";
import type { Pipe } from "../types/pipe";

type PipeRow = {
  id: string;
  name: string;
  description: string | null;
  type: Pipe["type"];
  status: Pipe["status"];
  is_template: boolean;
  created_at: string;
  updated_at: string;
};

export type SelectDatasetStepOutput = {
  step_key: "select_dataset";
  status: "completed";
  dataset_artifact_id: string;
  dataset_source_id: string;
  provider: "huggingface" | "airtable" | "google_sheets";
  source_label: string;
  row_count: number;
  column_count: number;
  columns: unknown[];
  eligibility: {
    eligible: boolean;
    blocking_issues: unknown[];
    warnings: unknown[];
  };
  storage: {
    format: "json";
    uri: string;
  };
};

export type CleanDataStepOutput = {
  step_key: "clean_data";
  status: "completed";
  cleaned_dataset_artifact_id: string;
  previous_dataset_artifact_id: string;
  rows_before: number;
  rows_after: number;
  columns_before: number;
  columns_after: number;
  missing_values_before: number;
  missing_values_after: number;
  duplicate_rows_removed: number;
  excluded_feature_columns: string[];
  storage: { format: "json"; uri: string };
};


export type SplitDataStepOutput = {
  step_key: "split_data";
  status: "completed";
  split_dataset_artifact_id: string;
  previous_cleaned_dataset_artifact_id: string;
  rows_total: number;
  train_rows: number;
  validation_rows: number;
  test_rows: number;
  split_config: {
    train_pct: number;
    validation_pct: number;
    test_pct: number;
    shuffle: boolean;
    random_seed: number;
    stratify: "auto_pending_target" | false;
  };
  storage: { format: "json"; uri: string };
};


export type ChooseTargetStepOutput = {
  step_key: "choose_target";
  status: "completed";
  target_config_artifact_id: string;
  previous_split_dataset_artifact_id: string;
  target_column: string;
  detected_task_type: "tabular_classification" | "tabular_regression";
  pipe_type: "tabular_classification" | "tabular_regression";
  task_type_mismatch: boolean;
  feature_columns: string[];
  excluded_feature_columns: string[];
  storage: { format: "json"; uri: string };
};


export type TrainModelsStepOutput = {
  step_key: "train_models";
  status: "completed";
  trained_models_artifact_id: string;
  previous_target_config_artifact_id: string;
  task_type: "tabular_classification" | "tabular_regression";
  target_column: string;
  recommended_model_id: string;
  recommended_model_name: string;
  primary_metric_name: string;
  primary_metric_value: number;
  model_count: number;
  storage: { format: "json"; uri: string };
};



export type ReviewResultsStepOutput = {
  step_key: "review_results";
  status: "completed";
  review_results_artifact_id: string;
  previous_trained_models_artifact_id: string;
  recommended_model_name: string;
  primary_metric_name: string;
  primary_metric_value: number;
  review_acknowledged?: boolean;
  review_acknowledged_at?: string;
  storage: { format: "json"; uri: string };
};





export type TestPredictionStepOutput = {
  step_key: "test_prediction";
  status: "completed";
  test_prediction_artifact_id: string;
  previous_review_results_artifact_id: string;
  prediction: string | number | boolean;
  confidence: number | null;
  model_name: string;
  storage: { format: "json"; uri: string };
};

export type BuilderStepKey =
  | "select_dataset"
  | "clean_data"
  | "split_data"
  | "choose_target"
  | "train_models"
  | "review_results"
  | "test_prediction"
  | "publish_pipe";

export type PipeCardMetadata = {
  completedStepCount: number;
  totalStepCount: 8;
  currentStepLabel: string;
  nextStepLabel: string | null;
  datasetLabel: string | null;
  targetColumn: string | null;
  taskType: string | null;
  recommendedModelName: string | null;
  primaryMetricName: string | null;
  primaryMetricValue: number | null;
  lastPrediction: string | number | boolean | null;
  summary: string;
};

export type PipeWithCardMetadata = {
  pipe: Pipe;
  metadata: PipeCardMetadata;
};

type StepOutputRow = {
  pipe_id: string;
  step_key: BuilderStepKey | string;
  status: string;
  output: Record<string, unknown> | null;
};

const BUILDER_STEPS: Array<{ key: BuilderStepKey; label: string }> = [
  { key: "select_dataset", label: "Select dataset" },
  { key: "clean_data", label: "Clean data" },
  { key: "split_data", label: "Split data" },
  { key: "choose_target", label: "Choose target" },
  { key: "train_models", label: "Train models" },
  { key: "review_results", label: "Review results" },
  { key: "test_prediction", label: "Test prediction" },
  { key: "publish_pipe", label: "Publish pipe" },
];

export type ArtifactRecord = {
  id: string;
  content: unknown;
  metadata: Record<string, unknown> | null;
};


function asString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function predictionPhrase(datasetLabel: string | null, targetColumn: string | null): string | null {
  if (!targetColumn) return null;
  const dataset = (datasetLabel ?? "").toLowerCase();
  const target = targetColumn.toLowerCase().replaceAll("_", "").replaceAll(".", "");
  if (dataset.includes("adult") && target === "income") return "predict whether income is <=50K or >50K";
  if (dataset.includes("heart") && ["hasheartdisease", "hashearthdisease", "target", "disease"].includes(target)) return "predict whether a patient may have heart disease";
  if ((dataset.includes("ames") || dataset.includes("housing")) && ["saleprice", "salepriceusd", "price"].includes(target)) return "predict house sale price";
  return `predict ${targetColumn}`;
}

function reviewingPhrase(datasetLabel: string | null, targetColumn: string | null): string {
  const phrase = predictionPhrase(datasetLabel, targetColumn) ?? `predict ${targetColumn ?? "the target"}`;
  return phrase.startsWith("predict ") ? phrase.replace("predict ", "predicting ") : phrase;
}

function buildPipeSummary(pipe: Pipe, metadata: Omit<PipeCardMetadata, "summary">): string {
  const pipeKind = pipe.type === "tabular_regression" ? "regression" : pipe.type === "tabular_classification" ? "classification" : pipe.type.replaceAll("_", " ");
  if (metadata.lastPrediction !== null) {
    return `Last tested prediction: ${String(metadata.lastPrediction)}.`;
  }
  if (metadata.datasetLabel && metadata.targetColumn && metadata.recommendedModelName && metadata.completedStepCount >= 6) {
    return `Reviewed ${metadata.recommendedModelName} on ${metadata.datasetLabel} for ${reviewingPhrase(metadata.datasetLabel, metadata.targetColumn)}.`;
  }
  if (metadata.datasetLabel && metadata.targetColumn && metadata.recommendedModelName) {
    return `Trained on ${metadata.datasetLabel} to ${predictionPhrase(metadata.datasetLabel, metadata.targetColumn) ?? `predict ${metadata.targetColumn}`}. Recommended model: ${metadata.recommendedModelName}.`;
  }
  if (metadata.datasetLabel && metadata.targetColumn) {
    return `Pipeline using ${metadata.datasetLabel} to ${predictionPhrase(metadata.datasetLabel, metadata.targetColumn) ?? `predict ${metadata.targetColumn}`}.`;
  }
  if (metadata.datasetLabel) {
    return `Pipeline using ${metadata.datasetLabel}. Choose a target column next.`;
  }
  return `Draft ${pipeKind} pipe. Select a dataset to begin.`;
}

function buildPipeCardMetadata(pipe: Pipe, outputs: StepOutputRow[]): PipeCardMetadata {
  const byStep = new Map(outputs.filter((row) => row.status === "completed").map((row) => [row.step_key, row.output ?? {}]));
  const selectOutput = byStep.get("select_dataset") ?? {};
  const targetOutput = byStep.get("choose_target") ?? {};
  const trainOutput = byStep.get("train_models") ?? {};
  const reviewOutput = byStep.get("review_results") ?? {};
  const testOutput = byStep.get("test_prediction") ?? {};
  const isStepComplete = (stepKey: BuilderStepKey) => {
    if (!byStep.has(stepKey)) return false;
    if (stepKey !== "review_results") return true;
    const reviewAcknowledged = byStep.get("review_results")?.review_acknowledged;
    return reviewAcknowledged === undefined || reviewAcknowledged === true;
  };
  const completedStepCount = BUILDER_STEPS.filter((step) => isStepComplete(step.key)).length;
  const firstIncomplete = BUILDER_STEPS.find((step) => !isStepComplete(step.key));
  const baseMetadata = {
    completedStepCount,
    totalStepCount: 8 as const,
    currentStepLabel: firstIncomplete?.label ?? "Complete",
    nextStepLabel: firstIncomplete?.label ?? null,
    datasetLabel: asString(selectOutput.source_label),
    targetColumn: asString(targetOutput.target_column),
    taskType: asString(targetOutput.detected_task_type) ?? asString(trainOutput.task_type),
    recommendedModelName: asString(reviewOutput.recommended_model_name) ?? asString(trainOutput.recommended_model_name),
    primaryMetricName: asString(reviewOutput.primary_metric_name) ?? asString(trainOutput.primary_metric_name),
    primaryMetricValue: asNumber(reviewOutput.primary_metric_value) ?? asNumber(trainOutput.primary_metric_value),
    lastPrediction: typeof testOutput.prediction === "string" || typeof testOutput.prediction === "number" || typeof testOutput.prediction === "boolean" ? testOutput.prediction : null,
  };
  return { ...baseMetadata, summary: buildPipeSummary(pipe, baseMetadata) };
}

function mapPipe(pipe: PipeRow): Pipe {
  return {
    id: pipe.id,
    name: pipe.name,
    description: pipe.description ?? "",
    type: pipe.type,
    status: pipe.status,
    isTemplate: pipe.is_template,
    createdAt: pipe.created_at,
    updatedAt: pipe.updated_at,
  };
}

export async function getPipes(): Promise<Pipe[]> {
  const { data, error } = await supabase
    .from("pipes")
    .select("id, name, description, type, status, is_template, created_at, updated_at")
    .order("updated_at", { ascending: false });

  if (error) throw error;

  return ((data ?? []) as PipeRow[]).map(mapPipe);
}



export async function getPipesWithCardMetadata(): Promise<PipeWithCardMetadata[]> {
  const pipes = await getPipes();
  const pipeIds = pipes.map((pipe) => pipe.id);
  if (!pipeIds.length) return [];

  const { data, error } = await supabase
    .from("pipe_step_outputs")
    .select("pipe_id, step_key, status, output")
    .in("pipe_id", pipeIds)
    .eq("status", "completed");

  if (error) throw error;

  const outputsByPipe = new Map<string, StepOutputRow[]>();
  for (const row of (data ?? []) as StepOutputRow[]) {
    const current = outputsByPipe.get(row.pipe_id) ?? [];
    current.push(row);
    outputsByPipe.set(row.pipe_id, current);
  }

  return pipes.map((pipe) => ({
    pipe,
    metadata: buildPipeCardMetadata(pipe, outputsByPipe.get(pipe.id) ?? []),
  }));
}

export async function getPipeById(pipeId: string): Promise<Pipe | null> {
  const { data, error } = await supabase
    .from("pipes")
    .select("id, name, description, type, status, is_template, created_at, updated_at")
    .eq("id", pipeId)
    .maybeSingle();

  if (error) throw error;
  if (!data) return null;

  return mapPipe(data as PipeRow);
}

export async function deletePipe(pipeId: string): Promise<void> {
  const { error } = await supabase.from("pipes").delete().eq("id", pipeId);
  if (error) throw error;
}

export async function updatePipeName(pipeId: string, name: string): Promise<Pipe> {
  const trimmed = name.trim();
  if (!trimmed) throw new Error("Pipe name cannot be empty.");

  const { data, error } = await supabase
    .from("pipes")
    .update({ name: trimmed })
    .eq("id", pipeId)
    .select("id, name, description, type, status, is_template, created_at, updated_at")
    .single();

  if (error) throw error;
  return mapPipe(data as PipeRow);
}

export async function getStepOutput<T>(pipeId: string, stepKey: string): Promise<T | null> {
  const { data, error } = await supabase
    .from("pipe_step_outputs")
    .select("output, status")
    .eq("pipe_id", pipeId)
    .eq("step_key", stepKey)
    .maybeSingle();

  if (error) throw error;
  if (!data || data.status !== "completed") return null;

  return data.output as T;
}

export async function getSelectDatasetStepOutput(pipeId: string): Promise<SelectDatasetStepOutput | null> {
  return getStepOutput<SelectDatasetStepOutput>(pipeId, "select_dataset");
}

export async function getCleanDataStepOutput(pipeId: string): Promise<CleanDataStepOutput | null> {
  return getStepOutput<CleanDataStepOutput>(pipeId, "clean_data");
}

export async function getSplitDataStepOutput(pipeId: string): Promise<SplitDataStepOutput | null> {
  return getStepOutput<SplitDataStepOutput>(pipeId, "split_data");
}


export async function getChooseTargetStepOutput(pipeId: string): Promise<ChooseTargetStepOutput | null> {
  return getStepOutput<ChooseTargetStepOutput>(pipeId, "choose_target");
}


export async function getTrainModelsStepOutput(pipeId: string): Promise<TrainModelsStepOutput | null> {
  return getStepOutput<TrainModelsStepOutput>(pipeId, "train_models");
}

export async function getReviewResultsStepOutput(pipeId: string): Promise<ReviewResultsStepOutput | null> {
  return getStepOutput<ReviewResultsStepOutput>(pipeId, "review_results");
}

export function isReviewResultsAcknowledged(output: ReviewResultsStepOutput | null): boolean {
  if (!output) return false;
  return output.review_acknowledged === undefined || output.review_acknowledged === true;
}

export async function acknowledgeReviewResults(pipeId: string): Promise<ReviewResultsStepOutput> {
  const current = await getReviewResultsStepOutput(pipeId);
  if (!current) throw new Error("Review results have not been generated yet.");
  const acknowledgedOutput: ReviewResultsStepOutput = {
    ...current,
    review_acknowledged: true,
    review_acknowledged_at: new Date().toISOString(),
  };
  const { data, error } = await supabase
    .from("pipe_step_outputs")
    .update({ output: acknowledgedOutput })
    .eq("pipe_id", pipeId)
    .eq("step_key", "review_results")
    .select("output")
    .single();

  if (error) throw error;
  return (data.output ?? acknowledgedOutput) as ReviewResultsStepOutput;
}

export async function getTestPredictionStepOutput(pipeId: string): Promise<TestPredictionStepOutput | null> {
  return getStepOutput<TestPredictionStepOutput>(pipeId, "test_prediction");
}

export async function getArtifactById(artifactId: string): Promise<ArtifactRecord | null> {
  const { data, error } = await supabase
    .from("artifacts")
    .select("id, content, metadata")
    .eq("id", artifactId)
    .maybeSingle();

  if (error) throw error;
  if (!data) return null;

  return data as ArtifactRecord;
}

export async function createDraftPipe(
  ownerId: string,
  pipeType: "tabular_classification" | "tabular_regression",
): Promise<Pipe> {
  const name = pipeType === "tabular_regression"
    ? "Untitled regression pipe"
    : "Untitled classification pipe";

  const { data, error } = await supabase
    .from("pipes")
    .insert({
      owner_id: ownerId,
      name,
      description: "Draft pipe created by builder",
      type: pipeType,
      status: "draft",
      is_template: false,
    })
    .select("id, name, description, type, status, is_template, created_at, updated_at")
    .single();

  if (error) throw error;

  return mapPipe(data as PipeRow);
}
