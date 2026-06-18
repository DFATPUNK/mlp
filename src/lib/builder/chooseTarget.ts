import type { BuilderPipeType } from "../../types/pipe";
import type { ColumnSummary, TargetAnalysis } from "../../types/builder";
import { isMissingValue } from "./cleanData";

const COMMON_TARGET_NAMES = ["target", "label", "class", "outcome", "y", "income", "price", "saleprice", "sale_price", "churn"];
const MAX_SAMPLE_VALUES = 5;

function isPlainObject(value: unknown) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizedString(value: unknown) {
  return String(value).trim().toLowerCase();
}

function isDateLike(value: unknown) {
  if (typeof value !== "string") return false;
  const trimmed = value.trim();
  if (!trimmed) return false;
  return /^\d{4}-\d{1,2}-\d{1,2}/.test(trimmed) && !Number.isNaN(Date.parse(trimmed));
}

function detectColumnType(values: unknown[]): ColumnSummary["detected_type"] {
  const present = values.filter((value) => !isMissingValue(value));
  if (!present.length) return "unknown";
  if (present.some((value) => isPlainObject(value) || Array.isArray(value))) return "unknown";
  if (present.every((value) => typeof value === "boolean" || ["true", "false"].includes(normalizedString(value)))) return "boolean";
  if (present.every((value) => typeof value === "number" || (typeof value === "string" && value.trim() !== "" && !Number.isNaN(Number(value))))) return "numeric";
  if (present.every(isDateLike)) return "datetime";
  const stringValues = present.filter((value) => typeof value === "string") as string[];
  const averageLength = stringValues.reduce((sum, value) => sum + value.length, 0) / Math.max(stringValues.length, 1);
  if (stringValues.length === present.length && averageLength > 80) return "text";
  return "categorical";
}

function uniqueValues(values: unknown[]) {
  const seen = new Map<string, unknown>();
  values.filter((value) => !isMissingValue(value)).forEach((value) => {
    const key = JSON.stringify(value);
    if (!seen.has(key)) seen.set(key, value);
  });
  return [...seen.values()];
}

export function summarizeColumns(rows: Record<string, unknown>[]): ColumnSummary[] {
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  return columns.map((name) => {
    const values = rows.map((row) => row[name]);
    const unique = uniqueValues(values);
    const detectedType = detectColumnType(values);
    const stringValues = values.filter((value) => typeof value === "string") as string[];
    const averageLength = stringValues.reduce((sum, value) => sum + value.length, 0) / Math.max(stringValues.length, 1);
    const uniqueRatio = rows.length ? unique.length / rows.length : 0;
    const missingCount = values.filter((value) => isMissingValue(value)).length;

    return {
      name,
      detected_type: detectedType,
      row_count: rows.length,
      missing_count: missingCount,
      unique_count: unique.length,
      unique_ratio: uniqueRatio,
      sample_values: unique.slice(0, MAX_SAMPLE_VALUES),
      is_id_like: name.toLowerCase().endsWith("id") || uniqueRatio > 0.95,
      is_constant: unique.length <= 1,
      has_many_missing_values: rows.length > 0 && missingCount / rows.length > 0.2,
      is_long_text: detectedType === "text" || averageLength > 120,
    };
  });
}

export function inferTaskType(summary: ColumnSummary): { taskType: BuilderPipeType; reason: string; warnings: string[] } {
  if (summary.detected_type === "boolean") {
    return { taskType: "tabular_classification", reason: "This column contains true/false values.", warnings: [] };
  }

  if (summary.detected_type === "numeric") {
    if (summary.unique_count <= Math.min(10, Math.max(2, Math.floor(summary.row_count * 0.05)))) {
      return {
        taskType: "tabular_classification",
        reason: "This column has only a few distinct numeric values, so MLP treats it like categories.",
        warnings: ["This column is numeric but only has a few distinct values. It may represent categories rather than a continuous number."],
      };
    }
    return { taskType: "tabular_regression", reason: "This column contains many numeric values, so MLP treats it as something to predict as a number.", warnings: [] };
  }

  return { taskType: "tabular_classification", reason: "This column contains categories.", warnings: [] };
}

export function analyzeTargetColumn(rows: Record<string, unknown>[], columnName: string, pipeType: BuilderPipeType): TargetAnalysis {
  const summary = summarizeColumns(rows).find((column) => column.name === columnName);
  if (!summary) {
    return {
      detected_type: "unknown",
      unique_count: 0,
      unique_ratio: 0,
      missing_count: 0,
      sample_values: [],
      is_constant: false,
      is_id_like: false,
      is_long_text: false,
      warnings: [],
      blocking_reasons: ["This column was not found in the training split."],
      reason: "Choose a column from the training split.",
      detected_task_type: pipeType,
    };
  }

  const inferred = inferTaskType(summary);
  const warnings = [...inferred.warnings];
  const blockingReasons: string[] = [];

  if (summary.is_constant) blockingReasons.push("This target has only one value. A model cannot learn what to predict.");
  if (summary.is_long_text) blockingReasons.push("Text prediction is not supported in this MVP. Choose a category or number column instead.");
  if (summary.has_many_missing_values || summary.missing_count > 0) warnings.push("This target has missing values. Rows with missing target values cannot be used for training.");
  if (summary.is_id_like) warnings.push("This column looks like an ID. It has many unique values and is probably not a useful prediction target.");
  if (inferred.taskType !== pipeType) {
    warnings.push(pipeType === "tabular_classification"
      ? "You created a classification pipe, but this target looks like a regression target. MLP will save this choice and training can decide how strict to be later."
      : "You created a regression pipe, but this target looks like a classification target. MLP will save this choice and training can decide how strict to be later.");
  }

  return {
    detected_type: summary.detected_type,
    unique_count: summary.unique_count,
    unique_ratio: summary.unique_ratio,
    missing_count: summary.missing_count,
    sample_values: summary.sample_values,
    is_constant: summary.is_constant,
    is_id_like: summary.is_id_like,
    is_long_text: summary.is_long_text,
    warnings,
    blocking_reasons: blockingReasons,
    reason: inferred.reason,
    detected_task_type: inferred.taskType,
  };
}

export function findRecommendedTarget(columns: ColumnSummary[], targetHint?: string | null) {
  const normalizedColumns = new Map(columns.map((column) => [column.name.toLowerCase(), column.name]));
  if (targetHint) {
    const hinted = normalizedColumns.get(targetHint.toLowerCase());
    if (hinted) return { columnName: hinted, reason: "MLP suggests this target because the dataset metadata names it as the likely prediction column." };
  }

  for (const commonName of COMMON_TARGET_NAMES) {
    const match = normalizedColumns.get(commonName);
    if (match) return { columnName: match, reason: "MLP suggests this target because its name is commonly used for prediction columns." };
  }

  return null;
}

export function buildTargetConfig(input: {
  previousSplitDatasetArtifactId: string;
  targetColumn: string;
  pipeType: BuilderPipeType;
  columnSummaries: ColumnSummary[];
  targetAnalysis: TargetAnalysis;
  excludedFeatureColumns: string[];
}) {
  const allColumns = input.columnSummaries.map((column) => column.name);
  const excluded = new Set(input.excludedFeatureColumns);
  const featureColumns = allColumns.filter((column) => column !== input.targetColumn && !excluded.has(column));
  const taskTypeMismatch = input.targetAnalysis.detected_task_type !== input.pipeType;

  return {
    previous_split_dataset_artifact_id: input.previousSplitDatasetArtifactId,
    target_column: input.targetColumn,
    detected_task_type: input.targetAnalysis.detected_task_type,
    pipe_type: input.pipeType,
    task_type_mismatch: taskTypeMismatch,
    feature_columns: featureColumns,
    excluded_feature_columns: [...excluded],
    target_analysis: input.targetAnalysis,
    column_summaries: input.columnSummaries,
  };
}
