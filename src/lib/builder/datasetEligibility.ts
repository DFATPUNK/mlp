import type { ColumnProfile, DatasetEligibility, DatasetIssue } from "../../types/builder";

export const DATASET_LIMITS = {
  maxRows: 1000,
  minRows: 50,
  maxColumns: 50,
  maxGlobalMissingRatio: 0.4,
  maxColumnMissingRatio: 0.8,
  maxUniqueRatioForCategorical: 0.5,
  maxTextLengthForTabular: 120,
};

export function getDatasetEligibility(input: {
  rowCount: number;
  columns: ColumnProfile[];
  missingValues: { globalMissingRatio: number };
  candidateTargetColumns: string[];
}): DatasetEligibility {
  const blocking_issues: DatasetIssue[] = [];
  const warnings: DatasetIssue[] = [];

  if (input.rowCount > DATASET_LIMITS.maxRows) blocking_issues.push({ code: "too_many_rows", message: "This dataset is too large for the MVP.", severity: "blocking" });
  if (input.rowCount < DATASET_LIMITS.minRows) blocking_issues.push({ code: "too_few_rows", message: "This dataset needs at least 50 rows.", severity: "blocking" });
  if (input.columns.length > DATASET_LIMITS.maxColumns) blocking_issues.push({ code: "too_many_columns", message: "This dataset has too many columns for the MVP.", severity: "blocking" });
  if (input.columns.some((column) => column.type === "object")) blocking_issues.push({ code: "nested_data", message: "Nested/object columns are not supported.", severity: "blocking" });
  if (!input.columns.some((column) => column.type !== "object" && column.type !== "unknown")) blocking_issues.push({ code: "no_usable_columns", message: "We could not find usable columns.", severity: "blocking" });
  if (!input.candidateTargetColumns.length) blocking_issues.push({ code: "no_target_candidate", message: "We could not find a possible target column yet.", severity: "blocking" });

  input.columns.forEach((column) => {
    if (column.name.toLowerCase().endsWith("id") || column.uniqueRatio > 0.95) warnings.push({ code: "id_like", message: "This column looks like an ID and may be excluded later.", severity: "warning", column: column.name });
    if (column.missingRatio > DATASET_LIMITS.maxColumnMissingRatio) warnings.push({ code: "high_missing", message: "This column has many missing values.", severity: "warning", column: column.name });
    if (column.type === "categorical" && column.uniqueRatio > DATASET_LIMITS.maxUniqueRatioForCategorical) warnings.push({ code: "high_cardinality", message: "This categorical column has many unique values.", severity: "warning", column: column.name });
    if (column.type === "datetime") warnings.push({ code: "likely_datetime", message: "This column looks like a date/time column.", severity: "warning", column: column.name });
    if (column.type === "text") warnings.push({ code: "long_text", message: "This column has long text values and may be excluded later.", severity: "warning", column: column.name });
  });

  if (input.missingValues.globalMissingRatio > DATASET_LIMITS.maxGlobalMissingRatio) warnings.push({ code: "global_missing", message: "This dataset has high overall missing values.", severity: "warning" });

  return { eligible: blocking_issues.length === 0, blocking_issues, warnings };
}
