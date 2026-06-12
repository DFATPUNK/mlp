export type BuilderStepKey =
  | "select_dataset"
  | "clean_data"
  | "split_data"
  | "choose_target"
  | "train_models"
  | "review_results"
  | "test_prediction"
  | "publish_pipe";

export type DatasetProvider = "huggingface" | "airtable" | "google_sheets";

export type IssueSeverity = "blocking" | "warning";

export type DatasetIssue = {
  code: string;
  message: string;
  severity: IssueSeverity;
  column?: string;
};

export type ColumnProfile = {
  name: string;
  type: "number" | "boolean" | "categorical" | "text" | "datetime" | "unknown" | "object";
  missingCount: number;
  missingRatio: number;
  uniqueCount: number;
  uniqueRatio: number;
  sampleValues: Array<string | number | boolean | null>;
};

export type DatasetEligibility = {
  eligible: boolean;
  blocking_issues: DatasetIssue[];
  warnings: DatasetIssue[];
};

export type DatasetProfile = {
  rowCount: number;
  columnCount: number;
  columns: ColumnProfile[];
  missingValues: { globalMissingRatio: number };
  candidateTargetColumns: string[];
  eligibility: DatasetEligibility;
  preview: Record<string, unknown>[];
};

export type ProviderConnection = {
  id: string;
  provider: DatasetProvider;
  providerAccountId: string;
  providerAccountLabel: string;
  scopes: string[];
  expiresAt: string | null;
};

export type ProviderDataset = {
  externalId: string;
  name: string;
  url?: string;
  sourceLabel: string;
  sourceConfig: Record<string, unknown>;
  rows: Record<string, unknown>[];
};

export type CleaningStrategy =
  | "median"
  | "mean"
  | "most_frequent"
  | "unknown"
  | "remove_rows"
  | "leave_as_is";

export type ColumnCleaningPlan = {
  name: string;
  detected_type: ColumnProfile["type"];
  issues: string[];
  missing_count: number;
  special_missing_token_count?: number;
  selected_strategy?: CleaningStrategy;
  recommended_strategy?: CleaningStrategy;
  recommended_role?: "feature" | "exclude_from_features" | "target_candidate";
};

export type CleaningPlan = {
  duplicateRows: {
    action: "remove" | "keep";
  };
  columns: ColumnCleaningPlan[];
};

export type CleaningResult = {
  rows_before: number;
  rows_after: number;
  columns_before: number;
  columns_after: number;
  missing_values_before: number;
  missing_values_after: number;
  duplicate_rows_before: number;
  duplicate_rows_removed: number;
  excluded_feature_columns: string[];
};

export type SplitConfig = {
  train_pct: number;
  validation_pct: number;
  test_pct: number;
  shuffle: boolean;
  random_seed: number;
  stratify: "auto_pending_target" | false;
};

export type SplitResult = {
  rows_total: number;
  train_rows: number;
  validation_rows: number;
  test_rows: number;
};

export type ColumnSummary = {
  name: string;
  detected_type: "numeric" | "boolean" | "categorical" | "datetime" | "text" | "unknown";
  row_count: number;
  missing_count: number;
  unique_count: number;
  unique_ratio: number;
  sample_values: unknown[];
  is_id_like: boolean;
  is_constant: boolean;
  has_many_missing_values: boolean;
  is_long_text: boolean;
};

export type TargetAnalysis = {
  detected_type: ColumnSummary["detected_type"];
  unique_count: number;
  unique_ratio: number;
  missing_count: number;
  sample_values: unknown[];
  is_constant: boolean;
  is_id_like: boolean;
  is_long_text: boolean;
  warnings: string[];
  blocking_reasons: string[];
  reason: string;
  detected_task_type: "tabular_classification" | "tabular_regression";
};

export type TargetConfig = {
  previous_split_dataset_artifact_id: string;
  target_column: string;
  detected_task_type: "tabular_classification" | "tabular_regression";
  pipe_type: "tabular_classification" | "tabular_regression";
  task_type_mismatch: boolean;
  feature_columns: string[];
  excluded_feature_columns: string[];
  target_analysis: TargetAnalysis;
  column_summaries: ColumnSummary[];
};
