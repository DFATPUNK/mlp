import type { CleaningPlan, CleaningResult, CleaningStrategy, ColumnCleaningPlan } from "../../types/builder";
import { profileDataset } from "./datasetProfiling";

const SPECIAL_MISSING_TOKENS = new Set(["", "?", "na", "n/a", "null", "none", "unknown"]);

type QualityAudit = {
  totalMissingValues: number;
  duplicateRows: number;
  columnsWithMissingValues: string[];
  idLikeColumns: string[];
  datetimeColumns: string[];
  longTextColumns: string[];
  highCardinalityCategoricalColumns: string[];
  constantColumns: string[];
  unsupportedColumns: string[];
};

export type AppliedCleaning = {
  cleanedRows: Record<string, unknown>[];
  cleaningResult: CleaningResult;
  profileBefore: ReturnType<typeof profileDataset>;
  profileAfter: ReturnType<typeof profileDataset>;
};

function isMissingValue(value: unknown, treatSpecialTokens: boolean) {
  if (value === null || value === undefined) return true;
  if (!treatSpecialTokens || typeof value !== "string") return false;
  return SPECIAL_MISSING_TOKENS.has(value.trim().toLowerCase());
}

function rowKey(row: Record<string, unknown>) {
  return JSON.stringify(
    Object.keys(row)
      .sort()
      .map((key) => [key, row[key]]),
  );
}

function countDuplicateRows(rows: Record<string, unknown>[]) {
  const seen = new Set<string>();
  let duplicates = 0;

  rows.forEach((row) => {
    const key = rowKey(row);
    if (seen.has(key)) duplicates += 1;
    else seen.add(key);
  });

  return duplicates;
}

function countMissingValues(rows: Record<string, unknown>[], treatSpecialTokens: boolean) {
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  return rows.reduce(
    (total, row) => total + columns.filter((column) => isMissingValue(row[column], treatSpecialTokens)).length,
    0,
  );
}

function specialMissingTokenCount(rows: Record<string, unknown>[], column: string) {
  return rows.filter((row) => typeof row[column] === "string" && SPECIAL_MISSING_TOKENS.has((row[column] as string).trim().toLowerCase())).length;
}

function median(values: number[]) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function mean(values: number[]) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function mostFrequent(values: unknown[]) {
  const counts = new Map<string, { value: unknown; count: number }>();
  values.forEach((value) => {
    const key = JSON.stringify(value);
    counts.set(key, { value, count: (counts.get(key)?.count ?? 0) + 1 });
  });

  return [...counts.values()].sort((a, b) => b.count - a.count)[0]?.value ?? null;
}

export function auditDataQuality(rows: Record<string, unknown>[]): QualityAudit {
  const profile = profileDataset(rows);

  return {
    totalMissingValues: countMissingValues(rows, true),
    duplicateRows: countDuplicateRows(rows),
    columnsWithMissingValues: profile.columns.filter((column) => column.missingCount + specialMissingTokenCount(rows, column.name) > 0).map((column) => column.name),
    idLikeColumns: profile.columns.filter((column) => column.name.toLowerCase().endsWith("id") || column.uniqueRatio > 0.95).map((column) => column.name),
    datetimeColumns: profile.columns.filter((column) => column.type === "datetime").map((column) => column.name),
    longTextColumns: profile.columns.filter((column) => column.type === "text").map((column) => column.name),
    highCardinalityCategoricalColumns: profile.columns.filter((column) => column.type === "categorical" && column.uniqueRatio > 0.5).map((column) => column.name),
    constantColumns: profile.columns.filter((column) => column.uniqueCount <= 1).map((column) => column.name),
    unsupportedColumns: profile.columns.filter((column) => column.type === "object" || column.type === "unknown").map((column) => column.name),
  };
}

export function generateCleaningPlan(rows: Record<string, unknown>[]): CleaningPlan {
  const profile = profileDataset(rows);
  const audit = auditDataQuality(rows);

  const columns: ColumnCleaningPlan[] = profile.columns.map((column) => {
    const specialCount = specialMissingTokenCount(rows, column.name);
    const missingCount = column.missingCount + specialCount;
    const issues: string[] = [];
    let recommended_strategy: CleaningStrategy | undefined;
    let recommended_role: ColumnCleaningPlan["recommended_role"] = "feature";

    if (missingCount > 0) {
      issues.push("missing_values");
      if (column.type === "number") recommended_strategy = "median";
      else if (column.type === "boolean") recommended_strategy = "most_frequent";
      else recommended_strategy = "unknown";
    }
    if (specialCount > 0) issues.push("special_missing_tokens");
    if (audit.idLikeColumns.includes(column.name)) {
      issues.push("id_like");
      recommended_role = "exclude_from_features";
    }
    if (audit.constantColumns.includes(column.name)) {
      issues.push("constant");
      recommended_role = "exclude_from_features";
    }
    if (audit.datetimeColumns.includes(column.name) || audit.longTextColumns.includes(column.name)) recommended_role = "exclude_from_features";
    if (audit.highCardinalityCategoricalColumns.includes(column.name)) issues.push("high_cardinality");
    if (audit.unsupportedColumns.includes(column.name)) issues.push("unsupported");

    return {
      name: column.name,
      detected_type: column.type,
      issues,
      missing_count: missingCount,
      special_missing_token_count: specialCount,
      selected_strategy: recommended_strategy,
      recommended_strategy,
      recommended_role,
    };
  });

  return {
    duplicateRows: { action: audit.duplicateRows > 0 ? "remove" : "keep" },
    columns,
  };
}

export function applyCleaningPlan(rows: Record<string, unknown>[], plan: CleaningPlan): AppliedCleaning {
  const profileBefore = profileDataset(rows);
  const missingBefore = countMissingValues(rows, true);
  const duplicateRowsBefore = countDuplicateRows(rows);

  let cleanedRows = rows.map((row) => ({ ...row }));
  const columnsToDropRowsFor = new Set(plan.columns.filter((column) => column.selected_strategy === "remove_rows").map((column) => column.name));

  if (plan.duplicateRows.action === "remove") {
    const seen = new Set<string>();
    cleanedRows = cleanedRows.filter((row) => {
      const key = rowKey(row);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  if (columnsToDropRowsFor.size) {
    cleanedRows = cleanedRows.filter((row) => [...columnsToDropRowsFor].every((column) => !isMissingValue(row[column], true)));
  }

  plan.columns.forEach((columnPlan) => {
    const strategy = columnPlan.selected_strategy;
    if (!strategy || strategy === "leave_as_is" || strategy === "remove_rows") return;

    const values = cleanedRows.map((row) => row[columnPlan.name]).filter((value) => !isMissingValue(value, true));
    let fillValue: unknown = null;

    if (strategy === "median") fillValue = median(values.filter((value): value is number => typeof value === "number"));
    if (strategy === "mean") fillValue = mean(values.filter((value): value is number => typeof value === "number"));
    if (strategy === "most_frequent") fillValue = mostFrequent(values);
    if (strategy === "unknown") fillValue = "Unknown";

    cleanedRows = cleanedRows.map((row) => ({
      ...row,
      [columnPlan.name]: isMissingValue(row[columnPlan.name], true) ? fillValue : row[columnPlan.name],
    }));
  });

  const profileAfter = profileDataset(cleanedRows);
  const excluded = plan.columns.filter((column) => column.recommended_role === "exclude_from_features").map((column) => column.name);

  return {
    cleanedRows,
    profileBefore,
    profileAfter,
    cleaningResult: {
      rows_before: rows.length,
      rows_after: cleanedRows.length,
      columns_before: profileBefore.columnCount,
      columns_after: profileAfter.columnCount,
      missing_values_before: missingBefore,
      missing_values_after: countMissingValues(cleanedRows, false),
      duplicate_rows_before: duplicateRowsBefore,
      duplicate_rows_removed: duplicateRowsBefore - countDuplicateRows(cleanedRows),
      excluded_feature_columns: excluded,
    },
  };
}
