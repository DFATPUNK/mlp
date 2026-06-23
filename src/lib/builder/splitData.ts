import type { SplitConfig, SplitResult } from "../../types/builder";

export type SplitRowsResult = {
  trainRows: Record<string, unknown>[];
  validationRows: Record<string, unknown>[];
  testRows: Record<string, unknown>[];
  splitResult: SplitResult;
};

export function buildDefaultSplitConfig(): SplitConfig {
  return {
    train_pct: 70,
    validation_pct: 15,
    test_pct: 15,
    shuffle: true,
    random_seed: 42,
    stratify: "auto_pending_target",
  };
}

function mulberry32(seed: number) {
  return function nextRandom() {
    let value = seed += 0x6d2b79f5;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function withOriginalRowIndex(row: Record<string, unknown>, index: number) {
  if (Object.prototype.hasOwnProperty.call(row, "original_row_index")) return { ...row };
  return { original_row_index: index, ...row };
}

export function createDeterministicShuffle<T>(items: T[], seed: number): T[] {
  const shuffled = [...items];
  const random = mulberry32(seed);

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }

  return shuffled;
}

export function validateSplitConfig(config: SplitConfig, rowCount: number): string[] {
  const errors: string[] = [];
  const totalPct = config.train_pct + config.validation_pct + config.test_pct;
  const trainRows = Math.floor(rowCount * (config.train_pct / 100));
  const validationRows = Math.floor(rowCount * (config.validation_pct / 100));
  const testRows = rowCount - trainRows - validationRows;

  if (totalPct !== 100) errors.push("Split percentages must add up to 100%.");
  if (config.train_pct <= 0) errors.push("Training percentage must be greater than 0%.");
  if (config.validation_pct < 0) errors.push("Validation percentage must be 0% or greater.");
  if (config.test_pct <= 0) errors.push("Test percentage must be greater than 0%.");
  if (rowCount < 2) errors.push("The dataset needs at least 2 rows to create train and test splits.");
  if (trainRows <= 0) errors.push("The training split would be empty.");
  if (testRows <= 0) errors.push("The test split would be empty.");
  if (config.validation_pct > 0 && validationRows <= 0) errors.push("The validation split would be empty.");

  return errors;
}

export function getSplitWarnings(config: SplitConfig, rowCount: number, pipeType: "tabular_classification" | "tabular_regression" | null): string[] {
  const warnings: string[] = [];
  const trainRows = Math.floor(rowCount * (config.train_pct / 100));
  const validationRows = Math.floor(rowCount * (config.validation_pct / 100));
  const testRows = rowCount - trainRows - validationRows;

  if (rowCount < 50) warnings.push("This dataset is small. The test set may not reliably represent real performance.");
  if (validationRows > 0 && validationRows < 10) warnings.push("The validation split has very few rows. Results may be unstable.");
  if (testRows > 0 && testRows < 10) warnings.push("The test split has very few rows. Results may be unstable.");
  if (pipeType === "tabular_classification") warnings.push("Stratification will be handled after target selection if needed.");
  if (trainRows < 10) warnings.push("The training split has very few rows. Model training may be unstable.");

  return warnings;
}

export function splitRows(rows: Record<string, unknown>[], config: SplitConfig): SplitRowsResult {
  const indexedRows = rows.map(withOriginalRowIndex);
  const orderedRows = config.shuffle ? createDeterministicShuffle(indexedRows, config.random_seed) : indexedRows;
  const rowsTotal = orderedRows.length;
  const trainCount = Math.floor(rowsTotal * (config.train_pct / 100));
  const validationCount = Math.floor(rowsTotal * (config.validation_pct / 100));
  const trainRows = orderedRows.slice(0, trainCount);
  const validationRows = orderedRows.slice(trainCount, trainCount + validationCount);
  const testRows = orderedRows.slice(trainCount + validationCount);

  return {
    trainRows,
    validationRows,
    testRows,
    splitResult: {
      rows_total: rowsTotal,
      train_rows: trainRows.length,
      validation_rows: validationRows.length,
      test_rows: testRows.length,
    },
  };
}
