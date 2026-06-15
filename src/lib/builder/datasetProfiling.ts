import type { ColumnProfile, DatasetProfile } from "../../types/builder";
import { getDatasetEligibility } from "./datasetEligibility";

function getColumnType(values: unknown[]): ColumnProfile["type"] {
  const defined = values.filter((value) => value !== null && value !== undefined);
  if (!defined.length) return "unknown";
  if (defined.some((value) => typeof value === "object")) return "object";
  if (defined.every((value) => typeof value === "number")) return "number";
  if (defined.every((value) => typeof value === "boolean")) return "boolean";

  const strings = defined.filter((value) => typeof value === "string") as string[];
  if (strings.length === defined.length) {
    if (strings.every((value) => !Number.isNaN(Date.parse(value)))) return "datetime";
    if (strings.some((value) => value.length > 120)) return "text";
    return "categorical";
  }

  return "unknown";
}

export function profileDataset(rows: Record<string, unknown>[]): DatasetProfile {
  const preview = rows.slice(0, 8);
  const columnNames = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  const columns = columnNames.map((name) => {
    const values = rows.map((row) => row[name]);
    const missingCount = values.filter((value) => value === null || value === undefined || value === "").length;
    const uniqueCount = new Set(values.filter((value) => value !== null && value !== undefined && value !== "").map((v) => JSON.stringify(v))).size;

    return {
      name,
      type: getColumnType(values),
      missingCount,
      missingRatio: rows.length ? missingCount / rows.length : 0,
      uniqueCount,
      uniqueRatio: rows.length ? uniqueCount / rows.length : 0,
      sampleValues: values.slice(0, 5).map((value) =>
        value === null || value === undefined ? null : (value as string | number | boolean),
      ),
    };
  });

  const filledCellCount = rows.reduce((acc, row) => {
    return acc + columnNames.filter((column) => row[column] !== null && row[column] !== undefined && row[column] !== "").length;
  }, 0);
  const totalCells = rows.length * Math.max(columnNames.length, 1);
  const missingValues = {
    globalMissingRatio: 1 - filledCellCount / totalCells,
  };
  const candidateTargetColumns = columns
    .filter((column) => column.type === "categorical" || column.type === "number")
    .filter((column) => column.uniqueCount >= 2 && column.uniqueRatio < 0.9)
    .map((column) => column.name);

  return {
    rowCount: rows.length,
    columnCount: columnNames.length,
    columns,
    missingValues,
    candidateTargetColumns,
    eligibility: getDatasetEligibility({ rowCount: rows.length, columns, missingValues, candidateTargetColumns }),
    preview,
  };
}
