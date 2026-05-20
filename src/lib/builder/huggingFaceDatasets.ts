import type { ProviderDataset } from "../../types/builder";

const HF_API = "https://datasets-server.huggingface.co";
const PAGE_SIZE = 100;
export const HF_MAX_ROWS = 1000;

export type HuggingFaceAllowlistedDataset = {
  datasetId: string;
  config: string;
  split: string;
  label: string;
  description: string;
  recommendedTaskType?: "tabular_classification" | "tabular_regression";
  targetHint?: string;
  selectedColumns?: string[];
  maxRows?: number;
};

export type HuggingFaceLoadMeta = {
  datasetKey: string;
  datasetId: string;
  config: string;
  split: string;
  loadedRows: number;
  numRowsTotal: number | null;
  fromCache: boolean;
  loadedAt: string;
  projection: {
    enabled: boolean;
    selectedCount: number;
    keptCount: number;
    missingColumns: string[];
  };
};

export const HUGGING_FACE_ALLOWLIST: HuggingFaceAllowlistedDataset[] = [
  {
    datasetId: "scikit-learn/adult-census-income",
    config: "default",
    split: "train",
    label: "Adult Census Income",
    description: "Income prediction from census-like tabular features.",
    recommendedTaskType: "tabular_classification",
    targetHint: "income",
  },
  {
    datasetId: "mstz/heart",
    config: "hungary",
    split: "train",
    label: "Heart Disease",
    description: "Binary classification dataset for heart disease risk.",
    recommendedTaskType: "tabular_classification",
    targetHint: "has_hearth_disease",
  },
  {
    datasetId: "cloderic/ames_iowa_housing",
    config: "default",
    split: "train",
    label: "Ames Housing",
    description: "House price regression from tabular housing features.",
    recommendedTaskType: "tabular_regression",
    targetHint: "saleprice",
    selectedColumns: [
      "lot_area",
      "overall_qual",
      "overall_cond",
      "year_built",
      "gr_liv_area",
      "garage_cars",
      "garage_area",
      "saleprice",
    ],
  },
];

type ViewerRow = { row: Record<string, unknown> };
type ViewerResponse = { rows: ViewerRow[]; num_rows_total?: number; error?: string };

export function getHuggingFaceDatasetKey(item: Pick<HuggingFaceAllowlistedDataset, "datasetId" | "config" | "split">): string {
  return `${item.datasetId}:${item.config}:${item.split}`;
}

export function listHuggingFaceDatasets(): ProviderDataset[] {
  return HUGGING_FACE_ALLOWLIST.map((item) => ({
    externalId: `hf:${item.datasetId}:${item.config}:${item.split}`,
    name: item.label,
    sourceLabel: `Hugging Face Datasets • ${item.label}`,
    sourceConfig: {
      dataset_id: item.datasetId,
      config: item.config,
      split: item.split,
      dataset_key: getHuggingFaceDatasetKey(item),
      description: item.description,
      recommended_task_type: item.recommendedTaskType ?? null,
      target_hint: item.targetHint ?? null,
      selected_columns: item.selectedColumns ?? null,
      max_rows: item.maxRows ?? null,
    },
    rows: [],
  }));
}

export function getDatasetKey(datasetId: string, config: string, split: string): string {
  return `${datasetId}:${config}:${split}`;
}

export function validateSelectedColumns(rows: Record<string, unknown>[], selectedColumns: string[]): { keptColumns: string[]; missingColumns: string[] } {
  const sampleKeys = new Set(rows.flatMap((row) => Object.keys(row)));
  const keptColumns = selectedColumns.filter((column) => sampleKeys.has(column));
  const missingColumns = selectedColumns.filter((column) => !sampleKeys.has(column));
  return { keptColumns, missingColumns };
}

export function projectRowsToSelectedColumns(rows: Record<string, unknown>[], selectedColumns: string[]): { projectedRows: Record<string, unknown>[]; keptColumns: string[]; missingColumns: string[] } {
  const { keptColumns, missingColumns } = validateSelectedColumns(rows, selectedColumns);
  const projectedRows = rows.map((row) => {
    const projected: Record<string, unknown> = {};
    keptColumns.forEach((column) => {
      projected[column] = row[column] ?? null;
    });
    return projected;
  });
  return { projectedRows, keptColumns, missingColumns };
}

export async function fetchHuggingFaceDatasetRows(dataset: ProviderDataset, options?: { signal?: AbortSignal; maxRows?: number }): Promise<{ rows: Record<string, unknown>[]; meta: HuggingFaceLoadMeta }> {
  const source = dataset.sourceConfig as {
    dataset_id: string;
    config: string;
    split: string;
    selected_columns?: string[] | null;
    max_rows?: number | null;
    target_hint?: string | null;
  };

  const maxRows = options?.maxRows ?? source.max_rows ?? HF_MAX_ROWS;
  const datasetKey = getDatasetKey(source.dataset_id, source.config, source.split);
  const allRows: Record<string, unknown>[] = [];
  let numRowsTotal: number | null = null;

  for (let offset = 0; offset < maxRows; offset += PAGE_SIZE) {
    if (options?.signal?.aborted) throw new DOMException("Aborted", "AbortError");

    const url = new URL(`${HF_API}/rows`);
    url.searchParams.set("dataset", source.dataset_id);
    url.searchParams.set("config", source.config);
    url.searchParams.set("split", source.split);
    url.searchParams.set("offset", String(offset));
    url.searchParams.set("length", String(PAGE_SIZE));

    const response = await fetch(url.toString(), { signal: options?.signal });

    if (!response.ok) {
      let body: string | undefined;
      try {
        body = await response.text();
      } catch {
        body = undefined;
      }
      console.error("[HF rows fetch failed]", {
        datasetId: source.dataset_id,
        config: source.config,
        split: source.split,
        offset,
        requestUrl: url.toString(),
        status: response.status,
        responseBody: body,
      });
      throw new Error(`Failed to fetch Hugging Face rows (${response.status}).`);
    }

    const payload = (await response.json()) as ViewerResponse;
    if (typeof payload.num_rows_total === "number") numRowsTotal = payload.num_rows_total;

    const pageRows = (payload.rows ?? []).map((item) => normalizeRow(item.row));
    if (!pageRows.length) break;

    allRows.push(...pageRows);
    if (allRows.length >= maxRows) break;
    if (numRowsTotal !== null && offset + PAGE_SIZE >= numRowsTotal) break;
  }

  let finalRows = allRows.slice(0, maxRows);
  let keptCount = 0;
  let missingColumns: string[] = [];
  const selectedColumns = source.selected_columns ?? [];

  if (selectedColumns.length > 0) {
    const projected = projectRowsToSelectedColumns(finalRows, selectedColumns);
    finalRows = projected.projectedRows;
    keptCount = projected.keptColumns.length;
    missingColumns = projected.missingColumns;

    if (missingColumns.length > 0) {
      console.warn("[HF selectedColumns missing]", {
        datasetId: source.dataset_id,
        config: source.config,
        split: source.split,
        missingColumns,
      });
    }
  }

  return {
    rows: finalRows,
    meta: {
      datasetKey,
      datasetId: source.dataset_id,
      config: source.config,
      split: source.split,
      loadedRows: finalRows.length,
      numRowsTotal,
      fromCache: false,
      loadedAt: new Date().toISOString(),
      projection: {
        enabled: selectedColumns.length > 0,
        selectedCount: selectedColumns.length,
        keptCount,
        missingColumns,
      },
    },
  };
}

function normalizeRow(row: Record<string, unknown>): Record<string, unknown> {
  const normalized: Record<string, unknown> = {};
  Object.entries(row).forEach(([key, value]) => {
    if (value === null || value === undefined) normalized[key] = null;
    else if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") normalized[key] = value;
    else normalized[key] = JSON.stringify(value);
  });
  return normalized;
}
