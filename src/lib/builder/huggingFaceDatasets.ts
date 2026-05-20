import type { ProviderDataset } from "../../types/builder";

const HF_API = "https://datasets-server.huggingface.co";
const PAGE_SIZE = 100;
export const HF_MAX_ROWS = 1000;

type HuggingFaceAllowlistedDataset = {
  datasetId: string;
  config: string;
  split: string;
  label: string;
  description: string;
  recommendedTaskType?: "tabular_classification" | "tabular_regression";
  targetHint?: string;
};

export const HUGGING_FACE_ALLOWLIST: HuggingFaceAllowlistedDataset[] = [
  { datasetId: "scikit-learn/adult-census-income", config: "default", split: "train", label: "Adult Census Income", description: "Income prediction from census-like tabular features.", recommendedTaskType: "tabular_classification", targetHint: "income" },
  { datasetId: "mstz/heart-disease", config: "default", split: "train", label: "Heart Disease", description: "Binary classification for heart disease risk.", recommendedTaskType: "tabular_classification", targetHint: "target" },
  { datasetId: "scikit-learn/ames-housing", config: "default", split: "train", label: "Ames Housing", description: "House price regression from tabular features.", recommendedTaskType: "tabular_regression", targetHint: "SalePrice" },
];

type ViewerRow = { row: Record<string, unknown> };
type ViewerResponse = { rows: ViewerRow[]; num_rows_total?: number };

export function listHuggingFaceDatasets(): ProviderDataset[] {
  return HUGGING_FACE_ALLOWLIST.map((item) => ({
    externalId: `hf:${item.datasetId}:${item.config}:${item.split}`,
    name: item.label,
    sourceLabel: `Hugging Face Datasets • ${item.label}`,
    sourceConfig: {
      dataset_id: item.datasetId,
      config: item.config,
      split: item.split,
      dataset_key: getDatasetKey(item.datasetId, item.config, item.split),
      description: item.description,
      recommended_task_type: item.recommendedTaskType ?? null,
      target_hint: item.targetHint ?? null,
    },
    rows: [],
  }));
}

export function getDatasetKey(datasetId: string, config: string, split: string): string {
  return `${datasetId}:${config}:${split}`;
}

export async function fetchHuggingFaceDatasetRows(
  dataset: ProviderDataset,
  options?: { signal?: AbortSignal; maxRows?: number },
): Promise<Record<string, unknown>[]> {
  const source = dataset.sourceConfig as { dataset_id: string; config: string; split: string };
  const maxRows = options?.maxRows ?? HF_MAX_ROWS;
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
    if (!response.ok) throw new Error(`Failed to fetch Hugging Face rows (${response.status}).`);

    const payload = (await response.json()) as ViewerResponse;
    if (typeof payload.num_rows_total === "number") numRowsTotal = payload.num_rows_total;

    const pageRows = (payload.rows ?? []).map((item) => normalizeRow(item.row));
    if (!pageRows.length) break;

    allRows.push(...pageRows);

    if (allRows.length >= maxRows) break;
    if (numRowsTotal !== null && offset + PAGE_SIZE >= numRowsTotal) break;
  }

  return allRows.slice(0, maxRows);
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
