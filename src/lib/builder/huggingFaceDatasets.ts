import type { ProviderDataset } from "../../types/builder";

const HF_API = "https://datasets-server.huggingface.co";
const PAGE_SIZE = 100;
const MAX_ROWS = 1000;

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
    datasetId: "mstz/heart-disease",
    config: "default",
    split: "train",
    label: "Heart Disease",
    description: "Binary classification for heart disease risk.",
    recommendedTaskType: "tabular_classification",
    targetHint: "target",
  },
  {
    datasetId: "scikit-learn/ames-housing",
    config: "default",
    split: "train",
    label: "Ames Housing",
    description: "House price regression from tabular features.",
    recommendedTaskType: "tabular_regression",
    targetHint: "SalePrice",
  },
];

type ViewerRow = { row: Record<string, unknown> };

type ViewerResponse = { rows: ViewerRow[] };

export function listHuggingFaceDatasets(): ProviderDataset[] {
  return HUGGING_FACE_ALLOWLIST.map((item) => ({
    externalId: `hf:${item.datasetId}:${item.config}:${item.split}`,
    name: item.label,
    sourceLabel: `Hugging Face Datasets • ${item.label}`,
    sourceConfig: {
      dataset_id: item.datasetId,
      config: item.config,
      split: item.split,
      description: item.description,
      recommended_task_type: item.recommendedTaskType ?? null,
      target_hint: item.targetHint ?? null,
    },
    rows: [],
  }));
}

export async function fetchHuggingFaceDatasetRows(dataset: ProviderDataset): Promise<Record<string, unknown>[]> {
  const source = dataset.sourceConfig as {
    dataset_id: string;
    config: string;
    split: string;
  };

  const allRows: Record<string, unknown>[] = [];

  for (let offset = 0; offset < MAX_ROWS; offset += PAGE_SIZE) {
    const url = new URL(`${HF_API}/rows`);
    url.searchParams.set("dataset", source.dataset_id);
    url.searchParams.set("config", source.config);
    url.searchParams.set("split", source.split);
    url.searchParams.set("offset", String(offset));
    url.searchParams.set("length", String(PAGE_SIZE));

    const response = await fetch(url.toString());
    if (!response.ok) {
      throw new Error(`Failed to fetch Hugging Face rows (${response.status}).`);
    }

    const payload = (await response.json()) as ViewerResponse;
    const normalized = (payload.rows ?? []).map((item) => normalizeRow(item.row));
    allRows.push(...normalized);

    if ((payload.rows ?? []).length < PAGE_SIZE) break;
  }

  return allRows.slice(0, MAX_ROWS);
}

function normalizeRow(row: Record<string, unknown>): Record<string, unknown> {
  const normalized: Record<string, unknown> = {};
  Object.entries(row).forEach(([key, value]) => {
    if (value === null || value === undefined) {
      normalized[key] = null;
      return;
    }
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      normalized[key] = value;
      return;
    }
    normalized[key] = JSON.stringify(value);
  });
  return normalized;
}
