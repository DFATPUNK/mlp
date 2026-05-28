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
    .order("created_at", { ascending: true });

  if (error) throw error;

  return ((data ?? []) as PipeRow[]).map(mapPipe);
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

export async function getSelectDatasetStepOutput(
  pipeId: string,
): Promise<SelectDatasetStepOutput | null> {
  const { data, error } = await supabase
    .from("pipe_step_outputs")
    .select("output, status")
    .eq("pipe_id", pipeId)
    .eq("step_key", "select_dataset")
    .maybeSingle();

  if (error) throw error;
  if (!data || data.status !== "completed") return null;

  return data.output as SelectDatasetStepOutput;
}
