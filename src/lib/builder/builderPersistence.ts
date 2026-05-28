import { supabase } from "../supabaseClient";
import type { DatasetProfile, DatasetProvider, ProviderConnection, ProviderDataset } from "../../types/builder";

export async function persistSelectDatasetStep(input: {
  userId: string;
  pipeType: string;
  pipeId?: string | null;
  provider: DatasetProvider;
  connection: ProviderConnection;
  dataset: ProviderDataset;
  profile: DatasetProfile;
}) {
  const pipeName = `${input.pipeType === "tabular_regression" ? "Regression" : "Classification"} pipe`;

  let pipeId = input.pipeId ?? null;
  if (!pipeId) {
    const { data: pipe, error: pipeError } = await supabase.from("pipes").insert({ owner_id: input.userId, name: pipeName, description: "Draft pipe created by builder", type: input.pipeType, status: "draft", is_template: false }).select("id").single();
    if (pipeError) throw pipeError;
    pipeId = pipe.id;
  }

  const { data: source, error: sourceError } = await supabase.from("dataset_sources").insert({ user_id: input.userId, provider: input.provider, connection_id: input.connection.id, external_id: input.dataset.externalId, external_name: input.dataset.name, external_url: input.dataset.url ?? null, source_config: input.dataset.sourceConfig }).select("id").single();
  if (sourceError) throw sourceError;

  const snapshot = { rows: input.dataset.rows, profile: input.profile };
  const storageUri = `supabase://artifacts/dataset/${pipeId}/${source.id}`;

  const { data: artifact, error: artifactError } = await supabase
    .from("artifacts")
    .insert({
      pipe_id: pipeId,
      artifact_type: "dataset_snapshot",
      kind: "dataset_snapshot",
      name: `${input.dataset.sourceLabel} snapshot`,
      content: snapshot,
      metadata: {
        provider: input.provider,
        source_label: input.dataset.sourceLabel,
        row_count: input.profile.rowCount,
        column_count: input.profile.columnCount,
      },
    })
    .select("id")
    .single();
  if (artifactError) throw artifactError;

  const output = { step_key: "select_dataset", status: "completed", dataset_artifact_id: artifact.id, dataset_source_id: source.id, provider: input.provider, source_label: input.dataset.sourceLabel, row_count: input.profile.rowCount, column_count: input.profile.columnCount, columns: input.profile.columns, eligibility: input.profile.eligibility, storage: { format: "json", uri: storageUri } };

  const profileInsert = await supabase.from("dataset_profiles").insert({ pipe_id: pipeId, dataset_source_id: source.id, row_count: input.profile.rowCount, column_count: input.profile.columnCount, columns: input.profile.columns, missing_values: input.profile.missingValues, eligibility: input.profile.eligibility, preview: input.profile.preview });
  if (profileInsert.error) throw profileInsert.error;

  const stepOutput = await supabase.from("pipe_step_outputs").upsert({ pipe_id: pipeId, step_key: "select_dataset", artifact_id: artifact.id, output, status: "completed" }, { onConflict: "pipe_id,step_key" }).select("id").single();
  if (stepOutput.error) throw stepOutput.error;

  return { pipeId, artifactId: artifact.id, output };
}
