import { supabase } from "../supabaseClient";
import type { DatasetProfile, DatasetProvider, ProviderConnection, ProviderDataset } from "../../types/builder";

export async function persistSelectDatasetStep(input: {
  userId: string;
  pipeId: string;
  provider: DatasetProvider;
  connection: ProviderConnection;
  dataset: ProviderDataset;
  profile: DatasetProfile;
}) {
  if (!input.pipeId) {
    throw new Error("Select Dataset persistence requires an existing pipe_id.");
  }

  let sourceId: string | null = null;
  let profileId: string | null = null;
  let artifactId: string | null = null;

  // TODO: Move Select Dataset persistence to a transactional Supabase RPC before alpha.
  // Until then, the catch block performs best-effort cleanup for this save attempt.
  try {
    const { data: source, error: sourceError } = await supabase
      .from("dataset_sources")
      .insert({ pipe_id: input.pipeId, user_id: input.userId, provider: input.provider, connection_id: input.connection.id, external_id: input.dataset.externalId, external_name: input.dataset.name, external_url: input.dataset.url ?? null, source_config: input.dataset.sourceConfig })
      .select("id")
      .single();
    if (sourceError) throw sourceError;
    sourceId = source.id;

    const { data: datasetProfile, error: profileError } = await supabase
      .from("dataset_profiles")
      .insert({ pipe_id: input.pipeId, dataset_source_id: sourceId, row_count: input.profile.rowCount, column_count: input.profile.columnCount, columns: input.profile.columns, missing_values: input.profile.missingValues, eligibility: input.profile.eligibility, preview: input.profile.preview })
      .select("id")
      .single();
    if (profileError) throw profileError;
    profileId = datasetProfile.id;

    const snapshot = { rows: input.dataset.rows, profile: input.profile };
    const storageUri = `supabase://artifacts/dataset/${input.pipeId}/${sourceId}`;
    const { data: artifact, error: artifactError } = await supabase
      .from("artifacts")
      .insert({
        pipe_id: input.pipeId,
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
    artifactId = artifact.id;

    const output = { step_key: "select_dataset", status: "completed", dataset_artifact_id: artifactId, dataset_source_id: sourceId, provider: input.provider, source_label: input.dataset.sourceLabel, row_count: input.profile.rowCount, column_count: input.profile.columnCount, columns: input.profile.columns, eligibility: input.profile.eligibility, storage: { format: "json", uri: storageUri } };
    const stepOutput = await supabase.from("pipe_step_outputs").upsert({ pipe_id: input.pipeId, step_key: "select_dataset", artifact_id: artifactId, output, status: "completed" }, { onConflict: "pipe_id,step_key" }).select("id").single();
    if (stepOutput.error) throw stepOutput.error;

    return { pipeId: input.pipeId, artifactId, output };
  } catch (error) {
    await cleanupPartialSelectDatasetPersistence({ sourceId, profileId, artifactId });
    throw error;
  }
}

async function cleanupPartialSelectDatasetPersistence(input: {
  sourceId: string | null;
  profileId: string | null;
  artifactId: string | null;
}) {
  if (input.artifactId) await supabase.from("artifacts").delete().eq("id", input.artifactId);
  if (input.profileId) await supabase.from("dataset_profiles").delete().eq("id", input.profileId);
  if (input.sourceId) await supabase.from("dataset_sources").delete().eq("id", input.sourceId);
}

export async function persistCleanDataStep(input: {
  pipeId: string;
  provider: DatasetProvider;
  sourceLabel: string;
  previousDatasetArtifactId: string;
  cleanedRows: Record<string, unknown>[];
  cleaningPlan: unknown;
  cleaningResult: {
    rows_before: number;
    rows_after: number;
    columns_before: number;
    columns_after: number;
    missing_values_before: number;
    missing_values_after: number;
    duplicate_rows_removed: number;
    excluded_feature_columns: string[];
  };
  profileBefore: unknown;
  profileAfter: DatasetProfile;
}) {
  const storageUri = `supabase://artifacts/cleaned_dataset/${input.pipeId}/${input.previousDatasetArtifactId}`;
  const content = {
    rows: input.cleanedRows,
    previous_dataset_artifact_id: input.previousDatasetArtifactId,
    cleaning_plan: input.cleaningPlan,
    cleaning_result: input.cleaningResult,
    profile_before: input.profileBefore,
    profile_after: input.profileAfter,
  };

  const { data: artifact, error: artifactError } = await supabase
    .from("artifacts")
    .insert({
      pipe_id: input.pipeId,
      artifact_type: "cleaned_dataset",
      kind: "cleaned_dataset",
      name: `${input.sourceLabel} cleaned dataset`,
      content,
      metadata: {
        source_label: input.sourceLabel,
        provider: input.provider,
        row_count: input.cleaningResult.rows_after,
        column_count: input.cleaningResult.columns_after,
        previous_dataset_artifact_id: input.previousDatasetArtifactId,
      },
    })
    .select("id")
    .single();

  if (artifactError) throw artifactError;

  const output = {
    step_key: "clean_data",
    status: "completed",
    cleaned_dataset_artifact_id: artifact.id,
    previous_dataset_artifact_id: input.previousDatasetArtifactId,
    rows_before: input.cleaningResult.rows_before,
    rows_after: input.cleaningResult.rows_after,
    columns_before: input.cleaningResult.columns_before,
    columns_after: input.cleaningResult.columns_after,
    missing_values_before: input.cleaningResult.missing_values_before,
    missing_values_after: input.cleaningResult.missing_values_after,
    duplicate_rows_removed: input.cleaningResult.duplicate_rows_removed,
    excluded_feature_columns: input.cleaningResult.excluded_feature_columns,
    storage: { format: "json", uri: storageUri },
  };

  const stepOutput = await supabase
    .from("pipe_step_outputs")
    .upsert(
      { pipe_id: input.pipeId, step_key: "clean_data", artifact_id: artifact.id, output, status: "completed" },
      { onConflict: "pipe_id,step_key" },
    )
    .select("id")
    .single();

  if (stepOutput.error) {
    // Best-effort cleanup until this multi-write operation moves into an RPC.
    await supabase.from("artifacts").delete().eq("id", artifact.id);
    throw stepOutput.error;
  }

  return { artifactId: artifact.id, output };
}
