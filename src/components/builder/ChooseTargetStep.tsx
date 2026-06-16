import { useEffect, useMemo, useState } from "react";
import { persistChooseTargetStep } from "../../lib/builder/builderPersistence";
import { analyzeTargetColumn, buildTargetConfig, findRecommendedTarget, summarizeColumns } from "../../lib/builder/chooseTarget";
import { getArtifactById, type ChooseTargetStepOutput, type SplitDataStepOutput } from "../../lib/pipes";
import type { ColumnSummary, TargetConfig } from "../../types/builder";
import type { BuilderPipeType } from "../../types/pipe";

type ChooseTargetStepProps = {
  pipeId: string;
  pipeType: BuilderPipeType | null;
  splitDataOutput: SplitDataStepOutput | null;
  initialChooseTargetOutput: ChooseTargetStepOutput | null;
  onCompleted: (output: ChooseTargetStepOutput) => void;
  onBackToSplitData: () => void;
};

type SplitArtifactContent = {
  previous_cleaned_dataset_artifact_id?: string;
  splits?: {
    train?: Record<string, unknown>[];
  };
};

type CleanedArtifactContent = {
  cleaning_plan?: {
    columns?: Array<{ name?: string; recommended_role?: string }>;
  };
  cleaning_result?: {
    excluded_feature_columns?: string[];
  };
};

function isSplitArtifactContent(content: unknown): content is SplitArtifactContent {
  return typeof content === "object" && content !== null && Array.isArray((content as SplitArtifactContent).splits?.train);
}

function getExcludedFeatureColumns(content: unknown) {
  if (typeof content !== "object" || content === null) return [];
  const cleaned = content as CleanedArtifactContent;
  const fromResult = cleaned.cleaning_result?.excluded_feature_columns ?? [];
  const fromPlan = cleaned.cleaning_plan?.columns
    ?.filter((column) => column.recommended_role === "exclude_from_features" && column.name)
    .map((column) => column.name as string) ?? [];
  return Array.from(new Set([...fromResult, ...fromPlan]));
}

function taskLabel(taskType: BuilderPipeType) {
  return taskType === "tabular_regression" ? "Regression" : "Classification";
}

export function ChooseTargetStep({ pipeId, pipeType, splitDataOutput, initialChooseTargetOutput, onCompleted, onBackToSplitData }: ChooseTargetStepProps) {
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [columns, setColumns] = useState<ColumnSummary[]>([]);
  const [targetColumn, setTargetColumn] = useState("");
  const [recommendationReason, setRecommendationReason] = useState<string | null>(null);
  const [excludedFeatureColumns, setExcludedFeatureColumns] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedOutput, setSavedOutput] = useState<ChooseTargetStepOutput | null>(null);

  useEffect(() => {
    if (!splitDataOutput?.split_dataset_artifact_id) return;
    let mounted = true;

    void Promise.resolve()
      .then(() => {
        if (!mounted) return null;
        setLoading(true);
        setLoadError(null);
        return getArtifactById(splitDataOutput.split_dataset_artifact_id);
      })
      .then(async (artifact) => {
        if (!mounted) return;
        if (!artifact || !isSplitArtifactContent(artifact.content)) {
          setLoadError("We could not load the split dataset artifact.");
          return;
        }

        const trainRows = artifact.content.splits?.train ?? [];
        if (!trainRows.length) {
          setLoadError("The training split is empty, so MLP cannot choose a target yet.");
          return;
        }

        const columnSummaries = summarizeColumns(trainRows);
        let excluded: string[] = [];
        if (artifact.content.previous_cleaned_dataset_artifact_id) {
          try {
            const cleanedArtifact = await getArtifactById(artifact.content.previous_cleaned_dataset_artifact_id);
            excluded = getExcludedFeatureColumns(cleanedArtifact?.content);
          } catch {
            excluded = [];
          }
        }

        if (!mounted) return;
        const recommended = findRecommendedTarget(columnSummaries);
        setRows(trainRows);
        setColumns(columnSummaries);
        setExcludedFeatureColumns(excluded);
        if (recommended) {
          setTargetColumn(recommended.columnName);
          setRecommendationReason(recommended.reason);
        }
      })
      .catch(() => {
        if (!mounted) return;
        setLoadError("We could not load the split dataset artifact.");
      })
      .finally(() => {
        if (!mounted) return;
        setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, [splitDataOutput?.split_dataset_artifact_id]);

  const output = savedOutput ?? initialChooseTargetOutput;
  const targetAnalysis = useMemo(() => {
    if (!targetColumn || !pipeType) return null;
    return analyzeTargetColumn(rows, targetColumn, pipeType);
  }, [pipeType, rows, targetColumn]);
  const targetConfig = useMemo<TargetConfig | null>(() => {
    if (!splitDataOutput || !targetColumn || !targetAnalysis || !pipeType) return null;
    return buildTargetConfig({
      previousSplitDatasetArtifactId: splitDataOutput.split_dataset_artifact_id,
      targetColumn,
      pipeType,
      columnSummaries: columns,
      targetAnalysis,
      excludedFeatureColumns,
    });
  }, [columns, excludedFeatureColumns, pipeType, splitDataOutput, targetAnalysis, targetColumn]);
  const featureCount = targetConfig?.feature_columns.length ?? 0;
  const blockingReasons = targetAnalysis?.blocking_reasons ?? [];
  const warnings = targetAnalysis?.warnings ?? [];

  async function handleApproveTarget() {
    if (!targetConfig || blockingReasons.length > 0) return;
    setSaving(true);
    setSaveError(null);
    try {
      const result = await persistChooseTargetStep({ pipeId, targetConfig });
      const nextOutput = result.output as ChooseTargetStepOutput;
      setSavedOutput(nextOutput);
      onCompleted(nextOutput);
    } catch {
      setSaveError("Unable to save this target. Please try again.");
    } finally {
      setSaving(false);
    }
  }

  if (!splitDataOutput) {
    return <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Split your dataset before choosing a target.</h2><button type="button" onClick={onBackToSplitData} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Back to Split data</button></section>;
  }

  if (output) {
    return <section className="mt-6 rounded-3xl border border-emerald-200 bg-emerald-50 p-6"><h2 className="text-lg font-semibold text-emerald-900">Prediction target saved.</h2><p className="mt-2 text-sm text-emerald-800">Choose Target is complete. Train Models is coming soon.</p><dl className="mt-4 grid gap-2 text-sm text-emerald-900 md:grid-cols-2"><div><dt className="font-medium">Target column</dt><dd>{output.target_column}</dd></div><div><dt className="font-medium">Detected task type</dt><dd>{taskLabel(output.detected_task_type)}</dd></div><div><dt className="font-medium">Feature columns</dt><dd>{output.feature_columns.length}</dd></div><div><dt className="font-medium">Mismatch warning</dt><dd>{output.task_type_mismatch ? "Pipe type and target type differ" : "None"}</dd></div><div className="md:col-span-2"><dt className="font-medium">Target config artifact ID</dt><dd className="font-mono text-xs">{output.target_config_artifact_id}</dd></div></dl><button type="button" className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Continue to Train models</button></section>;
  }

  if (loading) return <p className="mt-6 text-sm text-black/50">Loading split dataset artifact…</p>;
  if (loadError) return <p className="mt-6 rounded-2xl bg-red-500/10 px-4 py-3 text-sm text-red-700">{loadError}</p>;

  return <div>
    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">What is a target?</h2><p className="mt-2 text-sm text-black/60">The target is the column your pipe will learn to predict.</p><ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-black/60"><li>If you want to predict whether a customer will churn, choose the churn column.</li><li>If you want to predict a price, choose the price column.</li><li>If you want to predict a category, choose the category column.</li></ul></section>

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><label className="text-sm font-medium">Which column should the pipe predict?<select value={targetColumn} onChange={(event) => { setTargetColumn(event.target.value); setRecommendationReason(null); setSaveError(null); }} className="mt-2 block w-full rounded-2xl border border-black/10 bg-white px-4 py-3"><option value="">Choose a column…</option>{columns.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}</select></label>{recommendationReason && targetColumn ? <p className="mt-3 rounded-2xl bg-blue-500/10 px-4 py-3 text-sm text-blue-800">{recommendationReason}</p> : null}</section>

    {targetAnalysis && targetConfig ? <section className="mt-6 grid gap-6 lg:grid-cols-2"><div className="rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Detected task</h2><p className="mt-2 text-2xl font-semibold">{taskLabel(targetAnalysis.detected_task_type)}</p><p className="mt-2 text-sm text-black/60">{targetAnalysis.reason}</p>{targetConfig.task_type_mismatch ? <p className="mt-3 rounded-2xl bg-amber-500/10 px-4 py-3 text-sm text-amber-800">You created a {pipeType === "tabular_regression" ? "regression" : "classification"} pipe, but this target looks like a {targetAnalysis.detected_task_type === "tabular_regression" ? "regression" : "classification"} target.</p> : null}</div><div className="rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Feature summary</h2><p className="mt-2 text-sm text-black/60">MLP will use <strong>{featureCount}</strong> columns as inputs and predict <strong>{targetColumn}</strong>.</p>{excludedFeatureColumns.length ? <p className="mt-3 text-sm text-black/50">Recommended to exclude later: {excludedFeatureColumns.join(", ")}</p> : null}{featureCount < 2 ? <p className="mt-3 rounded-2xl bg-amber-500/10 px-4 py-3 text-sm text-amber-800">There may not be enough useful columns to train a model.</p> : null}</div></section> : null}

    {targetAnalysis && (warnings.length || blockingReasons.length) ? <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Warnings</h2><ul className="mt-3 list-disc space-y-1 pl-5 text-sm">{blockingReasons.map((reason) => <li key={reason} className="text-red-700">{reason}</li>)}{warnings.map((warning) => <li key={warning} className="text-amber-700">{warning}</li>)}</ul></section> : null}

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><button type="button" onClick={handleApproveTarget} disabled={!targetConfig || blockingReasons.length > 0 || saving} className="rounded-full bg-black px-4 py-2 text-sm font-medium text-white disabled:bg-black/30">{saving ? "Saving…" : "Approve target"}</button>{saveError ? <p className="mt-3 text-sm text-red-700">{saveError}</p> : null}</section>
  </div>;
}
