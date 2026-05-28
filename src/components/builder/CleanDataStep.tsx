import { useEffect, useMemo, useState } from "react";
import { persistCleanDataStep } from "../../lib/builder/builderPersistence";
import { applyCleaningPlan, auditDataQuality, generateCleaningPlan, type AppliedCleaning } from "../../lib/builder/cleanData";
import { getArtifactById, getCleanDataStepOutput, type CleanDataStepOutput, type SelectDatasetStepOutput } from "../../lib/pipes";
import type { CleaningPlan, CleaningStrategy } from "../../types/builder";

type DatasetArtifactContent = {
  rows?: Record<string, unknown>[];
};

type CleanDataStepProps = {
  pipeId: string | null;
  selectDatasetOutput: SelectDatasetStepOutput | null;
  initialCleanOutput: CleanDataStepOutput | null;
  onCompleted: (output: CleanDataStepOutput) => void;
  onBackToSelectDataset: () => void;
};

const strategyLabels: Record<CleaningStrategy, string> = {
  median: "Use median — recommended",
  mean: "Use mean",
  most_frequent: "Use most frequent value",
  unknown: "Use “Unknown” — recommended",
  remove_rows: "Remove rows with missing values",
  leave_as_is: "Leave as is",
};

function isRowsContent(content: unknown): content is DatasetArtifactContent {
  return typeof content === "object" && content !== null && Array.isArray((content as DatasetArtifactContent).rows);
}

function previewColumns(rows: Record<string, unknown>[]) {
  return Object.keys(rows[0] ?? {}).slice(0, 8);
}

export function CleanDataStep({ pipeId, selectDatasetOutput, initialCleanOutput, onCompleted, onBackToSelectDataset }: CleanDataStepProps) {
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [cleaningPlan, setCleaningPlan] = useState<CleaningPlan | null>(null);
  const [appliedCleaning, setAppliedCleaning] = useState<AppliedCleaning | null>(null);
  const [cleanOutput, setCleanOutput] = useState<CleanDataStepOutput | null>(initialCleanOutput);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setCleanOutput(initialCleanOutput);
  }, [initialCleanOutput]);

  useEffect(() => {
    if (!selectDatasetOutput?.dataset_artifact_id) return;
    let mounted = true;
    setLoading(true);
    setLoadError(null);

    getArtifactById(selectDatasetOutput.dataset_artifact_id)
      .then((artifact) => {
        if (!mounted) return;
        if (!artifact || !isRowsContent(artifact.content)) {
          setLoadError("We could not load the selected dataset artifact.");
          return;
        }
        const artifactRows = artifact.content.rows ?? [];
        setRows(artifactRows);
        setCleaningPlan(generateCleaningPlan(artifactRows));
      })
      .catch(() => {
        if (!mounted) return;
        setLoadError("We could not load the selected dataset artifact.");
      })
      .finally(() => {
        if (!mounted) return;
        setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, [selectDatasetOutput?.dataset_artifact_id]);

  const audit = useMemo(() => auditDataQuality(rows), [rows]);
  const issueColumns = useMemo(() => cleaningPlan?.columns.filter((column) => column.issues.length || column.recommended_role === "exclude_from_features") ?? [], [cleaningPlan]);

  function updateColumnStrategy(columnName: string, selectedStrategy: CleaningStrategy) {
    setCleaningPlan((current) => current ? {
      ...current,
      columns: current.columns.map((column) => column.name === columnName ? { ...column, selected_strategy: selectedStrategy } : column),
    } : current);
  }

  function updateDuplicateAction(action: "remove" | "keep") {
    setCleaningPlan((current) => current ? { ...current, duplicateRows: { action } } : current);
  }

  function handleApplyCleaning() {
    if (!cleaningPlan) return;
    setAppliedCleaning(applyCleaningPlan(rows, cleaningPlan));
  }

  async function handleApproveCleanedData() {
    if (!pipeId || !selectDatasetOutput || !cleaningPlan || !appliedCleaning) return;
    setSaving(true);
    try {
      const result = await persistCleanDataStep({
        pipeId,
        provider: selectDatasetOutput.provider,
        sourceLabel: selectDatasetOutput.source_label,
        previousDatasetArtifactId: selectDatasetOutput.dataset_artifact_id,
        cleanedRows: appliedCleaning.cleanedRows,
        cleaningPlan,
        cleaningResult: appliedCleaning.cleaningResult,
        profileBefore: appliedCleaning.profileBefore,
        profileAfter: appliedCleaning.profileAfter,
      });
      const nextOutput = result.output as CleanDataStepOutput;
      setCleanOutput(nextOutput);
      onCompleted(nextOutput);
    } finally {
      setSaving(false);
    }
  }

  useEffect(() => {
    if (!pipeId || cleanOutput) return;
    let mounted = true;
    getCleanDataStepOutput(pipeId).then((output) => {
      if (!mounted || !output) return;
      setCleanOutput(output);
      onCompleted(output);
    });
    return () => {
      mounted = false;
    };
  }, [pipeId, cleanOutput, onCompleted]);

  if (!selectDatasetOutput) {
    return <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Select a dataset before cleaning data.</h2><button type="button" onClick={onBackToSelectDataset} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Back to Select dataset</button></section>;
  }

  if (cleanOutput) {
    return <section className="mt-6 rounded-3xl border border-emerald-200 bg-emerald-50 p-6"><h2 className="text-lg font-semibold text-emerald-900">Cleaned dataset saved.</h2><p className="mt-2 text-sm text-emerald-800">Clean Data is complete. Split Data is coming soon.</p><dl className="mt-4 grid gap-2 text-sm text-emerald-900 md:grid-cols-2"><div><dt className="font-medium">Rows after cleaning</dt><dd>{cleanOutput.rows_after}</dd></div><div><dt className="font-medium">Columns after cleaning</dt><dd>{cleanOutput.columns_after}</dd></div><div><dt className="font-medium">Missing values after</dt><dd>{cleanOutput.missing_values_after}</dd></div><div><dt className="font-medium">Duplicates removed</dt><dd>{cleanOutput.duplicate_rows_removed}</dd></div><div className="md:col-span-2"><dt className="font-medium">Cleaned dataset artifact ID</dt><dd className="font-mono text-xs">{cleanOutput.cleaned_dataset_artifact_id}</dd></div></dl><button type="button" className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Continue to Split data</button></section>;
  }

  if (loading) return <p className="mt-6 text-sm text-black/50">Loading selected dataset artifact…</p>;
  if (loadError) return <p className="mt-6 rounded-2xl bg-red-500/10 px-4 py-3 text-sm text-red-700">{loadError}</p>;

  return <div>
    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Input dataset summary</h2><dl className="mt-4 grid gap-3 text-sm md:grid-cols-2"><div><dt className="text-black/50">Source</dt><dd>{selectDatasetOutput.source_label}</dd></div><div><dt className="text-black/50">Provider</dt><dd>{selectDatasetOutput.provider}</dd></div><div><dt className="text-black/50">Rows</dt><dd>{selectDatasetOutput.row_count}</dd></div><div><dt className="text-black/50">Columns</dt><dd>{selectDatasetOutput.column_count}</dd></div><div className="md:col-span-2"><dt className="text-black/50">Dataset artifact ID</dt><dd className="font-mono text-xs">{selectDatasetOutput.dataset_artifact_id}</dd></div></dl></section>

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Data quality summary</h2><div className="mt-4 grid gap-3 text-sm md:grid-cols-2 lg:grid-cols-4"><p>Total missing values: <strong>{audit.totalMissingValues}</strong></p><p>Duplicate rows: <strong>{audit.duplicateRows}</strong></p><p>Columns with missing values: <strong>{audit.columnsWithMissingValues.length}</strong></p><p>Columns flagged for later: <strong>{[...new Set([...audit.idLikeColumns, ...audit.datetimeColumns, ...audit.longTextColumns, ...audit.constantColumns])].length}</strong></p></div><div className="mt-4 flex flex-wrap gap-2 text-xs text-black/60">{audit.idLikeColumns.map((column) => <span key={column} className="rounded-full border border-black/10 px-3 py-1">ID-like: {column}</span>)}{audit.datetimeColumns.map((column) => <span key={column} className="rounded-full border border-black/10 px-3 py-1">Datetime: {column}</span>)}{audit.longTextColumns.map((column) => <span key={column} className="rounded-full border border-black/10 px-3 py-1">Long text: {column}</span>)}{audit.highCardinalityCategoricalColumns.map((column) => <span key={column} className="rounded-full border border-black/10 px-3 py-1">Many categories: {column}</span>)}{audit.constantColumns.map((column) => <span key={column} className="rounded-full border border-black/10 px-3 py-1">Constant: {column}</span>)}</div></section>

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Recommended cleaning plan</h2>{issueColumns.length === 0 && audit.duplicateRows === 0 ? <p className="mt-3 text-sm text-black/60">This dataset looks clean. MLP does not need to change anything before splitting.</p> : null}{audit.duplicateRows > 0 ? <div className="mt-4 rounded-2xl border border-black/10 p-4"><p className="font-medium">Duplicate rows</p><p className="mt-1 text-sm text-black/60">MLP found {audit.duplicateRows} duplicate rows and recommends removing them.</p><select value={cleaningPlan?.duplicateRows.action ?? "remove"} onChange={(event) => updateDuplicateAction(event.target.value as "remove" | "keep")} className="mt-3 rounded-xl border border-black/10 bg-white px-3 py-2 text-sm"><option value="remove">Remove duplicates — recommended</option><option value="keep">Keep duplicates</option></select></div> : null}<div className="mt-4 grid gap-4">{issueColumns.map((column) => <div key={column.name} className="rounded-2xl border border-black/10 p-4"><p className="font-medium">Column “{column.name}”</p><p className="mt-1 text-sm text-black/60">{column.missing_count > 0 ? `${column.missing_count} values look missing. ` : ""}{column.special_missing_token_count ? "Some values look missing even though they are stored as text. " : ""}{column.recommended_role === "exclude_from_features" ? "MLP recommends not using this column for training later." : ""}</p>{column.recommended_strategy ? <select value={column.selected_strategy} onChange={(event) => updateColumnStrategy(column.name, event.target.value as CleaningStrategy)} className="mt-3 rounded-xl border border-black/10 bg-white px-3 py-2 text-sm"><option value={column.recommended_strategy}>{strategyLabels[column.recommended_strategy]}</option><option value="most_frequent">Use most frequent value</option><option value="remove_rows">Remove rows with missing values</option><option value="leave_as_is">Leave as is</option>{column.detected_type === "number" ? <><option value="mean">Use mean</option><option value="median">Use median</option></> : null}</select> : null}</div>)}</div></section>

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Apply cleaning</h2><button type="button" onClick={handleApplyCleaning} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Apply cleaning plan</button>{appliedCleaning ? <div className="mt-6"><h3 className="font-semibold">Before / after summary</h3><div className="mt-3 grid gap-3 text-sm md:grid-cols-2"><div className="rounded-2xl border border-black/10 p-4"><p className="font-medium">Before</p><p>Rows: {appliedCleaning.cleaningResult.rows_before}</p><p>Columns: {appliedCleaning.cleaningResult.columns_before}</p><p>Missing values: {appliedCleaning.cleaningResult.missing_values_before}</p><p>Duplicate rows: {appliedCleaning.cleaningResult.duplicate_rows_before}</p></div><div className="rounded-2xl border border-black/10 p-4"><p className="font-medium">After</p><p>Rows: {appliedCleaning.cleaningResult.rows_after}</p><p>Columns: {appliedCleaning.cleaningResult.columns_after}</p><p>Missing values: {appliedCleaning.cleaningResult.missing_values_after}</p><p>Duplicates removed: {appliedCleaning.cleaningResult.duplicate_rows_removed}</p></div></div><div className="mt-4 overflow-x-auto"><table className="min-w-full text-sm"><thead><tr>{previewColumns(appliedCleaning.cleanedRows).map((column) => <th key={column} className="border-b border-black/10 px-3 py-2 text-left font-medium">{column}</th>)}</tr></thead><tbody>{appliedCleaning.cleanedRows.slice(0, 5).map((row, index) => <tr key={index}>{previewColumns(appliedCleaning.cleanedRows).map((column) => <td key={column} className="border-b border-black/5 px-3 py-2 text-black/70">{String(row[column])}</td>)}</tr>)}</tbody></table></div><button type="button" onClick={handleApproveCleanedData} disabled={saving} className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white disabled:bg-black/30">{saving ? "Saving…" : "Approve cleaned data"}</button></div> : null}</section>
  </div>;
}
