import { useEffect, useMemo, useState } from "react";
import { persistSplitDataStep } from "../../lib/builder/builderPersistence";
import { buildDefaultSplitConfig, getSplitWarnings, splitRows, validateSplitConfig, type SplitRowsResult } from "../../lib/builder/splitData";
import { getArtifactById, type CleanDataStepOutput, type SplitDataStepOutput } from "../../lib/pipes";
import type { SplitConfig } from "../../types/builder";
import type { BuilderPipeType } from "../../types/pipe";

type SplitDatasetStepProps = {
  pipeId: string;
  pipeType: BuilderPipeType | null;
  cleanDataOutput: CleanDataStepOutput | null;
  initialSplitOutput: SplitDataStepOutput | null;
  onCompleted: (output: SplitDataStepOutput) => void;
  onBackToCleanData: () => void;
};

type CleanedDatasetContent = {
  rows?: Record<string, unknown>[];
};

function isCleanedDatasetContent(content: unknown): content is CleanedDatasetContent {
  return typeof content === "object" && content !== null && Array.isArray((content as CleanedDatasetContent).rows);
}

export function SplitDatasetStep({ pipeId, pipeType, cleanDataOutput, initialSplitOutput, onCompleted, onBackToCleanData }: SplitDatasetStepProps) {
  const [rows, setRows] = useState<Record<string, unknown>[]>([]);
  const [config, setConfig] = useState<SplitConfig>(() => buildDefaultSplitConfig());
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [splitResult, setSplitResult] = useState<SplitRowsResult | null>(null);
  const [savedSplitOutput, setSavedSplitOutput] = useState<SplitDataStepOutput | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (!cleanDataOutput?.cleaned_dataset_artifact_id) return;
    let mounted = true;

    void Promise.resolve()
      .then(() => {
        if (!mounted) return null;
        setLoading(true);
        setLoadError(null);
        return getArtifactById(cleanDataOutput.cleaned_dataset_artifact_id);
      })
      .then((artifact) => {
        if (!mounted) return;
        if (!artifact || !isCleanedDatasetContent(artifact.content)) {
          setLoadError("We could not load the cleaned dataset artifact.");
          return;
        }
        setRows(artifact.content.rows ?? []);
      })
      .catch(() => {
        if (!mounted) return;
        setLoadError("We could not load the cleaned dataset artifact.");
      })
      .finally(() => {
        if (!mounted) return;
        setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, [cleanDataOutput?.cleaned_dataset_artifact_id]);

  const validationErrors = useMemo(() => validateSplitConfig(config, rows.length), [config, rows.length]);
  const warnings = useMemo(() => getSplitWarnings(config, rows.length, pipeType), [config, rows.length, pipeType]);

  function updateNumberConfig(key: "train_pct" | "validation_pct" | "test_pct" | "random_seed", value: string) {
    setConfig((current) => ({ ...current, [key]: Number(value) }));
  }

  function handleSplitDataset() {
    if (validationErrors.length) return;
    setSplitResult(splitRows(rows, config));
    setSaveError(null);
  }

  async function handleApproveSplit() {
    if (!cleanDataOutput || !splitResult) return;
    setSaving(true);
    setSaveError(null);
    try {
      const result = await persistSplitDataStep({
        pipeId,
        previousCleanedDatasetArtifactId: cleanDataOutput.cleaned_dataset_artifact_id,
        splitConfig: config,
        trainRows: splitResult.trainRows,
        validationRows: splitResult.validationRows,
        testRows: splitResult.testRows,
        splitResult: splitResult.splitResult,
      });
      const output = result.output as SplitDataStepOutput;
      setSavedSplitOutput(output);
      onCompleted(output);
    } catch {
      setSaveError("Unable to save this split. Please try again.");
    } finally {
      setSaving(false);
    }
  }

  const splitOutput = savedSplitOutput ?? initialSplitOutput;

  if (!cleanDataOutput) {
    return <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Clean your dataset before splitting it.</h2><button type="button" onClick={onBackToCleanData} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Back to Clean data</button></section>;
  }

  if (splitOutput) {
    return <section className="mt-6 rounded-3xl border border-emerald-200 bg-emerald-50 p-6"><h2 className="text-lg font-semibold text-emerald-900">Dataset split saved.</h2><p className="mt-2 text-sm text-emerald-800">Split Data is complete. Choose Target is coming soon.</p><dl className="mt-4 grid gap-2 text-sm text-emerald-900 md:grid-cols-2"><div><dt className="font-medium">Training rows</dt><dd>{splitOutput.train_rows}</dd></div><div><dt className="font-medium">Validation rows</dt><dd>{splitOutput.validation_rows}</dd></div><div><dt className="font-medium">Test rows</dt><dd>{splitOutput.test_rows}</dd></div><div><dt className="font-medium">Split</dt><dd>{splitOutput.split_config.train_pct}% / {splitOutput.split_config.validation_pct}% / {splitOutput.split_config.test_pct}%</dd></div><div className="md:col-span-2"><dt className="font-medium">Split dataset artifact ID</dt><dd className="font-mono text-xs">{splitOutput.split_dataset_artifact_id}</dd></div></dl><button type="button" className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Continue to Choose target</button></section>;
  }

  if (loading) return <p className="mt-6 text-sm text-black/50">Loading cleaned dataset artifact…</p>;
  if (loadError) return <p className="mt-6 rounded-2xl bg-red-500/10 px-4 py-3 text-sm text-red-700">{loadError}</p>;

  return <div>
    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Recommended split</h2><p className="mt-2 text-sm text-black/60">MLP will split your cleaned dataset into training, validation, and test sets before model training.</p><div className="mt-5 grid gap-3 md:grid-cols-3"><div className="rounded-2xl border border-black/10 p-4"><p className="font-medium">Training data</p><p className="mt-1 text-3xl font-semibold">{config.train_pct}%</p><p className="mt-2 text-sm text-black/50">Used to teach the model.</p></div><div className="rounded-2xl border border-black/10 p-4"><p className="font-medium">Validation data</p><p className="mt-1 text-3xl font-semibold">{config.validation_pct}%</p><p className="mt-2 text-sm text-black/50">Used to compare model options.</p></div><div className="rounded-2xl border border-black/10 p-4"><p className="font-medium">Test data</p><p className="mt-1 text-3xl font-semibold">{config.test_pct}%</p><p className="mt-2 text-sm text-black/50">Used for final evaluation.</p></div></div>{warnings.length ? <ul className="mt-4 list-disc space-y-1 pl-5 text-sm text-amber-700">{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}</section>

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><button type="button" onClick={() => setAdvancedOpen((value) => !value)} className="text-sm font-medium text-black/60 hover:text-black">{advancedOpen ? "Hide advanced settings" : "Show advanced settings"}</button>{advancedOpen ? <div className="mt-4 grid gap-4 md:grid-cols-2"><label className="text-sm">Training percentage<input type="number" value={config.train_pct} onChange={(event) => updateNumberConfig("train_pct", event.target.value)} className="mt-1 block w-full rounded-xl border border-black/10 bg-white px-3 py-2" /></label><label className="text-sm">Validation percentage<input type="number" value={config.validation_pct} onChange={(event) => updateNumberConfig("validation_pct", event.target.value)} className="mt-1 block w-full rounded-xl border border-black/10 bg-white px-3 py-2" /></label><label className="text-sm">Test percentage<input type="number" value={config.test_pct} onChange={(event) => updateNumberConfig("test_pct", event.target.value)} className="mt-1 block w-full rounded-xl border border-black/10 bg-white px-3 py-2" /></label><label className="text-sm">Random seed<input type="number" value={config.random_seed} onChange={(event) => updateNumberConfig("random_seed", event.target.value)} className="mt-1 block w-full rounded-xl border border-black/10 bg-white px-3 py-2" /></label><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={config.shuffle} onChange={(event) => setConfig((current) => ({ ...current, shuffle: event.target.checked }))} /> Shuffle rows</label><p className="text-sm text-black/50">Stratification: after target selection if needed.</p></div> : null}</section>

    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Split dataset</h2>{validationErrors.length ? <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-red-700">{validationErrors.map((error) => <li key={error}>{error}</li>)}</ul> : null}<button type="button" onClick={handleSplitDataset} disabled={validationErrors.length > 0} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white disabled:bg-black/30">Split dataset</button>{splitResult ? <div className="mt-6"><h3 className="font-semibold">Split summary</h3><div className="mt-3 grid gap-3 text-sm md:grid-cols-4"><p className="rounded-2xl border border-black/10 p-4">Total rows<br /><strong>{splitResult.splitResult.rows_total}</strong></p><p className="rounded-2xl border border-black/10 p-4">Training rows<br /><strong>{splitResult.splitResult.train_rows}</strong></p><p className="rounded-2xl border border-black/10 p-4">Validation rows<br /><strong>{splitResult.splitResult.validation_rows}</strong></p><p className="rounded-2xl border border-black/10 p-4">Test rows<br /><strong>{splitResult.splitResult.test_rows}</strong></p></div><button type="button" onClick={handleApproveSplit} disabled={saving} className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white disabled:bg-black/30">{saving ? "Saving…" : "Approve split"}</button>{saveError ? <p className="mt-3 text-sm text-red-700">{saveError}</p> : null}</div> : null}</section>
  </div>;
}
