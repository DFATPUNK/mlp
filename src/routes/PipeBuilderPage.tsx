import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { CleanDataStep } from "../components/builder/CleanDataStep";
import { ChooseTargetStep } from "../components/builder/ChooseTargetStep";
import { SplitDatasetStep } from "../components/builder/SplitDatasetStep";
import { TrainModelsStep } from "../components/builder/TrainModelsStep";
import { ReviewResultsStep } from "../components/builder/ReviewResultsStep";
import { TestPredictionStep } from "../components/builder/TestPredictionStep";
import { PublishPipeStep } from "../components/builder/PublishPipeStep";
import { useAuth } from "../lib/auth";
import { persistSelectDatasetStep } from "../lib/builder/builderPersistence";
import { profileDataset } from "../lib/builder/datasetProfiling";
import {
  fetchHuggingFaceDatasetRows,
  getDatasetKey
} from "../lib/builder/huggingFaceDatasets";
import { getMockConnections, getProviderDatasets } from "../lib/builder/mockDatasetProviders";
import { getChooseTargetStepOutput, getCleanDataStepOutput, getPipeById, getPublishPipeStepOutput, getReviewResultsStepOutput, getSelectDatasetStepOutput, getSplitDataStepOutput, getTestPredictionStepOutput, getTrainModelsStepOutput, isReviewResultsAcknowledged, type ChooseTargetStepOutput, type CleanDataStepOutput, type PublishPipeStepOutput, type ReviewResultsStepOutput, type SelectDatasetStepOutput, type SplitDataStepOutput, type TestPredictionStepOutput, type TrainModelsStepOutput } from "../lib/pipes";
import type { DatasetProvider } from "../types/builder";
import type { BuilderPipeType } from "../types/pipe";

const steps = ["Select dataset", "Clean data", "Split data", "Choose target", "Train models", "Review results", "Test prediction", "Publish pipe"];
const providerLabels: Record<DatasetProvider, string> = { huggingface: "Hugging Face Datasets", airtable: "Airtable", google_sheets: "Google Sheets" };

export function PipeBuilderPage() {
  const { user } = useAuth();
  const { pipeId } = useParams();
  const [loadedPipeType, setLoadedPipeType] = useState<BuilderPipeType | null>(null);
  const [existingStepOutput, setExistingStepOutput] = useState<SelectDatasetStepOutput | null>(null);
  const [cleanDataOutput, setCleanDataOutput] = useState<CleanDataStepOutput | null>(null);
  const [splitDataOutput, setSplitDataOutput] = useState<SplitDataStepOutput | null>(null);
  const [chooseTargetOutput, setChooseTargetOutput] = useState<ChooseTargetStepOutput | null>(null);
  const [trainModelsOutput, setTrainModelsOutput] = useState<TrainModelsStepOutput | null>(null);
  const [reviewResultsOutput, setReviewResultsOutput] = useState<ReviewResultsStepOutput | null>(null);
  const [testPredictionOutput, setTestPredictionOutput] = useState<TestPredictionStepOutput | null>(null);
  const [publishPipeOutput, setPublishPipeOutput] = useState<PublishPipeStepOutput | null>(null);
  const [activeStep, setActiveStep] = useState<"select_dataset" | "clean_data" | "split_data" | "choose_target" | "train_models" | "review_results" | "test_prediction" | "publish_pipe">("select_dataset");


  const [provider, setProvider] = useState<DatasetProvider>("huggingface");
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("");
  const [loadingRows, setLoadingRows] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [rowsError, setRowsError] = useState<string | null>(null);
  const [loadedRows, setLoadedRows] = useState<Record<string, unknown>[]>([]);
  const [acceptedArtifactId, setAcceptedArtifactId] = useState<string | null>(null);


  const hfRowsCacheRef = useRef<Map<string, { rows: Record<string, unknown>[] }>>(new Map());
  const [reloadCounter, setReloadCounter] = useState(0);

  const connections = useMemo(() => getMockConnections(provider), [provider]);
  const datasets = useMemo(() => getProviderDatasets(provider), [provider]);
  const selectedDataset = useMemo(() => datasets.find((dataset) => dataset.externalId === selectedDatasetId) ?? datasets[0], [datasets, selectedDatasetId]);

  useEffect(() => {
    if (!pipeId) return;
    let mounted = true;

    void Promise.all([getPipeById(pipeId), getSelectDatasetStepOutput(pipeId), getCleanDataStepOutput(pipeId), getSplitDataStepOutput(pipeId), getChooseTargetStepOutput(pipeId), getTrainModelsStepOutput(pipeId), getReviewResultsStepOutput(pipeId), getTestPredictionStepOutput(pipeId), getPublishPipeStepOutput(pipeId)]).then(([pipe, stepOutput, cleanOutput, splitOutput, targetOutput, trainOutput, reviewOutput, testOutput, publishOutput]) => {
      if (!mounted) return;
      if (pipe?.type === "tabular_classification" || pipe?.type === "tabular_regression") setLoadedPipeType(pipe.type);
      setExistingStepOutput(stepOutput);
      setCleanDataOutput(cleanOutput);
      setSplitDataOutput(splitOutput);
      setChooseTargetOutput(targetOutput);
      setTrainModelsOutput(trainOutput);
      setReviewResultsOutput(reviewOutput);
      setTestPredictionOutput(testOutput);
      setPublishPipeOutput(publishOutput);
      if (publishOutput?.publication_status === "live") setActiveStep("publish_pipe");
      else if (testOutput) setActiveStep("publish_pipe");
      else if (isReviewResultsAcknowledged(reviewOutput)) setActiveStep("test_prediction");
      else if (reviewOutput) setActiveStep("review_results");
      else if (trainOutput) setActiveStep("review_results");
      else if (targetOutput) setActiveStep("train_models");
      else if (splitOutput) setActiveStep("choose_target");
      else if (cleanOutput) setActiveStep("split_data");
      else if (stepOutput) setActiveStep("clean_data");
      if (stepOutput?.provider) setProvider(stepOutput.provider);
      if (stepOutput?.dataset_artifact_id) setAcceptedArtifactId(stepOutput.dataset_artifact_id);
    });

    return () => {
      mounted = false;
    };
  }, [pipeId]);

  const selectedDatasetKey = useMemo(() => {
    if (provider !== "huggingface") return selectedDataset.externalId;
    const [, datasetId = "", config = "default", split = "train"] = selectedDataset.externalId.split(":");
    return getDatasetKey(datasetId, config, split);
  }, [provider, selectedDataset.externalId]);

  useEffect(() => {
    if (existingStepOutput) return;
    if (provider !== "huggingface") {
      setRowsError(null);
      setLoadingRows(false);
      setLoadedRows(selectedDataset.rows);
      return;
    }

    const cached = hfRowsCacheRef.current.get(selectedDatasetKey);
    if (cached && reloadCounter === 0) {
      setRowsError(null);
      setLoadingRows(false);
      setLoadedRows(cached.rows);
      return;
    }

    const controller = new AbortController();
    setRowsError(null);
    setLoadingRows(true);

    void fetchHuggingFaceDatasetRows(selectedDataset, { signal: controller.signal })
      .then(({ rows }) => {
        hfRowsCacheRef.current.set(selectedDatasetKey, { rows });
        setLoadedRows(rows);
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setRowsError("Could not load this dataset right now. Please try reload.");
        setLoadedRows([]);
        })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingRows(false);
      });

    return () => controller.abort();
  }, [provider, selectedDatasetKey, reloadCounter, selectedDataset, existingStepOutput]);

  const profile = useMemo(() => profileDataset(loadedRows), [loadedRows]);

  async function handleUseDataset() {
    if (!user || !pipeId || !profile.eligibility.eligible) return;
    setSaving(true);
    setSaveError(null);
    try {
      const result = await persistSelectDatasetStep({ userId: user.id, pipeId, provider, connection: connections[0], dataset: { ...selectedDataset, rows: loadedRows }, profile });
      setAcceptedArtifactId(result.artifactId);
      setExistingStepOutput(result.output as SelectDatasetStepOutput);
      setActiveStep("clean_data");
    } catch {
      setSaveError("Unable to save this dataset. Please try again.");
    } finally {
      setSaving(false);
    }
  }

  if (!pipeId) {
    return <p className="text-sm text-red-700">This builder needs an existing draft pipe. Start from Create a pipe.</p>;
  }

  const reviewAcknowledged = isReviewResultsAcknowledged(reviewResultsOutput);

  return <div className="grid gap-8 lg:grid-cols-[280px_1fr]"><aside className="rounded-3xl border border-black/10 bg-white/60 p-5"><p className="text-xs uppercase tracking-[0.2em] text-black/40">Builder steps</p><ol className="mt-4 space-y-3">{steps.map((step, idx) => { const isActive = (idx === 0 && activeStep === "select_dataset") || (idx === 1 && activeStep === "clean_data") || (idx === 2 && activeStep === "split_data") || (idx === 3 && activeStep === "choose_target") || (idx === 4 && activeStep === "train_models") || (idx === 5 && activeStep === "review_results") || (idx === 6 && activeStep === "test_prediction") || (idx === 7 && activeStep === "publish_pipe"); const isCompleted = (idx === 0 && !!existingStepOutput) || (idx === 1 && !!cleanDataOutput) || (idx === 2 && !!splitDataOutput) || (idx === 3 && !!chooseTargetOutput) || (idx === 4 && !!trainModelsOutput) || (idx === 5 && reviewAcknowledged) || (idx === 6 && !!testPredictionOutput) || (idx === 7 && publishPipeOutput?.publication_status === "live"); return <li key={step} className={`rounded-2xl px-3 py-2 text-sm ${isActive ? "bg-black text-white" : isCompleted ? "bg-emerald-500/10 text-emerald-800" : "text-black/60"}`}>{idx + 1}. {step}{isCompleted ? " — completed" : !isActive && idx > 7 ? " — coming soon" : ""}</li>; })}</ol></aside>
    <div><Link to="/app/pipes" className="text-sm text-black/50 hover:text-black">← Back to pipes</Link><h1 className="mt-4 text-4xl font-semibold tracking-tight">{activeStep === "publish_pipe" ? "Step 8: Publish pipe" : activeStep === "test_prediction" ? "Step 7: Test prediction" : activeStep === "review_results" ? "Step 6: Review results" : activeStep === "train_models" ? "Step 5: Train models" : activeStep === "choose_target" ? "Step 4: Choose what to predict" : activeStep === "split_data" ? "Step 3: Split data" : activeStep === "clean_data" ? "Step 2: Clean data" : "Step 1: Select dataset"}</h1>{loadedPipeType ? <p className="mt-2 text-xs uppercase tracking-[0.16em] text-black/40">{loadedPipeType.replaceAll("_", " ")}</p> : null}<p className="mt-2 text-black/60">{activeStep === "publish_pipe" ? "Publish this tested model as a secure endpoint for workflows." : activeStep === "test_prediction" ? "Use one concrete example to test the trained pipe and inspect workflow-ready output." : activeStep === "review_results" ? "Review the recommended model, metrics, charts, and validation examples from real training results." : activeStep === "train_models" ? "MLP will train and compare real baseline models, then recommend the best one." : activeStep === "choose_target" ? "Choose the column your pipe should learn to predict. MLP will detect whether this is a classification or regression problem." : activeStep === "split_data" ? "MLP will split your cleaned dataset into training, validation, and test sets before model training." : activeStep === "clean_data" ? "MLP checks your dataset for common issues and recommends safe fixes before training." : "Connect data, preview it, and confirm it is eligible for the MVP before continuing."}</p>
      {existingStepOutput && activeStep === "select_dataset" ? <section className="mt-6 rounded-3xl border border-emerald-200 bg-emerald-50 p-6"><h2 className="font-semibold text-emerald-900">Dataset selected</h2><p className="mt-2 text-sm text-emerald-800">This draft already has a dataset artifact. You can continue to the next step.</p><dl className="mt-4 grid gap-2 text-sm text-emerald-900"><div><dt className="font-medium">Provider</dt><dd>{existingStepOutput.provider}</dd></div><div><dt className="font-medium">Source</dt><dd>{existingStepOutput.source_label}</dd></div><div><dt className="font-medium">Rows</dt><dd>{existingStepOutput.row_count}</dd></div><div><dt className="font-medium">Columns</dt><dd>{existingStepOutput.column_count}</dd></div><div><dt className="font-medium">Dataset artifact ID</dt><dd className="font-mono text-xs">{existingStepOutput.dataset_artifact_id}</dd></div></dl><button type="button" onClick={() => setActiveStep("clean_data")} className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Continue to Clean data</button></section> : null}
      {activeStep === "publish_pipe" ? <PublishPipeStep pipeId={pipeId} selectDatasetOutput={existingStepOutput} trainModelsOutput={trainModelsOutput} reviewResultsOutput={reviewResultsOutput} testPredictionOutput={testPredictionOutput} initialPublishOutput={publishPipeOutput} onCompleted={setPublishPipeOutput} onBackToReviewResults={() => setActiveStep("review_results")} onBackToTestPrediction={() => setActiveStep("test_prediction")} /> : null}
      {activeStep === "test_prediction" ? <TestPredictionStep pipeId={pipeId} reviewResultsOutput={reviewResultsOutput} initialTestPredictionOutput={testPredictionOutput} onCompleted={setTestPredictionOutput} onContinueToPublish={() => setActiveStep("publish_pipe")} onBackToReviewResults={() => setActiveStep("review_results")} /> : null}
      {activeStep === "review_results" ? <ReviewResultsStep pipeId={pipeId} trainModelsOutput={trainModelsOutput} initialReviewResultsOutput={reviewResultsOutput} onCompleted={(output) => setReviewResultsOutput(output)} onAcknowledged={(output) => { setReviewResultsOutput(output); setActiveStep("test_prediction"); }} onBackToTrainModels={() => setActiveStep("train_models")} /> : null}
      {activeStep === "train_models" ? <TrainModelsStep pipeId={pipeId} chooseTargetOutput={chooseTargetOutput} splitDataOutput={splitDataOutput} initialTrainModelsOutput={trainModelsOutput} onCompleted={(output) => { setTrainModelsOutput(output); setActiveStep("review_results"); }} onBackToChooseTarget={() => setActiveStep("choose_target")} /> : null}
      {activeStep === "choose_target" ? <ChooseTargetStep pipeId={pipeId} pipeType={loadedPipeType} splitDataOutput={splitDataOutput} initialChooseTargetOutput={chooseTargetOutput} onCompleted={(output) => { setChooseTargetOutput(output); setActiveStep("train_models"); }} onBackToSplitData={() => setActiveStep("split_data")} /> : null}
      {activeStep === "split_data" ? <SplitDatasetStep pipeId={pipeId} pipeType={loadedPipeType} cleanDataOutput={cleanDataOutput} initialSplitOutput={splitDataOutput} onCompleted={(output) => { setSplitDataOutput(output); setActiveStep("choose_target"); }} onBackToCleanData={() => setActiveStep("clean_data")} /> : null}
      {activeStep === "clean_data" ? <CleanDataStep pipeId={pipeId} selectDatasetOutput={existingStepOutput} initialCleanOutput={cleanDataOutput} onCompleted={(output) => { setCleanDataOutput(output); setActiveStep("split_data"); }} onBackToSelectDataset={() => setActiveStep("select_dataset")} /> : null}
      {!existingStepOutput && activeStep === "select_dataset" ? <><section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="font-semibold">1) Connect data</h2><div className="mt-4 grid gap-3 md:grid-cols-3">{(["huggingface", "airtable", "google_sheets"] as DatasetProvider[]).map((item) => <button key={item} onClick={() => { setProvider(item); setSelectedDatasetId(""); setRowsError(null); }} className={`rounded-2xl border px-4 py-3 text-left ${provider === item ? "border-black bg-white" : "border-black/10"}`}><p className="font-medium">{providerLabels[item]}</p><p className="text-xs text-black/50">{item === "huggingface" ? "Public datasets mode" : "Alpha mock, OAuth coming soon"}</p></button>)}</div>
        <div className="mt-4 rounded-2xl border border-black/10 p-4"><p className="text-sm text-black/50">Connected account</p><p className="mt-1 font-medium">{connections[0].providerAccountLabel}</p></div>
      </section>
      <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="font-semibold">2) Choose a dataset source</h2><div className="mt-4 grid gap-3">{datasets.map((dataset) => <button key={dataset.externalId} onClick={() => setSelectedDatasetId(dataset.externalId)} className={`rounded-2xl border p-4 text-left ${selectedDataset.externalId === dataset.externalId ? "border-black bg-white" : "border-black/10"}`}><p className="font-medium">{dataset.name}</p><p className="text-sm text-black/50">{dataset.sourceLabel}</p></button>)}</div></section>
      <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><div className="flex items-center justify-between"><h2 className="font-semibold">3) Preview data</h2><button type="button" onClick={() => { hfRowsCacheRef.current.delete(selectedDatasetKey); setReloadCounter((value) => value + 1); }} disabled={loadingRows} className="rounded-full border border-black/15 px-3 py-1 text-xs text-black/60 disabled:opacity-50">Reload data</button></div>{loadingRows ? <p className="mt-3 text-sm text-black/50">Loading rows…</p> : null}{rowsError ? <p className="mt-3 rounded-2xl bg-red-500/10 px-4 py-3 text-sm text-red-700">{rowsError}</p> : null}{!loadingRows && !rowsError ? <div className="mt-4 overflow-x-auto"><table className="min-w-full text-sm"><thead><tr>{Object.keys(profile.preview[0] ?? {}).map((column) => <th key={column} className="border-b border-black/10 px-3 py-2 text-left font-medium">{column}</th>)}</tr></thead><tbody>{profile.preview.slice(0, 5).map((row, idx) => <tr key={idx}>{Object.entries(row).map(([k, v]) => <td key={k} className="border-b border-black/5 px-3 py-2 text-black/70">{String(v)}</td>)}</tr>)}</tbody></table></div> : null}</section>
      <section className="mt-6 grid gap-6 lg:grid-cols-2"><div className="rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="font-semibold">4) Dataset profile</h2><p className="mt-3 text-sm text-black/60">{profile.rowCount} rows • {profile.columnCount} columns</p></div><div className="rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="font-semibold">5) Eligibility</h2><p className={`mt-2 text-sm font-medium ${profile.eligibility.eligible ? "text-emerald-700" : "text-red-700"}`}>{profile.eligibility.eligible ? "This dataset is eligible." : "This dataset is not eligible yet."}</p><ul className="mt-3 list-disc space-y-1 pl-4 text-sm text-red-700">{profile.eligibility.blocking_issues.map((issue) => <li key={`${issue.code}-${issue.column ?? ""}`}>{issue.message}</li>)}</ul><button onClick={handleUseDataset} disabled={!profile.eligibility.eligible || saving || loadingRows || !!rowsError} className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-black/20">{saving ? "Saving…" : "Use this dataset"}</button>{saveError ? <p className="mt-3 text-sm text-red-700">{saveError}</p> : null}{acceptedArtifactId ? <div className="mt-4 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800">Dataset saved. Continue to Clean data.<p className="mt-2 text-xs">dataset_artifact_id: {acceptedArtifactId}</p></div> : null}</div></section></> : null}
    </div></div>;
}
