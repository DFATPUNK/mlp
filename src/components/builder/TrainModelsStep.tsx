import { useMemo, useState } from "react";
import { supabase } from "../../lib/supabaseClient";
import type { ChooseTargetStepOutput, SplitDataStepOutput, TrainModelsStepOutput } from "../../lib/pipes";

type TrainModelsStepProps = {
  pipeId: string;
  chooseTargetOutput: ChooseTargetStepOutput | null;
  splitDataOutput: SplitDataStepOutput | null;
  initialTrainModelsOutput: TrainModelsStepOutput | null;
  onCompleted: (output: TrainModelsStepOutput) => void;
  onBackToChooseTarget: () => void;
};

type TrainModelsResponse = {
  trained_models_artifact_id: string;
  recommended_model_id: string;
  recommended_model_name: string;
  primary_metric_name: string;
  primary_metric_value: number;
  model_count: number;
  task_type: "tabular_classification" | "tabular_regression";
  target_column: string;
};

function taskLabel(taskType: "tabular_classification" | "tabular_regression") {
  return taskType === "tabular_regression" ? "Regression" : "Classification";
}

function metricLabel(metricName: string) {
  if (metricName === "mae") return "Average error";
  if (metricName === "f1_macro") return "F1 score";
  return metricName.replaceAll("_", " ");
}

function formatMetric(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

export function TrainModelsStep({ pipeId, chooseTargetOutput, splitDataOutput, initialTrainModelsOutput, onCompleted, onBackToChooseTarget }: TrainModelsStepProps) {
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusIndex, setStatusIndex] = useState(0);
  const [savedOutput, setSavedOutput] = useState<TrainModelsStepOutput | null>(null);
  const output = savedOutput ?? initialTrainModelsOutput;
  const statuses = ["Preparing features", "Training models", "Comparing validation results"];
  const summary = useMemo(() => {
    if (!chooseTargetOutput || !splitDataOutput) return null;
    return {
      targetColumn: chooseTargetOutput.target_column,
      taskType: chooseTargetOutput.detected_task_type,
      trainRows: splitDataOutput.train_rows,
      validationRows: splitDataOutput.validation_rows,
      featureCount: chooseTargetOutput.feature_columns.length,
    };
  }, [chooseTargetOutput, splitDataOutput]);

  async function handleTrainModels() {
    if (!chooseTargetOutput) return;
    setTraining(true);
    setError(null);
    setStatusIndex(0);
    const timer = window.setInterval(() => setStatusIndex((current) => Math.min(current + 1, statuses.length - 1)), 1800);
    try {
      const trainingApiUrl = import.meta.env.VITE_ML_TRAINING_API_URL as string | undefined;
      if (!trainingApiUrl) throw new Error("Training service is not configured.");
      const { data } = await supabase.auth.getSession();
      const token = data.session?.access_token;
      if (!token) throw new Error("Please sign in again before training models.");
      const response = await fetch(`${trainingApiUrl.replace(/\/$/, "")}/train-models`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ pipe_id: pipeId, target_config_artifact_id: chooseTargetOutput.target_config_artifact_id }),
      });
      const payload = await response.json() as TrainModelsResponse & { error?: string };
      if (!response.ok) throw new Error(payload.error ?? "Training failed.");
      const nextOutput: TrainModelsStepOutput = {
        step_key: "train_models",
        status: "completed",
        trained_models_artifact_id: payload.trained_models_artifact_id,
        previous_target_config_artifact_id: chooseTargetOutput.target_config_artifact_id,
        task_type: payload.task_type,
        target_column: payload.target_column,
        recommended_model_id: payload.recommended_model_id,
        recommended_model_name: payload.recommended_model_name,
        primary_metric_name: payload.primary_metric_name,
        primary_metric_value: payload.primary_metric_value,
        model_count: payload.model_count,
        storage: { format: "json", uri: `artifact:${payload.trained_models_artifact_id}` },
      };
      setSavedOutput(nextOutput);
      onCompleted(nextOutput);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to train models. Please try again.");
    } finally {
      window.clearInterval(timer);
      setTraining(false);
    }
  }

  if (!chooseTargetOutput) {
    return <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Choose a target before training models.</h2><button type="button" onClick={onBackToChooseTarget} className="mt-4 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Back to Choose target</button></section>;
  }

  if (output) {
    return <section className="mt-6 rounded-3xl border border-emerald-200 bg-emerald-50 p-6"><h2 className="text-lg font-semibold text-emerald-900">Models trained.</h2><p className="mt-2 text-sm text-emerald-800">Train Models is complete. Review Results is coming soon.</p><dl className="mt-4 grid gap-2 text-sm text-emerald-900 md:grid-cols-2"><div><dt className="font-medium">Recommended model</dt><dd>{output.recommended_model_name}</dd>{output.recommended_model_name.toLowerCase().includes("random forest") ? <p className="mt-1 text-xs text-emerald-700">Random Forest combines predictions from many decision trees. You can inspect how it reached individual predictions in Step 7.</p> : null}</div><div><dt className="font-medium">Primary metric</dt><dd>{metricLabel(output.primary_metric_name)}: {formatMetric(output.primary_metric_value)}</dd></div><div><dt className="font-medium">Models trained</dt><dd>{output.model_count}</dd></div><div><dt className="font-medium">Target</dt><dd>{output.target_column}</dd></div><div className="md:col-span-2"><dt className="font-medium">Trained models artifact ID</dt><dd className="font-mono text-xs">{output.trained_models_artifact_id}</dd></div></dl><button type="button" className="mt-5 rounded-full bg-black px-4 py-2 text-sm font-medium text-white">Continue to Review results</button></section>;
  }

  return <div>
    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><h2 className="text-lg font-semibold">Train real baseline models</h2><p className="mt-2 text-sm text-black/60">MLP will train and compare real baseline models, then recommend the best one.</p>{summary ? <dl className="mt-4 grid gap-3 text-sm md:grid-cols-2"><div><dt className="text-black/50">Target column</dt><dd>{summary.targetColumn}</dd></div><div><dt className="text-black/50">Task type</dt><dd>{taskLabel(summary.taskType)}</dd></div><div><dt className="text-black/50">Training rows</dt><dd>{summary.trainRows}</dd></div><div><dt className="text-black/50">Validation rows</dt><dd>{summary.validationRows}</dd></div><div><dt className="text-black/50">Feature columns</dt><dd>{summary.featureCount}</dd></div></dl> : null}</section>
    <section className="mt-6 rounded-3xl border border-black/10 bg-white/60 p-6"><button type="button" onClick={handleTrainModels} disabled={training} className="rounded-full bg-black px-4 py-2 text-sm font-medium text-white disabled:bg-black/30">{training ? "Training…" : "Train models"}</button>{training ? <p className="mt-4 text-sm text-black/60">{statuses[statusIndex]}</p> : null}{error ? <p className="mt-4 rounded-2xl bg-red-500/10 px-4 py-3 text-sm text-red-700">{error}</p> : null}</section>
  </div>;
}
