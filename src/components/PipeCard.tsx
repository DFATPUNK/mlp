import { useEffect, useState, type KeyboardEvent } from "react";
import { Link } from "react-router-dom";
import type { PipeCardMetadata } from "../lib/pipes";
import type { Pipe } from "../types/pipe";

type PipeCardProps = {
  pipe: Pipe;
  metadata: PipeCardMetadata;
  onDelete?: (pipe: Pipe) => void;
  onRename?: (pipe: Pipe, name: string) => Promise<void>;
  deleting?: boolean;
};

function metricLabel(metricName: string | null) {
  if (!metricName) return null;
  if (metricName === "mae") return "Average error";
  if (metricName === "rmse") return "RMSE";
  if (metricName === "r2") return "R²";
  if (metricName === "f1_macro" || metricName === "f1_weighted") return "F1 score";
  return metricName.replaceAll("_", " ");
}

function formatMetric(value: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "—";
}

export function PipeCard({ pipe, metadata, onDelete, onRename, deleting = false }: PipeCardProps) {
  const isDraft = pipe.status === "draft" && !pipe.isTemplate;
  const canRename = !pipe.isTemplate && !!onRename;
  const primaryHref = isDraft ? `/app/pipes/${pipe.id}/builder` : `/app/pipes/${pipe.id}`;
  const primaryLabel = isDraft ? "Resume editing" : "Open pipe";
  const [editing, setEditing] = useState(false);
  const [draftName, setDraftName] = useState(pipe.name);
  const [savingName, setSavingName] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);

  useEffect(() => {
    if (!editing) setDraftName(pipe.name);
  }, [pipe.name, editing]);

  async function saveName() {
    if (!canRename) return;
    const trimmed = draftName.trim();
    if (!trimmed) {
      setRenameError("Pipe name cannot be empty.");
      return;
    }
    if (trimmed === pipe.name) {
      setEditing(false);
      setRenameError(null);
      return;
    }
    setSavingName(true);
    setRenameError(null);
    try {
      await onRename(pipe, trimmed);
      setEditing(false);
    } catch {
      setRenameError("Unable to rename this pipe.");
    } finally {
      setSavingName(false);
    }
  }

  function handleNameKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      event.preventDefault();
      void saveName();
    }
    if (event.key === "Escape") {
      event.preventDefault();
      setDraftName(pipe.name);
      setRenameError(null);
      setEditing(false);
    }
  }

  return (
    <article className="group rounded-3xl border border-black/10 bg-white/60 p-6 shadow-sm transition hover:-translate-y-0.5 hover:bg-white">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap gap-2">
            <span className="rounded-full border border-black/10 bg-white/70 px-3 py-1 text-xs font-medium uppercase tracking-[0.14em] text-black/50">
              {pipe.type.replaceAll("_", " ")}
            </span>
            <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-700">
              {pipe.status}
            </span>
          </div>

          <div className="mt-4">
            {editing ? (
              <div>
                <input
                  value={draftName}
                  onChange={(event) => setDraftName(event.target.value)}
                  onKeyDown={handleNameKeyDown}
                  onBlur={() => void saveName()}
                  disabled={savingName}
                  autoFocus
                  className="w-full rounded-2xl border border-black/15 bg-white px-3 py-2 text-2xl font-semibold tracking-tight outline-none focus:border-black disabled:opacity-60"
                />
                <p className="mt-2 text-xs text-black/40">Press Enter to save or Escape to cancel.</p>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <h2 className="min-w-0 break-words text-2xl font-semibold tracking-tight">{pipe.name}</h2>
                {canRename ? (
                  <button
                    type="button"
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      setEditing(true);
                    }}
                    className="shrink-0 rounded-full border border-black/10 px-3 py-1 text-xs font-medium text-black/60 transition hover:border-black/30"
                  >
                    Rename
                  </button>
                ) : null}
              </div>
            )}
            {savingName ? <p className="mt-2 text-xs text-black/50">Saving name…</p> : null}
            {renameError ? <p className="mt-2 text-xs text-red-700">{renameError}</p> : null}
          </div>
        </div>
      </div>

      <p className="mt-5 rounded-2xl bg-black/5 px-4 py-3 text-sm leading-6 text-black/70">{metadata.summary}</p>

      <dl className="mt-5 grid gap-3 text-sm md:grid-cols-2">
        <div>
          <dt className="text-black/45">Dataset</dt>
          <dd className="font-medium text-black/80">{metadata.datasetLabel ?? "Not selected yet"}</dd>
        </div>
        <div>
          <dt className="text-black/45">Target</dt>
          <dd className="font-medium text-black/80">{metadata.targetColumn ?? "Not selected yet"}</dd>
        </div>
        <div>
          <dt className="text-black/45">Steps completed</dt>
          <dd className="font-medium text-black/80">{metadata.completedStepCount}/{metadata.totalStepCount}</dd>
        </div>
        <div>
          <dt className="text-black/45">{metadata.completedStepCount === 0 ? "Current step" : "Next"}</dt>
          <dd className="font-medium text-black/80">{metadata.nextStepLabel ?? metadata.currentStepLabel}</dd>
        </div>
      </dl>

      <div className="mt-5 rounded-2xl border border-black/10 bg-white/60 px-4 py-3 text-sm">
        <p className="text-black/45">Recommended model</p>
        <p className="mt-1 font-medium text-black/80">{metadata.recommendedModelName ?? "Not trained yet"}</p>
        {metadata.recommendedModelName ? (
          <p className="mt-1 text-xs text-black/50">
            {metricLabel(metadata.primaryMetricName)}: {formatMetric(metadata.primaryMetricValue)}
          </p>
        ) : null}
      </div>

      <div className="mt-8 flex flex-wrap gap-3">
        <Link
          to={primaryHref}
          className="rounded-full bg-black px-4 py-2 text-sm font-medium text-white transition hover:bg-black/80"
        >
          {primaryLabel}
        </Link>

        <Link
          to={`/app/pipes/${pipe.id}`}
          className="rounded-full border border-black/10 px-4 py-2 text-sm font-medium transition hover:border-black/30"
        >
          Test
        </Link>

        {!pipe.isTemplate && onDelete ? (
          <button
            type="button"
            disabled={deleting}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onDelete(pipe);
            }}
            className="rounded-full border border-red-500/20 bg-red-500/5 px-4 py-2 text-sm font-medium text-red-700 transition hover:bg-red-500/10 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {deleting ? "Deleting…" : "Delete"}
          </button>
        ) : null}
      </div>
    </article>
  );
}
