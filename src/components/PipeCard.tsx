import { Link } from "react-router-dom";
import type { Pipe } from "../types/pipe";

type PipeCardProps = {
  pipe: Pipe;
  onDelete?: (pipe: Pipe) => void;
  deleting?: boolean;
};

export function PipeCard({ pipe, onDelete, deleting = false }: PipeCardProps) {
  const isDraft = pipe.status === "draft" && !pipe.isTemplate;
  const primaryHref = isDraft ? `/app/pipes/${pipe.id}/builder` : `/app/pipes/${pipe.id}`;
  const primaryLabel = isDraft ? "Resume editing" : "Open pipe";

  return (
    <article className="group rounded-3xl border border-black/10 bg-white/60 p-6 shadow-sm transition hover:-translate-y-0.5 hover:bg-white">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-black/40">
            {pipe.type.replaceAll("_", " ")}
          </p>

          <h2 className="mt-4 text-2xl font-semibold tracking-tight">{pipe.name}</h2>
        </div>

        <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-700">
          {pipe.status}
        </span>
      </div>

      <p className="mt-5 max-w-xl text-sm leading-6 text-black/60">{pipe.description}</p>

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
