import { Link } from "react-router-dom";
import { PipeCard } from "../components/PipeCard";
import { mockPipes } from "../data/mockPipes";

export function PipesPage() {
  return (
    <div>
      <section className="flex flex-col justify-between gap-8 border-b border-black/10 pb-10 md:flex-row md:items-end">
        <div>
          <p className="text-sm font-medium uppercase tracking-[0.2em] text-black/40">
            NLP dashboard
          </p>

          <h1 className="mt-4 max-w-3xl text-5xl font-semibold tracking-[-0.05em]">
            Build, test, and publish machine learning pipes.
          </h1>

          <p className="mt-6 max-w-2xl text-base leading-7 text-black/60">
            Start with a production-ready image classifier template, then create
            tabular ML pipelines from connected SaaS data.
          </p>
        </div>

        <Link
          to="/app/pipes/new"
          className="inline-flex w-fit rounded-full bg-black px-5 py-3 text-sm font-medium text-white transition hover:bg-black/80"
        >
          Create a pipe
        </Link>
      </section>

      <section className="py-10">
        <div className="mb-5 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Production</h2>
          <span className="text-sm text-black/40">{mockPipes.length} pipe</span>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          {mockPipes.map((pipe) => (
            <PipeCard key={pipe.id} pipe={pipe} />
          ))}
        </div>
      </section>

      <section className="rounded-3xl border border-dashed border-black/15 p-8">
        <h2 className="text-lg font-semibold">Drafts</h2>
        <p className="mt-2 text-sm text-black/50">
          No custom pipes yet. Create your first tabular ML pipe in the next
          sprint.
        </p>
      </section>
    </div>
  );
}