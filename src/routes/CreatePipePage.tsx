import { Link } from "react-router-dom";

export function CreatePipePage() {
  return (
    <div>
      <Link to="/app/pipes" className="text-sm text-black/50 hover:text-black">
        ← Back to pipes
      </Link>

      <section className="mt-8 max-w-4xl">
        <p className="text-sm font-medium uppercase tracking-[0.2em] text-black/40">
          Create a pipe
        </p>

        <h1 className="mt-4 text-5xl font-semibold tracking-[-0.05em]">
          Start from connected SaaS data.
        </h1>

        <p className="mt-6 max-w-2xl text-base leading-7 text-black/60">
          The visual builder will let you select a data source, clean the data,
          split it, train models, compare results, and publish a reusable ML
          pipe.
        </p>
      </section>

      <section className="mt-10 grid gap-5 md:grid-cols-2">
        <button className="rounded-3xl border border-black/10 bg-white/60 p-6 text-left transition hover:-translate-y-0.5 hover:bg-white">
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-black/40">
            Coming soon
          </p>
          <h2 className="mt-4 text-2xl font-semibold">Tabular classification</h2>
          <p className="mt-3 text-sm leading-6 text-black/60">
            Predict a category such as churn, lead quality, ticket priority, or
            product type.
          </p>
        </button>

        <button className="rounded-3xl border border-black/10 bg-white/60 p-6 text-left transition hover:-translate-y-0.5 hover:bg-white">
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-black/40">
            Coming soon
          </p>
          <h2 className="mt-4 text-2xl font-semibold">Tabular regression</h2>
          <p className="mt-3 text-sm leading-6 text-black/60">
            Predict a number such as price, revenue, delivery time, or demand.
          </p>
        </button>
      </section>
    </div>
  );
}