import { Link } from "react-router-dom";
import type { BuilderPipeType } from "../types/pipe";

const pipeOptions: {
  type: BuilderPipeType;
  title: string;
  eyebrow: string;
  description: string;
  examples: string[];
}[] = [
  {
    type: "tabular_classification",
    title: "Tabular classification",
    eyebrow: "Predict a category",
    description:
      "Use this when you want the pipe to choose between labels such as churn/no churn, qualified/not qualified, or priority levels.",
    examples: ["Lead quality", "Customer churn", "Ticket priority"],
  },
  {
    type: "tabular_regression",
    title: "Tabular regression",
    eyebrow: "Predict a number",
    description:
      "Use this when you want the pipe to estimate a numeric value such as price, revenue, demand, or delivery time.",
    examples: ["Revenue forecast", "Price estimate", "Delivery time"],
  },
];

export function CreatePipePage() {
  return (
    <div>
      <Link to="/app/pipes" className="text-sm text-black/50 hover:text-black">
        ← Back to pipes
      </Link>

      <section className="mt-8 grid gap-8 border-b border-black/10 pb-10 lg:grid-cols-[1.1fr_0.9fr]">
        <div>
          <p className="text-sm font-medium uppercase tracking-[0.2em] text-black/40">
            Create a pipe
          </p>

          <h1 className="mt-4 max-w-4xl text-5xl font-semibold tracking-[-0.05em]">
            Start from connected SaaS data.
          </h1>

          <p className="mt-6 max-w-2xl text-base leading-7 text-black/60">
            The builder will guide you step by step: select a dataset, clean the
            data, split it, prepare features, train models, compare results, and
            publish a reusable ML pipe.
          </p>
        </div>

        <aside className="rounded-3xl border border-black/10 bg-white/60 p-6">
          <h2 className="text-lg font-semibold">MVP data sources</h2>

          <div className="mt-5 flex flex-wrap gap-2">
            {["Google Sheets", "Airtable", "Hugging Face Dataset"].map(
              (source) => (
                <span
                  key={source}
                  className="rounded-full border border-black/10 px-3 py-1 text-sm text-black/60"
                >
                  {source}
                </span>
              ),
            )}
          </div>

          <p className="mt-5 text-sm leading-6 text-black/50">
            Uploads are intentionally excluded. MLP is designed as a middle layer
            between SaaS tools and ML pipelines.
          </p>
        </aside>
      </section>

      <section className="mt-10">
        <div className="mb-5">
          <h2 className="text-lg font-semibold">Choose your pipe type</h2>
          <p className="mt-2 text-sm text-black/50">
            For the MVP, only small tabular datasets are supported.
          </p>
        </div>

        <div className="grid gap-5 md:grid-cols-2">
          {pipeOptions.map((option) => (
            <Link
              key={option.type}
              to={`/app/pipes/new/${option.type}`}
              className="group rounded-3xl border border-black/10 bg-white/60 p-6 text-left shadow-sm transition hover:-translate-y-0.5 hover:bg-white"
            >
              <p className="text-xs font-medium uppercase tracking-[0.18em] text-black/40">
                {option.eyebrow}
              </p>

              <h3 className="mt-4 text-2xl font-semibold tracking-tight">
                {option.title}
              </h3>

              <p className="mt-3 text-sm leading-6 text-black/60">
                {option.description}
              </p>

              <div className="mt-6 flex flex-wrap gap-2">
                {option.examples.map((example) => (
                  <span
                    key={example}
                    className="rounded-full border border-black/10 px-3 py-1 text-xs text-black/50"
                  >
                    {example}
                  </span>
                ))}
              </div>

              <div className="mt-8 text-sm font-medium">
                Open builder{" "}
                <span className="inline-block transition group-hover:translate-x-1">
                  →
                </span>
              </div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}