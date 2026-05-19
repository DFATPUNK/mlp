import { Link, useParams } from "react-router-dom";
import type { BuilderPipeType } from "../types/pipe";

const builderSteps = [
  {
    number: "01",
    title: "Select dataset",
    description: "Connect Google Sheets, Airtable, or Hugging Face Dataset.",
  },
  {
    number: "02",
    title: "Clean data",
    description: "Impute missing values and encode categorical columns.",
  },
  {
    number: "03",
    title: "Split dataset",
    description: "Create train, validation, and test sets.",
  },
  {
    number: "04",
    title: "Prepare features",
    description: "Choose the target column and confirm task type.",
  },
  {
    number: "05",
    title: "Train models",
    description: "Compare multiple sklearn models on the same dataset.",
  },
  {
    number: "06",
    title: "Evaluate model",
    description: "Generate metrics and matplotlib charts.",
  },
  {
    number: "07",
    title: "Test prediction",
    description: "Run a sample prediction before publishing.",
  },
  {
    number: "08",
    title: "Publish pipe",
    description: "Expose the trained pipe as a reusable endpoint.",
  },
];

const pipeTypeCopy: Record<
  BuilderPipeType,
  {
    title: string;
    description: string;
  }
> = {
  tabular_classification: {
    title: "Tabular Classification Builder",
    description:
      "Build a pipe that predicts a category from connected SaaS data.",
  },
  tabular_regression: {
    title: "Tabular Regression Builder",
    description: "Build a pipe that predicts a number from connected SaaS data.",
  },
};

export function PipeBuilderPlaceholderPage() {
  const { pipeType } = useParams();

  const normalizedPipeType = isBuilderPipeType(pipeType)
    ? pipeType
    : "tabular_classification";

  const copy = pipeTypeCopy[normalizedPipeType];

  return (
    <div>
      <Link
        to="/app/pipes/new"
        className="text-sm text-black/50 hover:text-black"
      >
        ← Back to pipe types
      </Link>

      <section className="mt-8 grid gap-8 border-b border-black/10 pb-10 lg:grid-cols-[1.1fr_0.9fr]">
        <div>
          <p className="text-sm font-medium uppercase tracking-[0.2em] text-black/40">
            Builder placeholder
          </p>

          <h1 className="mt-4 max-w-4xl text-5xl font-semibold tracking-[-0.05em]">
            {copy.title}
          </h1>

          <p className="mt-6 max-w-2xl text-base leading-7 text-black/60">
            {copy.description}
          </p>
        </div>

        <aside className="rounded-3xl border border-black/10 bg-white/60 p-6">
          <h2 className="text-lg font-semibold">Builder coming soon</h2>

          <p className="mt-4 text-sm leading-6 text-black/60">
            This route is intentionally prepared before implementing the full
            editor. The next product step is to turn each builder section into a
            testable Zapier-like node.
          </p>

          <div className="mt-6 rounded-2xl bg-black p-4 font-mono text-xs text-white">
            <pre>{`/app/pipes/new/${normalizedPipeType}`}</pre>
          </div>
        </aside>
      </section>

      <section className="mt-10">
        <div className="mb-5">
          <h2 className="text-lg font-semibold">Planned builder flow</h2>
          <p className="mt-2 text-sm text-black/50">
            Each step will be configurable and testable independently.
          </p>
        </div>

        <div className="grid gap-4">
          {builderSteps.map((step) => (
            <div
              key={step.number}
              className="grid gap-4 rounded-3xl border border-black/10 bg-white/60 p-5 md:grid-cols-[80px_1fr_auto] md:items-center"
            >
              <span className="font-mono text-sm text-black/35">
                {step.number}
              </span>

              <div>
                <h3 className="font-semibold">{step.title}</h3>
                <p className="mt-1 text-sm leading-6 text-black/50">
                  {step.description}
                </p>
              </div>

              <span className="w-fit rounded-full border border-black/10 px-3 py-1 text-xs text-black/40">
                Coming soon
              </span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function isBuilderPipeType(value: string | undefined): value is BuilderPipeType {
  return value === "tabular_classification" || value === "tabular_regression";
}