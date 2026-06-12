import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getPipeById } from "../lib/pipes";
import type { Pipe, SchemaField, WasteClassifierOutput } from "../types/pipe";

const wasteClassifierInputSchema: SchemaField[] = [
  {
    name: "image_url",
    type: "url",
    description: "Public URL of the waste image to classify.",
    required: false,
  },
  {
    name: "image_file",
    type: "file",
    description: "Image file sent directly to the pipe.",
    required: false,
  },
];

const wasteClassifierOutputSchema: SchemaField[] = [
  {
    name: "predicted_category",
    type: "string",
    description:
      "The most likely waste category predicted by the image classifier.",
    required: true,
  },
  {
    name: "confidence",
    type: "number",
    description: "Confidence score between 0 and 1.",
    required: true,
  },
  {
    name: "alternative_categories",
    type: "array",
    description: "Other possible categories with their confidence scores.",
    required: true,
  },
];

const mockPrediction: WasteClassifierOutput = {
  predicted_category: "cardboard",
  confidence: 0.91,
  alternative_categories: [
    { label: "paper", confidence: 0.06 },
    { label: "other_trash", confidence: 0.03 },
  ],
};

const categories = [
  "Aluminium",
  "Cardboard",
  "Glass",
  "Biodegradable",
  "Other trash",
];

export function PipeDetailPage() {
  const { pipeId } = useParams();
  const [pipe, setPipe] = useState<Pipe | null>(null);
  const [loading, setLoading] = useState(true);
  const [testStatus, setTestStatus] = useState<
    "idle" | "running" | "completed"
  >("idle");
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "failed">(
    "idle",
  );

  const endpoint = useMemo(() => {
    if (!pipeId) return "";

    return `https://api.mlp.jeremybrunet.com/pipes/${pipeId}/predict`;
  }, [pipeId]);

  useEffect(() => {
    let mounted = true;

    if (!pipeId) {
      setLoading(false);
      return;
    }

    getPipeById(pipeId)
      .then((item) => {
        if (!mounted) return;
        setPipe(item);
      })
      .finally(() => {
        if (!mounted) return;
        setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, [pipeId]);

  async function handleTestPipe() {
    setTestStatus("running");

    window.setTimeout(() => {
      setTestStatus("completed");
    }, 700);
  }

  async function handleCopyEndpoint() {
    try {
      await navigator.clipboard.writeText(endpoint);
      setCopyStatus("copied");

      window.setTimeout(() => {
        setCopyStatus("idle");
      }, 1800);
    } catch {
      setCopyStatus("failed");
    }
  }

  if (loading) {
    return <p className="text-sm text-black/50">Loading pipe…</p>;
  }

  if (!pipe) {
    return (
      <div>
        <h1 className="text-3xl font-semibold">Pipe not found</h1>
        <Link className="mt-6 inline-block text-sm underline" to="/app/pipes">
          Back to pipes
        </Link>
      </div>
    );
  }

  return (
    <div>
      <Link to="/app/pipes" className="text-sm text-black/50 hover:text-black">
        ← Back to pipes
      </Link>

      <section className="mt-8 grid gap-8 lg:grid-cols-[1.1fr_0.9fr]">
        <div>
          <p className="text-sm font-medium uppercase tracking-[0.2em] text-black/40">
            Published template
          </p>

          <div className="mt-4 flex flex-wrap items-center gap-3">
            <h1 className="text-5xl font-semibold tracking-[-0.05em]">
              {pipe.name}
            </h1>

            <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-xs font-medium uppercase tracking-[0.14em] text-emerald-700">
              {pipe.status}
            </span>
          </div>

          <p className="mt-6 max-w-2xl text-base leading-7 text-black/60">
            {pipe.description}
          </p>

          <div className="mt-8 flex flex-wrap gap-3">
            {pipe.status === "draft" && !pipe.isTemplate ? (
              <Link
                to={`/app/pipes/${pipe.id}/builder`}
                className="rounded-full bg-black px-5 py-3 text-sm font-medium text-white transition hover:bg-black/80"
              >
                Resume editing
              </Link>
            ) : (
            <button
              type="button"
              onClick={handleTestPipe}
              disabled={testStatus === "running"}
              className="rounded-full bg-black px-5 py-3 text-sm font-medium text-white transition hover:bg-black/80 disabled:cursor-not-allowed disabled:bg-black/40"
            >
              {testStatus === "running" ? "Testing…" : "Test pipe"}
            </button>
            )}

            <button
              type="button"
              onClick={handleCopyEndpoint}
              className="rounded-full border border-black/10 px-5 py-3 text-sm font-medium transition hover:border-black/30"
            >
              {copyStatus === "copied"
                ? "Endpoint copied"
                : copyStatus === "failed"
                  ? "Copy failed"
                  : "Copy endpoint"}
            </button>
          </div>

          <div className="mt-8 rounded-3xl border border-black/10 bg-white/60 p-5">
            <p className="text-xs font-medium uppercase tracking-[0.18em] text-black/40">
              Mock endpoint
            </p>
            <p className="mt-3 break-all font-mono text-sm text-black/70">
              {endpoint}
            </p>
          </div>
        </div>

        <aside className="rounded-3xl border border-black/10 bg-white/60 p-6 shadow-sm">
          <h2 className="text-lg font-semibold">What this pipe does</h2>

          <p className="mt-4 text-sm leading-6 text-black/60">
            This template receives a waste image and returns a predicted
            category. In the future, this output can be mapped into SaaS
            actions like moving a file, creating a record, sending an alert, or
            routing an item for review.
          </p>

          <div className="mt-6 flex flex-wrap gap-2">
            {categories.map((category) => (
              <span
                key={category}
                className="rounded-full border border-black/10 px-3 py-1 text-sm text-black/60"
              >
                {category}
              </span>
            ))}
          </div>
        </aside>
      </section>

      <section className="mt-10 grid gap-6 lg:grid-cols-2">
        <SchemaCard title="Input schema" fields={wasteClassifierInputSchema} />
        <SchemaCard title="Output schema" fields={wasteClassifierOutputSchema} />
      </section>

      <section className="mt-10 grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-3xl border border-black/10 bg-white/60 p-6">
          <h2 className="text-lg font-semibold">How this pipe can be used</h2>

          <div className="mt-6 space-y-5">
            <UseCaseStep
              number="1"
              title="Trigger"
              description="A new image is added to a Google Drive folder."
            />
            <UseCaseStep
              number="2"
              title="Prediction"
              description="MLP runs the Waste Image Classifier on the image."
            />
            <UseCaseStep
              number="3"
              title="Action"
              description="The file is moved into the right folder based on the predicted category."
            />
          </div>
        </div>

        <div className="rounded-3xl border border-black/10 bg-white/60 p-6">
          <div className="flex items-center justify-between gap-4">
            <h2 className="text-lg font-semibold">Latest test result</h2>
            <span className="text-sm text-black/40">
              {testStatus === "completed"
                ? "Mock result"
                : "Run a test to see output"}
            </span>
          </div>

          {testStatus === "completed" ? (
            <div className="mt-5 rounded-2xl bg-black p-5 font-mono text-sm text-white">
              <pre>{JSON.stringify(mockPrediction, null, 2)}</pre>
            </div>
          ) : (
            <div className="mt-5 rounded-2xl border border-dashed border-black/15 p-8 text-sm leading-6 text-black/50">
              Click <span className="font-medium text-black">Test pipe</span>{" "}
              to simulate an image classification run and generate mappable
              output data.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function SchemaCard({
  title,
  fields,
}: {
  title: string;
  fields: SchemaField[];
}) {
  return (
    <div className="rounded-3xl border border-black/10 bg-white/60 p-6 shadow-sm">
      <h2 className="text-lg font-semibold">{title}</h2>

      <div className="mt-6 space-y-4">
        {fields.map((field) => (
          <div key={field.name} className="border-b border-black/10 pb-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <span className="font-mono text-sm font-medium">
                {field.name}
              </span>

              <div className="flex items-center gap-2">
                {field.required ? (
                  <span className="rounded-full bg-black px-2 py-1 text-xs text-white">
                    required
                  </span>
                ) : (
                  <span className="rounded-full border border-black/10 px-2 py-1 text-xs text-black/40">
                    optional
                  </span>
                )}

                <span className="rounded-full border border-black/10 px-2 py-1 text-xs text-black/50">
                  {field.type}
                </span>
              </div>
            </div>

            <p className="mt-2 text-sm leading-6 text-black/50">
              {field.description}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

function UseCaseStep({
  number,
  title,
  description,
}: {
  number: string;
  title: string;
  description: string;
}) {
  return (
    <div className="flex gap-4">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-black text-sm font-medium text-white">
        {number}
      </div>

      <div>
        <h3 className="font-medium">{title}</h3>
        <p className="mt-1 text-sm leading-6 text-black/50">{description}</p>
      </div>
    </div>
  );
}