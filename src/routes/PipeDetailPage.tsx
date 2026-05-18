import { Link, useParams } from "react-router-dom";
import { mockPipes } from "../data/mockPipes";

const mockPrediction = {
  predicted_category: "cardboard",
  confidence: 0.91,
  alternative_categories: [
    { label: "paper", confidence: 0.06 },
    { label: "other_trash", confidence: 0.03 },
  ],
};

export function PipeDetailPage() {
  const { pipeId } = useParams();
  const pipe = mockPipes.find((item) => item.id === pipeId);

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

          <h1 className="mt-4 text-5xl font-semibold tracking-[-0.05em]">
            {pipe.name}
          </h1>

          <p className="mt-6 max-w-2xl text-base leading-7 text-black/60">
            {pipe.description}
          </p>

          <div className="mt-8 flex flex-wrap gap-3">
            <button className="rounded-full bg-black px-5 py-3 text-sm font-medium text-white transition hover:bg-black/80">
              Test pipe
            </button>

            <button className="rounded-full border border-black/10 px-5 py-3 text-sm font-medium transition hover:border-black/30">
              Copy endpoint
            </button>
          </div>
        </div>

        <aside className="rounded-3xl border border-black/10 bg-white/60 p-6 shadow-sm">
          <h2 className="text-lg font-semibold">Output schema</h2>

          <div className="mt-6 space-y-4 text-sm">
            <SchemaRow name="predicted_category" type="string" />
            <SchemaRow name="confidence" type="number" />
            <SchemaRow name="alternative_categories" type="array" />
          </div>
        </aside>
      </section>

      <section className="mt-10 grid gap-6 lg:grid-cols-2">
        <div className="rounded-3xl border border-black/10 bg-white/60 p-6">
          <h2 className="text-lg font-semibold">Categories</h2>

          <div className="mt-5 flex flex-wrap gap-2">
            {["Aluminium", "Cardboard", "Glass", "Biodegradable", "Other trash"].map(
              (category) => (
                <span
                  key={category}
                  className="rounded-full border border-black/10 px-3 py-1 text-sm text-black/60"
                >
                  {category}
                </span>
              ),
            )}
          </div>
        </div>

        <div className="rounded-3xl border border-black/10 bg-white/60 p-6">
          <h2 className="text-lg font-semibold">Latest test result</h2>

          <div className="mt-5 rounded-2xl bg-black p-5 font-mono text-sm text-white">
            <pre>{JSON.stringify(mockPrediction, null, 2)}</pre>
          </div>
        </div>
      </section>
    </div>
  );
}

function SchemaRow({ name, type }: { name: string; type: string }) {
  return (
    <div className="flex items-center justify-between border-b border-black/10 pb-3">
      <span className="font-medium">{name}</span>
      <span className="text-black/40">{type}</span>
    </div>
  );
}