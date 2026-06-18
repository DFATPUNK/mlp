import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { createDraftPipe } from "../lib/pipes";
import type { BuilderPipeType } from "../types/pipe";

function isBuilderPipeType(pipeType: string | undefined): pipeType is BuilderPipeType {
  return pipeType === "tabular_classification" || pipeType === "tabular_regression";
}

export function NewPipeRedirectPage() {
  const { user } = useAuth();
  const { pipeType } = useParams();
  const navigate = useNavigate();
  const creationStarted = useRef(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!user || creationStarted.current) return;

    if (!isBuilderPipeType(pipeType)) return;

    creationStarted.current = true;

    void createDraftPipe(user.id, pipeType)
      .then((pipe) => {
        navigate(`/app/pipes/${pipe.id}/builder`, { replace: true });
      })
      .catch(() => {
        creationStarted.current = false;
        setErrorMessage("We could not create your draft pipe. Please try again.");
      });
  }, [navigate, pipeType, user]);

  const visibleError = isBuilderPipeType(pipeType)
    ? errorMessage
    : "Choose a supported pipe type before starting the builder.";

  if (visibleError) {
    return (
      <section className="rounded-3xl border border-red-500/20 bg-red-500/10 p-6">
        <h1 className="text-lg font-semibold text-red-800">Unable to create pipe</h1>
        <p className="mt-2 text-sm text-red-700">{visibleError}</p>
        <Link to="/app/pipes/new" className="mt-4 inline-block text-sm font-medium text-red-800 underline">
          Back to pipe types
        </Link>
      </section>
    );
  }

  return <p className="text-sm text-black/50">Creating pipe…</p>;
}
