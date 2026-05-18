import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";

type LoginMode = "sign-in" | "request-access";

export function LoginPage() {
  const navigate = useNavigate();
  const { signIn } = useAuth();

  const [mode, setMode] = useState<LoginMode>("sign-in");
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [message, setMessage] = useState("");
  const [password, setPassword] = useState("");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSignIn(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setErrorMessage(null);
    setStatusMessage(null);

    try {
      await signIn(email, password);
      navigate("/app/pipes");
    } catch {
      setErrorMessage("Unable to sign in. Check your email and password.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleRequestAccess(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setErrorMessage(null);
    setStatusMessage(null);

    const webhookUrl = import.meta.env
      .VITE_N8N_REQUEST_ACCESS_WEBHOOK_URL as string | undefined;

    if (!webhookUrl) {
      setSubmitting(false);
      setErrorMessage("Request access webhook is not configured.");
      return;
    }

    try {
      const response = await fetch(webhookUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email,
          name,
          message,
          source: "mlp-login-page",
          requested_at: new Date().toISOString(),
        }),
      });

      if (!response.ok) {
        throw new Error("Webhook request failed.");
      }

      setStatusMessage(
        "Request sent. If accepted, your account will be created manually.",
      );
      setName("");
      setMessage("");
    } catch {
      setErrorMessage("Unable to send request. Please try again later.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#f7f4ed] text-[#0b0b0b]">
      <div className="mx-auto grid min-h-screen max-w-7xl grid-cols-1 px-6 lg:grid-cols-[1.1fr_0.9fr]">
        <section className="flex flex-col justify-center py-16">
          <p className="mb-6 text-sm font-medium uppercase tracking-[0.2em] text-black/50">
            Invite-only MVP
          </p>

          <h1 className="max-w-3xl text-5xl font-semibold tracking-[-0.05em] md:text-7xl">
            Build machine learning pipes without code.
          </h1>

          <p className="mt-8 max-w-xl text-lg leading-8 text-black/60">
            Connect SaaS data, train small ML pipelines, test predictions, and
            use them as intelligent actions in your workflows.
          </p>
        </section>

        <section className="flex items-center justify-center py-16">
          <div className="w-full max-w-md rounded-3xl border border-black/10 bg-white/60 p-8 shadow-sm backdrop-blur">
            <div>
              <h2 className="text-2xl font-semibold tracking-tight">
                {mode === "sign-in" ? "Log in" : "Request access"}
              </h2>
              <p className="mt-2 text-sm text-black/50">
                {mode === "sign-in"
                  ? "Access is restricted to invited users."
                  : "Tell us who you are. We review access manually."}
              </p>
            </div>

            <div className="mt-6 grid grid-cols-2 rounded-full border border-black/10 bg-black/5 p-1">
              <button
                type="button"
                onClick={() => {
                  setMode("sign-in");
                  setErrorMessage(null);
                  setStatusMessage(null);
                }}
                className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                  mode === "sign-in"
                    ? "bg-black text-white"
                    : "text-black/50 hover:text-black"
                }`}
              >
                Sign in
              </button>

              <button
                type="button"
                onClick={() => {
                  setMode("request-access");
                  setErrorMessage(null);
                  setStatusMessage(null);
                }}
                className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                  mode === "request-access"
                    ? "bg-black text-white"
                    : "text-black/50 hover:text-black"
                }`}
              >
                Request access
              </button>
            </div>

            {mode === "sign-in" ? (
              <form onSubmit={handleSignIn} className="mt-8 space-y-5">
                <Field
                  label="Email"
                  type="email"
                  value={email}
                  onChange={setEmail}
                  required
                />

                <Field
                  label="Password"
                  type="password"
                  value={password}
                  onChange={setPassword}
                  required
                />

                <SubmitButton submitting={submitting}>
                  Log in
                </SubmitButton>
              </form>
            ) : (
              <form onSubmit={handleRequestAccess} className="mt-8 space-y-5">
                <Field
                  label="Email"
                  type="email"
                  value={email}
                  onChange={setEmail}
                  required
                />

                <Field
                  label="Name"
                  type="text"
                  value={name}
                  onChange={setName}
                  required
                />

                <label className="block">
                  <span className="text-sm font-medium">Message</span>
                  <textarea
                    className="mt-2 min-h-28 w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm outline-none transition focus:border-black"
                    value={message}
                    onChange={(event) => setMessage(event.target.value)}
                    placeholder="Why do you want to test MLP?"
                  />
                </label>

                <SubmitButton submitting={submitting}>
                  Request access
                </SubmitButton>
              </form>
            )}

            {statusMessage ? (
              <p className="mt-5 rounded-2xl bg-emerald-500/10 px-4 py-3 text-sm text-emerald-700">
                {statusMessage}
              </p>
            ) : null}

            {errorMessage ? (
              <p className="mt-5 rounded-2xl bg-red-500/10 px-4 py-3 text-sm text-red-700">
                {errorMessage}
              </p>
            ) : null}
          </div>
        </section>
      </div>
    </main>
  );
}

function Field({
  label,
  type,
  value,
  onChange,
  required,
}: {
  label: string;
  type: string;
  value: string;
  onChange: (value: string) => void;
  required?: boolean;
}) {
  return (
    <label className="block">
      <span className="text-sm font-medium">{label}</span>
      <input
        className="mt-2 w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm outline-none transition focus:border-black"
        type={type}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        required={required}
      />
    </label>
  );
}

function SubmitButton({
  children,
  submitting,
}: {
  children: React.ReactNode;
  submitting: boolean;
}) {
  return (
    <button
      type="submit"
      disabled={submitting}
      className="w-full rounded-full bg-black px-5 py-3 text-sm font-medium text-white transition hover:bg-black/80 disabled:cursor-not-allowed disabled:bg-black/40"
    >
      {submitting ? "Please wait…" : children}
    </button>
  );
}