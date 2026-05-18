import { useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";

export function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("demo@nlp.local");
  const [password, setPassword] = useState("password");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    localStorage.setItem("nlp_demo_auth", "true");
    navigate("/app/pipes");
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
          <form
            onSubmit={handleSubmit}
            className="w-full max-w-md rounded-3xl border border-black/10 bg-white/60 p-8 shadow-sm backdrop-blur"
          >
            <div>
              <h2 className="text-2xl font-semibold tracking-tight">Log in</h2>
              <p className="mt-2 text-sm text-black/50">
                Access is restricted to invited users.
              </p>
            </div>

            <div className="mt-8 space-y-5">
              <label className="block">
                <span className="text-sm font-medium">Email</span>
                <input
                  className="mt-2 w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm outline-none transition focus:border-black"
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                />
              </label>

              <label className="block">
                <span className="text-sm font-medium">Password</span>
                <input
                  className="mt-2 w-full rounded-2xl border border-black/10 bg-white px-4 py-3 text-sm outline-none transition focus:border-black"
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                />
              </label>
            </div>

            <button
              type="submit"
              className="mt-8 w-full rounded-full bg-black px-5 py-3 text-sm font-medium text-white transition hover:bg-black/80"
            >
              Log in
            </button>

            <p className="mt-5 text-center text-xs text-black/40">
              Demo auth only. Supabase Auth comes in Sprint 2.
            </p>
          </form>
        </section>
      </div>
    </main>
  );
}