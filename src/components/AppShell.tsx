import { Link, Outlet, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";

export function AppShell() {
  const location = useLocation();
  const navigate = useNavigate();
  const { signOut, user } = useAuth();

  async function handleSignOut() {
    await signOut();
    navigate("/login");
  }

  const navItems = [
    { label: "Pipes", href: "/app/pipes" },
    { label: "Templates", href: "/app/pipes" },
    { label: "Runs", href: "/app/pipes" },
  ];

  return (
    <div className="min-h-screen bg-[#f7f4ed] text-[#0b0b0b]">
      <header className="border-b border-black/10">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <Link to="/app/pipes" className="text-lg font-semibold tracking-tight">
            MLP
          </Link>

          <nav className="hidden items-center gap-8 text-sm text-black/60 md:flex">
            {navItems.map((item) => {
              const active = location.pathname === item.href;

              return (
                <Link
                  key={item.label}
                  to={item.href}
                  className={active ? "text-black" : "hover:text-black"}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <div className="flex items-center gap-3">
            <span className="hidden text-sm text-black/40 lg:inline">
              {user?.email}
            </span>

            <Link
              to="/app/pipes/new"
              className="rounded-full bg-black px-4 py-2 text-sm font-medium text-white transition hover:bg-black/80"
            >
              Create a pipe
            </Link>

            <button
              type="button"
              onClick={handleSignOut}
              className="rounded-full border border-black/10 px-4 py-2 text-sm font-medium transition hover:border-black/30"
            >
              Log out
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-10">
        <Outlet />
      </main>
    </div>
  );
}