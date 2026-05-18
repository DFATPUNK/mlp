import { Link, Outlet, useLocation } from "react-router-dom";

export function AppShell() {
  const location = useLocation();

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
            NLP
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

          <Link
            to="/app/pipes/new"
            className="rounded-full bg-black px-4 py-2 text-sm font-medium text-white transition hover:bg-black/80"
          >
            Create a pipe
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-10">
        <Outlet />
      </main>
    </div>
  );
}