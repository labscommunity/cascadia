import { Outlet, useLocation } from "react-router-dom";

import { Nav } from "./Nav";

export function Layout() {
  const { pathname } = useLocation();
  // The streams showcase is the one dark, full-bleed route. It sets the
  // attribute Tailwind's `dark:` variant keys on, pins the shell to the
  // viewport so the tile grid scrolls inside its own box instead of growing
  // the page, and drops the footer. Every other route renders as before.
  const showcase = pathname.startsWith("/streams");

  return (
    <div
      data-theme={showcase ? "dark" : undefined}
      className={showcase ? "h-screen overflow-hidden flex flex-col" : "min-h-screen flex flex-col"}
    >
      <Nav dark={showcase} />
      <main className={showcase ? "flex-1 min-h-0 flex flex-col" : "flex-1"}>
        <Outlet />
      </main>
      {showcase ? null : (
        <footer className="border-t border-rule-2 mt-16">
          <div className="max-w-container mx-auto px-6 py-6 flex items-center justify-between label-mono">
            <span>cascadia · cluster</span>
            <span className="text-ink-low/70">distributed LLM inference for Intel hardware</span>
          </div>
        </footer>
      )}
    </div>
  );
}
