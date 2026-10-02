import { Outlet } from "react-router-dom";

import { Nav } from "./Nav";

export function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <Nav />
      <main className="flex-1">
        <Outlet />
      </main>
      <footer className="border-t border-rule-2 mt-16">
        <div className="max-w-container mx-auto px-6 py-6 flex items-center justify-between label-mono">
          <span>cascadia · cluster</span>
          <span className="text-ink-low/70">distributed LLM inference for Intel hardware</span>
        </div>
      </footer>
    </div>
  );
}
