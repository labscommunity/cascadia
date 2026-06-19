import { NavLink } from "react-router-dom";

export function Nav() {
  return (
    <header className="sticky top-0 z-10 bg-paper/85 backdrop-blur border-b border-rule">
      <div className="max-w-container mx-auto px-6 h-14 flex items-center gap-8">
        <NavLink
          to="/"
          aria-label="Cascadia"
          className="shrink-0 hover:opacity-80 transition-opacity"
        >
          <img src="/logo.svg" alt="Cascadia" width={140} height={30} className="h-[26px] w-auto" />
        </NavLink>
        <nav className="flex items-center gap-1">
          <NavItem to="/" end label="Cluster" />
          <NavItem to="/chat" label="Chat" />
        </nav>
        <div className="ml-auto flex items-center gap-2 label-mono">
          <span className="pulse-dot" aria-hidden />
          <span>live</span>
        </div>
      </div>
    </header>
  );
}

function NavItem({ to, label, end }: { to: string; label: string; end?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `px-3 py-1.5 rounded-sm label-mono transition-colors ${
          isActive ? "text-ink bg-mint-wash" : "hover:text-ink"
        }`
      }
    >
      {label}
    </NavLink>
  );
}
