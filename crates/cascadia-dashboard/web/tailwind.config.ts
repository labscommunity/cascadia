import type { Config } from "tailwindcss";

// Palette + typography ported from cascadia-website's design system. Light
// mode only — see the rationale in src/styles/globals.css. Tailwind tokens
// here are the source of truth; the CSS variables in globals.css mirror
// them only for places where utilities aren't enough (e.g. pseudo-elements,
// keyframes with rgb(... / alpha) syntax).
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#F9FAF7",
        "paper-2": "#F4F4F4",
        "mint-wash": "#ECF7F5",
        mint: "#B7EECF",
        "mint-bright": "#78F3C8",
        celadon: "#86CCC3",
        persian: "#23A392",
        pine: "#2A8378",
        brunswick: "#00463E",
        oxford: "#13233A",
        ink: "#13233A",
        "ink-dim": "#4A5568",
        "ink-low": "#8A94A3",
        rule: "#D9DCD4",
        "rule-2": "#E8EAE4",
        // Semantic state colors — the Cascadia status mapping (cascadia.to /
        // cascadia-agentic-workflows): ok = persian, warn = amber, alert =
        // red; cold = ink-low for unmeasured/idle.
        "state-ok": "#23A392",
        "state-warm": "#B54708",
        "state-cold": "#8A94A3",
        "state-error": "#B42318",
      },
      fontFamily: {
        // LT Cushion (the brand display face) is a paid font; Fraunces is the
        // open stand-in used here, matching cascadia.to's fallback intent.
        display: [
          '"Fraunces Variable"',
          '"Iowan Old Style"',
          '"Apple Garamond"',
          "Georgia",
          "serif",
        ],
        // Body face matches cascadia.to / cascadia-agentic-workflows: Lato.
        sans: ['"Lato"', "-apple-system", "BlinkMacSystemFont", '"Segoe UI"', "sans-serif"],
        mono: [
          '"JetBrains Mono Variable"',
          "ui-monospace",
          '"SFMono-Regular"',
          "Menlo",
          "monospace",
        ],
      },
      // Type scale lifted verbatim from cascadia.to's globals.css clamps.
      fontSize: {
        display: ["clamp(36px, 5vw, 64px)", { lineHeight: "1.05", letterSpacing: "-0.02em" }],
        h2: ["clamp(30px, 4vw, 48px)", { lineHeight: "1.1", letterSpacing: "-0.015em" }],
        h3: ["clamp(22px, 2.6vw, 32px)", { lineHeight: "1.2", letterSpacing: "-0.01em" }],
        h4: ["clamp(18px, 1.9vw, 24px)", { lineHeight: "1.3" }],
      },
      boxShadow: {
        card: "0 1px 0 rgba(19, 35, 58, 0.03), 0 8px 24px -16px rgba(19, 35, 58, 0.10)",
        elev: "0 24px 56px -28px rgba(19, 35, 58, 0.18)",
      },
      maxWidth: {
        container: "1280px",
      },
      spacing: {
        gutter: "40px",
      },
      borderRadius: {
        sm: "4px",
        md: "8px",
        lg: "12px",
      },
    },
  },
  plugins: [],
} satisfies Config;
