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
        // Semantic state colors. The dashboard surfaces a richer set of node
        // states than the cascadia-website palette did (cold / warm / live /
        // error), so this is where we depart from the site palette.
        "state-ok": "#23A392",
        "state-warm": "#B54708",
        "state-cold": "#8A94A3",
        "state-error": "#D92D20",
      },
      fontFamily: {
        display: [
          '"Fraunces Variable"',
          '"Iowan Old Style"',
          '"Apple Garamond"',
          "Georgia",
          "serif",
        ],
        sans: [
          '"Inter Variable"',
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          "sans-serif",
        ],
        mono: [
          '"JetBrains Mono Variable"',
          "ui-monospace",
          '"SFMono-Regular"',
          "Menlo",
          "monospace",
        ],
      },
      fontSize: {
        display: ["clamp(36px, 5vw, 64px)", { lineHeight: "1.05", letterSpacing: "-0.02em" }],
        h2: ["clamp(28px, 4vw, 44px)", { lineHeight: "1.1", letterSpacing: "-0.015em" }],
        h3: ["clamp(20px, 2.4vw, 28px)", { lineHeight: "1.2", letterSpacing: "-0.01em" }],
        h4: ["clamp(16px, 1.6vw, 20px)", { lineHeight: "1.3" }],
      },
      boxShadow: {
        card: "0 1px 0 rgba(19, 35, 58, 0.03), 0 8px 24px -16px rgba(19, 35, 58, 0.10)",
        elev: "0 24px 56px -28px rgba(19, 35, 58, 0.18)",
      },
      maxWidth: {
        container: "1320px",
      },
      spacing: {
        gutter: "32px",
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
