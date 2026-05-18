import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The Rust API listens on whatever port `tahoma worker --api` was given;
// override via VITE_API_PROXY=http://host:port to point at a remote
// cluster. Default matches the demo's `tahoma worker --api 0.0.0.0:8000`.
const API_PROXY = process.env.VITE_API_PROXY ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: API_PROXY, changeOrigin: true },
      "/v1": { target: API_PROXY, changeOrigin: true },
      "/health": { target: API_PROXY, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2020",
    sourcemap: false,
  },
});
