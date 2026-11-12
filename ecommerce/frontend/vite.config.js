import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server on 5173 (Streamlit uses 8501, the API uses 8000). /api is proxied to the FastAPI backend.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } } },
  build: { outDir: "dist", chunkSizeWarningLimit: 900 },
});
