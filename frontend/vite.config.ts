import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxies /api to the FastAPI backend during dev, so the frontend never
// hardcodes localhost:8000 - one fewer thing to change between local dev
// and wherever this ends up deployed.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
