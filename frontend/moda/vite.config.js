import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/static/moda/",
  build: {
    outDir: "../../static/moda",
    emptyOutDir: true
  }
});
