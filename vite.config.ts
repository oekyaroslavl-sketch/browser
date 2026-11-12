import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  root: "public",
  base: "./",
  build: {
    outDir: "../src-tauri/gen/dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: path.resolve(__dirname, "public/app.html"),      // main browser shell
        newtab: path.resolve(__dirname, "public/index.html"),  // new-tab page
        blocked: path.resolve(__dirname, "public/blocked.html"), // block interstitial
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "public"),
    },
  },
  server: {
    port: 5173,
    host: "0.0.0.0",
  },
});
