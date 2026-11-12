import { defineConfig, type Plugin } from "vite";
import path from "path";
import fs from "node:fs";

/**
 * Копирует статические файлы из public/ в каталог сборки.
 *
 * Причина: public/ одновременно является `root`, и когда задан явный
 * `build.rollupOptions.input`, Vite не считает его publicDir и не копирует
 * содержимое. В итоге в gen/dist не оказывалось filter-rules.json,
 * fetch('./filter-rules.json') возвращал 404, state.filterRulesJson
 * оставался undefined — и блокировка трекеров молча не работала.
 */
function copyStaticAssets(): Plugin {
  const files = ["filter-rules.json", "blocked.html", "error.html"];
  return {
    name: "difri-copy-static-assets",
    apply: "build",
    closeBundle() {
      const outDir = path.resolve(__dirname, "src-tauri/gen/dist");
      for (const file of files) {
        const from = path.resolve(__dirname, "public", file);
        const to = path.join(outDir, file);
        if (!fs.existsSync(from)) continue;
        fs.mkdirSync(path.dirname(to), { recursive: true });
        fs.copyFileSync(from, to);
      }
      // Иконки и svg нужны интерфейсу и в проде.
      for (const dir of ["svg"]) {
        const from = path.resolve(__dirname, "public", dir);
        if (!fs.existsSync(from)) continue;
        fs.cpSync(from, path.join(outDir, dir), { recursive: true });
      }
      for (const file of fs.readdirSync(path.resolve(__dirname, "public"))) {
        if (/\.(png|ico)$/i.test(file)) {
          fs.copyFileSync(
            path.resolve(__dirname, "public", file),
            path.join(outDir, file)
          );
        }
      }
    },
  };
}

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
  plugins: [copyStaticAssets()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "public"),
    },
  },
  server: {
    port: 5173,
    host: "0.0.0.0",
    allowedHosts: true,
  },
  preview: {
    port: 4173,
    host: "0.0.0.0",
    allowedHosts: true,
  },
});
