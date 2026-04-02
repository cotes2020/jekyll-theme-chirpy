import { defineConfig } from "vite";

const chzzkOpenApiProxy = {
  "/__chzzk_openapi": {
    target: "https://openapi.chzzk.naver.com",
    changeOrigin: true,
    secure: true,
    rewrite: (path: string) => path.replace(/^\/__chzzk_openapi/, "")
  }
};

export default defineConfig({
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    /**
     * 브라우저 → openapi.chzzk.naver.com 직접 호출은 CORS로 실패하는 경우가 많음.
     * 같은 출처로 프록시해 `fetch` / Session API가 동작하도록 함.
     */
    proxy: chzzkOpenApiProxy
  },
  preview: {
    port: 1420,
    strictPort: true,
    proxy: chzzkOpenApiProxy
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target:
      process.env.TAURI_ENV_PLATFORM === "windows" ? "chrome105" : "safari13",
    minify: process.env.TAURI_DEBUG ? false : "esbuild",
    sourcemap: !!process.env.TAURI_DEBUG
  }
});
