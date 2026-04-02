import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"]
  },
  /** 로컬 `.env` 가 vitest에 박히지 않도록(테스트는 `process.env` + stubEnv 사용) */
  define: {
    "import.meta.env.VITE_CHAT_FEED": JSON.stringify(""),
    "import.meta.env.VITE_CHZZK_ACCESS_TOKEN": JSON.stringify(""),
    "import.meta.env.VITE_CHZZK_SESSION": JSON.stringify("")
  }
});
