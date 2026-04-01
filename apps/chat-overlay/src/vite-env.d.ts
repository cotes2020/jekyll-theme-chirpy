/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_CHAT_FEED?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
