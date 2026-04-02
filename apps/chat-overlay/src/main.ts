import { listen } from "@tauri-apps/api/event";
import { chatFeedKindFromEnv, createChatFeed } from "./chat/createChatFeed";
import type { ChatLine } from "./chat/types";
import { initThemeEditor } from "./themeEditor";
import { getCurrentWindow } from "@tauri-apps/api/window";

const MAX_LINES = 40;

function authorHue(author: string): number {
  let h = 0;
  for (let i = 0; i < author.length; i += 1) {
    h = (h * 31 + author.charCodeAt(i)) >>> 0;
  }
  return h % 360;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

function appendLine(container: HTMLElement, line: ChatLine): void {
  const row = el("div", "line line--enter");
  row.style.setProperty("--author-hue", String(authorHue(line.author)));
  const author = el("span", "author", line.author);
  const text = el("span", "text", line.text);
  row.appendChild(author);
  row.appendChild(text);
  container.appendChild(row);
  while (container.children.length > MAX_LINES) {
    container.removeChild(container.firstChild!);
  }
  requestAnimationFrame(() => {
    row.classList.add("line--visible");
  });
}

const log = document.getElementById("log");
if (!log) {
  throw new Error("#log not found");
}

initThemeEditor();

/** 짧은 시간에 같은 닉+본문이 연달아 오면 한 줄만 (확장 중복 전송 완화) */
let lastIngestDup: { key: string; at: number } | null = null;
const INGEST_DEDUP_MS = 120;

/** KarmoWebExtension 등 → Tauri `POST /ingest` → `extension-ingest` */
void listen<{ author: string; text: string; ts: number }>("extension-ingest", (event) => {
  const p = event.payload;
  const key = `${p.author}\u0000${p.text}`;
  const now = Date.now();
  if (
    lastIngestDup &&
    lastIngestDup.key === key &&
    now - lastIngestDup.at < INGEST_DEDUP_MS
  ) {
    return;
  }
  lastIngestDup = { key, at: now };
  appendLine(log, {
    id: `ext-${p.ts}-${Math.random().toString(36).slice(2, 10)}`,
    author: p.author,
    text: p.text,
    ts: p.ts
  });
}).catch(() => {
  /* Vite 미리보기 등 Tauri 없음 */
});

const moveHandle = document.querySelector<HTMLButtonElement>(".move-handle");
const resizeHandle = document.querySelector<HTMLButtonElement>(".resize-handle");

const appWindow = getCurrentWindow();

void listen<{ visible: boolean }>("layout-edit", (e) => {
  document.body.classList.toggle("layout-edit", e.payload.visible);
}).catch(() => {
  /* Tauri 밖 미리보기 */
});

if (moveHandle) {
  moveHandle.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    void appWindow.startDragging().catch((err) => {
      console.error("[chat-overlay] startDragging 실패:", err);
    });
  });
}

if (resizeHandle) {
  resizeHandle.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    void appWindow.startResizeDragging("SouthEast").catch((err) => {
      console.error("[chat-overlay] startResizeDragging 실패:", err);
    });
  });
}

const feed = createChatFeed(chatFeedKindFromEnv());
const unsub = feed.subscribe((line) => {
  appendLine(log, line);
});

window.addEventListener("beforeunload", () => {
  unsub();
  feed.destroy?.();
});
