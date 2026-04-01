import type { ChatLine } from "../chat/types.js";

const MAX_LINES = 40;

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
  const row = el("div", "line");
  const author = el("span", "author", line.author);
  const text = el("span", "text", line.text);
  row.appendChild(author);
  row.appendChild(text);
  container.appendChild(row);
  while (container.children.length > MAX_LINES) {
    container.removeChild(container.firstChild!);
  }
}

const log = document.getElementById("log");
if (!log) {
  throw new Error("#log not found");
}

const unsub = window.chatOverlay.onChatLine((line) => {
  appendLine(log, line);
});

window.addEventListener("beforeunload", () => {
  unsub();
});
