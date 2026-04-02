/** 본문(body)에 붙는 채팅 표시 옵션 — 프리셋과 독립적으로 조합 가능 */

import type { PresetId } from "./presets";

export interface ChatAppearanceOptions {
  hideAuthor: boolean;
  lineDarkBg: boolean;
  textAlignCenter: boolean;
}

export const DEFAULT_CHAT_OPTIONS: ChatAppearanceOptions = {
  hideAuthor: false,
  lineDarkBg: false,
  textAlignCenter: false
};

const LEGACY_PRESET_CLASS = "chat-preset-dark-chat-only";

export function applyChatOptions(opts: ChatAppearanceOptions): void {
  document.body.classList.remove(LEGACY_PRESET_CLASS);
  document.body.classList.toggle("chat-opt-hide-author", opts.hideAuthor);
  document.body.classList.toggle("chat-opt-line-dark-bg", opts.lineDarkBg);
  document.body.classList.toggle("chat-opt-text-center", opts.textAlignCenter);
}

export function parseChatOptions(
  o: Record<string, unknown>,
  preset: PresetId
): ChatAppearanceOptions {
  const hasKeys =
    "hideAuthor" in o || "lineDarkBg" in o || "textAlignCenter" in o;
  if (!hasKeys && preset === "dark_chat_only") {
    return { hideAuthor: true, lineDarkBg: true, textAlignCenter: false };
  }
  return {
    hideAuthor: Boolean(o.hideAuthor),
    lineDarkBg: Boolean(o.lineDarkBg),
    textAlignCenter: Boolean(o.textAlignCenter)
  };
}
