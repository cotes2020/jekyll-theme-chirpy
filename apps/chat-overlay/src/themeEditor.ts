import { getCurrentWindow } from "@tauri-apps/api/window";
import { listen } from "@tauri-apps/api/event";
import {
  applyChatOptions,
  DEFAULT_CHAT_OPTIONS,
  parseChatOptions,
  type ChatAppearanceOptions
} from "./chatOptions";
import { isPresetId, type PresetId } from "./presets";

const STORAGE_KEY = "chatOverlayTheme";

const SLUG_TO_VAR: Record<string, string> = {
  family: "--chat-font-family",
  size: "--chat-font-size",
  weight: "--chat-font-weight",
  lineHeight: "--chat-line-height",
  textColor: "--chat-text-color",
  authorSat: "--author-saturation",
  authorLight: "--author-lightness",
  lineGap: "--chat-line-gap",
  logPadding: "--chat-log-padding"
};

const DEFAULTS: Record<string, string> = {
  family: '"Malgun Gothic", "Segoe UI", system-ui, sans-serif',
  size: "14px",
  weight: "400",
  lineHeight: "1.35",
  textColor: "rgba(245, 245, 250, 0.95)",
  authorSat: "62%",
  authorLight: "72%",
  lineGap: "6px",
  logPadding: "12px 14px 16px"
};

const SLUGS = Object.keys(SLUG_TO_VAR);

/** 빈 문자열은 무시하고 DEFAULTS 유지 (저장·불러오기 깨짐 방지) */
function mergeThemeData(raw: Record<string, string>): Record<string, string> {
  const out = { ...DEFAULTS };
  for (const s of SLUGS) {
    const v = raw[s]?.trim();
    if (v) {
      out[s] = v;
    }
  }
  return out;
}

function slugToCssVars(data: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const s of SLUGS) {
    out[SLUG_TO_VAR[s]] = data[s] ?? DEFAULTS[s];
  }
  return out;
}

function applyCssVars(vars: Record<string, string>) {
  const root = document.documentElement;
  for (const [k, v] of Object.entries(vars)) {
    root.style.setProperty(k, v);
  }
}

function clearCssVars() {
  const root = document.documentElement;
  for (const v of Object.values(SLUG_TO_VAR)) {
    root.style.removeProperty(v);
  }
}

function parseStoredObject(raw: string): {
  preset: PresetId;
  slugRaw: Record<string, string>;
  chatOpts: ChatAppearanceOptions;
} {
  const o = JSON.parse(raw) as Record<string, unknown>;
  const preset =
    typeof o.preset === "string" && isPresetId(o.preset) ? o.preset : "default";
  const slugRaw: Record<string, string> = {};
  for (const s of SLUGS) {
    if (typeof o[s] === "string") {
      slugRaw[s] = o[s];
    }
  }
  const chatOpts = parseChatOptions(o, preset);
  return { preset, slugRaw, chatOpts };
}

export function applyStoredTheme(): void {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      applyChatOptions(DEFAULT_CHAT_OPTIONS);
      return;
    }
    const { slugRaw, chatOpts } = parseStoredObject(raw);
    applyChatOptions(chatOpts);
    applyCssVars(slugToCssVars(mergeThemeData(slugRaw)));
  } catch {
    applyChatOptions(DEFAULT_CHAT_OPTIONS);
  }
}

function getPresetFromUI(panel: HTMLElement): PresetId {
  const sel = panel.querySelector<HTMLSelectElement>("#theme-preset");
  if (sel && isPresetId(sel.value)) {
    return sel.value;
  }
  return "default";
}

function setPresetInUI(panel: HTMLElement, id: PresetId): void {
  const sel = panel.querySelector<HTMLSelectElement>("#theme-preset");
  if (sel) {
    sel.value = id;
  }
}

function getChatOptionsFromUI(panel: HTMLElement): ChatAppearanceOptions {
  const hide = panel.querySelector<HTMLInputElement>("#theme-opt-hide-author");
  const dark = panel.querySelector<HTMLInputElement>("#theme-opt-line-dark-bg");
  const center = panel.querySelector<HTMLInputElement>("#theme-opt-text-center");
  return {
    hideAuthor: Boolean(hide?.checked),
    lineDarkBg: Boolean(dark?.checked),
    textAlignCenter: Boolean(center?.checked)
  };
}

function setChatOptionsInUI(panel: HTMLElement, o: ChatAppearanceOptions): void {
  const hide = panel.querySelector<HTMLInputElement>("#theme-opt-hide-author");
  const dark = panel.querySelector<HTMLInputElement>("#theme-opt-line-dark-bg");
  const center = panel.querySelector<HTMLInputElement>("#theme-opt-text-center");
  if (hide) {
    hide.checked = o.hideAuthor;
  }
  if (dark) {
    dark.checked = o.lineDarkBg;
  }
  if (center) {
    center.checked = o.textAlignCenter;
  }
}

function readForm(panel: HTMLElement): Record<string, string> {
  const out: Record<string, string> = {};
  for (const s of SLUGS) {
    const el = panel.querySelector<HTMLInputElement | HTMLSelectElement>(
      `[data-theme-slug="${s}"]`
    );
    if (el && "value" in el) {
      out[s] = el.value.trim();
    }
  }
  return out;
}

function writeForm(panel: HTMLElement, data: Record<string, string>) {
  const merged = mergeThemeData(data);
  for (const s of SLUGS) {
    const el = panel.querySelector<HTMLInputElement | HTMLSelectElement>(
      `[data-theme-slug="${s}"]`
    );
    if (el) {
      el.value = merged[s];
    }
  }
  syncPreview(panel);
}

function syncPreview(panel: HTMLElement) {
  const wrap = panel.querySelector<HTMLElement>(".theme-editor__preview-inner");
  if (!wrap) {
    return;
  }
  const data = mergeThemeData(readForm(panel));
  const vars = slugToCssVars(data);
  for (const [k, v] of Object.entries(vars)) {
    wrap.style.setProperty(k, v);
  }
  wrap.querySelector<HTMLElement>(".author")?.style.setProperty("--author-hue", "210");

  const opts = getChatOptionsFromUI(panel);
  const preview = panel.querySelector(".theme-editor__preview");
  preview?.classList.toggle("theme-editor__preview--hide-author", opts.hideAuthor);
  preview?.classList.toggle("theme-editor__preview--line-dark-bg", opts.lineDarkBg);
  preview?.classList.toggle("theme-editor__preview--text-center", opts.textAlignCenter);
}

let panelEl: HTMLElement | null = null;

function openEditor() {
  if (!panelEl) {
    return;
  }
  panelEl.hidden = false;
  void getCurrentWindow()
    .setIgnoreCursorEvents(false)
    .catch(() => {});
}

function closeEditor() {
  if (!panelEl) {
    return;
  }
  panelEl.hidden = true;
}

function toggleEditor() {
  if (!panelEl) {
    return;
  }
  if (panelEl.hidden) {
    openEditor();
  } else {
    closeEditor();
  }
}

export function initThemeEditor(): void {
  panelEl = document.getElementById("theme-editor");
  if (!panelEl) {
    return;
  }

  applyStoredTheme();

  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const { preset, slugRaw, chatOpts } = parseStoredObject(raw);
      setPresetInUI(panelEl, preset);
      setChatOptionsInUI(panelEl, chatOpts);
      writeForm(panelEl, slugRaw);
    } else {
      setPresetInUI(panelEl, "default");
      setChatOptionsInUI(panelEl, DEFAULT_CHAT_OPTIONS);
      writeForm(panelEl, DEFAULTS);
    }
  } catch {
    setPresetInUI(panelEl, "default");
    setChatOptionsInUI(panelEl, DEFAULT_CHAT_OPTIONS);
    writeForm(panelEl, DEFAULTS);
  }

  panelEl.querySelector("#theme-save")?.addEventListener("click", () => {
    const preset = getPresetFromUI(panelEl!);
    const chatOpts = getChatOptionsFromUI(panelEl!);
    const data = mergeThemeData(readForm(panelEl!));
    applyChatOptions(chatOpts);
    applyCssVars(slugToCssVars(data));
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        preset,
        hideAuthor: chatOpts.hideAuthor,
        lineDarkBg: chatOpts.lineDarkBg,
        textAlignCenter: chatOpts.textAlignCenter,
        ...data
      })
    );
    closeEditor();
  });

  panelEl.querySelector("#theme-reset")?.addEventListener("click", () => {
    localStorage.removeItem(STORAGE_KEY);
    clearCssVars();
    applyChatOptions(DEFAULT_CHAT_OPTIONS);
    setPresetInUI(panelEl!, "default");
    setChatOptionsInUI(panelEl!, DEFAULT_CHAT_OPTIONS);
    writeForm(panelEl!, DEFAULTS);
  });

  panelEl.querySelector("#theme-close")?.addEventListener("click", () => {
    closeEditor();
  });

  const onFormTweak = (e: Event) => {
    const t = e.target as HTMLElement;
    if (t.id === "theme-preset" && isPresetId((t as HTMLSelectElement).value)) {
      if ((t as HTMLSelectElement).value === "dark_chat_only") {
        const cur = getChatOptionsFromUI(panelEl!);
        setChatOptionsInUI(panelEl!, { ...cur, hideAuthor: true, lineDarkBg: true });
      }
    }
    if (
      t.closest("[data-theme-slug]") ||
      t.id === "theme-preset" ||
      t.id === "theme-opt-hide-author" ||
      t.id === "theme-opt-line-dark-bg" ||
      t.id === "theme-opt-text-center"
    ) {
      syncPreview(panelEl!);
    }
  };
  panelEl.addEventListener("input", onFormTweak);
  panelEl.addEventListener("change", onFormTweak);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && panelEl && !panelEl.hidden) {
      e.preventDefault();
      closeEditor();
    }
    if (e.ctrlKey && e.shiftKey && e.key === ",") {
      e.preventDefault();
      toggleEditor();
    }
  });

  void listen("theme-editor-toggle", () => {
    toggleEditor();
  }).catch(() => {});

  panelEl.querySelector("#theme-editor-backdrop")?.addEventListener("click", () => {
    closeEditor();
  });
}
