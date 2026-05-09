export type ThemeGlobal = {
  DARK: string;
  LIGHT: string;
  ID: string;
  switchable: boolean;
  visualState: string;
  getThemeMapper(light: string, dark: string): Record<string, string>;
};

export type TocbotGlobal = {
  init(options: unknown): void;
  refresh(options: unknown): void;
};

export type MermaidGlobal = {
  initialize(config: unknown): void;
  init(config: unknown, selector: string): void;
};

// `object` (more specific than unknown) — GLightbox 라이브러리는 instance object 반환.
// null 과의 union 사용처 (img-popup.ts) 가 의미 있도록 unknown → object (KL-031 B1.3).
export type GLightboxInstance = object;
export type GLightboxGlobal = (options: { selector: string }) => GLightboxInstance;

export type DayjsGlobal = {
  locale(locale: string): void;
  extend(plugin: unknown): void;
  unix(ts: number): { format(formatStr?: string): string };
};

export type ClipboardJSEvent = { trigger: Element; clearSelection(): void };
export type ClipboardJSGlobal = new (
  selector: string,
  options: { target: (trigger: Element) => Element }
) => { on(event: 'success', callback: (e: ClipboardJSEvent) => void): void };

export function requiredGlobal<T>(name: string): T {
  const value = (window as unknown as Record<string, unknown>)[name];
  if (!value) {
    throw new Error(`Missing required global: ${name}`);
  }
  return value as T;
}
