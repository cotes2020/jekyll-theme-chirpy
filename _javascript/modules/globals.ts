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

export type GLightboxInstance = unknown;
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

function requiredGlobal<T>(name: string): T {
  const value = (window as unknown as Record<string, unknown>)[name];
  if (!value) {
    throw new Error(`Missing required global: ${name}`);
  }
  return value as T;
}

export const Theme = requiredGlobal<ThemeGlobal>('Theme');
export const tocbot = requiredGlobal<TocbotGlobal>('tocbot');
export const mermaid = requiredGlobal<MermaidGlobal>('mermaid');
export const GLightbox = requiredGlobal<GLightboxGlobal>('GLightbox');
export const dayjs = requiredGlobal<DayjsGlobal>('dayjs');
export const ClipboardJS = requiredGlobal<ClipboardJSGlobal>('ClipboardJS');

export function dayjsLocalizedFormatPlugin(): unknown {
  return (window as Window & { dayjs_plugin_localizedFormat?: unknown })
    .dayjs_plugin_localizedFormat;
}
