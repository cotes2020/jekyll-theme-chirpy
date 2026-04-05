/**
 * KarmoLabAI — SSOT for Gemini-related model IDs and display catalog.
 * No browser, no fetch, no SDK: safe for KarmoLab (esbuild → browser) and Node (yawnbot, scripts).
 */

export type ModelProvider = 'gemini' | 'geminiImage' | 'imagen';

export interface ModelEntry {
  id: string;
  name: string;
  /** When true, used as UI/API default for that provider bucket */
  isDefault?: boolean;
}

/** Mirrors former `MODELS` in apps/karmolab/src/gemini.ts */
export const MODEL_CATALOG: Record<ModelProvider, ModelEntry[]> = {
  gemini: [
    { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash', isDefault: true },
    { id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro' },
    { id: 'gemini-2.0-flash', name: 'Gemini 2 Flash' },
    { id: 'gemini-2.0-flash-lite', name: 'Gemini 2 Flash Lite' },
    { id: 'gemini-2.5-flash-lite', name: 'Gemini 2.5 Flash Lite' },
    { id: 'gemini-3-flash-preview', name: 'Gemini 3 Flash' },
    { id: 'gemini-3.1-pro-preview', name: 'Gemini 3.1 Pro' },
    { id: 'gemini-3.1-flash-lite-preview', name: 'Gemini 3.1 Flash Lite' },
  ],
  geminiImage: [
    { id: 'gemini-2.5-flash-image', name: 'Nano Banana (Gemini 2.5 Flash Image)', isDefault: true },
    { id: 'gemini-3-pro-image-preview', name: 'Nano Banana Pro (Gemini 3 Pro Image)' },
    { id: 'gemini-3.1-flash-image-preview', name: 'Nano Banana 2 (Gemini 3.1 Flash Image)' },
  ],
  imagen: [
    { id: 'imagen-4.0-generate-001', name: 'Imagen 4 Generate', isDefault: true },
    { id: 'imagen-4.0-ultra-generate-001', name: 'Imagen 4 Ultra Generate' },
    { id: 'imagen-4.0-fast-generate-001', name: 'Imagen 4 Fast Generate' },
  ],
};

export function getDefaultModelId(provider: ModelProvider): string {
  const models = MODEL_CATALOG[provider];
  if (!models?.length) return '';
  const def = models.find((m) => m.isDefault);
  return def ? def.id : models[0].id;
}

/** Text Gemini when `GEMINI_MODEL` / UI does not override (matches catalog default). */
export const DEFAULT_GEMINI_TEXT_MODEL_ID = getDefaultModelId('gemini');
