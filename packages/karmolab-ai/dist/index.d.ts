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
export declare const MODEL_CATALOG: Record<ModelProvider, ModelEntry[]>;
export declare function getDefaultModelId(provider: ModelProvider): string;
/** Text Gemini when `GEMINI_MODEL` / UI does not override (matches catalog default). */
export declare const DEFAULT_GEMINI_TEXT_MODEL_ID: string;
