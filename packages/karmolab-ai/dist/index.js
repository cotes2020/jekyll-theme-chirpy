"use strict";
/**
 * KarmoLabAI — SSOT for Gemini-related model IDs and display catalog.
 * No browser, no fetch, no SDK: safe for KarmoLab (esbuild → browser) and Node (yawnbot, scripts).
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.DEFAULT_GEMINI_TEXT_MODEL_ID = exports.MODEL_CATALOG = void 0;
exports.getDefaultModelId = getDefaultModelId;
/** Mirrors former `MODELS` in apps/karmolab/src/gemini.ts */
exports.MODEL_CATALOG = {
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
function getDefaultModelId(provider) {
    const models = exports.MODEL_CATALOG[provider];
    if (!models?.length)
        return '';
    const def = models.find((m) => m.isDefault);
    return def ? def.id : models[0].id;
}
/** Text Gemini when `GEMINI_MODEL` / UI does not override (matches catalog default). */
exports.DEFAULT_GEMINI_TEXT_MODEL_ID = getDefaultModelId('gemini');
