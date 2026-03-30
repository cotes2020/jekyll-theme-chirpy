/** Shared types for KarmoLab TS sources (imported by src/; erased at build) */

export interface ImageConvertOptions {
  outputMime: string;
  quality?: number;
  maxLongSide?: number;
  background?: string;
  fillAlpha?: boolean;
  smoothing?: 'low' | 'medium' | 'high';
}

export interface KarmoLabImageConvertAPI {
  MIME_PNG: string;
  MIME_JPEG: string;
  MIME_WEBP: string;
  extFromMime: (mime: string) => string;
  isRasterImageFile: (file: File) => boolean;
  isSvgFile: (file: File) => boolean;
  supportsWebpOutput: () => boolean;
  computeDimensions: (nw: number, nh: number, maxLong: number) => { w: number; h: number };
  imageToCanvas: (img: HTMLImageElement, opts: ImageConvertOptions) => HTMLCanvasElement | null;
  canvasToBlob: (canvas: HTMLCanvasElement, mime: string, quality?: number) => Promise<Blob>;
  convertImage: (img: HTMLImageElement, opts: ImageConvertOptions) => Promise<Blob>;
  loadImageFromFile: (file: File) => Promise<{
    img: HTMLImageElement;
    objectUrl: string;
    file: File;
  }>;
  baseNameFromFile: (file: File) => string;
  revokeObjectUrl: (u: string | undefined) => void;
}

export interface RandomGenTopic {
  id: string;
  label: string;
  group: string;
  items?: string[];
  generator?: () => string | { name: string; sub?: string };
}

export interface KarmoLabImageBatchRecipe {
  steps: Array<{ type: string; opts?: ImageConvertOptions }>;
}

export interface KarmoLabImageBatchHooks {
  signal?: AbortSignal;
  onItemStart?: (i: number, file: File, total: number) => void;
  onItemDone?: (i: number, file: File, blob: Blob, total: number) => void;
  onItemError?: (i: number, file: File, err: Error, total: number) => void;
}

export interface KarmoLabImageBatchAPI {
  StepType: { CONVERT: string };
  recipeConvert: (opts: ImageConvertOptions) => KarmoLabImageBatchRecipe;
  processFile: (
    IC: KarmoLabImageConvertAPI,
    file: File,
    recipe: KarmoLabImageBatchRecipe,
    signal?: AbortSignal
  ) => Promise<Blob>;
  processFilesSequential: (
    IC: KarmoLabImageConvertAPI,
    files: File[],
    recipe: KarmoLabImageBatchRecipe,
    hooks?: KarmoLabImageBatchHooks
  ) => Promise<{
    results: Array<{ ok: boolean; file: File; blob?: Blob; error?: unknown }>;
    aborted: boolean;
  }>;
  downloadResultsSequential: (
    results: Array<{ ok: boolean; file: File; blob?: Blob }>,
    IC: KarmoLabImageConvertAPI,
    outputMime: string,
    delayMs?: number
  ) => Promise<void>;
}
