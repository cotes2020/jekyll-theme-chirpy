/**
 * KarmoLab — 이미지 배치 파이프라인 (도구 공통)
 * 현재 지원: convert 단계 1개 (KarmoLabImageConvert.convertImage 와 동일 옵션)
 * 이후 단계 타입을 recipe.steps 에 추가해 확장.
 */
import type {
  ImageConvertOptions,
  KarmoLabImageBatchAPI,
  KarmoLabImageBatchHooks,
  KarmoLabImageBatchRecipe,
  KarmoLabImageConvertAPI
} from '../../../types/karmolab';

(function (global: Window) {
  const StepType = {
    CONVERT: 'convert'
  };

  function assertRecipe(recipe: KarmoLabImageBatchRecipe | null | undefined): void {
    if (!recipe || !Array.isArray(recipe.steps) || recipe.steps.length === 0) {
      throw new Error('KarmoLabImageBatch: recipe.steps 가 비어 있지 않아야 합니다.');
    }
  }

  function recipeConvert(opts: ImageConvertOptions): KarmoLabImageBatchRecipe {
    return { steps: [{ type: StepType.CONVERT, opts: opts }] };
  }

  function processFile(
    IC: KarmoLabImageConvertAPI,
    file: File,
    recipe: KarmoLabImageBatchRecipe,
    signal?: AbortSignal
  ): Promise<Blob> {
    assertRecipe(recipe);
    if (!recipe || recipe.steps.length !== 1) {
      return Promise.reject(new Error('KarmoLabImageBatch: 현재는 변환 단계 1개만 지원합니다.'));
    }
    const step = recipe.steps[0];
    if (step.type !== StepType.CONVERT || !step.opts) {
      return Promise.reject(new Error('KarmoLabImageBatch: 지원하지 않는 단계입니다.'));
    }
    if (signal && signal.aborted) {
      return Promise.reject(new DOMException('Aborted', 'AbortError'));
    }
    const convertOpts = step.opts;
    return IC.loadImageFromFile(file).then(function (res) {
      const url = res.objectUrl;
      if (signal && signal.aborted) {
        IC.revokeObjectUrl(url);
        return Promise.reject(new DOMException('Aborted', 'AbortError'));
      }
      return IC.convertImage(res.img, convertOpts).finally(function () {
        IC.revokeObjectUrl(url);
      });
    });
  }

  function processFilesSequential(
    IC: KarmoLabImageConvertAPI,
    files: File[],
    recipe: KarmoLabImageBatchRecipe,
    hooks?: KarmoLabImageBatchHooks
  ): Promise<{
    results: Array<{ ok: boolean; file: File; blob?: Blob; error?: unknown }>;
    aborted: boolean;
  }> {
    const h = hooks || {};
    const results: Array<{ ok: boolean; file: File; blob?: Blob; error?: unknown }> = [];
    const total = files.length;
    let i = 0;

    function next(): Promise<{
      results: Array<{ ok: boolean; file: File; blob?: Blob; error?: unknown }>;
      aborted: boolean;
    }> {
      if (h.signal && h.signal.aborted) {
        return Promise.resolve({ results: results, aborted: true });
      }
      if (i >= total) {
        return Promise.resolve({ results: results, aborted: false });
      }
      const file = files[i];
      const index = i;
      i += 1;
      if (h.onItemStart) h.onItemStart(index, file, total);
      return processFile(IC, file, recipe, h.signal).then(
        function (blob) {
          results.push({ ok: true, file: file, blob: blob });
          if (h.onItemDone) h.onItemDone(index, file, blob, total);
          return next();
        },
        function (err: Error) {
          results.push({ ok: false, file: file, error: err });
          if (h.onItemError) h.onItemError(index, file, err, total);
          return next();
        }
      );
    }
    return next();
  }

  function downloadResultsSequential(
    results: Array<{ ok: boolean; file: File; blob?: Blob }>,
    IC: KarmoLabImageConvertAPI,
    outputMime: string,
    delayMs?: number
  ): Promise<void> {
    const dm = delayMs == null ? 380 : delayMs;
    const ok = results.filter(function (r) {
      return r.ok && r.blob;
    });
    return ok.reduce(function (chain, r) {
      return chain.then(function () {
        return new Promise<void>(function (resolve) {
          const blob = r.blob!;
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = IC.baseNameFromFile(r.file) + '.' + IC.extFromMime(outputMime);
          a.click();
          setTimeout(function () {
            try {
              URL.revokeObjectURL(url);
            } catch {
              /* noop */
            }
            setTimeout(resolve, dm);
          }, 80);
        });
      });
    }, Promise.resolve());
  }

  const api: KarmoLabImageBatchAPI = {
    StepType: StepType,
    recipeConvert: recipeConvert,
    processFile: processFile,
    processFilesSequential: processFilesSequential,
    downloadResultsSequential: downloadResultsSequential
  };

  global.KarmoLabImageBatch = api;
})(typeof window !== 'undefined' ? window : (globalThis as unknown as Window));
