/**
 * KarmoLab — 이미지 배치 파이프라인 (도구 공통)
 * 현재 지원: convert 단계 1개 (KarmoLabImageConvert.convertImage 와 동일 옵션)
 * 이후 단계 타입을 recipe.steps 에 추가해 확장.
 */
(function (global) {
    'use strict';

    var StepType = {
        CONVERT: 'convert'
    };

    function assertRecipe(recipe) {
        if (!recipe || !Array.isArray(recipe.steps) || recipe.steps.length === 0) {
            throw new Error('KarmoLabImageBatch: recipe.steps 가 비어 있지 않아야 합니다.');
        }
    }

    /**
     * @param {typeof global.KarmoLabImageConvert} IC
     * @param {{ outputMime: string, quality?: number, maxLongSide?: number, background?: string, fillAlpha?: boolean, smoothing?: string }} opts
     */
    function recipeConvert(opts) {
        return { steps: [{ type: StepType.CONVERT, opts: opts }] };
    }

    /**
     * 단일 파일 · 단일 변환 단계 (확장 시 steps 순차 실행으로 일반화 가능)
     * @param {AbortSignal} [signal]
     */
    function processFile(IC, file, recipe, signal) {
        assertRecipe(recipe);
        if (recipe.steps.length !== 1) {
            return Promise.reject(new Error('KarmoLabImageBatch: 현재는 변환 단계 1개만 지원합니다.'));
        }
        var step = recipe.steps[0];
        if (step.type !== StepType.CONVERT || !step.opts) {
            return Promise.reject(new Error('KarmoLabImageBatch: 지원하지 않는 단계입니다.'));
        }
        if (signal && signal.aborted) {
            return Promise.reject(new DOMException('Aborted', 'AbortError'));
        }
        return IC.loadImageFromFile(file).then(function (res) {
            var url = res.objectUrl;
            if (signal && signal.aborted) {
                IC.revokeObjectUrl(url);
                return Promise.reject(new DOMException('Aborted', 'AbortError'));
            }
            return IC.convertImage(res.img, step.opts).finally(function () {
                IC.revokeObjectUrl(url);
            });
        });
    }

    /**
     * @param {File[]} files
     * @param {{ steps: Array<{ type: string, opts?: * }> }} recipe
     * @param {{ signal?: AbortSignal, onItemStart?: (i:number, file:File, total:number)=>void, onItemDone?: (i:number, file:File, blob:Blob, total:number)=>void, onItemError?: (i:number, file:File, err:Error, total:number)=>void }} [hooks]
     * @returns {Promise<{ results: Array<{ok:boolean,file:File,blob?:Blob,error?:*}>, aborted: boolean }>}
     */
    function processFilesSequential(IC, files, recipe, hooks) {
        hooks = hooks || {};
        var results = [];
        var total = files.length;
        var i = 0;

        function next() {
            if (hooks.signal && hooks.signal.aborted) {
                return Promise.resolve({ results: results, aborted: true });
            }
            if (i >= total) {
                return Promise.resolve({ results: results, aborted: false });
            }
            var file = files[i];
            var index = i;
            i += 1;
            if (hooks.onItemStart) hooks.onItemStart(index, file, total);
            return processFile(IC, file, recipe, hooks.signal).then(
                function (blob) {
                    results.push({ ok: true, file: file, blob: blob });
                    if (hooks.onItemDone) hooks.onItemDone(index, file, blob, total);
                    return next();
                },
                function (err) {
                    results.push({ ok: false, file: file, error: err, blob: null });
                    if (hooks.onItemError) hooks.onItemError(index, file, err, total);
                    return next();
                }
            );
        }
        return next();
    }

    /**
     * 성공 항목만 순차 다운로드 (브라우저 팝업 정책 완화용 간격)
     */
    function downloadResultsSequential(results, IC, outputMime, delayMs) {
        delayMs = delayMs == null ? 380 : delayMs;
        var ok = results.filter(function (r) {
            return r.ok && r.blob;
        });
        return ok.reduce(function (chain, r) {
            return chain.then(function () {
                return new Promise(function (resolve) {
                    var url = URL.createObjectURL(r.blob);
                    var a = document.createElement('a');
                    a.href = url;
                    a.download = IC.baseNameFromFile(r.file) + '.' + IC.extFromMime(outputMime);
                    a.click();
                    setTimeout(function () {
                        try {
                            URL.revokeObjectURL(url);
                        } catch (_) {}
                        setTimeout(resolve, delayMs);
                    }, 80);
                });
            });
        }, Promise.resolve());
    }

    global.KarmoLabImageBatch = {
        StepType: StepType,
        recipeConvert: recipeConvert,
        processFile: processFile,
        processFilesSequential: processFilesSequential,
        downloadResultsSequential: downloadResultsSequential
    };
})(typeof window !== 'undefined' ? window : this);
