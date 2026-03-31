type ImageConvertOutputMime = string;

type ImageConvertSettings = {
    outFmt: 'png' | 'jpeg' | 'webp';
    quality: number; // 5..100
    maxPreset: '' | 'custom' | string;
    maxCustom: number; // 64..16384
    bg: string; // #RRGGBB
    fillAlpha: boolean;
    smoothing: 'high' | 'medium' | 'low';
};

type ImageConvertConvertOpts = {
    outputMime: ImageConvertOutputMime;
    quality: number; // 0..1
    maxLongSide: number; // 0 => keep
    background: string; // #RRGGBB
    fillAlpha: boolean;
    smoothing: 'high' | 'medium' | 'low';
};

type ImageConvertLoadResult = { img: HTMLImageElement; objectUrl: string; file: File };

type ImageConvertCore = {
    MIME_JPEG: ImageConvertOutputMime;
    MIME_PNG: ImageConvertOutputMime;
    MIME_WEBP: ImageConvertOutputMime;
    supportsWebpOutput: () => boolean;
    extFromMime: (mime: ImageConvertOutputMime) => string;
    baseNameFromFile: (file: File) => string;
    loadImageFromFile: (file: File) => Promise<ImageConvertLoadResult>;
    convertImage: (img: HTMLImageElement, opts: ImageConvertConvertOpts) => Promise<Blob>;
    revokeObjectUrl: (url: string | null) => void;
};

type ImageBatchResultItem = { ok: boolean; file?: File; error?: unknown; blob?: Blob; name?: string };
type ImageBatchProcessOutput = { results: ImageBatchResultItem[]; aborted: boolean };
type ImageBatch = {
    recipeConvert: (opts: ImageConvertConvertOpts) => unknown;
    processFilesSequential: (
        ic: ImageConvertCore,
        files: File[],
        recipe: unknown,
        opts: { signal: AbortSignal; onItemStart?: (idx: number, file: File, total: number) => void }
    ) => Promise<ImageBatchProcessOutput>;
    downloadResultsSequential: (results: ImageBatchResultItem[], ic: ImageConvertCore, mime: ImageConvertOutputMime) => Promise<void>;
};

(function (): void {
    const IC = (window as unknown as { KarmoLabImageConvert?: ImageConvertCore }).KarmoLabImageConvert;
    if (!IC) {
        console.error('KarmoLabImageConvert missing');
        return;
    }
    const core: ImageConvertCore = IC;

    const STORAGE_KEY = 'karmolab_imageconvert_settings_v1';

    const DEFAULTS: ImageConvertSettings = {
        outFmt: 'png',
        quality: 92,
        maxPreset: '',
        maxCustom: 1920,
        bg: '#ffffff',
        fillAlpha: false,
        smoothing: 'high'
    };

    function loadSettings(): ImageConvertSettings {
        let o: Record<string, unknown> = {};
        try {
            o = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}') || {};
        } catch (_) {}
        return {
            outFmt: (o.outFmt === 'jpeg' || o.outFmt === 'webp' || o.outFmt === 'png') ? o.outFmt : DEFAULTS.outFmt,
            quality: Math.min(100, Math.max(5, parseInt(String(o.quality ?? ''), 10) || DEFAULTS.quality)),
            maxPreset: typeof o.maxPreset === 'string' ? o.maxPreset : DEFAULTS.maxPreset,
            maxCustom: Math.min(16384, Math.max(64, parseInt(String(o.maxCustom ?? ''), 10) || DEFAULTS.maxCustom)),
            bg: typeof o.bg === 'string' && /^#[0-9a-fA-F]{6}$/.test(o.bg) ? o.bg : DEFAULTS.bg,
            fillAlpha: !!o.fillAlpha,
            smoothing: (o.smoothing === 'low' || o.smoothing === 'medium') ? o.smoothing : DEFAULTS.smoothing
        };
    }

    function saveSettings(s: ImageConvertSettings): void {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
        } catch (_) {}
    }

    function mimeForFmt(fmt: ImageConvertSettings['outFmt']): ImageConvertOutputMime {
        if (fmt === 'jpeg') return core.MIME_JPEG;
        if (fmt === 'webp') return core.MIME_WEBP;
        return core.MIME_PNG;
    }

    function maxLongFromUI(preset: ImageConvertSettings['maxPreset'], customVal: number): number {
        if (preset === 'custom') return customVal > 0 ? customVal : 0;
        if (!preset) return 0;
        const n = parseInt(preset, 10);
        return n > 0 ? n : 0;
    }

    Mdd.injectCSS('imageconvert', `
        .imc-root {
            --imc-preview-slot-h: min(55vh, 560px);
            display:flex; flex-direction:column; gap:16px; max-width:min(100%, 1080px); margin:0 auto; padding:8px 0 28px; position:relative;
        }
        .imc-drop {
            border:2px dashed var(--border); border-radius:var(--radius-lg); padding:32px 22px;
            text-align:center; background:var(--bg-secondary); color:var(--text-secondary);
            cursor:pointer; transition:border-color var(--transition), background var(--transition);
        }
        .imc-drop:hover, .imc-drop.imc-drag { border-color:var(--accent); background:var(--accent-subtle); }
        .imc-drop-title { font-size:var(--font-size-sm); font-weight:600; color:var(--text-primary); margin-bottom:6px; }
        .imc-drop-hint { font-size:var(--font-size-xs); color:var(--text-tertiary); line-height:1.45; }
        .imc-panel {
            border:1px solid var(--border); border-radius:var(--radius-lg); padding:16px 18px;
            background:var(--bg-secondary); display:none; flex-direction:column; gap:14px;
        }
        .imc-panel.imc-visible { display:flex; }
        .imc-preview-row { display:grid; grid-template-columns:1fr 1fr; gap:14px; align-items:stretch; }
        .imc-preview-col { min-width:0; display:flex; flex-direction:column; }
        @media (max-width:640px) { .imc-preview-row { grid-template-columns:1fr; } }
        .imc-preview-caption {
            font-size:var(--font-size-2xs); font-weight:600; color:var(--text-tertiary);
            margin-bottom:6px; letter-spacing:0.02em;
        }
        .imc-preview-caption .imc-zoom-hint { font-weight:500; opacity:0.75; }
        .imc-preview-wrap {
            border-radius:var(--radius-md); overflow:hidden; background:var(--bg-tertiary);
            border:1px solid var(--border);
            height:var(--imc-preview-slot-h);
            width:100%; box-sizing:border-box;
            display:grid; grid-template:1fr / 1fr; place-items:stretch;
            background-image: linear-gradient(45deg, var(--bg-primary) 25%, transparent 25%),
                linear-gradient(-45deg, var(--bg-primary) 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, var(--bg-primary) 75%),
                linear-gradient(-45deg, transparent 75%, var(--bg-primary) 75%);
            background-size: 12px 12px;
            background-position: 0 0, 0 6px, 6px -6px, -6px 0px;
        }
        .imc-preview-wrap > * { grid-area:1 / 1; min-width:0; min-height:0; }
        .imc-preview-wrap img {
            box-sizing:border-box; display:block;
            width:100%; height:100%;
            object-fit:contain; cursor:zoom-in;
        }
        .imc-preview-wrap .imc-out-empty {
            cursor:default; box-sizing:border-box;
            display:flex; align-items:center; justify-content:center;
            margin:0;
        }
        .imc-out-empty {
            padding:20px 14px; text-align:center; font-size:var(--font-size-xs);
            color:var(--text-tertiary); line-height:1.5;
        }
        .imc-preview-out-img { display:none; }
        .imc-preview-out-img.imc-visible { display:block; }
        .imc-meta { font-size:var(--font-size-2xs); color:var(--text-tertiary); font-family:monospace; line-height:1.5; }
        .imc-section-title {
            font-size:var(--font-size-2xs); font-weight:700; text-transform:uppercase; letter-spacing:0.04em;
            color:var(--text-tertiary); margin:4px 0 2px;
        }
        .imc-grid { display:grid; grid-template-columns:100px 1fr; gap:10px 14px; align-items:center; }
        @media (max-width:560px) { .imc-grid { grid-template-columns:1fr; } }
        .imc-label { font-size:var(--font-size-xs); font-weight:600; color:var(--text-secondary); }
        .imc-format { display:flex; gap:8px; flex-wrap:wrap; }
        .imc-format label {
            display:inline-flex; align-items:center; gap:6px; font-size:var(--font-size-xs);
            cursor:pointer; color:var(--text-secondary); padding:6px 10px; border-radius:var(--radius-sm);
            border:1px solid var(--border); background:var(--bg-tertiary);
        }
        .imc-format input { accent-color:var(--accent); }
        .imc-format label.imc-fmt-off { opacity:0.4; pointer-events:none; }
        .imc-format label:has(input:checked) { border-color:var(--accent); color:var(--text-primary); background:var(--accent-subtle); }
        .imc-quality { display:flex; align-items:center; gap:10px; width:100%; max-width:320px; }
        .imc-quality input[type="range"] { flex:1; accent-color:var(--accent); }
        .imc-select, .imc-num {
            padding:6px 10px; font-size:var(--font-size-xs); border-radius:var(--radius-sm);
            border:1px solid var(--border); background:var(--bg-tertiary); color:var(--text-primary); font-family:inherit;
            max-width:100%;
        }
        .imc-num { width:100px; }
        .imc-color { width:44px; height:32px; padding:2px; border:1px solid var(--border); border-radius:var(--radius-sm); cursor:pointer; background:var(--bg-tertiary); }
        .imc-row-inline { display:flex; flex-wrap:wrap; align-items:center; gap:10px; }
        .imc-check { display:inline-flex; align-items:center; gap:8px; font-size:var(--font-size-xs); color:var(--text-secondary); cursor:pointer; }
        .imc-check input { accent-color:var(--accent); }
        .imc-actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:4px; }
        .imc-note { font-size:var(--font-size-2xs); color:var(--text-tertiary); line-height:1.55; margin:0; }
        .imc-resample-block.imc-off { display:none; }
        .imc-batch {
            margin-top:12px; padding-top:14px; border-top:1px solid var(--border);
            display:flex; flex-direction:column; gap:10px;
        }
        .imc-batch.imc-off { display:none; }
        .imc-batch-status { font-size:var(--font-size-2xs); color:var(--text-tertiary); line-height:1.45; margin:0; }
        .imc-embed.imc-root { max-width:100%; padding:4px 0 16px; }
        .imc-embed-bar {
            display:flex; flex-wrap:wrap; align-items:center; gap:10px 14px;
            padding:10px 12px; margin-bottom:10px; border-radius:var(--radius-md);
            border:1px solid var(--border); background:var(--bg-secondary);
            font-size:var(--font-size-xs); color:var(--text-secondary);
        }
        .imc-embed-bar-text { flex:1; min-width:140px; line-height:1.45; }
        .imc-lightbox {
            display:none; position:fixed; inset:0; z-index:100000;
            box-sizing:border-box; margin:0; padding:0;
            min-height:100vh; min-height:100dvh;
            background:rgba(0,0,0,0.92); backdrop-filter:blur(4px);
            cursor:pointer;
        }
        .imc-lightbox.imc-open {
            display:grid; grid-template:1fr / 1fr; place-items:stretch;
        }
        .imc-lightbox img {
            grid-area:1 / 1; box-sizing:border-box;
            width:100%; height:100%; min-width:0; min-height:0;
            object-fit:contain; object-position:center;
            cursor:pointer; user-select:none;
        }
    `);

    type BuildOptions = { embed?: boolean; onSyncCanvas?: () => void };

    const ImageConvertApp = {
        build: function (container: HTMLElement, opts?: BuildOptions) {
            const o = opts || {};
            const embed = !!o.embed;
            const webpOk = core.supportsWebpOutput();
            var dropBlock = embed
                ? '<div class="imc-embed-bar" id="imcEmbedBar">' +
                  '<span class="imc-embed-bar-text">편집 캔버스가 원본입니다. 캔버스를 수정한 뒤 여기에 반영하려면 버튼을 누르세요.</span>' +
                  '<button type="button" class="btn btn-ghost" id="imcSyncCanvas">캔버스 다시 불러오기</button>' +
                  '</div>'
                : '<div class="imc-drop" id="imcDrop" tabindex="0" role="button" aria-label="이미지 파일 선택">' +
                  '<div class="imc-drop-title">이미지를 여기에 놓거나 클릭해서 선택</div>' +
                  '<div class="imc-drop-hint">PNG · JPEG · WebP · GIF · BMP · 변환은 브라우저 안에서만 (업로드 없음)<br>GIF는 첫 프레임만 변환됩니다. 캔버스를 거치면서 EXIF 등 메타데이터는 제거됩니다.</div>' +
                  '</div>';

            container.innerHTML =
                '<div class="imc-root' + (embed ? ' imc-embed' : '') + '">' +
                dropBlock +
                '<input type="file" id="imcInput" accept="image/png,image/jpeg,image/webp,image/gif,image/bmp,.png,.jpg,.jpeg,.webp,.gif,.bmp" hidden>' +
                '<div class="imc-panel" id="imcPanel">' +
                '<div class="imc-preview-row">' +
                '<div class="imc-preview-col">' +
                '<div class="imc-preview-caption">원본 <span class="imc-zoom-hint">· 클릭하면 확대</span></div>' +
                '<div class="imc-preview-wrap" title="클릭하면 크게 봅니다"><img id="imcPreview" alt="원본 미리보기"></div>' +
                '<div class="imc-meta" id="imcMeta" style="margin-top:6px"></div>' +
                '</div>' +
                '<div class="imc-preview-col">' +
                '<div class="imc-preview-caption">변환 결과 <span class="imc-zoom-hint">· 클릭하면 확대</span></div>' +
                '<div class="imc-preview-wrap" title="미리보기 클릭 시 크게 봅니다">' +
                '<div class="imc-out-empty" id="imcOutEmpty">옵션을 맞춘 뒤 「변환 미리보기」를 누르면 여기에 표시됩니다.</div>' +
                '<img class="imc-preview-out-img" id="imcPreviewOut" alt="변환 미리보기">' +
                '</div>' +
                '<div class="imc-meta" id="imcOutMeta" style="margin-top:6px"></div>' +
                '</div></div>' +
                '<p class="imc-section-title">출력</p>' +
                '<div class="imc-grid">' +
                '<span class="imc-label">파일 형식</span>' +
                '<div class="imc-format">' +
                '<label><input type="radio" name="imcFmt" value="png"> PNG</label>' +
                '<label><input type="radio" name="imcFmt" value="jpeg"> JPEG</label>' +
                '<label class="' + (webpOk ? '' : 'imc-fmt-off') + '"><input type="radio" name="imcFmt" value="webp"' + (webpOk ? '' : ' disabled') + '> WebP</label>' +
                '</div>' +
                '<span class="imc-label" id="imcQlLabel">품질</span>' +
                '<div class="imc-quality"><input type="range" id="imcQuality" min="5" max="100" value="92"><span class="imc-meta" id="imcQualityVal">92%</span></div>' +
                '</div>' +
                '<p class="imc-section-title">크기</p>' +
                '<div class="imc-grid">' +
                '<span class="imc-label">긴 변 제한</span>' +
                '<div class="imc-row-inline">' +
                '<select class="imc-select" id="imcMaxPreset">' +
                '<option value="">원본 크기 유지</option>' +
                '<option value="8192">8192 px 이하</option>' +
                '<option value="4096">4096 px 이하</option>' +
                '<option value="2560">2560 px 이하</option>' +
                '<option value="1920">1920 px 이하</option>' +
                '<option value="1280">1280 px 이하</option>' +
                '<option value="1024">1024 px 이하</option>' +
                '<option value="800">800 px 이하</option>' +
                '<option value="640">640 px 이하</option>' +
                '<option value="512">512 px 이하</option>' +
                '<option value="384">384 px 이하</option>' +
                '<option value="256">256 px 이하</option>' +
                '<option value="custom">사용자 지정…</option>' +
                '</select>' +
                '<input type="number" class="imc-num" id="imcMaxCustom" min="64" max="16384" value="1920" title="긴 변 최대(px)" style="display:none">' +
                '</div></div>' +
                '<p class="imc-section-title">색 / 투명</p>' +
                '<div class="imc-grid">' +
                '<span class="imc-label">배경색</span>' +
                '<div class="imc-row-inline">' +
                '<input type="color" class="imc-color" id="imcBg" value="#ffffff">' +
                '<label class="imc-check"><input type="checkbox" id="imcFillAlpha"> JPEG 외 형식에서도 투명·반투명을 이 색으로 채우기</label>' +
                '</div></div>' +
                '<div class="imc-resample-block" id="imcResampleBlock">' +
                '<p class="imc-section-title">리샘플</p>' +
                '<div class="imc-grid">' +
                '<span class="imc-label">스케일 품질</span>' +
                '<select class="imc-select" id="imcSmooth">' +
                '<option value="high">고품질 · 부드럽게</option>' +
                '<option value="medium">보통</option>' +
                '<option value="low">빠름 · 픽셀/계단 느낌</option>' +
                '</select>' +
                '</div></div>' +
                '<p class="imc-note" style="margin-top:2px">PNG 압축 단계 같은 세부 값은 브라우저에서 조절할 수 없어요. 형식·해상도·품질로 맞추고, 긴 변을 줄일 때만 스케일(리샘플) 품질을 고를 수 있어요. 설정은 이 브라우저에 저장됩니다.</p>' +
                '<div class="imc-actions">' +
                '<button type="button" class="btn btn-ghost" id="imcPreviewBtn">변환 미리보기</button>' +
                '<button type="button" class="btn btn-primary" id="imcDownload">다운로드</button>' +
                '<button type="button" class="btn btn-ghost" id="imcClear">다른 파일</button>' +
                '</div>' +
                '<p class="imc-note" id="imcFootNote"></p>' +
                '<div class="imc-batch" id="imcBatchRoot">' +
                '<p class="imc-section-title">여러 파일</p>' +
                '<p class="imc-note" style="margin:0">위 출력·크기 옵션과 동일하게 연속 변환합니다. 브라우저에 따라 저장 창이 파일마다 뜰 수 있어요.</p>' +
                '<input type="file" id="imcBatchInput" accept="image/png,image/jpeg,image/webp,image/gif,image/bmp,.png,.jpg,.jpeg,.webp,.gif,.bmp" multiple hidden>' +
                '<div class="imc-actions" style="margin-top:0">' +
                '<button type="button" class="btn btn-ghost" id="imcBatchPick">파일 선택…</button>' +
                '<button type="button" class="btn btn-primary" id="imcBatchRun" disabled>일괄 변환 후 저장</button>' +
                '<button type="button" class="btn btn-ghost" id="imcBatchCancel" disabled style="display:none">취소</button>' +
                '</div>' +
                '<p class="imc-batch-status" id="imcBatchStatus">선택된 파일: 없음</p>' +
                '</div>' +
                '</div>' +
                '<div class="imc-lightbox" id="imcLightbox" tabindex="-1" aria-hidden="true" role="dialog" aria-label="이미지 확대 · 화면 아무 곳이나 눌러 닫기">' +
                '<img id="imcLightboxImg" alt="">' +
                '</div>' +
                '</div>';

            const drop = container.querySelector<HTMLElement>('#imcDrop');
            const input = container.querySelector<HTMLInputElement>('#imcInput')!;
            const panel = container.querySelector<HTMLElement>('#imcPanel')!;
            const preview = container.querySelector<HTMLImageElement>('#imcPreview')!;
            const meta = container.querySelector<HTMLElement>('#imcMeta')!;
            const quality = container.querySelector<HTMLInputElement>('#imcQuality')!;
            const qualityVal = container.querySelector<HTMLElement>('#imcQualityVal')!;
            const qlLabel = container.querySelector<HTMLElement>('#imcQlLabel')!;
            const maxPreset = container.querySelector<HTMLSelectElement>('#imcMaxPreset')!;
            const maxCustom = container.querySelector<HTMLInputElement>('#imcMaxCustom')!;
            const bgInput = container.querySelector<HTMLInputElement>('#imcBg')!;
            const fillAlpha = container.querySelector<HTMLInputElement>('#imcFillAlpha')!;
            const smoothSel = container.querySelector<HTMLSelectElement>('#imcSmooth')!;
            const resampleBlock = container.querySelector<HTMLElement>('#imcResampleBlock');
            const previewBtn = container.querySelector<HTMLButtonElement>('#imcPreviewBtn')!;
            const downloadBtn = container.querySelector<HTMLButtonElement>('#imcDownload')!;
            const clearBtn = container.querySelector<HTMLButtonElement>('#imcClear')!;
            const foot = container.querySelector<HTMLElement>('#imcFootNote')!;
            const outEmpty = container.querySelector<HTMLElement>('#imcOutEmpty')!;
            const previewOut = container.querySelector<HTMLImageElement>('#imcPreviewOut')!;
            const outMeta = container.querySelector<HTMLElement>('#imcOutMeta')!;
            const lightbox = container.querySelector<HTMLElement>('#imcLightbox')!;
            const lightboxImg = container.querySelector<HTMLImageElement>('#imcLightboxImg')!;

            let st = loadSettings();
            let current: { img: HTMLImageElement | null; objectUrl: string | null; file: File | null; baseName: string } = { img: null, objectUrl: null, file: null, baseName: 'image' };
            let outBlob: Blob | null = null;
            let outPreviewUrl: string | null = null;
            let lastPreviewKey: string | null = null;

            var EMPTY_OUT =
                '옵션을 맞춘 뒤 「변환 미리보기」를 누르면 여기에 표시됩니다.';
            var STALE_OUT = '설정이 바뀌었어요. 「변환 미리보기」를 다시 눌러 주세요.';

            function revokeOutPreview(): void {
                if (outPreviewUrl) {
                    try {
                        URL.revokeObjectURL(outPreviewUrl);
                    } catch (_) {}
                    outPreviewUrl = null;
                }
                outBlob = null;
                lastPreviewKey = null;
            }

            function closeLightbox(): void {
                if (!lightbox.classList.contains('imc-open')) return;
                lightbox.classList.remove('imc-open');
                lightbox.setAttribute('aria-hidden', 'true');
                lightboxImg.removeAttribute('src');
                lightboxImg.alt = '';
                document.removeEventListener('keydown', onLightboxEscape);
                document.body.style.overflow = '';
            }

            function onLightboxEscape(e: KeyboardEvent): void {
                if (e.key === 'Escape') closeLightbox();
            }

            function openLightbox(src: string, altText?: string): void {
                if (!src || !lightbox || !lightboxImg) return;
                lightboxImg.src = src;
                lightboxImg.alt = altText || '';
                lightbox.classList.add('imc-open');
                lightbox.setAttribute('aria-hidden', 'false');
                document.addEventListener('keydown', onLightboxEscape);
                document.body.style.overflow = 'hidden';
                try {
                    lightbox.focus();
                } catch (_) {}
            }

            function invalidateOutPreview(stale: boolean): void {
                closeLightbox();
                revokeOutPreview();
                outEmpty.textContent = stale ? STALE_OUT : EMPTY_OUT;
                outEmpty.style.display = 'block';
                previewOut.removeAttribute('src');
                previewOut.classList.remove('imc-visible');
                outMeta.textContent = '';
            }

            function settingsKey(): string {
                return JSON.stringify({
                    f: st.outFmt,
                    q: st.quality,
                    mp: st.maxPreset,
                    mc: st.maxCustom,
                    bg: st.bg,
                    fa: st.fillAlpha,
                    sm: st.smoothing
                });
            }

            function getConvertOptsFromSt(): ImageConvertConvertOpts {
                return {
                    outputMime: mimeForFmt(st.outFmt),
                    quality: st.quality / 100,
                    maxLongSide: maxLongFromUI(st.maxPreset, st.maxCustom),
                    background: st.bg,
                    fillAlpha: st.fillAlpha,
                    smoothing: st.smoothing
                };
            }

            function triggerDownloadBlob(blob: Blob, mime: ImageConvertOutputMime): void {
                var url = URL.createObjectURL(blob);
                var a = document.createElement('a');
                a.href = url;
                a.download = current.baseName + '.' + core.extFromMime(mime);
                a.click();
                setTimeout(function () {
                    URL.revokeObjectURL(url);
                }, 2000);
            }

            function onSettingsChanged(): void {
                persistFromForm();
                if (lastPreviewKey !== null) invalidateOutPreview(true);
            }

            function persistFromForm(): void {
                st.outFmt = (container.querySelector<HTMLInputElement>('input[name="imcFmt"]:checked')!.value as ImageConvertSettings['outFmt']);
                st.quality = parseInt(quality.value, 10);
                st.maxPreset = maxPreset.value;
                st.maxCustom = parseInt(maxCustom.value, 10) || DEFAULTS.maxCustom;
                st.bg = bgInput.value;
                st.fillAlpha = fillAlpha.checked;
                st.smoothing = (smoothSel.value as ImageConvertSettings['smoothing']);
                saveSettings(st);
            }

            function applySettingsToForm(): void {
                container.querySelectorAll<HTMLInputElement>('input[name="imcFmt"]').forEach(function (r) {
                    r.checked = r.value === st.outFmt;
                });
                if (st.outFmt === 'webp' && !webpOk) {
                    const png = container.querySelector<HTMLInputElement>('input[name="imcFmt"][value="png"]');
                    if (png) png.checked = true;
                    st.outFmt = 'png';
                    saveSettings(st);
                }
                quality.value = String(st.quality);
                qualityVal.textContent = st.quality + '%';
                maxPreset.value = st.maxPreset;
                maxCustom.value = String(st.maxCustom);
                bgInput.value = st.bg;
                fillAlpha.checked = st.fillAlpha;
                smoothSel.value = st.smoothing;
                syncMaxCustomVis();
                syncResampleVis();
                updateQualityRow();
            }

            function syncMaxCustomVis(): void {
                maxCustom.style.display = maxPreset.value === 'custom' ? 'inline-block' : 'none';
            }

            function syncResampleVis(): void {
                if (!resampleBlock) return;
                var resizing = maxPreset.value !== '';
                resampleBlock.classList.toggle('imc-off', !resizing);
            }

            function updateQualityRow(): void {
                const fmt = container.querySelector<HTMLInputElement>('input[name="imcFmt"]:checked')!.value as ImageConvertSettings['outFmt'];
                var lossy = fmt === 'jpeg' || fmt === 'webp';
                qlLabel.style.opacity = lossy ? '1' : '0.45';
                quality.disabled = !lossy;
                quality.style.opacity = lossy ? '1' : '0.45';
                qualityVal.style.opacity = lossy ? '1' : '0.45';
                var tail = ' 다운로드는 미리보기와 설정이 같으면 미리보기 결과를 그대로 저장해요.';
                if (fmt === 'jpeg') {
                    foot.textContent =
                        'JPEG는 알파 채널이 없습니다. 투명 영역은 위 배경색으로 채워집니다.' + tail;
                } else if (fmt === 'webp') {
                    foot.textContent =
                        'WebP는 투명을 유지할 수 있어요. 「투명을 배경색으로 채우기」를 켜면 불투명 WebP가 됩니다.' + tail;
                } else {
                    foot.textContent =
                        'PNG는 무손실입니다. 「투명을 배경색으로 채우기」를 켜면 알파가 제거된 PNG가 됩니다.' + tail;
                }
            }

            function showError(msg: string): void {
                Toolbox.showToast(msg, 'error', undefined);
                Mdd.linePreset('error', { msg: msg });
            }

            function revokeCurrent(): void {
                core.revokeObjectUrl(current.objectUrl);
                current = { img: null, objectUrl: null, file: null, baseName: 'image' };
            }

            function applyFile(file: File): void {
                core.loadImageFromFile(file).then(
                    function (res) {
                        revokeCurrent();
                        invalidateOutPreview(false);
                        current.img = res.img;
                        current.objectUrl = res.objectUrl;
                        current.file = res.file;
                        current.baseName = core.baseNameFromFile(res.file);
                        preview.src = res.objectUrl;
                        var w = res.img.naturalWidth;
                        var h = res.img.naturalHeight;
                        meta.innerHTML =
                            w +
                            ' × ' +
                            h +
                            ' px · ' +
                            (res.file.size / 1024).toFixed(1) +
                            ' KB · ' +
                            (res.file.type || 'unknown');
                        panel.classList.add('imc-visible');
                        Mdd.linePreset('success', { mood: 'happy', msg: '설정 맞추고 저장해요' });
                    },
                    function (err: unknown) {
                        const e = err as { message?: string } | null;
                        showError(e?.message || '오류');
                    }
                );
            }

            function pick(): void {
                input.click();
            }

            if (drop) {
                drop.addEventListener('click', pick);
                drop.addEventListener('keydown', function (e: KeyboardEvent) {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        pick();
                    }
                });
                (['dragenter', 'dragover'] as const).forEach(function (ev) {
                    drop.addEventListener(ev, function (e: Event) {
                        e.preventDefault();
                        e.stopPropagation();
                        drop.classList.add('imc-drag');
                    });
                });
                (['dragleave', 'drop'] as const).forEach(function (ev) {
                    drop.addEventListener(ev, function (e: Event) {
                        e.preventDefault();
                        e.stopPropagation();
                        drop.classList.remove('imc-drag');
                    });
                });
                drop.addEventListener('drop', function (e: DragEvent) {
                    const f = e.dataTransfer?.files?.[0];
                    if (f) applyFile(f);
                });
            }
            const syncBtn = container.querySelector<HTMLButtonElement>('#imcSyncCanvas');
            if (syncBtn) {
                if (embed && typeof o.onSyncCanvas === 'function') {
                    syncBtn.onclick = function () {
                        o.onSyncCanvas?.();
                    };
                } else {
                    syncBtn.style.display = 'none';
                }
            }
            input.addEventListener('change', function () {
                const f = input.files?.[0];
                if (f) applyFile(f);
                input.value = '';
            });

            preview.addEventListener('click', function () {
                if (preview.naturalWidth > 0 && preview.src) openLightbox(preview.src, '원본');
            });
            previewOut.addEventListener('click', function () {
                if (!previewOut.classList.contains('imc-visible')) return;
                if (previewOut.naturalWidth > 0 && previewOut.src) openLightbox(previewOut.src, '변환 결과');
            });
            lightbox.addEventListener('click', function () {
                closeLightbox();
            });

            container.querySelectorAll<HTMLInputElement>('input[name="imcFmt"]').forEach(function (r) {
                r.addEventListener('change', function () {
                    updateQualityRow();
                    onSettingsChanged();
                });
            });
            quality.addEventListener('input', function () {
                qualityVal.textContent = quality.value + '%';
            });
            quality.addEventListener('change', onSettingsChanged);
            maxPreset.addEventListener('change', function () {
                syncMaxCustomVis();
                syncResampleVis();
                onSettingsChanged();
            });
            maxCustom.addEventListener('change', onSettingsChanged);
            bgInput.addEventListener('change', onSettingsChanged);
            fillAlpha.addEventListener('change', onSettingsChanged);
            smoothSel.addEventListener('change', onSettingsChanged);

            previewBtn.addEventListener('click', function () {
                if (!current.img) return;
                persistFromForm();
                var key = settingsKey();
                var opts = getConvertOptsFromSt();
                previewBtn.disabled = true;
                core.convertImage(current.img, opts)
                    .then(function (blob) {
                        revokeOutPreview();
                        outBlob = blob;
                        outPreviewUrl = URL.createObjectURL(blob);
                        lastPreviewKey = key;
                        outEmpty.style.display = 'none';
                        previewOut.src = outPreviewUrl;
                        previewOut.classList.add('imc-visible');
                        var im = new Image();
                        im.onload = function () {
                            outMeta.textContent =
                                im.naturalWidth +
                                ' × ' +
                                im.naturalHeight +
                                ' px · 약 ' +
                                (blob.size / 1024).toFixed(1) +
                                ' KB';
                        };
                        im.src = outPreviewUrl;
                        Toolbox.showToast('미리보기를 갱신했어요', undefined, undefined);
                        Mdd.linePreset('success', { mood: 'happy', msg: '이렇게 나와요!' });
                    })
                    .catch(function () {
                        showError('변환에 실패했어요.');
                    })
                    .finally(function () {
                        previewBtn.disabled = false;
                    });
            });

            downloadBtn.addEventListener('click', function () {
                if (!current.img) return;
                persistFromForm();
                var key = settingsKey();
                var mime = mimeForFmt(st.outFmt);
                if (outBlob && lastPreviewKey === key) {
                    triggerDownloadBlob(outBlob, mime);
                    Toolbox.showToast('저장했어요', undefined, undefined);
                    Mdd.linePreset('success', { mood: 'happy', msg: '내려받기 완료!' });
                    return;
                }
                core.convertImage(current.img, getConvertOptsFromSt()).then(
                    function (blob) {
                        triggerDownloadBlob(blob, mime);
                        Toolbox.showToast('저장했어요', undefined, undefined);
                        Mdd.linePreset('success', { mood: 'happy', msg: '내려받기 완료!' });
                    },
                    function () {
                        showError('변환에 실패했어요.');
                    }
                );
            });

            clearBtn.addEventListener('click', function () {
                closeLightbox();
                preview.removeAttribute('src');
                panel.classList.remove('imc-visible');
                meta.textContent = '';
                invalidateOutPreview(false);
                revokeCurrent();
                Mdd.linePreset('tool_run', { mood: 'idle', msg: '다른 이미지를 골라요' });
            });

            const Batch = (window as unknown as { KarmoLabImageBatch?: ImageBatch }).KarmoLabImageBatch;
            const batchRoot = container.querySelector<HTMLElement>('#imcBatchRoot');
            const batchInput = container.querySelector<HTMLInputElement>('#imcBatchInput');
            const batchPick = container.querySelector<HTMLButtonElement>('#imcBatchPick');
            const batchRun = container.querySelector<HTMLButtonElement>('#imcBatchRun');
            const batchCancel = container.querySelector<HTMLButtonElement>('#imcBatchCancel');
            const batchStatus = container.querySelector<HTMLElement>('#imcBatchStatus');
            let batchFiles: File[] = [];
            let batchAbort: AbortController | null = null;

            function batchUiIdle(): void {
                batchAbort = null;
                if (batchRun) batchRun.disabled = batchFiles.length === 0;
                if (batchPick) batchPick.disabled = false;
                if (batchCancel) {
                    batchCancel.disabled = true;
                    batchCancel.style.display = 'none';
                }
                previewBtn.disabled = false;
                downloadBtn.disabled = false;
            }

            function batchStatusLine(): void {
                if (!batchStatus) return;
                batchStatus.textContent =
                    batchFiles.length === 0
                        ? '선택된 파일: 없음'
                        : '선택된 파일: ' + batchFiles.length + '개';
            }

            if (!Batch || !batchRoot || !batchInput || !batchPick || !batchRun || !batchCancel || !batchStatus) {
                if (batchRoot) batchRoot.classList.add('imc-off');
            } else {
                batchPick.addEventListener('click', function () {
                    batchInput.click();
                });
                batchInput.addEventListener('change', function () {
                    batchFiles = batchInput.files ? Array.from(batchInput.files) : [];
                    batchStatusLine();
                    batchRun.disabled = batchFiles.length === 0;
                    if (batchFiles.length) panel.classList.add('imc-visible');
                    batchInput.value = '';
                });
                batchCancel.addEventListener('click', function () {
                    if (batchAbort) batchAbort.abort();
                });
                batchRun.addEventListener('click', function () {
                    if (!batchFiles.length) return;
                    persistFromForm();
                    var mime = mimeForFmt(st.outFmt);
                    var recipe = Batch.recipeConvert(getConvertOptsFromSt());
                    batchAbort = new AbortController();
                    batchRun.disabled = true;
                    batchPick.disabled = true;
                    batchCancel.disabled = false;
                    batchCancel.style.display = 'inline-block';
                    previewBtn.disabled = true;
                    downloadBtn.disabled = true;
                    batchStatus.textContent = '변환 중… 0 / ' + batchFiles.length;
                    Batch.processFilesSequential(core, batchFiles, recipe, {
                        signal: batchAbort.signal,
                        onItemStart: function (idx, file, total) {
                            batchStatus.textContent =
                                '변환 중… ' + (idx + 1) + ' / ' + total + ' · ' + (file.name || '');
                        }
                    })
                        .then(function (out) {
                            var results = out.results;
                            var okc = results.filter(function (r) {
                                return r.ok;
                            }).length;
                            var failed = results.length - okc;
                            if (out.aborted) {
                                Toolbox.showToast('일괄 변환을 취소했어요.', undefined, undefined);
                                batchStatus.textContent =
                                    '취소됨 · 처리 ' + results.length + ' · 성공 ' + okc + ' · 실패 ' + failed;
                                return;
                            }
                            batchStatus.textContent =
                                '완료 · 성공 ' + okc + ' · 실패 ' + failed + (okc ? ' · 저장 창이 순서대로 열립니다' : '');
                            if (!okc) {
                                Toolbox.showToast('변환에 성공한 파일이 없어요.', undefined, undefined);
                                return;
                            }
                            return Batch.downloadResultsSequential(results, core, mime).then(function () {
                                Toolbox.showToast('일괄 저장 요청을 마쳤어요', undefined, undefined);
                                Mdd.linePreset('success', { mood: 'happy', msg: '모두 저장했어요!' });
                            });
                        })
                        .catch(function () {
                            showError('일괄 처리 중 오류가 났어요.');
                        })
                        .finally(function () {
                            batchUiIdle();
                            batchStatusLine();
                        });
                });
            }

            applySettingsToForm();
            Mdd.linePreset('tool_run', {
                mood: 'idle',
                msg: embed ? '형식·크기 맞춰서 내려받기' : '이미지 형식·크기·품질을 한곳에서',
            });

            return { applyFile: applyFile };
        }
    };

    (window as unknown as { KarmoLabImageConvertUI?: typeof ImageConvertApp }).KarmoLabImageConvertUI = ImageConvertApp;
})();
