(function () {
    /* ===== CSS ===== */
    Mdd.injectCSS('imageedit', `
        .ie-layout { display:flex; flex-direction:column; height:100%; gap:0; }

        /* Toolbar */
        .ie-toolbar {
            display:flex; align-items:center; gap:6px; padding:10px 14px;
            border-bottom:1px solid var(--border); flex-wrap:wrap;
        }
        .ie-toolbar-group { display:flex; gap:4px; align-items:center; }
        .ie-toolbar-sep { width:1px; height:22px; background:var(--border); margin:0 4px; }
        .ie-tb-btn {
            padding:5px 10px; font-size:var(--font-size-xs); font-weight:500; border:1px solid var(--border);
            border-radius:var(--radius-sm); background:var(--bg-tertiary); color:var(--text-secondary);
            cursor:pointer; transition:all var(--transition); font-family:inherit; white-space:nowrap;
        }
        .ie-tb-btn:hover { color:var(--text-primary); border-color:var(--accent); }
        .ie-tb-btn:disabled { opacity:0.35; cursor:not-allowed; }
        .ie-tb-btn.accent { background:var(--accent); color:#fff; border-color:var(--accent); }
        .ie-tb-btn.accent:hover { background:var(--accent-hover); }
        .ie-toolbar .ie-spacer { flex:1; }
        .ie-size-label { font-size:var(--font-size-2xs); color:var(--text-tertiary); font-family:monospace; }

        /* Import dropdown */
        .ie-dropdown { position:relative; }
        .ie-dropdown-menu {
            display:none; position:absolute; top:100%; right:0; margin-top:4px; z-index:100;
            min-width:180px; background:var(--bg-secondary); border:1px solid var(--border);
            border-radius:var(--radius-md); box-shadow:0 8px 32px rgba(0,0,0,0.5); overflow:hidden;
        }
        .ie-dropdown-menu.open { display:block; }
        .ie-dropdown-item {
            display:block; width:100%; padding:9px 14px; font-size:var(--font-size-xs);
            border:none; background:none; color:var(--text-secondary);
            cursor:pointer; text-align:left; font-family:inherit; transition:all var(--transition);
        }
        .ie-dropdown-item:hover { background:var(--bg-hover); color:var(--text-primary); }

        /* Body (tool sidebar + canvas) */
        .ie-body { display:flex; flex:1; overflow:hidden; }

        /* Tool sidebar */
        .ie-tools {
            width:52px; flex-shrink:0; display:flex; flex-direction:column; gap:2px;
            padding:8px 6px; border-right:1px solid var(--border); background:var(--bg-secondary);
        }
        .ie-tool-btn {
            width:40px; height:40px; display:flex; align-items:center; justify-content:center;
            border:1px solid transparent; border-radius:var(--radius-sm); background:none;
            color:var(--text-tertiary); cursor:pointer; transition:all var(--transition); font-size:16px;
        }
        .ie-tool-btn:hover { color:var(--text-primary); background:var(--bg-hover); }
        .ie-tool-btn.active { color:var(--accent); border-color:var(--accent); background:var(--accent-subtle); }
        .ie-tool-btn svg { width:18px; height:18px; }

        /* Canvas area */
        .ie-canvas-area {
            flex:1; display:flex; flex-direction:column; overflow:hidden;
        }
        .ie-canvas-wrap {
            flex:1; display:flex; align-items:center; justify-content:center;
            position:relative; overflow:hidden; background:var(--bg-primary);
            background-image: linear-gradient(45deg, var(--bg-tertiary) 25%, transparent 25%),
                              linear-gradient(-45deg, var(--bg-tertiary) 25%, transparent 25%),
                              linear-gradient(45deg, transparent 75%, var(--bg-tertiary) 75%),
                              linear-gradient(-45deg, transparent 75%, var(--bg-tertiary) 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
        }
        .ie-canvas-wrap canvas {
            max-width:95%; max-height:95%; object-fit:contain;
            border-radius:2px; box-shadow:0 2px 16px rgba(0,0,0,0.4);
        }
        .ie-placeholder {
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            gap:12px; color:var(--text-tertiary); text-align:center;
            position:absolute; inset:0; cursor:pointer; transition:background 0.2s;
        }
        .ie-placeholder:hover { background:rgba(139,124,246,0.05); }
        .ie-placeholder-icon { font-size:48px; opacity:0.25; }
        .ie-placeholder-text { font-size:var(--font-size-sm); }
        .ie-placeholder-sub { font-size:var(--font-size-xs); opacity:0.6; }

        /* Crop overlay */
        .ie-crop-overlay {
            position:absolute; pointer-events:none; display:none;
        }
        .ie-crop-overlay.active { display:block; pointer-events:auto; }
        .ie-crop-region {
            position:absolute; border:2px dashed var(--accent);
            box-shadow:0 0 0 9999px rgba(0,0,0,0.55); cursor:move;
        }
        .ie-crop-handle {
            position:absolute; width:10px; height:10px; background:var(--accent);
            border:1px solid #fff; border-radius:2px;
        }
        .ie-crop-handle.nw { top:-5px; left:-5px; cursor:nw-resize; }
        .ie-crop-handle.ne { top:-5px; right:-5px; cursor:ne-resize; }
        .ie-crop-handle.sw { bottom:-5px; left:-5px; cursor:sw-resize; }
        .ie-crop-handle.se { bottom:-5px; right:-5px; cursor:se-resize; }

        /* Options panel */
        .ie-options {
            min-height:56px; padding:10px 14px; border-top:1px solid var(--border);
            display:flex; align-items:center; gap:12px; flex-wrap:wrap;
            background:var(--bg-secondary);
        }
        .ie-opt-label { font-size:var(--font-size-xs); color:var(--text-secondary); font-weight:500; white-space:nowrap; }
        .ie-opt-input {
            width:72px; padding:4px 8px; font-size:var(--font-size-xs); font-family:monospace;
            border:1px solid var(--border); border-radius:var(--radius-sm);
            background:var(--bg-tertiary); color:var(--text-primary);
        }
        .ie-opt-range { width:120px; accent-color:var(--accent); }
        .ie-opt-range-val { font-size:var(--font-size-2xs); color:var(--text-tertiary); font-family:monospace; min-width:36px; }
        .ie-opt-check { accent-color:var(--accent); }
        .ie-opt-btn {
            padding:4px 12px; font-size:var(--font-size-xs); border:1px solid var(--border);
            border-radius:var(--radius-sm); background:var(--bg-tertiary); color:var(--text-secondary);
            cursor:pointer; font-family:inherit; transition:all var(--transition);
        }
        .ie-opt-btn:hover { border-color:var(--accent); color:var(--text-primary); }
        .ie-opt-btn.active { background:var(--accent); color:#fff; border-color:var(--accent); }
        .ie-apply-btn {
            padding:5px 16px; font-size:var(--font-size-xs); font-weight:600; border:none;
            border-radius:var(--radius-sm); background:var(--accent); color:#fff;
            cursor:pointer; font-family:inherit; transition:all var(--transition); margin-left:auto;
        }
        .ie-apply-btn:hover { background:var(--accent-hover); }

        /* Filter grid */
        .ie-filter-grid { display:flex; gap:6px; flex-wrap:wrap; }
        .ie-filter-card {
            width:68px; display:flex; flex-direction:column; align-items:center; gap:4px;
            padding:6px 4px; border-radius:var(--radius-sm); border:1px solid var(--border);
            background:var(--bg-tertiary); cursor:pointer; transition:all var(--transition);
        }
        .ie-filter-card:hover { border-color:var(--accent); }
        .ie-filter-card.active { border-color:var(--accent); background:var(--accent-subtle); }
        .ie-filter-thumb {
            width:48px; height:48px; border-radius:4px; object-fit:cover;
            background:var(--bg-primary);
        }
        .ie-filter-name { font-size:var(--font-size-2xs); color:var(--text-secondary); text-align:center; }

        /* Mask preview overlay */
        .ie-mask-preview-overlay {
            display:none; position:absolute; inset:0; z-index:50;
            flex-direction:column; align-items:center; justify-content:center;
            background:var(--bg-primary); padding:12px;
        }
        .ie-mask-preview-overlay.active { display:flex; }
        .ie-mask-preview-box {
            display:flex; flex-direction:column; align-items:center;
            max-width:95%; max-height:95%; box-shadow:0 2px 16px rgba(0,0,0,0.4);
            border-radius:8px; overflow:hidden; background:var(--bg-secondary);
        }
        .ie-mask-preview-imgwrap {
            position:relative; flex:1; min-height:0; display:flex; align-items:center; justify-content:center;
        }
        .ie-mask-preview-inner {
            position:relative; overflow:hidden;
        }
        .ie-mask-preview-layer {
            position:absolute; top:0; left:0; width:100%; height:100%;
        }
        .ie-mask-preview-layer canvas {
            position:absolute; top:0; left:0; display:block; pointer-events:none;
        }
        .ie-mask-preview-divider {
            position:absolute; top:0; bottom:0; width:4px; background:var(--accent);
            cursor:ew-resize; z-index:10; box-shadow:0 0 8px rgba(0,0,0,0.5);
            transform:translateX(-50%);
        }
        .ie-mask-preview-divider::before {
            content:''; position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
            width:24px; height:24px; background:var(--accent); border-radius:50%;
            border:2px solid #fff; box-shadow:0 1px 4px rgba(0,0,0,0.3);
        }
        .ie-mask-preview-sliderrow {
            display:flex; align-items:center; gap:10px; padding:10px 12px;
            width:100%; box-sizing:border-box; background:var(--bg-tertiary);
        }
        .ie-mask-preview-sliderrow span { font-size:11px; color:var(--text-secondary); white-space:nowrap; }
        .ie-mask-preview-sliderrow input[type="range"] { flex:1; min-width:80px; accent-color:var(--accent); }
        .ie-mask-preview-close {
            padding:4px 12px; border:none; border-radius:var(--radius-sm);
            background:var(--accent); color:#fff; cursor:pointer; font-size:12px;
        }

        /* URL dialog */
        .ie-url-dialog {
            display:none; position:fixed; inset:0; z-index:200;
            background:rgba(0,0,0,0.7); backdrop-filter:blur(3px);
            align-items:center; justify-content:center;
        }
        .ie-url-dialog.open { display:flex; }
        .ie-url-dialog-box {
            background:var(--bg-secondary); border:1px solid var(--border);
            border-radius:var(--radius-lg); padding:24px; width:400px; max-width:90vw;
        }
        .ie-url-dialog-title { font-size:14px; font-weight:600; margin-bottom:12px; }
        .ie-url-dialog-input {
            width:100%; padding:8px 12px; font-size:var(--font-size-xs); border:1px solid var(--border);
            border-radius:var(--radius-sm); background:var(--bg-tertiary); color:var(--text-primary);
            font-family:inherit;
        }
        .ie-url-dialog-actions { display:flex; gap:8px; margin-top:14px; justify-content:flex-end; }

        /* Library picker dialog */
        .ie-lib-dialog {
            display:none; position:fixed; inset:0; z-index:200;
            background:rgba(0,0,0,0.7); backdrop-filter:blur(3px);
            align-items:center; justify-content:center;
        }
        .ie-lib-dialog.open { display:flex; }
        .ie-lib-dialog-box {
            background:var(--bg-secondary); border:1px solid var(--border);
            border-radius:var(--radius-lg); padding:24px; width:560px; max-width:92vw; max-height:80vh;
            display:flex; flex-direction:column;
        }
        .ie-lib-dialog-title { font-size:14px; font-weight:600; margin-bottom:12px; }
        .ie-lib-grid {
            display:grid; grid-template-columns:repeat(auto-fill, minmax(100px,1fr));
            gap:8px; overflow-y:auto; flex:1; padding:2px;
        }
        .ie-lib-thumb-card {
            aspect-ratio:1; border-radius:var(--radius-sm); overflow:hidden;
            border:2px solid var(--border); cursor:pointer; transition:all var(--transition);
        }
        .ie-lib-thumb-card:hover { border-color:var(--accent); transform:translateY(-2px); }
        .ie-lib-thumb-card img { width:100%; height:100%; object-fit:cover; display:block; }
        .ie-lib-empty { text-align:center; color:var(--text-tertiary); padding:40px 20px; font-size:var(--font-size-sm); }

        /* Drop zone highlight */
        .ie-canvas-wrap.dragover::after {
            content:'이미지를 여기에 놓으세요'; position:absolute; inset:0;
            display:flex; align-items:center; justify-content:center;
            background:rgba(139,124,246,0.15); border:2px dashed var(--accent);
            border-radius:var(--radius-md); font-size:14px; color:var(--accent);
            font-weight:600; z-index:50; pointer-events:none;
        }

        /* Size warning dialog */
        .ie-warn-dialog {
            display:none; position:fixed; inset:0; z-index:200;
            background:rgba(0,0,0,0.7); backdrop-filter:blur(3px);
            align-items:center; justify-content:center;
        }
        .ie-warn-dialog.open { display:flex; }
        .ie-warn-dialog-box {
            background:var(--bg-secondary); border:1px solid var(--border);
            border-radius:var(--radius-lg); padding:24px; width:400px; max-width:90vw;
        }
        .ie-warn-dialog-title { font-size:14px; font-weight:600; margin-bottom:8px; }
        .ie-warn-dialog-text { font-size:var(--font-size-xs); color:var(--text-secondary); margin-bottom:14px; line-height:1.6; }
        .ie-warn-dialog-actions { display:flex; gap:8px; justify-content:flex-end; }

        /* Background removal overlay */
        .ie-rembg-overlay {
            position:absolute; inset:0; z-index:60;
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            background:rgba(0,0,0,0.6); backdrop-filter:blur(3px); gap:14px;
        }
        .ie-rembg-spinner {
            width:40px; height:40px; border:3px solid rgba(255,255,255,0.15);
            border-top-color:var(--accent); border-radius:50%;
            animation:ie-spin 0.8s linear infinite;
        }
        @keyframes ie-spin { to { transform:rotate(360deg); } }
        .ie-rembg-status { color:#fff; font-size:var(--font-size-sm); font-weight:500; text-align:center; line-height:1.5; }
        .ie-rembg-bar-wrap {
            width:220px; height:6px; background:rgba(255,255,255,0.15); border-radius:3px; overflow:hidden;
        }
        .ie-rembg-bar {
            height:100%; background:var(--accent); border-radius:3px; transition:width 0.3s;
            width:100%; transform-origin:left;
            animation:ie-pulse 1.5s ease-in-out infinite;
        }
        .ie-rembg-bar.determinate { animation:none; }
        @keyframes ie-pulse {
            0%,100% { transform:scaleX(0.15); opacity:0.5; }
            50% { transform:scaleX(1); opacity:1; }
        }
        .ie-rembg-note { font-size:var(--font-size-2xs)!important; color:var(--text-tertiary)!important; font-weight:400!important; }

        /* Rembg mode tabs */
        .ie-rembg-tabs { display:flex; flex-wrap:wrap; gap:2px; width:100%; overflow-x:auto; }
        .ie-rembg-tab {
            flex:0 1 auto; min-width:52px; padding:5px 6px; font-size:var(--font-size-xs); font-weight:500; text-align:center; white-space:nowrap;
            border:1px solid var(--border); border-radius:var(--radius-sm);
            background:var(--bg-tertiary); color:var(--text-secondary);
            cursor:pointer; transition:all var(--transition); font-family:inherit;
        }
        .ie-rembg-tab:hover { color:var(--text-primary); border-color:var(--accent); }
        .ie-rembg-tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }

        /* Brush overlay */
        .ie-brush-overlay {
            position:absolute; cursor:crosshair; touch-action:none; z-index:40;
        }
        .ie-caption-overlay {
            position:absolute; cursor:move; touch-action:none; z-index:35;
            pointer-events:auto;
        }
        .ie-sticker-overlay {
            position:absolute; cursor:move; touch-action:none; z-index:35;
            pointer-events:auto;
        }

        /* Chromakey */
        .ie-chroma-swatch {
            width:24px; height:24px; border-radius:4px; border:2px solid var(--border);
            display:inline-block; vertical-align:middle; background:#888;
        }

        @media (max-width:768px) {
            .ie-body { flex-direction:column; }
            .ie-tools { width:100%; flex-direction:row; overflow-x:auto; border-right:none; border-bottom:1px solid var(--border); padding:4px 6px; }
            .ie-tool-btn { width:36px; height:36px; flex-shrink:0; }
            .ie-options { flex-wrap:wrap; }
        }
    `);

    /* ===== Constants ===== */
    const MAX_HISTORY = 20;
    const MAX_PIXELS = 4096 * 4096;
    const WARN_PIXELS = 3000 * 3000;

    /* ===== State ===== */
    let canvas, ctx;
    let history = [], historyIdx = -1;
    let activeTool = 'crop';
    let cropState = null;
    let cropAspect = null;
    let adjustValues = { brightness: 100, contrast: 100, saturate: 100, hue: 0 };
    let jpegQuality = 0.92;
    let freeRotateDeg = 0;
    let hasPendingPreview = false;
    let rembgMode = 'chroma';
    let rembgModel = 'isnet_quint8';
    let rembgOutput = 'foreground';
    let rembgMaxSize = 1024;
    let rembgBusy = false;
    let chromaColor = null;
    let chromaTolerance = 30;
    let chromaFeather = 5;
    let brushMode = 'bg';
    let brushSize = 20;
    let brushDrawing = false;

    /* ===== Helpers ===== */
    function pushHistory() {
        if (!canvas || canvas.width === 0) return;
        historyIdx++;
        history.length = historyIdx;
        history.push({
            data: ctx.getImageData(0, 0, canvas.width, canvas.height),
            w: canvas.width,
            h: canvas.height
        });
        if (history.length > MAX_HISTORY) {
            history.shift();
            historyIdx--;
        }
        updateUndoRedoButtons();
    }

    function undo() {
        if (historyIdx <= 0) return;
        historyIdx--;
        restoreFromHistory();
    }

    function redo() {
        if (historyIdx >= history.length - 1) return;
        historyIdx++;
        restoreFromHistory();
    }

    function restoreFromHistory() {
        const snap = history[historyIdx];
        if (!snap) return;
        canvas.width = snap.w;
        canvas.height = snap.h;
        ctx.putImageData(snap.data, 0, 0);
        updateSizeLabel();
        updateUndoRedoButtons();
    }

    function updateUndoRedoButtons() {
        const undoBtn = document.getElementById('ieUndoBtn');
        const redoBtn = document.getElementById('ieRedoBtn');
        if (undoBtn) undoBtn.disabled = historyIdx <= 0;
        if (redoBtn) redoBtn.disabled = historyIdx >= history.length - 1;
    }

    function updateSizeLabel() {
        const el = document.getElementById('ieSizeLabel');
        if (el && canvas && canvas.width > 0) {
            const mp = ((canvas.width * canvas.height) / 1e6).toFixed(1);
            el.textContent = canvas.width + ' × ' + canvas.height + '  (' + mp + 'MP)';
        } else if (el) {
            el.textContent = '';
        }
    }

    function hasImage() {
        return canvas && canvas.width > 0 && canvas.height > 0 && history.length > 0;
    }

    function requireImage() {
        if (hasImage()) return true;
        Toolbox.showToast('먼저 이미지를 불러오세요.', 'error');
        return false;
    }

    function closeAllDropdowns() {
        document.querySelectorAll('.ie-dropdown-menu').forEach(m => m.classList.remove('open'));
    }

    /* ===== Image Loading ===== */
    function loadImageFromSrc(src) {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            const pixels = img.naturalWidth * img.naturalHeight;

            if (pixels > MAX_PIXELS) {
                showSizeWarning(img, true);
                return;
            }
            if (pixels > WARN_PIXELS) {
                showSizeWarning(img, false);
                return;
            }
            commitImageLoad(img);
        };
        img.onerror = () => {
            Toolbox.showToast('이미지를 불러올 수 없습니다.', 'error');
            Mdd.setMood('sad'); Mdd.say('이미지 로드 실패냥...');
        };
        img.src = src;
    }

    function commitImageLoad(img) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        history = [];
        historyIdx = -1;
        pushHistory();
        updateSizeLabel();
        hidePlaceholder();
        selectTool(activeTool);
        Mdd.setMood('happy'); Mdd.say('이미지 불러왔다냥!');
    }

    function commitImageLoadDownscaled(img, targetW, targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
        ctx.drawImage(img, 0, 0, targetW, targetH);
        history = [];
        historyIdx = -1;
        pushHistory();
        updateSizeLabel();
        hidePlaceholder();
        selectTool(activeTool);
        Mdd.setMood('happy'); Mdd.say('이미지를 축소해서 불러왔다냥!');
    }

    function showSizeWarning(img, forceDownscale) {
        const dialog = document.getElementById('ieWarnDialog');
        if (!dialog) { commitImageLoad(img); return; }

        const w = img.naturalWidth, h = img.naturalHeight;
        const mp = ((w * h) / 1e6).toFixed(1);
        const maxSide = Math.max(w, h);
        const targetMax = 2048;
        const scale = targetMax / maxSide;
        const dw = Math.round(w * scale), dh = Math.round(h * scale);

        document.getElementById('ieWarnText').innerHTML =
            '이미지 크기: <strong>' + w + ' × ' + h + '</strong> (' + mp + 'MP)<br>' +
            (forceDownscale
                ? '너무 큰 이미지입니다. 편집을 위해 <strong>' + dw + ' × ' + dh + '</strong>로 축소합니다.'
                : '큰 이미지는 메모리를 많이 사용합니다. 축소하시겠습니까?<br>축소 시: <strong>' + dw + ' × ' + dh + '</strong>');

        const btnOrig = document.getElementById('ieWarnOriginal');
        const btnDown = document.getElementById('ieWarnDownscale');
        const btnCancel = document.getElementById('ieWarnCancel');

        if (forceDownscale) {
            btnOrig.style.display = 'none';
        } else {
            btnOrig.style.display = '';
        }

        btnOrig.onclick = () => { dialog.classList.remove('open'); commitImageLoad(img); };
        btnDown.onclick = () => { dialog.classList.remove('open'); commitImageLoadDownscaled(img, dw, dh); };
        btnCancel.onclick = () => { dialog.classList.remove('open'); };
        dialog.onclick = (e) => { if (e.target === dialog) dialog.classList.remove('open'); };
        dialog.classList.add('open');
    }

    function loadImageFromFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            Toolbox.showToast('이미지 파일이 아닙니다.', 'error');
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => loadImageFromSrc(e.target.result);
        reader.readAsDataURL(file);
    }

    function hidePlaceholder() {
        const ph = document.getElementById('iePlaceholder');
        if (ph) ph.style.display = 'none';
    }

    /* ===== Tool Panels ===== */
    const CROP_RATIOS = [
        { label: '자유', value: null },
        { label: '1:1', value: 1 },
        { label: '4:3', value: 4 / 3 },
        { label: '3:4', value: 3 / 4 },
        { label: '16:9', value: 16 / 9 },
        { label: '9:16', value: 9 / 16 },
    ];

    function buildCropOptions(container) {
        container.innerHTML = '';

        CROP_RATIOS.forEach(r => {
            const btn = document.createElement('button');
            btn.className = 'ie-opt-btn' + (cropAspect === r.value ? ' active' : '');
            btn.textContent = r.label;
            btn.onclick = () => {
                cropAspect = r.value;
                container.querySelectorAll('.ie-opt-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                if (hasImage()) { destroyCrop(); initCrop(); }
            };
            container.appendChild(btn);
        });

        const sep = document.createElement('span');
        sep.className = 'ie-toolbar-sep';
        container.appendChild(sep);

        const info = document.createElement('span');
        info.className = 'ie-opt-label';
        info.textContent = '캔버스 위에서 드래그하여 영역을 선택하세요';
        info.id = 'ieCropInfo';
        container.appendChild(info);

        const applyBtn = document.createElement('button');
        applyBtn.className = 'ie-apply-btn';
        applyBtn.textContent = '자르기 적용';
        applyBtn.onclick = applyCrop;
        container.appendChild(applyBtn);
    }

    function buildResizeOptions(container) {
        container.innerHTML = `
            <span class="ie-opt-label">너비</span>
            <input type="number" class="ie-opt-input" id="ieResizeW" min="1">
            <span class="ie-opt-label">높이</span>
            <input type="number" class="ie-opt-input" id="ieResizeH" min="1">
            <label style="display:flex;align-items:center;gap:4px;cursor:pointer;">
                <input type="checkbox" class="ie-opt-check" id="ieResizeLock" checked>
                <span class="ie-opt-label">비율 유지</span>
            </label>
            <span class="ie-toolbar-sep"></span>
            <button class="ie-opt-btn" data-pct="50">50%</button>
            <button class="ie-opt-btn" data-pct="75">75%</button>
            <button class="ie-opt-btn" data-pct="150">150%</button>
            <button class="ie-opt-btn" data-pct="200">200%</button>
            <button class="ie-apply-btn" id="ieResizeApply">적용</button>`;

        requestAnimationFrame(() => {
            if (!hasImage()) return;
            const wInput = document.getElementById('ieResizeW');
            const hInput = document.getElementById('ieResizeH');
            const lockCb = document.getElementById('ieResizeLock');
            const aspect = canvas.width / canvas.height;
            wInput.value = canvas.width;
            hInput.value = canvas.height;

            wInput.oninput = () => {
                if (lockCb.checked) hInput.value = Math.round(parseInt(wInput.value) / aspect) || '';
            };
            hInput.oninput = () => {
                if (lockCb.checked) wInput.value = Math.round(parseInt(hInput.value) * aspect) || '';
            };

            container.querySelectorAll('[data-pct]').forEach(btn => {
                btn.onclick = () => {
                    const p = parseInt(btn.dataset.pct) / 100;
                    wInput.value = Math.round(canvas.width * p);
                    hInput.value = Math.round(canvas.height * p);
                };
            });

            document.getElementById('ieResizeApply').onclick = applyResize;
        });
    }

    function buildRotateOptions(container) {
        freeRotateDeg = 0;
        container.innerHTML = `
            <button class="ie-opt-btn" id="ieRot90cw">↻ 90°</button>
            <button class="ie-opt-btn" id="ieRot90ccw">↺ -90°</button>
            <button class="ie-opt-btn" id="ieRot180">⟳ 180°</button>
            <button class="ie-opt-btn" id="ieFlipH">↔ 수평</button>
            <button class="ie-opt-btn" id="ieFlipV">↕ 수직</button>
            <span class="ie-toolbar-sep"></span>
            <span class="ie-opt-label">자유 회전</span>
            <input type="range" class="ie-opt-range" id="ieRotRange" min="-180" max="180" value="0" style="width:140px;">
            <input type="number" class="ie-opt-input" id="ieRotDeg" value="0" min="-360" max="360" style="width:56px;">
            <span class="ie-opt-label">°</span>
            <button class="ie-apply-btn" id="ieRotApply">적용</button>`;

        requestAnimationFrame(() => {
            document.getElementById('ieRot90cw').onclick = () => applyRotate(90);
            document.getElementById('ieRot90ccw').onclick = () => applyRotate(-90);
            document.getElementById('ieRot180').onclick = () => applyRotate(180);
            document.getElementById('ieFlipH').onclick = () => applyFlip('h');
            document.getElementById('ieFlipV').onclick = () => applyFlip('v');

            const range = document.getElementById('ieRotRange');
            const degInput = document.getElementById('ieRotDeg');

            const syncPreview = (deg) => {
                freeRotateDeg = deg;
                if (canvas) {
                    canvas.style.transform = deg ? 'rotate(' + deg + 'deg)' : '';
                    hasPendingPreview = deg !== 0;
                }
            };

            range.oninput = () => {
                const v = parseInt(range.value);
                degInput.value = v;
                syncPreview(v);
            };
            degInput.oninput = () => {
                const v = parseInt(degInput.value) || 0;
                range.value = Math.max(-180, Math.min(180, v));
                syncPreview(v);
            };

            document.getElementById('ieRotApply').onclick = () => {
                if (!freeRotateDeg) return;
                canvas.style.transform = '';
                hasPendingPreview = false;
                applyFreeRotate(freeRotateDeg);
                freeRotateDeg = 0;
                range.value = 0;
                degInput.value = 0;
            };
        });
    }

    function buildAdjustOptions(container) {
        adjustValues = { brightness: 100, contrast: 100, saturate: 100, hue: 0 };
        const sliders = [
            { id: 'brightness', label: '밝기', min: 0, max: 200, val: 100, unit: '%' },
            { id: 'contrast', label: '대비', min: 0, max: 200, val: 100, unit: '%' },
            { id: 'saturate', label: '채도', min: 0, max: 200, val: 100, unit: '%' },
            { id: 'hue', label: '색조', min: -180, max: 180, val: 0, unit: '°' },
        ];
        container.innerHTML = sliders.map(s => `
            <span class="ie-opt-label">${s.label}</span>
            <input type="range" class="ie-opt-range" id="ieAdj_${s.id}" min="${s.min}" max="${s.max}" value="${s.val}">
            <span class="ie-opt-range-val" id="ieAdjVal_${s.id}">${s.val}${s.unit}</span>
        `).join('') + '<button class="ie-opt-btn" id="ieAdjReset">초기화</button><button class="ie-apply-btn" id="ieAdjApply">적용</button>';

        requestAnimationFrame(() => {
            sliders.forEach(s => {
                const range = document.getElementById('ieAdj_' + s.id);
                const valEl = document.getElementById('ieAdjVal_' + s.id);
                range.oninput = () => {
                    adjustValues[s.id] = parseInt(range.value);
                    valEl.textContent = range.value + s.unit;
                    previewAdjust();
                };
            });
            document.getElementById('ieAdjReset').onclick = () => {
                sliders.forEach(s => {
                    document.getElementById('ieAdj_' + s.id).value = s.val;
                    document.getElementById('ieAdjVal_' + s.id).textContent = s.val + s.unit;
                    adjustValues[s.id] = s.val;
                });
                previewAdjust();
            };
            document.getElementById('ieAdjApply').onclick = applyAdjust;
        });
    }

    const FILTERS = [
        { id: 'none', name: '원본', css: 'none' },
        { id: 'grayscale', name: '흑백', css: 'grayscale(100%)' },
        { id: 'sepia', name: '세피아', css: 'sepia(100%)' },
        { id: 'blur', name: '블러', css: 'blur(3px)' },
        { id: 'invert', name: '반전', css: 'invert(100%)' },
        { id: 'vintage', name: '빈티지', css: 'sepia(40%) contrast(90%) brightness(110%) saturate(80%)' },
        { id: 'cool', name: '쿨톤', css: 'saturate(80%) hue-rotate(180deg) brightness(105%)' },
        { id: 'warm', name: '따뜻한', css: 'sepia(20%) saturate(140%) brightness(105%)' },
    ];

    function buildFilterOptions(container) {
        container.innerHTML = '';
        const grid = document.createElement('div');
        grid.className = 'ie-filter-grid';

        FILTERS.forEach(f => {
            const card = document.createElement('div');
            card.className = 'ie-filter-card' + (f.id === 'none' ? ' active' : '');
            card.dataset.filterId = f.id;

            const thumb = document.createElement('canvas');
            thumb.className = 'ie-filter-thumb';
            thumb.width = 48;
            thumb.height = 48;
            if (hasImage()) {
                const tCtx = thumb.getContext('2d');
                tCtx.filter = f.css === 'none' ? '' : f.css;
                const scale = Math.min(48 / canvas.width, 48 / canvas.height);
                const w = canvas.width * scale, h = canvas.height * scale;
                tCtx.drawImage(canvas, (48 - w) / 2, (48 - h) / 2, w, h);
            }

            const name = document.createElement('div');
            name.className = 'ie-filter-name';
            name.textContent = f.name;

            card.appendChild(thumb);
            card.appendChild(name);
            card.onclick = () => {
                grid.querySelectorAll('.ie-filter-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                previewFilter(f.css);
            };
            grid.appendChild(card);
        });

        container.appendChild(grid);

        const applyBtn = document.createElement('button');
        applyBtn.className = 'ie-apply-btn';
        applyBtn.textContent = '필터 적용';
        applyBtn.onclick = applyFilter;
        container.appendChild(applyBtn);
    }

    /* ===== Operations ===== */
    function buildAdjustFilterStr() {
        const { brightness, contrast, saturate, hue } = adjustValues;
        let s = `brightness(${brightness}%) contrast(${contrast}%) saturate(${saturate}%)`;
        if (hue !== 0) s += ` hue-rotate(${hue}deg)`;
        return s;
    }

    function previewAdjust() {
        if (!canvas) return;
        canvas.style.filter = buildAdjustFilterStr();
        hasPendingPreview = true;
    }

    function applyAdjust() {
        if (!requireImage()) return;
        applyCanvasFilter(buildAdjustFilterStr());
        canvas.style.filter = '';
        hasPendingPreview = false;
        adjustValues = { brightness: 100, contrast: 100, saturate: 100, hue: 0 };
        Toolbox.showToast('조정 적용 완료');
    }

    function previewFilter(css) {
        if (!canvas) return;
        canvas.style.filter = css === 'none' ? '' : css;
        hasPendingPreview = css !== 'none';
    }

    function applyFilter() {
        if (!requireImage()) return;
        const activeCard = document.querySelector('.ie-filter-card.active');
        if (!activeCard) return;
        const fId = activeCard.dataset.filterId;
        const f = FILTERS.find(x => x.id === fId);
        if (!f || f.id === 'none') {
            canvas.style.filter = '';
            return;
        }
        applyCanvasFilter(f.css);
        canvas.style.filter = '';
        Toolbox.showToast('필터 적용 완료');
    }

    function applyCanvasFilter(filterStr) {
        const off = document.createElement('canvas');
        off.width = canvas.width;
        off.height = canvas.height;
        const offCtx = off.getContext('2d');
        offCtx.filter = filterStr;
        offCtx.drawImage(canvas, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(off, 0, 0);
        pushHistory();
    }

    function applyResize() {
        if (!requireImage()) return;
        const w = parseInt(document.getElementById('ieResizeW').value);
        const h = parseInt(document.getElementById('ieResizeH').value);
        if (!w || !h || w < 1 || h < 1) {
            Toolbox.showToast('유효한 크기를 입력하세요.', 'error');
            return;
        }
        const off = document.createElement('canvas');
        off.width = w;
        off.height = h;
        off.getContext('2d').drawImage(canvas, 0, 0, w, h);
        canvas.width = w;
        canvas.height = h;
        ctx.drawImage(off, 0, 0);
        pushHistory();
        updateSizeLabel();
        Toolbox.showToast('크기 조절 완료');
    }

    function applyRotate(deg) {
        if (!requireImage()) return;
        const radians = (deg * Math.PI) / 180;
        const absDeg = Math.abs(deg % 360);
        const swap = absDeg === 90 || absDeg === 270;
        const newW = swap ? canvas.height : canvas.width;
        const newH = swap ? canvas.width : canvas.height;
        const off = document.createElement('canvas');
        off.width = newW;
        off.height = newH;
        const offCtx = off.getContext('2d');
        offCtx.translate(newW / 2, newH / 2);
        offCtx.rotate(radians);
        offCtx.drawImage(canvas, -canvas.width / 2, -canvas.height / 2);
        canvas.width = newW;
        canvas.height = newH;
        ctx.drawImage(off, 0, 0);
        pushHistory();
        updateSizeLabel();
        Toolbox.showToast(deg + '° 회전 완료');
    }

    function applyFreeRotate(deg) {
        if (!requireImage()) return;
        const rad = (deg * Math.PI) / 180;
        const sin = Math.abs(Math.sin(rad)), cos = Math.abs(Math.cos(rad));
        const newW = Math.round(canvas.width * cos + canvas.height * sin);
        const newH = Math.round(canvas.width * sin + canvas.height * cos);
        const off = document.createElement('canvas');
        off.width = newW;
        off.height = newH;
        const offCtx = off.getContext('2d');
        offCtx.translate(newW / 2, newH / 2);
        offCtx.rotate(rad);
        offCtx.drawImage(canvas, -canvas.width / 2, -canvas.height / 2);
        canvas.width = newW;
        canvas.height = newH;
        ctx.drawImage(off, 0, 0);
        pushHistory();
        updateSizeLabel();
        Toolbox.showToast(deg + '° 회전 완료');
    }

    function applyFlip(dir) {
        if (!requireImage()) return;
        const off = document.createElement('canvas');
        off.width = canvas.width;
        off.height = canvas.height;
        const offCtx = off.getContext('2d');
        if (dir === 'h') {
            offCtx.translate(canvas.width, 0);
            offCtx.scale(-1, 1);
        } else {
            offCtx.translate(0, canvas.height);
            offCtx.scale(1, -1);
        }
        offCtx.drawImage(canvas, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(off, 0, 0);
        pushHistory();
        Toolbox.showToast(dir === 'h' ? '수평 뒤집기 완료' : '수직 뒤집기 완료');
    }

    /* ===== Background Removal — Common ===== */
    function destroyBrush() {
        const ov = document.getElementById('ieBrushOverlay');
        if (ov) { ov.style.display = 'none'; ov.onpointerdown = null; ov.onpointermove = null; ov.onpointerup = null; }
    }

    function buildRembgOptions(container) {
        destroyBrush();
        const modes = [
            { id: 'chroma', label: '🎨 크로마키' },
            { id: 'brush',  label: '🖌️ 브러시' },
            { id: 'ai',     label: '🤖 AI (ONNX)' },
            { id: 'gemini', label: '✨ Gemini' },
            { id: 'bgg',    label: '배경색' }
        ];
        container.innerHTML = `
            <div style="width:100%;display:flex;flex-direction:column;gap:8px;">
                <div class="ie-rembg-tabs">${modes.map(m =>
                    `<button class="ie-rembg-tab${rembgMode === m.id ? ' active' : ''}" data-rm="${m.id}">${m.label}</button>`
                ).join('')}</div>
                <div id="ieRembgBody" style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;"></div>
            </div>`;

        requestAnimationFrame(() => {
            container.querySelectorAll('.ie-rembg-tab').forEach(btn => {
                btn.onclick = () => {
                    rembgMode = btn.dataset.rm;
                    container.querySelectorAll('.ie-rembg-tab').forEach(b => b.classList.toggle('active', b.dataset.rm === rembgMode));
                    destroyBrush();
                    buildRembgBody();
                };
            });
            buildRembgBody();
        });
    }

    function buildRembgBody() {
        const body = document.getElementById('ieRembgBody');
        if (!body) return;
        switch (rembgMode) {
            case 'chroma': buildChromaBody(body); break;
            case 'brush':  buildBrushBody(body); break;
            case 'ai':     buildAiBody(body); break;
            case 'gemini': buildGeminiBody(body); break;
            case 'bgg':    buildBggBody(body); break;
        }
    }

    /* ===== Mode 1 — Chromakey ===== */
    function buildChromaBody(body) {
        body.innerHTML = `
            <span class="ie-opt-label">색상</span>
            <span class="ie-chroma-swatch" id="ieChromaSwatch"></span>
            <span class="ie-opt-label" id="ieChromaHex" style="font-family:monospace;min-width:60px;">(클릭으로 선택)</span>
            <span class="ie-toolbar-sep"></span>
            <span class="ie-opt-label">허용치</span>
            <input type="range" class="ie-opt-range" id="ieChromaTol" min="1" max="120" value="${chromaTolerance}" style="width:100px;">
            <span class="ie-opt-range-val" id="ieChromaTolVal">${chromaTolerance}</span>
            <span class="ie-toolbar-sep"></span>
            <span class="ie-opt-label">페더</span>
            <input type="range" class="ie-opt-range" id="ieChromaFeather" min="0" max="30" value="${chromaFeather}" style="width:80px;">
            <span class="ie-opt-range-val" id="ieChromaFeatherVal">${chromaFeather}</span>
            <button class="ie-apply-btn" id="ieChromaApply">적용</button>`;
        requestAnimationFrame(() => {
            const swatch = document.getElementById('ieChromaSwatch');
            if (chromaColor) swatch.style.backgroundColor = `rgb(${chromaColor.join(',')})`;

            document.getElementById('ieChromaTol').oninput = (e) => {
                chromaTolerance = parseInt(e.target.value);
                document.getElementById('ieChromaTolVal').textContent = chromaTolerance;
            };
            document.getElementById('ieChromaFeather').oninput = (e) => {
                chromaFeather = parseInt(e.target.value);
                document.getElementById('ieChromaFeatherVal').textContent = chromaFeather;
            };
            document.getElementById('ieChromaApply').onclick = applyChromakey;
        });
    }

    function onCanvasClickForChroma(e) {
        if (activeTool !== 'rembg' || rembgMode !== 'chroma' || !hasImage()) return;
        if (e.target !== canvas) return;
        const rect = canvas.getBoundingClientRect();
        const x = Math.round((e.clientX - rect.left) / rect.width * canvas.width);
        const y = Math.round((e.clientY - rect.top) / rect.height * canvas.height);
        const px = ctx.getImageData(Math.max(0, Math.min(x, canvas.width - 1)), Math.max(0, Math.min(y, canvas.height - 1)), 1, 1).data;
        chromaColor = [px[0], px[1], px[2]];
        const swatch = document.getElementById('ieChromaSwatch');
        if (swatch) swatch.style.backgroundColor = `rgb(${px[0]},${px[1]},${px[2]})`;
        const hexEl = document.getElementById('ieChromaHex');
        if (hexEl) hexEl.textContent = '#' + [px[0], px[1], px[2]].map(c => c.toString(16).padStart(2, '0')).join('');
    }

    function applyChromakey() {
        if (!requireImage()) return;
        if (!chromaColor) { Toolbox.showToast('먼저 이미지에서 제거할 색상을 클릭하세요.', 'error'); return; }
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const d = imgData.data;
        const [tr, tg, tb] = chromaColor;
        const tol = chromaTolerance, fth = Math.max(chromaFeather, 0.5);
        for (let i = 0; i < d.length; i += 4) {
            const dist = Math.sqrt((d[i] - tr) ** 2 + (d[i + 1] - tg) ** 2 + (d[i + 2] - tb) ** 2);
            if (dist < tol - fth) {
                d[i + 3] = 0;
            } else if (dist < tol + fth) {
                const alpha = Math.round(((dist - (tol - fth)) / (2 * fth)) * 255);
                d[i + 3] = Math.min(d[i + 3], alpha);
            }
        }
        ctx.putImageData(imgData, 0, 0);
        pushHistory();
        updateSizeLabel();
        Toolbox.showToast('크로마키 적용 완료');
        Mdd.setMood('happy'); Mdd.say('배경 날렸다냥!');
    }

    /* ===== Mode 2 — Brush ===== */
    function buildBrushBody(body) {
        body.innerHTML = `
            <span class="ie-opt-label">모드</span>
            <button class="ie-opt-btn${brushMode === 'bg' ? ' active' : ''}" id="ieBrushBg">🔴 배경 (제거)</button>
            <button class="ie-opt-btn${brushMode === 'fg' ? ' active' : ''}" id="ieBrushFg">🟢 전경 (유지)</button>
            <span class="ie-toolbar-sep"></span>
            <span class="ie-opt-label">크기</span>
            <input type="range" class="ie-opt-range" id="ieBrushSize" min="4" max="80" value="${brushSize}" style="width:100px;">
            <span class="ie-opt-range-val" id="ieBrushSizeVal">${brushSize}px</span>
            <button class="ie-opt-btn" id="ieBrushClear">지우기</button>
            <button class="ie-apply-btn" id="ieBrushApply">적용</button>`;
        requestAnimationFrame(() => {
            const bgBtn = document.getElementById('ieBrushBg');
            const fgBtn = document.getElementById('ieBrushFg');
            bgBtn.onclick = () => { brushMode = 'bg'; bgBtn.classList.add('active'); fgBtn.classList.remove('active'); };
            fgBtn.onclick = () => { brushMode = 'fg'; fgBtn.classList.add('active'); bgBtn.classList.remove('active'); };
            document.getElementById('ieBrushSize').oninput = (e) => {
                brushSize = parseInt(e.target.value);
                document.getElementById('ieBrushSizeVal').textContent = brushSize + 'px';
            };
            document.getElementById('ieBrushClear').onclick = () => {
                const ov = document.getElementById('ieBrushOverlay');
                if (ov) ov.getContext('2d').clearRect(0, 0, ov.width, ov.height);
            };
            document.getElementById('ieBrushApply').onclick = applyBrushSegmentation;
            if (hasImage()) initBrushOverlay();
        });
    }

    function initBrushOverlay() {
        const ov = document.getElementById('ieBrushOverlay');
        if (!ov || !hasImage()) return;
        const wrap = document.getElementById('ieCanvasWrap');
        const rect = canvas.getBoundingClientRect();
        const wrapRect = wrap.getBoundingClientRect();
        ov.style.display = 'block';
        ov.style.left = (rect.left - wrapRect.left) + 'px';
        ov.style.top = (rect.top - wrapRect.top) + 'px';
        ov.style.width = rect.width + 'px';
        ov.style.height = rect.height + 'px';
        ov.width = Math.round(rect.width);
        ov.height = Math.round(rect.height);
        const bCtx = ov.getContext('2d');
        bCtx.clearRect(0, 0, ov.width, ov.height);

        ov.onpointerdown = (e) => {
            brushDrawing = true;
            const r = ov.getBoundingClientRect();
            bCtx.beginPath();
            bCtx.moveTo(e.clientX - r.left, e.clientY - r.top);
            ov.setPointerCapture(e.pointerId);
            e.preventDefault();
        };
        ov.onpointermove = (e) => {
            if (!brushDrawing) return;
            const r = ov.getBoundingClientRect();
            bCtx.lineWidth = brushSize;
            bCtx.lineCap = 'round';
            bCtx.lineJoin = 'round';
            bCtx.strokeStyle = brushMode === 'fg' ? 'rgba(0,200,0,0.55)' : 'rgba(200,0,0,0.55)';
            bCtx.lineTo(e.clientX - r.left, e.clientY - r.top);
            bCtx.stroke();
            bCtx.beginPath();
            bCtx.moveTo(e.clientX - r.left, e.clientY - r.top);
            e.preventDefault();
        };
        ov.onpointerup = () => { brushDrawing = false; };
    }

    function applyBrushSegmentation() {
        if (!requireImage()) return;
        const ov = document.getElementById('ieBrushOverlay');
        if (!ov) return;
        const bCtx = ov.getContext('2d');
        const bData = bCtx.getImageData(0, 0, ov.width, ov.height).data;
        const scaleX = canvas.width / ov.width, scaleY = canvas.height / ov.height;

        const BINS = 16, SHIFT = 4, TOTAL = BINS * BINS * BINS;
        const fgH = new Float32Array(TOTAL), bgH = new Float32Array(TOTAL);
        let fgC = 0, bgC = 0;
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const px = imgData.data;

        for (let i = 0; i < bData.length; i += 4) {
            if (bData[i + 3] < 40) continue;
            const bx = (i / 4) % ov.width, by = Math.floor((i / 4) / ov.width);
            const cx = Math.min(Math.floor(bx * scaleX), canvas.width - 1);
            const cy = Math.min(Math.floor(by * scaleY), canvas.height - 1);
            const ci = (cy * canvas.width + cx) * 4;
            const bin = ((px[ci] >> SHIFT) * BINS + (px[ci + 1] >> SHIFT)) * BINS + (px[ci + 2] >> SHIFT);
            if (bData[i + 1] > bData[i]) { fgH[bin]++; fgC++; }
            else { bgH[bin]++; bgC++; }
        }

        if (!fgC || !bgC) { Toolbox.showToast('전경(🟢)과 배경(🔴)을 모두 칠해주세요.', 'error'); return; }
        for (let i = 0; i < TOTAL; i++) { fgH[i] /= fgC; bgH[i] /= bgC; }

        const EPS = 1e-7;
        for (let i = 0; i < px.length; i += 4) {
            const bin = ((px[i] >> SHIFT) * BINS + (px[i + 1] >> SHIFT)) * BINS + (px[i + 2] >> SHIFT);
            const fg = fgH[bin] + EPS, bg = bgH[bin] + EPS;
            px[i + 3] = Math.round((fg / (fg + bg)) * 255);
        }
        ctx.putImageData(imgData, 0, 0);
        destroyBrush();
        pushHistory();
        updateSizeLabel();
        Toolbox.showToast('브러시 분류 적용 완료');
        Mdd.setMood('happy'); Mdd.say('누끼 작업 완료냥!');
    }

    /* ===== Mode 3 — AI (ONNX) ===== */
    function buildAiBody(body) {
        body.innerHTML = `
            <span class="ie-opt-label">모델</span>
            <button class="ie-opt-btn${rembgModel === 'isnet_quint8' ? ' active' : ''}" id="ieRembgSmall">⚡ 빠른</button>
            <button class="ie-opt-btn${rembgModel === 'isnet_fp16' ? ' active' : ''}" id="ieRembgMedium">✨ 정밀</button>
            <span class="ie-toolbar-sep"></span>
            <span class="ie-opt-label">해상도</span>
            <select class="ie-opt-input" id="ieRembgSize" style="width:auto;">
                <option value="512"${rembgMaxSize === 512 ? ' selected' : ''}>512</option>
                <option value="768"${rembgMaxSize === 768 ? ' selected' : ''}>768</option>
                <option value="1024"${rembgMaxSize === 1024 ? ' selected' : ''}>1024</option>
            </select>
            <span class="ie-opt-label ie-rembg-note">처음 실행 시 모델 다운로드 · 느릴 수 있음</span>
            <button class="ie-apply-btn" id="ieRembgApply">🪄 배경 제거</button>`;
        requestAnimationFrame(() => {
            const s = document.getElementById('ieRembgSmall'), m = document.getElementById('ieRembgMedium');
            s.onclick = () => { rembgModel = 'isnet_quint8'; s.classList.add('active'); m.classList.remove('active'); };
            m.onclick = () => { rembgModel = 'isnet_fp16'; m.classList.add('active'); s.classList.remove('active'); };
            document.getElementById('ieRembgSize').onchange = (e) => { rembgMaxSize = parseInt(e.target.value); };
            document.getElementById('ieRembgApply').onclick = applyRemoveBackground;
        });
    }

    const REMBG_CDN = 'https://cdn.jsdelivr.net/npm/@imgly/background-removal@1.7.0/+esm';

    function rembgWorkerCode() {
        return `
            self.onmessage = async (e) => {
                try {
                    const { imageBlob, model, outputType } = e.data;
                    const mod = await import('${REMBG_CDN}');
                    const fn = mod.removeBackground || mod.default;
                    const progress = (key, current, total) => {
                        self.postMessage({ t: 'p', key, current, total });
                    };
                    let result;
                    try {
                        result = await fn(imageBlob, { model, device:'gpu', output:{format:'image/png',type:outputType}, progress });
                    } catch (_) {
                        self.postMessage({ t:'n' });
                        result = await fn(imageBlob, { model, device:'cpu', output:{format:'image/png',type:outputType}, progress });
                    }
                    self.postMessage({ t: 'd', blob: result });
                } catch (err) {
                    self.postMessage({ t: 'e', msg: err.message || String(err) });
                }
            };
        `;
    }

    function runInWorker(blob, model, onProgress) {
        const url = URL.createObjectURL(new Blob([rembgWorkerCode()], { type: 'text/javascript' }));
        const w = new Worker(url, { type: 'module' });
        let settled = false;
        const cleanup = () => { if (settled) return; settled = true; try { w.terminate(); } catch {} try { URL.revokeObjectURL(url); } catch {} };
        const promise = new Promise((resolve, reject) => {
            w.onmessage = (e) => {
                const d = e.data;
                if (d.t === 'p') onProgress(d.key, d.current, d.total);
                else if (d.t === 'n') { /* gpu fallback note */ }
                else if (d.t === 'd') { cleanup(); resolve(d.blob); }
                else if (d.t === 'e') { cleanup(); reject(new Error(d.msg)); }
            };
            w.onerror = (e) => { cleanup(); reject(new Error(e.message || 'Worker failed')); };
            w.postMessage({ imageBlob: blob, model, outputType: 'foreground' });
        });
        return { promise, cancel: cleanup };
    }

    async function runOnMainThread(blob, model, onProgress) {
        const mod = await import(REMBG_CDN);
        const fn = mod.removeBackground || mod.default;
        const run = (d) => fn(blob, { model, device: d, output: { format: 'image/png', type: 'foreground' }, progress: onProgress });
        try { return await run('gpu'); } catch { return await run('cpu'); }
    }

    function showRembgOverlay() {
        const wrap = document.getElementById('ieCanvasWrap');
        const ov = document.createElement('div');
        ov.className = 'ie-rembg-overlay'; ov.id = 'ieRembgOverlay';
        ov.innerHTML = `
            <div class="ie-rembg-spinner"></div>
            <div class="ie-rembg-status" id="ieRembgStatus">준비 중...</div>
            <div class="ie-rembg-status" style="font-size:var(--font-size-xs);font-weight:400;opacity:0.8" id="ieRembgElapsed">경과: 0.0s</div>
            <div class="ie-rembg-bar-wrap"><div class="ie-rembg-bar" id="ieRembgBar"></div></div>
            <button class="ie-tb-btn" id="ieRembgCancel" style="margin-top:6px;">취소</button>`;
        wrap.appendChild(ov);
        const start = performance.now();
        const timer = setInterval(() => {
            const el = document.getElementById('ieRembgElapsed');
            if (!el) return;
            const s = (performance.now() - start) / 1000;
            el.textContent = '경과: ' + s.toFixed(s < 10 ? 1 : 0) + 's';
        }, 200);
        return { timer, overlay: ov };
    }

    function cleanupOverlay(ui) {
        try { ui.overlay.remove(); } catch {}
        clearInterval(ui.timer);
        rembgBusy = false;
    }

    async function applyRemoveBackground() {
        if (!requireImage()) return;
        if (rembgBusy) { Toolbox.showToast('이미 처리 중입니다.', 'error'); return; }
        rembgBusy = true;

        const ui = showRembgOverlay();
        const stEl = () => document.getElementById('ieRembgStatus');
        const barEl = () => document.getElementById('ieRembgBar');
        let cancelFn = null;
        const cBtn = document.getElementById('ieRembgCancel');
        if (cBtn) cBtn.onclick = () => { if (cancelFn) cancelFn(); cleanupOverlay(ui); Toolbox.showToast('취소됨'); };

        const onProg = (key, cur, tot) => {
            if (!tot) return;
            const p = Math.round(cur / tot * 100);
            const b = barEl(); if (b) { b.classList.add('determinate'); b.style.width = p + '%'; }
            const s = stEl(); if (s) s.textContent = (key.includes('onnx') || key.includes('model') ? '모델 다운로드 ' : '처리 ') + p + '%';
        };

        try {
            const origW = canvas.width, origH = canvas.height;
            const longest = Math.max(origW, origH);
            let procW = origW, procH = origH;
            if (longest > rembgMaxSize) { const sc = rembgMaxSize / longest; procW = Math.round(origW * sc); procH = Math.round(origH * sc); }

            if (stEl()) stEl().textContent = '이미지 축소 중... (' + procW + '×' + procH + ')';
            const pc = document.createElement('canvas'); pc.width = procW; pc.height = procH;
            pc.getContext('2d').drawImage(canvas, 0, 0, procW, procH);
            const blob = await new Promise(r => pc.toBlob(r, 'image/png'));

            if (stEl()) stEl().textContent = 'AI 엔진 초기화 중...';
            let rb;
            try {
                const job = runInWorker(blob, rembgModel, onProg);
                cancelFn = job.cancel;
                rb = await job.promise;
            } catch (we) {
                if (stEl()) stEl().textContent = '배경 제거 중... (화면 멈출 수 있음)';
                const b = barEl(); if (b) { b.classList.remove('determinate'); b.style.width = ''; }
                const c = document.getElementById('ieRembgCancel'); if (c) { c.disabled = true; c.textContent = '취소 불가'; }
                await new Promise(r => requestAnimationFrame(() => setTimeout(r, 100)));
                rb = await runOnMainThread(blob, rembgModel, onProg);
            }

            const img = new Image(); const u = URL.createObjectURL(rb);
            await new Promise((ok, no) => { img.onload = ok; img.onerror = no; img.src = u; });

            if (procW !== origW) {
                const off = document.createElement('canvas'); off.width = origW; off.height = origH;
                const oc = off.getContext('2d');
                oc.drawImage(canvas, 0, 0);
                oc.globalCompositeOperation = 'destination-in';
                oc.drawImage(img, 0, 0, origW, origH);
                canvas.width = origW; canvas.height = origH;
                ctx.drawImage(off, 0, 0);
            } else {
                canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
                ctx.drawImage(img, 0, 0);
            }
            URL.revokeObjectURL(u);
            pushHistory(); updateSizeLabel(); cleanupOverlay(ui);
            Toolbox.showToast('배경 제거 완료!');
            Mdd.setMood('cheer'); Mdd.say('누끼 완성이다냥!');
        } catch (e) {
            console.error('BG removal error:', e);
            cleanupOverlay(ui);
            Toolbox.showToast('배경 제거 실패: ' + (e.message || e), 'error');
        }
    }

    /* ===== Mode 4 — Gemini ===== */
    let lastGeminiMaskDataUrl = null;

    function buildGeminiBody(body) {
        const hasKey = !!Gemini.getApiKey();
        body.innerHTML = `
            <span class="ie-opt-label">모델</span>
            <select class="ie-opt-input" id="ieGeminiModel" style="width:auto;">
                ${Gemini.MODELS.geminiImage.map(m => `<option value="${m.id}">${m.name}</option>`).join('')}
            </select>
            <span class="ie-toolbar-sep"></span>
            <span class="ie-opt-label ie-rembg-note">${hasKey ? '✅ API 키 설정됨' : '⚠️ API 키 필요 (홈 > 사용자 설정)'}</span>
            <div style="display:flex;gap:6px;flex-wrap:wrap;">
                <button class="ie-apply-btn" id="ieGeminiApply" ${hasKey ? '' : 'disabled'}>✨ Gemini 배경 제거</button>
                <button class="ie-apply-btn" id="ieGeminiDownloadMask" style="background:#555;" ${lastGeminiMaskDataUrl ? '' : 'disabled'}>🖼️ 마스크 다운로드</button>
            </div>`;
        requestAnimationFrame(() => {
            document.getElementById('ieGeminiApply').onclick = applyGeminiRemoveBg;
            document.getElementById('ieGeminiDownloadMask').onclick = downloadGeminiMask;
        });
    }

    function downloadGeminiMask() {
        if (!lastGeminiMaskDataUrl) { Toolbox.showToast('마스크가 없습니다. 먼저 배경 제거를 실행하세요.', 'error'); return; }
        const a = document.createElement('a');
        a.href = lastGeminiMaskDataUrl;
        a.download = 'gemini_mask.png';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        Toolbox.showToast('마스크 다운로드!');
    }

    async function applyGeminiRemoveBg() {
        if (!requireImage()) return;
        const key = Gemini.requireApiKey();
        if (!key) return;
        if (rembgBusy) { Toolbox.showToast('이미 처리 중입니다.', 'error'); return; }
        rembgBusy = true;

        const ui = showRembgOverlay();
        const stEl = () => document.getElementById('ieRembgStatus');
        const origW = canvas.width, origH = canvas.height;

        try {
            if (stEl()) stEl().textContent = '이미지 준비 중...';
            const maxSide = 1024;
            let dw = origW, dh = origH;
            const longest = Math.max(dw, dh);
            if (longest > maxSide) { const sc = maxSide / longest; dw = Math.round(dw * sc); dh = Math.round(dh * sc); }
            const tmp = document.createElement('canvas'); tmp.width = dw; tmp.height = dh;
            tmp.getContext('2d').drawImage(canvas, 0, 0, dw, dh);
            const base64 = tmp.toDataURL('image/png').split(',')[1];

            if (stEl()) stEl().textContent = 'Gemini에 마스크 요청 중...';
            const modelId = document.getElementById('ieGeminiModel')?.value || Gemini.MODELS.geminiImage[0].id;
            const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelId}:generateContent?key=${key}`;
            const reqBody = {
                contents: [{ parts: [
                    { text: 'Generate a binary segmentation mask for this image. The mask must be the EXACT same dimensions as the input image. The foreground subject should be pure white (#FFFFFF) and the background should be pure black (#000000). No gray, no gradients, no antialiasing — strictly black and white only. Output ONLY the mask image, nothing else.' },
                    { inlineData: { mimeType: 'image/png', data: base64 } }
                ]}],
                generationConfig: { responseModalities: ['TEXT', 'IMAGE'] }
            };

            const res = await Gemini.fetchWithRetry(url, reqBody);
            const data = await res.json();
            const imgPart = data.candidates?.[0]?.content?.parts?.find(p => p.inlineData);
            if (!imgPart) throw new Error('Gemini에서 마스크 응답이 없습니다.');

            if (stEl()) stEl().textContent = '마스크 적용 중...';
            const rawMaskUrl = `data:${imgPart.inlineData.mimeType || 'image/png'};base64,${imgPart.inlineData.data}`;
            const maskImg = new Image();
            await new Promise((ok, no) => { maskImg.onload = ok; maskImg.onerror = no; maskImg.src = rawMaskUrl; });

            const maskCvs = document.createElement('canvas');
            maskCvs.width = origW; maskCvs.height = origH;
            const mc = maskCvs.getContext('2d');
            mc.drawImage(maskImg, 0, 0, origW, origH);

            lastGeminiMaskDataUrl = maskCvs.toDataURL('image/png');
            const dlBtn = document.getElementById('ieGeminiDownloadMask');
            if (dlBtn) dlBtn.disabled = false;

            const maskData = mc.getImageData(0, 0, origW, origH);
            const md = maskData.data;
            for (let i = 0; i < md.length; i += 4) {
                const lum = md[i] * 0.299 + md[i + 1] * 0.587 + md[i + 2] * 0.114;
                md[i] = 255; md[i + 1] = 255; md[i + 2] = 255;
                md[i + 3] = Math.round(lum);
            }
            mc.putImageData(maskData, 0, 0);

            const off = document.createElement('canvas');
            off.width = origW; off.height = origH;
            const oc = off.getContext('2d');
            oc.drawImage(canvas, 0, 0);
            oc.globalCompositeOperation = 'destination-in';
            oc.drawImage(maskCvs, 0, 0);

            canvas.width = origW; canvas.height = origH;
            ctx.clearRect(0, 0, origW, origH);
            ctx.drawImage(off, 0, 0);

            pushHistory(); updateSizeLabel(); cleanupOverlay(ui);
            Toolbox.showToast('Gemini 마스크 적용 완료!');
            Mdd.setMood('cheer'); Mdd.say('Gemini가 마스크 만들어줬다냥!');
        } catch (e) {
            console.error('Gemini mask error:', e);
            cleanupOverlay(ui);
            Toolbox.showToast('Gemini 실패: ' + (e.message || e), 'error');
            Mdd.setMood('sad'); Mdd.say('Gemini 실패했다냥...');
        }
    }

    /* ===== Mode 5 — 배경색 변경 (Gemini) ===== */
    let bggBusy = false;

    function buildBggBody(body) {
        const hasKey = !!Gemini.getApiKey();
        body.innerHTML = `
            <span class="ie-opt-label">배경색</span>
            <input type="color" id="ieBggColor" value="#ffffff" style="width:40px;height:28px;padding:2px;border:1px solid var(--border);border-radius:4px;cursor:pointer;">
            <input type="text" id="ieBggColorText" placeholder="예: 하늘색, 그라데이션" style="width:120px;padding:4px 8px;font-size:var(--font-size-xs);border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--bg-tertiary);color:var(--text-primary);" title="색상명 또는 설명 입력">
            <span class="ie-toolbar-sep"></span>
            <span class="ie-opt-label">모델</span>
            <select class="ie-opt-input" id="ieBggModel" style="width:auto;">
                ${Gemini.MODELS.geminiImage.map(m => `<option value="${m.id}">${m.name}</option>`).join('')}
            </select>
            <span class="ie-toolbar-sep"></span>
            <span class="ie-opt-label ie-rembg-note">${hasKey ? '✅ API 키 설정됨' : '⚠️ API 키 필요'}</span>
            <button class="ie-apply-btn" id="ieBggApply" ${hasKey ? '' : 'disabled'}>🎨 Gemini 배경색 변경</button>`;
        requestAnimationFrame(() => {
            const textEl = document.getElementById('ieBggColorText');
            textEl?.addEventListener('input', () => {
                const v = textEl.value.trim();
                if (/^#[0-9a-fA-F]{6}$/.test(v)) document.getElementById('ieBggColor').value = v;
            });
            document.getElementById('ieBggApply').onclick = applyGeminiBgg;
        });
    }

    async function applyGeminiBgg() {
        if (!requireImage()) return;
        const key = Gemini.requireApiKey();
        if (!key) return;
        if (bggBusy) { Toolbox.showToast('이미 처리 중입니다.', 'error'); return; }
        bggBusy = true;

        const colorText = document.getElementById('ieBggColorText')?.value?.trim();
        const colorHex = document.getElementById('ieBggColor')?.value || '#ffffff';
        const colorDesc = colorText || colorHex;

        const ui = showRembgOverlay();
        const stEl = () => document.getElementById('ieRembgStatus');

        try {
            if (stEl()) stEl().textContent = '이미지 준비 중...';
            const maxSide = 1024;
            let dw = canvas.width, dh = canvas.height;
            const longest = Math.max(dw, dh);
            if (longest > maxSide) { const sc = maxSide / longest; dw = Math.round(dw * sc); dh = Math.round(dh * sc); }
            const tmp = document.createElement('canvas');
            tmp.width = dw; tmp.height = dh;
            tmp.getContext('2d').drawImage(canvas, 0, 0, dw, dh);
            const base64 = tmp.toDataURL('image/png').split(',')[1];

            if (stEl()) stEl().textContent = 'Gemini에 배경색 변경 요청 중...';
            const modelId = document.getElementById('ieBggModel')?.value || Gemini.MODELS.geminiImage[0].id;
            const colorPrompt = /^#[0-9a-fA-F]{6}$/.test(colorDesc)
                ? `solid color ${colorDesc}`
                : colorDesc;
            const prompt = `Change the background of this image to ${colorPrompt}. Keep the foreground subject exactly the same. Do not alter the subject's appearance, pose, or lighting. Replace only the background. Output the result as a PNG image.`;

            const result = await Gemini.callGeminiImage(prompt, modelId, { referenceImage: base64 });
            if (!result?.dataUrl) throw new Error('이미지 응답이 없습니다.');

            if (stEl()) stEl().textContent = '결과 적용 중...';
            const img = new Image();
            await new Promise((ok, no) => { img.onload = ok; img.onerror = no; img.src = result.dataUrl; });

            canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            pushHistory(); updateSizeLabel(); cleanupOverlay(ui);
            Toolbox.showToast('배경색 변경 완료!');
            Mdd.setMood('cheer'); Mdd.say('배경색 바꿔줬다냥!');
        } catch (e) {
            console.error('Gemini BGG error:', e);
            cleanupOverlay(ui);
            Toolbox.showToast('Gemini 실패: ' + (e.message || e), 'error');
            Mdd.setMood('sad'); Mdd.say('Gemini 실패했다냥...');
        } finally {
            bggBusy = false;
        }
    }

    /* ===== Caption Tool ===== */
    const CAPTION_PRESETS = [
        { id: 'impact', label: '흰 테두리 검정', icon: '⬛', font: 'Impact, "Noto Sans KR", Arial Black, sans-serif', fontSize: 0.07, fillStyle: '#000000', strokeStyle: '#FFFFFF', lineWidth: 4, position: 'bottom', uppercase: true, maxLines: 3 },
        { id: 'subtitle', label: '자막 스타일', icon: '🟡', font: '"Noto Sans KR", sans-serif', fontWeight: 'bold', fontSize: 0.055, fillStyle: '#FFFF00', strokeStyle: '#000000', lineWidth: 3, position: 'bottom', bgBar: 'rgba(0,0,0,0.5)', maxLines: 3 },
        { id: 'bubble', label: '만화 말풍선', icon: '💬', font: '"Comic Sans MS", "나눔손글씨 붓", cursive', fontSize: 0.06, fillStyle: '#000000', bubbleStyle: { fill: '#FFFFFF', stroke: '#000000', strokeWidth: 2, padding: 16 }, position: 'center', maxLines: 4 },
        { id: 'bar', label: '심플 하단바', icon: '▬', font: '"Noto Sans KR", sans-serif', fontSize: 0.05, fillStyle: '#FFFFFF', barBg: 'rgba(0,0,0,0.7)', padding: 20, position: 'bottom', maxLines: 2 }
    ];
    (function ensureCaptionFonts() {
        if (document.getElementById('ie-caption-fonts')) return;
        const link = document.createElement('link');
        link.id = 'ie-caption-fonts';
        link.rel = 'stylesheet';
        link.href = 'https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap';
        document.head.appendChild(link);
    })();
    function wrapCaptionLines(ctx, text, maxWidth) {
        if (!text || maxWidth <= 0) return [];
        const lines = [];
        const paragraphs = String(text).split(/\n/);
        for (const para of paragraphs) {
            let line = '';
            for (let i = 0; i < para.length; i++) {
                const test = line + para[i];
                const m = ctx.measureText(test);
                if (m.width > maxWidth && line.length > 0) { lines.push(line); line = para[i]; } else { line = test; }
            }
            if (line) lines.push(line);
        }
        return lines;
    }
    function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    }
    let captionPosX = 0.5, captionPosY = 0.8;
    let captionDragging = false;

    function renderCaptionOnCanvas(ctx, preset, text, canvasW, canvasH, positionOverride) {
        if (!text || !preset) return;
        const pad = Math.min(canvasW, canvasH) * 0.05;
        const maxWidth = canvasW - pad * 2;
        const useCustom = positionOverride && typeof positionOverride.x === 'number' && typeof positionOverride.y === 'number';
        const pos = useCustom ? 'custom' : (preset.position || 'bottom');
        const fontSize = Math.max(14, Math.floor(Math.min(canvasW, canvasH) * preset.fontSize));
        ctx.save();
        ctx.font = `${preset.fontWeight || ''} ${fontSize}px ${preset.font}`.trim();
        let displayText = text;
        if (preset.uppercase) displayText = displayText.toUpperCase();
        const lines = wrapCaptionLines(ctx, displayText, maxWidth).slice(0, preset.maxLines || 3);
        if (lines.length === 0) { ctx.restore(); return; }
        const lineHeight = fontSize * 1.3;
        const totalHeight = lines.length * lineHeight;
        let yBase, centerX, centerY;
        if (pos === 'custom') {
            centerX = positionOverride.x * canvasW;
            centerY = positionOverride.y * canvasH;
            yBase = centerY - (lines.length - 1) * lineHeight / 2;
        } else {
            centerX = canvasW / 2;
            if (pos === 'top') yBase = pad + lineHeight;
            else if (pos === 'center') yBase = (canvasH - totalHeight) / 2 + lineHeight;
            else yBase = canvasH - pad - totalHeight + lineHeight;
            centerY = yBase + (lines.length - 1) * lineHeight / 2;
        }
        if (preset.id === 'impact') {
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.strokeStyle = preset.strokeStyle;
            ctx.fillStyle = preset.fillStyle;
            ctx.lineWidth = Math.max(2, Math.floor(fontSize * 0.08));
            lines.forEach((line, i) => {
                const y = yBase + i * lineHeight;
                ctx.strokeText(line, centerX, y);
                ctx.fillText(line, centerX, y);
            });
        } else if (preset.id === 'subtitle') {
            const subYBase = pos === 'custom' ? (centerY - (lines.length - 1) * lineHeight / 2) : yBase;
            if (preset.bgBar) {
                const barH = totalHeight + pad * 2;
                const barY = pos === 'custom' ? (centerY - barH / 2) : (canvasH - barH);
                const barW = pos === 'custom' ? Math.min(canvasW, maxWidth + 40) : canvasW;
                const barX = pos === 'custom' ? (centerX - barW / 2) : 0;
                ctx.fillStyle = preset.bgBar;
                ctx.fillRect(barX, barY, barW, barH);
            }
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.strokeStyle = preset.strokeStyle;
            ctx.fillStyle = preset.fillStyle;
            ctx.lineWidth = Math.max(2, Math.floor(fontSize * 0.06));
            lines.forEach((line, i) => {
                const y = subYBase + i * lineHeight;
                ctx.strokeText(line, centerX, y);
                ctx.fillText(line, centerX, y);
            });
        } else if (preset.id === 'bubble') {
            const bubble = preset.bubbleStyle || {};
            const padding = bubble.padding || 16;
            const strokeW = bubble.strokeWidth || 2;
            const lineW = lines.reduce((w, l) => Math.max(w, ctx.measureText(l).width), 0);
            const boxW = lineW + padding * 2;
            const boxH = totalHeight + padding * 2;
            const boxX = pos === 'custom' ? (centerX - boxW / 2) : (canvasW - boxW) / 2;
            const boxY = pos === 'custom' ? (centerY - boxH / 2) : (canvasH - boxH) / 2;
            ctx.fillStyle = bubble.fill || '#FFFFFF';
            ctx.strokeStyle = bubble.stroke || '#000000';
            ctx.lineWidth = strokeW;
            const r = Math.min(boxW, boxH) * 0.2;
            roundRect(ctx, boxX, boxY, boxW, boxH, r);
            ctx.fill();
            ctx.stroke();
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = preset.fillStyle || '#000000';
            const tyBase = boxY + padding + lineHeight / 2;
            lines.forEach((line, i) => {
                ctx.fillText(line, centerX, tyBase + i * lineHeight);
            });
        } else if (preset.id === 'bar') {
            const barPad = preset.padding || 20;
            const barH = totalHeight + barPad * 2;
            const barY = pos === 'custom' ? (centerY - barH / 2) : (canvasH - barH);
            const barW = pos === 'custom' ? Math.min(canvasW, maxWidth + 40) : canvasW;
            const barX = pos === 'custom' ? (centerX - barW / 2) : 0;
            ctx.fillStyle = preset.barBg || 'rgba(0,0,0,0.7)';
            ctx.fillRect(barX, barY, barW, barH);
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = preset.fillStyle || '#FFFFFF';
            const byBase = barY + barPad + lineHeight / 2;
            lines.forEach((line, i) => {
                ctx.fillText(line, centerX, byBase + i * lineHeight);
            });
        }
        ctx.restore();
    }

    function destroyCaptionOverlay() {
        const ov = document.getElementById('ieCaptionOverlay');
        if (ov) {
            ov.style.display = 'none';
            ov.onpointerdown = null;
            ov.onpointermove = null;
            ov.onpointerup = null;
            ov.onpointerleave = null;
        }
    }

    function redrawCaptionPreview() {
        const ov = document.getElementById('ieCaptionOverlay');
        if (!ov || ov.style.display !== 'block' || !hasImage()) return;
        const presetId = document.querySelector('#ieCaptionPresets button.active')?.dataset?.preset || 'impact';
        const preset = CAPTION_PRESETS.find(p => p.id === presetId) || CAPTION_PRESETS[0];
        const text = document.getElementById('ieCaptionText')?.value?.trim() || '';
        const ovCtx = ov.getContext('2d');
        ovCtx.drawImage(canvas, 0, 0);
        if (text) {
            renderCaptionOnCanvas(ovCtx, preset, text, ov.width, ov.height, { x: captionPosX, y: captionPosY });
        }
    }

    function initCaptionOverlay() {
        const ov = document.getElementById('ieCaptionOverlay');
        if (!ov || !hasImage()) return;
        const wrap = document.getElementById('ieCanvasWrap');
        const rect = canvas.getBoundingClientRect();
        const wrapRect = wrap.getBoundingClientRect();
        ov.style.display = 'block';
        ov.style.left = (rect.left - wrapRect.left) + 'px';
        ov.style.top = (rect.top - wrapRect.top) + 'px';
        ov.style.width = rect.width + 'px';
        ov.style.height = rect.height + 'px';
        ov.width = canvas.width;
        ov.height = canvas.height;
        redrawCaptionPreview();
        const getCanvasCoords = (e) => {
            const r = ov.getBoundingClientRect();
            const x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
            const y = Math.max(0, Math.min(1, (e.clientY - r.top) / r.height));
            return { x, y };
        };
        ov.onpointerdown = (e) => {
            captionDragging = true;
            const coords = getCanvasCoords(e);
            captionPosX = coords.x;
            captionPosY = coords.y;
            redrawCaptionPreview();
            ov.setPointerCapture(e.pointerId);
            e.preventDefault();
        };
        ov.onpointermove = (e) => {
            if (!captionDragging) return;
            const coords = getCanvasCoords(e);
            captionPosX = coords.x;
            captionPosY = coords.y;
            redrawCaptionPreview();
            e.preventDefault();
        };
        ov.onpointerup = ov.onpointerleave = () => { captionDragging = false; };
    }
    function buildCaptionOptions(optPanel) {
        optPanel.innerHTML = `
            <div class="ie-opt-row" style="flex-wrap:wrap;gap:8px;">
                <span class="ie-opt-label">스타일</span>
                <div class="ie-filter-grid" id="ieCaptionPresets" style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;"></div>
            </div>
            <div class="ie-opt-row" style="flex-direction:column;align-items:stretch;">
                <span class="ie-opt-label">캡션 텍스트 <span style="color:var(--text-tertiary);font-weight:400;">(드래그로 위치 이동)</span></span>
                <textarea class="ie-opt-input" id="ieCaptionText" placeholder="캡션을 입력하세요 (줄바꿈 가능)" style="min-height:60px;resize:vertical;"></textarea>
            </div>
            <div class="ie-opt-row">
                <button class="ie-apply-btn" id="ieCaptionApply">캡션 적용</button>
            </div>`;
        const presetWrap = document.getElementById('ieCaptionPresets');
        CAPTION_PRESETS.forEach((p, i) => {
            const btn = document.createElement('button');
            btn.className = 'ie-opt-btn' + (i === 0 ? ' active' : '');
            btn.dataset.preset = p.id;
            btn.innerHTML = `<span style="font-size:1.2em;">${p.icon}</span><span class="ie-opt-label" style="font-size:var(--font-size-2xs);">${p.label}</span>`;
            btn.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:2px;padding:8px 4px;';
            presetWrap.appendChild(btn);
        });
        presetWrap.addEventListener('click', (e) => {
            const btn = e.target.closest('[data-preset]');
            if (!btn) return;
            presetWrap.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            redrawCaptionPreview();
        });
        const textEl = document.getElementById('ieCaptionText');
        textEl.addEventListener('input', () => redrawCaptionPreview());
        textEl.addEventListener('keyup', () => redrawCaptionPreview());
        if (document.fonts && document.fonts.load) {
            document.fonts.load('16px "Noto Sans KR"').catch(() => {});
        }
        document.getElementById('ieCaptionApply').onclick = applyCaption;
        if (hasImage()) initCaptionOverlay();
    }
    function applyCaption() {
        if (!requireImage()) return;
        const presetId = document.querySelector('#ieCaptionPresets button.active')?.dataset?.preset || 'impact';
        const preset = CAPTION_PRESETS.find(p => p.id === presetId) || CAPTION_PRESETS[0];
        const text = document.getElementById('ieCaptionText')?.value?.trim();
        if (!text) { Toolbox.showToast('캡션 텍스트를 입력하세요.', 'error'); return; }
        pushHistory();
        renderCaptionOnCanvas(ctx, preset, text, canvas.width, canvas.height, { x: captionPosX, y: captionPosY });
        document.getElementById('ieCaptionText').value = '';
        redrawCaptionPreview();
        updateSizeLabel();
        Toolbox.showToast('캡션 적용됨', 'success');
    }

    /* ===== Sticker Tool ===== */
    const STICKER_BASE = '/apps/karmolab/img/stickers';
    let stickerImages = [];
    let stickerPosX = 0.5, stickerPosY = 0.5;
    let stickerScale = 0.2;
    let stickerDragging = false;
    let selectedStickerImg = null;
    let selectedStickerFile = null;

    function destroyStickerOverlay() {
        const ov = document.getElementById('ieStickerOverlay');
        if (ov) {
            ov.style.display = 'none';
            ov.onpointerdown = null;
            ov.onpointermove = null;
            ov.onpointerup = null;
            ov.onpointerleave = null;
        }
    }

    function renderStickerOnCanvas(ctx, img, canvasW, canvasH, posX, posY, scale) {
        if (!img || !img.complete || img.naturalWidth === 0) return;
        const baseSize = Math.min(canvasW, canvasH) * scale;
        const w = img.naturalWidth;
        const h = img.naturalHeight;
        const aspect = w / h;
        let drawW, drawH;
        if (aspect >= 1) { drawW = baseSize; drawH = baseSize / aspect; }
        else { drawH = baseSize; drawW = baseSize * aspect; }
        const x = posX * canvasW - drawW / 2;
        const y = posY * canvasH - drawH / 2;
        ctx.drawImage(img, x, y, drawW, drawH);
    }

    function redrawStickerPreview() {
        const ov = document.getElementById('ieStickerOverlay');
        if (!ov || ov.style.display !== 'block' || !hasImage()) return;
        const ovCtx = ov.getContext('2d');
        ovCtx.drawImage(canvas, 0, 0);
        if (selectedStickerImg) {
            renderStickerOnCanvas(ovCtx, selectedStickerImg, ov.width, ov.height, stickerPosX, stickerPosY, stickerScale);
        }
    }

    function initStickerOverlay() {
        const ov = document.getElementById('ieStickerOverlay');
        if (!ov || !hasImage()) return;
        const wrap = document.getElementById('ieCanvasWrap');
        const rect = canvas.getBoundingClientRect();
        const wrapRect = wrap.getBoundingClientRect();
        ov.style.display = 'block';
        ov.style.left = (rect.left - wrapRect.left) + 'px';
        ov.style.top = (rect.top - wrapRect.top) + 'px';
        ov.style.width = rect.width + 'px';
        ov.style.height = rect.height + 'px';
        ov.width = canvas.width;
        ov.height = canvas.height;
        redrawStickerPreview();
        const getCanvasCoords = (e) => {
            const r = ov.getBoundingClientRect();
            return {
                x: Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
                y: Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))
            };
        };
        ov.onpointerdown = (e) => {
            stickerDragging = true;
            const coords = getCanvasCoords(e);
            stickerPosX = coords.x;
            stickerPosY = coords.y;
            redrawStickerPreview();
            ov.setPointerCapture(e.pointerId);
            e.preventDefault();
        };
        ov.onpointermove = (e) => {
            if (!stickerDragging) return;
            const coords = getCanvasCoords(e);
            stickerPosX = coords.x;
            stickerPosY = coords.y;
            redrawStickerPreview();
            e.preventDefault();
        };
        ov.onpointerup = ov.onpointerleave = () => { stickerDragging = false; };
    }

    function buildStickerOptions(optPanel) {
        optPanel.innerHTML = `
            <div class="ie-opt-row" style="flex-wrap:wrap;gap:8px;">
                <span class="ie-opt-label">스티커 <span style="color:var(--text-tertiary);font-weight:400;">(드래그로 위치 이동)</span></span>
                <div class="ie-filter-grid" id="ieStickerGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(48px,1fr));gap:6px;max-height:120px;overflow-y:auto;"></div>
            </div>
            <div class="ie-opt-row" style="align-items:center;gap:8px;">
                <span class="ie-opt-label">크기</span>
                <input type="range" class="ie-opt-range" id="ieStickerScale" min="5" max="50" value="20" style="width:100px;">
                <span id="ieStickerScaleVal" class="ie-opt-range-val">20%</span>
            </div>
            <div class="ie-opt-row">
                <button class="ie-apply-btn" id="ieStickerApply">스티커 적용</button>
            </div>`;
        const grid = document.getElementById('ieStickerGrid');
        const scaleSlider = document.getElementById('ieStickerScale');
        const scaleVal = document.getElementById('ieStickerScaleVal');
        scaleSlider.oninput = () => {
            stickerScale = scaleSlider.value / 100;
            scaleVal.textContent = scaleSlider.value + '%';
            redrawStickerPreview();
        };
        function loadStickers(files) {
                if (!Array.isArray(files) || files.length === 0) {
                    grid.innerHTML = '<span class="ie-opt-label" style="grid-column:1/-1;">스티커가 없습니다. img/stickers/ 폴더에 이미지 추가 후 stickers.json에 파일명을 적어주세요.</span>';
                    return;
                }
                stickerImages = [];
                files.forEach((file, i) => {
                    const img = new Image();
                    img.crossOrigin = 'anonymous';
                    const btn = document.createElement('button');
                    btn.className = 'ie-opt-btn' + (i === 0 ? ' active' : '');
                    btn.dataset.file = file;
                    btn.style.cssText = 'padding:4px;min-width:44px;min-height:44px;display:flex;align-items:center;justify-content:center;';
                    btn.innerHTML = '<img src="' + STICKER_BASE + '/' + encodeURIComponent(file) + '" alt="" style="max-width:40px;max-height:40px;object-fit:contain;" loading="lazy">';
                    btn.onclick = () => {
                        grid.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                        btn.classList.add('active');
                        if (btn._stickerImg) {
                            selectedStickerImg = btn._stickerImg;
                            selectedStickerFile = file;
                            redrawStickerPreview();
                        }
                    };
                    grid.appendChild(btn);
                    img.onload = () => {
                        btn._stickerImg = img;
                        stickerImages.push(img);
                        if (i === 0) { selectedStickerImg = img; selectedStickerFile = file; }
                        redrawStickerPreview();
                    };
                    img.onerror = () => { stickerImages.push(null); };
                    img.src = STICKER_BASE + '/' + encodeURIComponent(file);
                });
        }
        fetch(STICKER_BASE + '/stickers.json')
            .then(r => r.ok ? r.json() : [])
            .then(files => loadStickers(files))
            .catch(() => {
                grid.innerHTML = '<span class="ie-opt-label" style="grid-column:1/-1;">stickers.json을 불러올 수 없습니다.</span>';
            });
        document.getElementById('ieStickerApply').onclick = applySticker;
        if (hasImage()) initStickerOverlay();
    }

    function applySticker() {
        if (!requireImage()) return;
        if (!selectedStickerImg || !selectedStickerImg.complete) {
            Toolbox.showToast('스티커를 선택하세요.', 'error');
            return;
        }
        pushHistory();
        renderStickerOnCanvas(ctx, selectedStickerImg, canvas.width, canvas.height, stickerPosX, stickerPosY, stickerScale);
        redrawStickerPreview();
        updateSizeLabel();
        Toolbox.showToast('스티커 적용됨', 'success');
    }

    /* ===== Mask Apply Tool ===== */
    let maskImageData = null;
    let maskInvert = false;
    let maskPreviewActive = false;

    function buildMaskOptions(optPanel) {
        optPanel.innerHTML = `
            <div style="display:flex;flex-direction:column;gap:10px;width:100%;">
                <div style="padding-bottom:8px;border-bottom:1px solid var(--border);">
                    <span class="ie-opt-label" style="font-weight:600;margin-bottom:4px;">📁 외부 마스크 적용</span>
                    <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:4px;">
                        <button class="ie-apply-btn" id="ieMaskLoadFile" style="background:#555;">📂 파일</button>
                        <button class="ie-apply-btn" id="ieMaskPaste" style="background:#555;">📋 붙여넣기</button>
                        <div id="ieMaskPreviewWrap" style="display:none; align-items:center; gap:6px;">
                            <canvas id="ieMaskPreview" style="max-width:60px; max-height:40px; border:1px solid var(--border); border-radius:4px;"></canvas>
                            <span id="ieMaskInfo" class="ie-opt-label" style="font-size:var(--font-size-xs);"></span>
                        </div>
                        <span class="ie-opt-label">해석</span>
                        <select class="ie-opt-input" id="ieMaskInterpret" style="width:auto;">
                            <option value="alpha">알파 채널</option>
                            <option value="luminance">밝기</option>
                        </select>
                        <label id="ieMaskInvertLabel" style="display:none; align-items:center; gap:4px; font-size:var(--font-size-sm); color:var(--text-secondary); cursor:pointer;">
                            <input type="checkbox" id="ieMaskInvertCb"> 반전
                        </label>
                        <button class="ie-apply-btn" id="ieMaskPreviewBtn" disabled style="background:#666;">👁 미리보기</button>
                        <button class="ie-apply-btn" id="ieMaskApplyBtn" disabled>🎭 적용</button>
                    </div>
                </div>
                <div>
                    <span class="ie-opt-label" style="font-weight:600;margin-bottom:4px;">🎯 셀프 마스크 (현재 이미지 기준)</span>
                    <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:4px;">
                        <span class="ie-opt-label">대상</span>
                        <select class="ie-opt-input" id="ieSelfMaskTarget" style="width:auto;">
                            <option value="dark">검정 (어두운 영역 제거)</option>
                            <option value="light">흰색 (밝은 영역 제거)</option>
                        </select>
                        <span class="ie-opt-label">허용치</span>
                        <input type="range" class="ie-opt-range" id="ieSelfMaskTol" min="1" max="128" value="30" style="width:80px;">
                        <span id="ieSelfMaskTolVal" class="ie-opt-label" style="min-width:28px;">30</span>
                        <span class="ie-opt-label">페더</span>
                        <input type="range" class="ie-opt-range" id="ieSelfMaskFeather" min="0" max="64" value="10" style="width:60px;">
                        <span id="ieSelfMaskFeatherVal" class="ie-opt-label" style="min-width:28px;">10</span>
                        <button class="ie-apply-btn" id="ieSelfMaskApply">✂️ 제거</button>
                    </div>
                </div>
            </div>`;

        requestAnimationFrame(() => {
            const fileInput = document.createElement('input');
            fileInput.type = 'file'; fileInput.accept = 'image/*'; fileInput.style.display = 'none';
            optPanel.appendChild(fileInput);

            document.getElementById('ieMaskLoadFile').onclick = () => fileInput.click();
            fileInput.onchange = (e) => { if (e.target.files[0]) loadMaskFromFile(e.target.files[0]); };

            document.getElementById('ieMaskPaste').onclick = async () => {
                try {
                    const items = await navigator.clipboard.read();
                    for (const item of items) {
                        const imgType = item.types.find(t => t.startsWith('image/'));
                        if (imgType) { loadMaskFromFile(await item.getType(imgType)); return; }
                    }
                    Toolbox.showToast('클립보드에 이미지가 없습니다.', 'error');
                } catch { Toolbox.showToast('클립보드 읽기 실패', 'error'); }
            };

            document.getElementById('ieMaskInterpret').onchange = () => {
                if (maskPreviewActive) refreshMaskSplitPreview();
            };
            document.getElementById('ieMaskInvertCb').onchange = (e) => {
                maskInvert = e.target.checked;
                updateMaskThumbPreview();
                if (maskPreviewActive) refreshMaskSplitPreview();
            };
            document.getElementById('ieMaskPreviewBtn').onclick = showMaskSplitPreview;
            document.getElementById('ieMaskApplyBtn').onclick = applyMaskToImage;

            document.getElementById('ieSelfMaskTol').oninput = (e) => {
                document.getElementById('ieSelfMaskTolVal').textContent = e.target.value;
            };
            document.getElementById('ieSelfMaskFeather').oninput = (e) => {
                document.getElementById('ieSelfMaskFeatherVal').textContent = e.target.value;
            };
            document.getElementById('ieSelfMaskApply').onclick = applySelfMask;
        });
    }

    function loadMaskFromFile(fileOrBlob) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const cvs = document.createElement('canvas');
                cvs.width = img.naturalWidth; cvs.height = img.naturalHeight;
                cvs.getContext('2d').drawImage(img, 0, 0);
                maskImageData = cvs;
                updateMaskThumbPreview();
                const applyBtn = document.getElementById('ieMaskApplyBtn');
                const prevBtn = document.getElementById('ieMaskPreviewBtn');
                if (applyBtn) applyBtn.disabled = false;
                if (prevBtn) prevBtn.disabled = false;
                document.getElementById('ieMaskPreviewWrap').style.display = 'flex';
                document.getElementById('ieMaskInvertLabel').style.display = 'flex';
                Toolbox.showToast('마스크 이미지 로드 완료');
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(fileOrBlob);
    }

    function updateMaskThumbPreview() {
        if (!maskImageData) return;
        const preview = document.getElementById('ieMaskPreview');
        if (!preview) return;
        const w = maskImageData.width, h = maskImageData.height;
        const scale = Math.min(60 / w, 40 / h, 1);
        preview.width = Math.round(w * scale); preview.height = Math.round(h * scale);
        const pc = preview.getContext('2d');
        pc.drawImage(maskImageData, 0, 0, preview.width, preview.height);
        if (maskInvert) {
            pc.globalCompositeOperation = 'difference';
            pc.fillStyle = '#fff';
            pc.fillRect(0, 0, preview.width, preview.height);
            pc.globalCompositeOperation = 'source-over';
        }
        const info = document.getElementById('ieMaskInfo');
        if (info) info.textContent = `${w}×${h}${maskInvert ? ' (반전)' : ''}`;
    }

    function computeMaskedImageData() {
        const w = canvas.width, h = canvas.height;
        const useAlpha = (document.getElementById('ieMaskInterpret')?.value || 'alpha') === 'alpha';
        const src = ctx.getImageData(0, 0, w, h);
        const maskCvs = document.createElement('canvas');
        maskCvs.width = w; maskCvs.height = h;
        maskCvs.getContext('2d').drawImage(maskImageData, 0, 0, w, h);
        const mask = maskCvs.getContext('2d').getImageData(0, 0, w, h).data;

        const out = new Uint8ClampedArray(w * h * 4);
        const srcD = src.data;
        for (let i = 0; i < out.length; i += 4) {
            let f;
            if (useAlpha) {
                f = mask[i + 3] / 255;
            } else {
                const lum = mask[i] * 0.299 + mask[i + 1] * 0.587 + mask[i + 2] * 0.114;
                f = lum / 255;
            }
            if (maskInvert) f = 1 - f;
            out[i] = srcD[i];
            out[i + 1] = srcD[i + 1];
            out[i + 2] = srcD[i + 2];
            out[i + 3] = Math.round(srcD[i + 3] * f);
        }
        return new ImageData(out, w, h);
    }

    function showMaskSplitPreview() {
        if (!requireImage() || !maskImageData) return;
        maskPreviewActive = true;
        const overlay = document.getElementById('ieMaskPreviewOverlay');
        const box = document.getElementById('ieMaskPreviewBox');
        const inner = document.getElementById('ieMaskPreviewInner');
        const layerBefore = document.getElementById('ieMaskLayerBefore');
        const layerAfter = document.getElementById('ieMaskLayerAfter');
        const beforeCvs = document.getElementById('ieMaskBefore');
        const afterCvs = document.getElementById('ieMaskAfter');
        const divider = document.getElementById('ieMaskPreviewDivider');
        const rangeInput = document.getElementById('ieMaskPreviewRange');
        const closeBtn = document.getElementById('ieMaskPreviewClose');
        if (!overlay || !box || !inner || !layerBefore || !layerAfter || !beforeCvs || !afterCvs || !divider || !rangeInput) return;

        overlay.classList.add('active');
        canvas.style.visibility = 'hidden';

        const w = canvas.width, h = canvas.height;
        beforeCvs.width = w; beforeCvs.height = h;
        afterCvs.width = w; afterCvs.height = h;

        beforeCvs.getContext('2d').putImageData(ctx.getImageData(0, 0, w, h), 0, 0);
        afterCvs.getContext('2d').putImageData(computeMaskedImageData(), 0, 0);

        const ow = overlay.clientWidth || 400, oh = overlay.clientHeight || 300;
        const pad = 60;
        const scale = Math.min((ow - 24) / w, (oh - pad) / h, 1);
        const dispW = Math.round(w * scale), dispH = Math.round(h * scale);
        inner.style.width = dispW + 'px';
        inner.style.height = dispH + 'px';
        beforeCvs.style.width = dispW + 'px';
        beforeCvs.style.height = dispH + 'px';
        afterCvs.style.width = dispW + 'px';
        afterCvs.style.height = dispH + 'px';
        box.style.width = dispW + 'px';

        let splitPct = 50;
        rangeInput.value = 50;

        function updateSplit() {
            const pct = splitPct;
            layerBefore.style.clipPath = 'inset(0 ' + (100 - pct) + '% 0 0)';
            layerAfter.style.clipPath = 'inset(0 0 0 ' + pct + '%)';
            divider.style.left = pct + '%';
            rangeInput.value = Math.round(pct);
        }
        updateSplit();

        rangeInput.oninput = () => {
            splitPct = parseInt(rangeInput.value, 10);
            updateSplit();
        };

        let dragging = false;
        function onMove(e) {
            if (!dragging) return;
            const rect = inner.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 100;
            splitPct = Math.max(0, Math.min(100, x));
            updateSplit();
        }
        function onUp() { dragging = false; document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); }
        divider.onmousedown = (e) => { e.preventDefault(); dragging = true; document.addEventListener('mousemove', onMove); document.addEventListener('mouseup', onUp); };

        closeBtn.onclick = () => {
            overlay.classList.remove('active');
            canvas.style.visibility = '';
            maskPreviewActive = false;
        };
    }

    function refreshMaskSplitPreview() {
        if (!maskPreviewActive || !maskImageData) return;
        const beforeCvs = document.getElementById('ieMaskBefore');
        const afterCvs = document.getElementById('ieMaskAfter');
        if (!beforeCvs || !afterCvs) return;
        const w = canvas.width, h = canvas.height;
        beforeCvs.getContext('2d').putImageData(ctx.getImageData(0, 0, w, h), 0, 0);
        afterCvs.getContext('2d').putImageData(computeMaskedImageData(), 0, 0);
    }

    function destroyMaskPreview() {
        const overlay = document.getElementById('ieMaskPreviewOverlay');
        if (overlay) overlay.classList.remove('active');
        canvas.style.visibility = '';
        maskPreviewActive = false;
    }

    function applyMaskToImage() {
        if (!requireImage()) return;
        if (!maskImageData) { Toolbox.showToast('마스크 이미지를 먼저 불러오세요.', 'error'); return; }

        const masked = computeMaskedImageData();
        ctx.putImageData(masked, 0, 0);

        destroyMaskPreview();
        pushHistory(); updateSizeLabel();
        Toolbox.showToast('마스크 적용 완료!');
        Mdd.setMood('cheer'); Mdd.say('마스크 적용했다냥!');
    }

    function applySelfMask() {
        if (!requireImage()) return;
        const target = document.getElementById('ieSelfMaskTarget')?.value || 'dark';
        const tol = parseInt(document.getElementById('ieSelfMaskTol')?.value || '30', 10);
        const feather = parseInt(document.getElementById('ieSelfMaskFeather')?.value || '10', 10);

        const w = canvas.width, h = canvas.height;
        const imgData = ctx.getImageData(0, 0, w, h);
        const d = imgData.data;

        for (let i = 0; i < d.length; i += 4) {
            const lum = d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114;
            let alpha;
            if (target === 'dark') {
                if (lum <= tol) { alpha = 0; }
                else if (feather > 0 && lum < tol + feather) { alpha = (lum - tol) / feather; }
                else { alpha = 1; }
            } else {
                const invLum = 255 - lum;
                if (invLum <= tol) { alpha = 0; }
                else if (feather > 0 && invLum < tol + feather) { alpha = (invLum - tol) / feather; }
                else { alpha = 1; }
            }
            d[i + 3] = Math.round(d[i + 3] * alpha);
        }
        ctx.putImageData(imgData, 0, 0);

        pushHistory(); updateSizeLabel();
        const label = target === 'dark' ? '어두운' : '밝은';
        Toolbox.showToast(`${label} 영역 제거 완료!`);
        Mdd.setMood('cheer'); Mdd.say('깔끔하게 날렸다냥!');
    }

    /* ===== Crop ===== */
    function initCrop() {
        const overlay = document.getElementById('ieCropOverlay');
        if (!overlay || !hasImage()) return;
        const wrap = document.getElementById('ieCanvasWrap');
        const rect = canvas.getBoundingClientRect();
        const wrapRect = wrap.getBoundingClientRect();

        overlay.style.left = (rect.left - wrapRect.left) + 'px';
        overlay.style.top = (rect.top - wrapRect.top) + 'px';
        overlay.style.width = rect.width + 'px';
        overlay.style.height = rect.height + 'px';
        overlay.classList.add('active');

        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        let cw, ch, cx, cy;
        if (cropAspect) {
            const displayAspect = rect.width / rect.height;
            if (cropAspect > displayAspect) {
                cw = rect.width * 0.8;
                ch = cw / cropAspect;
            } else {
                ch = rect.height * 0.8;
                cw = ch * cropAspect;
            }
            cx = (rect.width - cw) / 2;
            cy = (rect.height - ch) / 2;
        } else {
            const pad = 0.1;
            cx = Math.round(rect.width * pad);
            cy = Math.round(rect.height * pad);
            cw = Math.round(rect.width * (1 - 2 * pad));
            ch = Math.round(rect.height * (1 - 2 * pad));
        }

        cropState = {
            x: Math.round(cx), y: Math.round(cy),
            w: Math.round(cw), h: Math.round(ch),
            scaleX, scaleY,
            maxW: rect.width, maxH: rect.height,
            dragging: null, startX: 0, startY: 0, startCrop: null
        };
        renderCropRegion();
    }

    function destroyCrop() {
        const overlay = document.getElementById('ieCropOverlay');
        if (overlay) overlay.classList.remove('active');
        cropState = null;
    }

    function renderCropRegion() {
        if (!cropState) return;
        const region = document.getElementById('ieCropRegion');
        if (!region) return;
        region.style.left = cropState.x + 'px';
        region.style.top = cropState.y + 'px';
        region.style.width = cropState.w + 'px';
        region.style.height = cropState.h + 'px';

        const info = document.getElementById('ieCropInfo');
        if (info) {
            const rw = Math.round(cropState.w * cropState.scaleX);
            const rh = Math.round(cropState.h * cropState.scaleY);
            info.textContent = `선택 영역: ${rw} × ${rh}`;
        }
    }

    function constrainCropAspect(baseX, baseY, dx, dy) {
        if (!cropAspect) return { dx, dy };
        const adx = Math.abs(dx), ady = Math.abs(dy);
        const dominant = adx > ady ? 'x' : 'y';
        if (dominant === 'x') {
            dy = Math.sign(dy || 1) * (adx / cropAspect);
        } else {
            dx = Math.sign(dx || 1) * (ady * cropAspect);
        }
        return { dx, dy };
    }

    function onCropPointerDown(e) {
        if (!cropState) return;
        const overlay = document.getElementById('ieCropOverlay');
        const oRect = overlay.getBoundingClientRect();
        const mx = e.clientX - oRect.left;
        const my = e.clientY - oRect.top;

        const target = e.target;
        if (target.classList.contains('ie-crop-handle')) {
            cropState.dragging = target.dataset.dir;
        } else if (target.id === 'ieCropRegion') {
            cropState.dragging = 'move';
        } else {
            cropState.dragging = 'new';
            cropState.x = mx;
            cropState.y = my;
            cropState.w = 0;
            cropState.h = 0;
        }
        cropState.startX = mx;
        cropState.startY = my;
        cropState.startCrop = { x: cropState.x, y: cropState.y, w: cropState.w, h: cropState.h };
        e.preventDefault();
    }

    function onCropPointerMove(e) {
        if (!cropState || !cropState.dragging) return;
        const overlay = document.getElementById('ieCropOverlay');
        const oRect = overlay.getBoundingClientRect();
        const mx = e.clientX - oRect.left;
        const my = e.clientY - oRect.top;
        const dx = mx - cropState.startX;
        const dy = my - cropState.startY;
        const s = cropState.startCrop;
        const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

        if (cropState.dragging === 'move') {
            cropState.x = clamp(s.x + dx, 0, cropState.maxW - s.w);
            cropState.y = clamp(s.y + dy, 0, cropState.maxH - s.h);
        } else if (cropState.dragging === 'new') {
            let rawW = mx - cropState.startX;
            let rawH = my - cropState.startY;
            if (cropAspect) {
                const c = constrainCropAspect(0, 0, rawW, rawH);
                rawW = c.dx; rawH = c.dy;
            }
            const x1 = clamp(Math.min(cropState.startX, cropState.startX + rawW), 0, cropState.maxW);
            const y1 = clamp(Math.min(cropState.startY, cropState.startY + rawH), 0, cropState.maxH);
            const x2 = clamp(Math.max(cropState.startX, cropState.startX + rawW), 0, cropState.maxW);
            const y2 = clamp(Math.max(cropState.startY, cropState.startY + rawH), 0, cropState.maxH);
            cropState.x = x1; cropState.y = y1;
            cropState.w = x2 - x1; cropState.h = y2 - y1;
        } else {
            const d = cropState.dragging;
            let nx = s.x, ny = s.y, nw = s.w, nh = s.h;
            if (cropAspect) {
                if (d === 'se') {
                    const c = constrainCropAspect(0, 0, dx, dy);
                    nw = clamp(s.w + c.dx, 10, cropState.maxW - s.x);
                    nh = nw / cropAspect;
                } else if (d === 'nw') {
                    const c = constrainCropAspect(0, 0, -dx, -dy);
                    nw = clamp(s.w + c.dx, 10, s.x + s.w);
                    nh = nw / cropAspect;
                    nx = s.x + s.w - nw;
                    ny = s.y + s.h - nh;
                } else if (d === 'ne') {
                    const c = constrainCropAspect(0, 0, dx, -dy);
                    nw = clamp(s.w + c.dx, 10, cropState.maxW - s.x);
                    nh = nw / cropAspect;
                    ny = s.y + s.h - nh;
                } else if (d === 'sw') {
                    const c = constrainCropAspect(0, 0, -dx, dy);
                    nw = clamp(s.w + c.dx, 10, s.x + s.w);
                    nh = nw / cropAspect;
                    nx = s.x + s.w - nw;
                }
            } else {
                if (d.includes('w')) { nx = clamp(s.x + dx, 0, s.x + s.w - 10); nw = s.w - (nx - s.x); }
                if (d.includes('e')) { nw = clamp(s.w + dx, 10, cropState.maxW - s.x); }
                if (d.includes('n')) { ny = clamp(s.y + dy, 0, s.y + s.h - 10); nh = s.h - (ny - s.y); }
                if (d.includes('s')) { nh = clamp(s.h + dy, 10, cropState.maxH - s.y); }
            }
            cropState.x = nx; cropState.y = ny; cropState.w = nw; cropState.h = nh;
        }
        renderCropRegion();
        e.preventDefault();
    }

    function onCropPointerUp() {
        if (cropState) cropState.dragging = null;
    }

    function applyCrop() {
        if (!cropState || !requireImage()) return;
        const sx = Math.round(cropState.x * cropState.scaleX);
        const sy = Math.round(cropState.y * cropState.scaleY);
        const sw = Math.max(1, Math.round(cropState.w * cropState.scaleX));
        const sh = Math.max(1, Math.round(cropState.h * cropState.scaleY));

        const imgData = ctx.getImageData(sx, sy, sw, sh);
        canvas.width = sw;
        canvas.height = sh;
        ctx.putImageData(imgData, 0, 0);
        pushHistory();
        updateSizeLabel();
        destroyCrop();
        if (activeTool === 'crop') initCrop();
        Toolbox.showToast('자르기 완료');
    }

    /* ===== Export ===== */
    function exportDownload(format) {
        if (!requireImage()) return;
        const mime = format === 'jpeg' ? 'image/jpeg' : 'image/png';
        const quality = format === 'jpeg' ? jpegQuality : undefined;
        canvas.toBlob(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'edited-image-' + Date.now() + '.' + format;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            const sizeKB = (blob.size / 1024).toFixed(1);
            Toolbox.showToast('다운로드 시작 (' + sizeKB + ' KB)');
        }, mime, quality);
    }

    async function exportClipboard() {
        if (!requireImage()) return;
        try {
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
            await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
            Toolbox.showToast('클립보드에 복사됨');
        } catch (e) {
            Toolbox.showToast('클립보드 복사 실패', 'error', e);
        }
    }

    async function exportToLibrary() {
        if (!requireImage()) return;
        try {
            const dataUrl = canvas.toDataURL('image/png');
            await ImageDB.save({
                id: 'edit_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6),
                url: dataUrl,
                prompt: '(이미지 편집)',
                model: 'imageedit',
                modelName: '이미지 편집',
                timestamp: Date.now(),
                tokens: null,
                elapsed: null
            });
            Toolbox.showToast('라이브러리에 저장됨');
            Mdd.setMood('cheer'); Mdd.say('라이브러리에 저장했다냥!');
        } catch (e) {
            Toolbox.showToast('저장 실패', 'error', e);
        }
    }

    /* ===== Tool Switching ===== */
    function selectTool(toolId) {
        if (hasPendingPreview && toolId !== activeTool) {
            if (!confirm('적용하지 않은 변경사항이 있습니다. 무시하고 전환하시겠습니까?')) return;
        }
        activeTool = toolId;
        canvas.style.filter = '';
        canvas.style.transform = '';
        hasPendingPreview = false;
        freeRotateDeg = 0;
        destroyCrop();
        destroyBrush();
        destroyCaptionOverlay();
        destroyStickerOverlay();
        destroyMaskPreview();

        document.querySelectorAll('.ie-tool-btn').forEach(b => b.classList.toggle('active', b.dataset.tool === toolId));

        const optPanel = document.getElementById('ieOptions');
        if (!optPanel) return;

        switch (toolId) {
            case 'crop': buildCropOptions(optPanel); if (hasImage()) initCrop(); break;
            case 'resize': buildResizeOptions(optPanel); break;
            case 'rotate': buildRotateOptions(optPanel); break;
            case 'adjust': buildAdjustOptions(optPanel); break;
            case 'filter': buildFilterOptions(optPanel); break;
            case 'rembg': buildRembgOptions(optPanel); break;
            case 'mask': buildMaskOptions(optPanel); break;
            case 'caption': buildCaptionOptions(optPanel); break;
            case 'sticker': buildStickerOptions(optPanel); break;
            default: optPanel.innerHTML = '';
        }
    }

    /* ===== Dialogs ===== */
    function showUrlDialog() {
        const d = document.getElementById('ieUrlDialog');
        if (d) { d.classList.add('open'); document.getElementById('ieUrlInput').value = ''; document.getElementById('ieUrlInput').focus(); }
    }

    function showLibDialog() {
        const d = document.getElementById('ieLibDialog');
        if (!d) return;
        d.classList.add('open');
        const grid = document.getElementById('ieLibGrid');
        grid.innerHTML = '<div class="ie-lib-empty">로딩 중...</div>';
        ImageDB.getAll().then(items => {
            if (items.length === 0) {
                grid.innerHTML = '<div class="ie-lib-empty">라이브러리가 비어 있습니다.</div>';
                return;
            }
            grid.innerHTML = '';
            items.forEach(item => {
                const card = document.createElement('div');
                card.className = 'ie-lib-thumb-card';
                card.innerHTML = '<img src="' + Toolbox.escapeHtml(item.url) + '" alt="" loading="lazy">';
                card.onclick = () => {
                    loadImageFromSrc(item.url);
                    d.classList.remove('open');
                };
                grid.appendChild(card);
            });
        }).catch(() => {
            grid.innerHTML = '<div class="ie-lib-empty">라이브러리를 불러올 수 없습니다.</div>';
        });
    }

    /* ===== Build ===== */
    function buildEditor(container) {
        Mdd.setMood('happy'); Mdd.say('이미지 편집이다냥!');

        container.innerHTML = `
            <div class="ie-layout">
                <div class="ie-toolbar">
                    <div class="ie-toolbar-group">
                        <button class="ie-tb-btn" id="ieUndoBtn" disabled title="되돌리기 (Ctrl+Z)">↩ 되돌리기</button>
                        <button class="ie-tb-btn" id="ieRedoBtn" disabled title="다시실행 (Ctrl+Y)">↪ 다시실행</button>
                        <button class="ie-tb-btn" id="ieResetBtn" title="원본으로 복원">⟲ 리셋</button>
                    </div>
                    <span class="ie-size-label" id="ieSizeLabel"></span>
                    <span class="ie-spacer"></span>
                    <div class="ie-toolbar-group">
                        <div class="ie-dropdown">
                            <button class="ie-tb-btn" id="ieImportBtn">📥 가져오기 ▾</button>
                            <div class="ie-dropdown-menu" id="ieImportMenu">
                                <button class="ie-dropdown-item" id="ieImportFile">📁 파일 업로드</button>
                                <button class="ie-dropdown-item" id="ieImportUrl">🔗 URL 입력</button>
                                <button class="ie-dropdown-item" id="ieImportLib">🖼️ 라이브러리에서</button>
                            </div>
                        </div>
                        <div class="ie-dropdown">
                            <button class="ie-tb-btn accent" id="ieExportBtn">📤 내보내기 ▾</button>
                            <div class="ie-dropdown-menu" id="ieExportMenu">
                                <button class="ie-dropdown-item" id="ieExDlPng">⬇️ PNG 다운로드</button>
                                <button class="ie-dropdown-item" id="ieExDlJpg">⬇️ JPEG 다운로드</button>
                                <div style="padding:6px 14px;display:flex;align-items:center;gap:6px;">
                                    <span style="font-size:var(--font-size-2xs);color:var(--text-tertiary);">JPEG 품질</span>
                                    <input type="range" id="ieJpegQuality" min="10" max="100" value="92" style="width:80px;accent-color:var(--accent);">
                                    <span id="ieJpegQualityVal" style="font-size:var(--font-size-2xs);color:var(--text-tertiary);font-family:monospace;min-width:28px;">92%</span>
                                </div>
                                <button class="ie-dropdown-item" id="ieExClip">📋 클립보드 복사</button>
                                <button class="ie-dropdown-item" id="ieExLib">🖼️ 라이브러리에 저장</button>
                            </div>
                        </div>
                    </div>
                    <input type="file" id="ieFileInput" accept="image/*" style="display:none">
                </div>

                <div class="ie-body">
                    <div class="ie-tools">
                        <button class="ie-tool-btn active" data-tool="crop" title="자르기">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/></svg>
                        </button>
                        <button class="ie-tool-btn" data-tool="resize" title="크기 조절">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
                        </button>
                        <button class="ie-tool-btn" data-tool="rotate" title="회전/뒤집기">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
                        </button>
                        <button class="ie-tool-btn" data-tool="adjust" title="밝기/대비/채도">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 0 20z"/></svg>
                        </button>
                        <button class="ie-tool-btn" data-tool="filter" title="필터">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>
                        </button>
                        <button class="ie-tool-btn" data-tool="rembg" title="배경 제거 (AI)">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 21l9.5-9.5"/><path d="M15 4l1 2 2 1-2 1-1 2-1-2-2-1 2-1z"/><path d="M19 10l.7 1.3 1.3.7-1.3.7-.7 1.3-.7-1.3-1.3-.7 1.3-.7z"/></svg>
                        </button>
                        <button class="ie-tool-btn" data-tool="mask" title="마스크 적용">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="12" cy="12" r="5"/><path d="M12 7v0M12 17v0M7 12h0M17 12h0"/></svg>
                        </button>
                        <button class="ie-tool-btn" data-tool="caption" title="캡션">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 7h16M4 12h10M4 17h12"/></svg>
                        </button>
                        <button class="ie-tool-btn" data-tool="sticker" title="스티커">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M8 12h8M12 8v8"/></svg>
                        </button>
                    </div>
                    <div class="ie-canvas-area">
                        <div class="ie-canvas-wrap" id="ieCanvasWrap">
                            <canvas id="ieCanvas"></canvas>
                            <div class="ie-placeholder" id="iePlaceholder">
                                <div class="ie-placeholder-icon">🖼️</div>
                                <div class="ie-placeholder-text">이미지를 불러오세요</div>
                                <div class="ie-placeholder-sub">클릭, 드래그&드롭, Ctrl+V 붙여넣기, 또는 '가져오기' 사용</div>
                            </div>
                            <canvas class="ie-brush-overlay" id="ieBrushOverlay" style="display:none;"></canvas>
                            <canvas class="ie-caption-overlay" id="ieCaptionOverlay" style="display:none;"></canvas>
                            <canvas class="ie-sticker-overlay" id="ieStickerOverlay" style="display:none;"></canvas>
                            <div class="ie-crop-overlay" id="ieCropOverlay">
                                <div class="ie-crop-region" id="ieCropRegion">
                                    <div class="ie-crop-handle nw" data-dir="nw"></div>
                                    <div class="ie-crop-handle ne" data-dir="ne"></div>
                                    <div class="ie-crop-handle sw" data-dir="sw"></div>
                                    <div class="ie-crop-handle se" data-dir="se"></div>
                                </div>
                            </div>
                            <div class="ie-mask-preview-overlay" id="ieMaskPreviewOverlay">
                                <div class="ie-mask-preview-box" id="ieMaskPreviewBox">
                                    <div class="ie-mask-preview-imgwrap" id="ieMaskPreviewImgwrap">
                                        <div class="ie-mask-preview-inner" id="ieMaskPreviewInner">
                                            <div class="ie-mask-preview-layer" id="ieMaskLayerBefore">
                                                <canvas id="ieMaskBefore"></canvas>
                                            </div>
                                            <div class="ie-mask-preview-layer" id="ieMaskLayerAfter">
                                                <canvas id="ieMaskAfter"></canvas>
                                            </div>
                                            <div class="ie-mask-preview-divider" id="ieMaskPreviewDivider"></div>
                                        </div>
                                    </div>
                                    <div class="ie-mask-preview-sliderrow" id="ieMaskPreviewSliderrow">
                                        <span>적용 전</span>
                                        <input type="range" id="ieMaskPreviewRange" min="0" max="100" value="50">
                                        <span>적용 후</span>
                                        <button class="ie-mask-preview-close" id="ieMaskPreviewClose">닫기</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="ie-options" id="ieOptions"></div>
                    </div>
                </div>
            </div>

            <div class="ie-url-dialog" id="ieUrlDialog">
                <div class="ie-url-dialog-box">
                    <div class="ie-url-dialog-title">URL에서 이미지 불러오기</div>
                    <input type="text" class="ie-url-dialog-input" id="ieUrlInput" placeholder="https://example.com/image.png">
                    <div class="ie-url-dialog-actions">
                        <button class="ie-tb-btn" id="ieUrlCancel">취소</button>
                        <button class="ie-tb-btn accent" id="ieUrlConfirm">불러오기</button>
                    </div>
                </div>
            </div>

            <div class="ie-lib-dialog" id="ieLibDialog">
                <div class="ie-lib-dialog-box">
                    <div class="ie-lib-dialog-title">라이브러리에서 이미지 선택</div>
                    <div class="ie-lib-grid" id="ieLibGrid"></div>
                    <div style="text-align:right; margin-top:12px;">
                        <button class="ie-tb-btn" id="ieLibClose">닫기</button>
                    </div>
                </div>
            </div>

            <div class="ie-warn-dialog" id="ieWarnDialog">
                <div class="ie-warn-dialog-box">
                    <div class="ie-warn-dialog-title">⚠️ 큰 이미지 감지</div>
                    <div class="ie-warn-dialog-text" id="ieWarnText"></div>
                    <div class="ie-warn-dialog-actions">
                        <button class="ie-tb-btn" id="ieWarnCancel">취소</button>
                        <button class="ie-tb-btn" id="ieWarnOriginal">원본 유지</button>
                        <button class="ie-tb-btn accent" id="ieWarnDownscale">축소</button>
                    </div>
                </div>
            </div>`;

        requestAnimationFrame(() => {
            canvas = document.getElementById('ieCanvas');
            ctx = canvas.getContext('2d');
            canvas.width = 0;
            canvas.height = 0;

            /* Toolbar buttons */
            document.getElementById('ieUndoBtn').onclick = undo;
            document.getElementById('ieRedoBtn').onclick = redo;
            document.getElementById('ieResetBtn').onclick = () => {
                if (!hasImage()) return;
                const first = history[0];
                if (!first) return;
                history = [];
                historyIdx = -1;
                canvas.width = first.w;
                canvas.height = first.h;
                ctx.putImageData(first.data, 0, 0);
                pushHistory();
                updateSizeLabel();
                canvas.style.filter = '';
                Toolbox.showToast('원본으로 복원');
                selectTool(activeTool);
            };

            /* Dropdown toggles — fix: close other menus first */
            const setupDropdown = (btnId, menuId) => {
                const btn = document.getElementById(btnId);
                const menu = document.getElementById(menuId);
                btn.onclick = (e) => {
                    e.stopPropagation();
                    const wasOpen = menu.classList.contains('open');
                    closeAllDropdowns();
                    if (!wasOpen) menu.classList.add('open');
                };
            };
            setupDropdown('ieImportBtn', 'ieImportMenu');
            setupDropdown('ieExportBtn', 'ieExportMenu');

            document.addEventListener('click', closeAllDropdowns);

            /* JPEG quality slider */
            const jpegSlider = document.getElementById('ieJpegQuality');
            const jpegVal = document.getElementById('ieJpegQualityVal');
            jpegSlider.oninput = () => {
                jpegQuality = parseInt(jpegSlider.value) / 100;
                jpegVal.textContent = jpegSlider.value + '%';
            };
            jpegSlider.onclick = (e) => e.stopPropagation();

            /* Import actions */
            const fileInput = document.getElementById('ieFileInput');
            document.getElementById('ieImportFile').onclick = () => fileInput.click();
            fileInput.onchange = () => {
                if (fileInput.files[0]) loadImageFromFile(fileInput.files[0]);
                fileInput.value = '';
            };
            document.getElementById('ieImportUrl').onclick = showUrlDialog;
            document.getElementById('ieImportLib').onclick = showLibDialog;

            /* Export actions */
            document.getElementById('ieExDlPng').onclick = () => exportDownload('png');
            document.getElementById('ieExDlJpg').onclick = () => exportDownload('jpeg');
            document.getElementById('ieExClip').onclick = exportClipboard;
            document.getElementById('ieExLib').onclick = exportToLibrary;

            /* Tool buttons */
            document.querySelectorAll('.ie-tool-btn').forEach(btn => {
                btn.onclick = () => selectTool(btn.dataset.tool);
            });

            /* Placeholder click => file picker */
            document.getElementById('iePlaceholder').onclick = () => fileInput.click();

            /* URL dialog */
            document.getElementById('ieUrlCancel').onclick = () => document.getElementById('ieUrlDialog').classList.remove('open');
            document.getElementById('ieUrlConfirm').onclick = () => {
                const url = document.getElementById('ieUrlInput').value.trim();
                if (!url) { Toolbox.showToast('URL을 입력하세요.', 'error'); return; }
                loadImageFromSrc(url);
                document.getElementById('ieUrlDialog').classList.remove('open');
            };
            document.getElementById('ieUrlInput').onkeydown = (e) => {
                if (e.key === 'Enter') document.getElementById('ieUrlConfirm').click();
            };
            document.getElementById('ieUrlDialog').onclick = (e) => {
                if (e.target.id === 'ieUrlDialog') document.getElementById('ieUrlDialog').classList.remove('open');
            };

            /* Library dialog */
            document.getElementById('ieLibClose').onclick = () => document.getElementById('ieLibDialog').classList.remove('open');
            document.getElementById('ieLibDialog').onclick = (e) => {
                if (e.target.id === 'ieLibDialog') document.getElementById('ieLibDialog').classList.remove('open');
            };

            /* Drag & Drop */
            const wrap = document.getElementById('ieCanvasWrap');
            wrap.ondragover = (e) => { e.preventDefault(); wrap.classList.add('dragover'); };
            wrap.ondragleave = () => wrap.classList.remove('dragover');
            wrap.ondrop = (e) => {
                e.preventDefault();
                wrap.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file) loadImageFromFile(file);
            };

            /* Clipboard paste */
            document.addEventListener('paste', (e) => {
                const page = document.getElementById('page-imageedit');
                if (!page || !page.classList.contains('active')) return;
                const items = e.clipboardData?.items;
                if (!items) return;
                for (let i = 0; i < items.length; i++) {
                    if (items[i].type.startsWith('image/')) {
                        loadImageFromFile(items[i].getAsFile());
                        e.preventDefault();
                        return;
                    }
                }
            });

            /* Keyboard shortcuts */
            document.addEventListener('keydown', (e) => {
                const page = document.getElementById('page-imageedit');
                if (!page || !page.classList.contains('active')) return;
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
                if (e.ctrlKey && e.key === 'z') { e.preventDefault(); undo(); }
                if (e.ctrlKey && e.key === 'y') { e.preventDefault(); redo(); }
            });

            /* Crop pointer events */
            const overlay = document.getElementById('ieCropOverlay');
            overlay.onpointerdown = onCropPointerDown;
            overlay.onpointermove = onCropPointerMove;
            overlay.onpointerup = onCropPointerUp;

            /* Chromakey color picker — canvas click */
            wrap.addEventListener('click', onCanvasClickForChroma);

            /* Window resize => re-init crop/brush/caption if active */
            window.addEventListener('resize', () => {
                if (activeTool === 'crop' && hasImage()) {
                    destroyCrop();
                    setTimeout(() => initCrop(), 100);
                }
                if (activeTool === 'rembg' && rembgMode === 'brush' && hasImage()) {
                    destroyBrush();
                    setTimeout(() => initBrushOverlay(), 100);
                }
                if (activeTool === 'caption' && hasImage()) {
                    destroyCaptionOverlay();
                    setTimeout(() => initCaptionOverlay(), 100);
                }
                if (activeTool === 'sticker' && hasImage()) {
                    destroyStickerOverlay();
                    setTimeout(() => initStickerOverlay(), 100);
                }
            });

            /* Initial tool */
            selectTool('crop');
        });
    }

    /* ===== Register ===== */
    Toolbox.register({
        id: 'imageedit',
        title: '이미지 편집',
        category: 'feature',
        desc: 'AI로 이미지를 편집·변형합니다',
        layout: 'full',
        icon: '<rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M9 3v18" stroke="currentColor" stroke-width="1.5"/><path d="M3 15h18" stroke="currentColor" stroke-width="1.5"/><circle cx="15" cy="9" r="2" stroke="currentColor" stroke-width="1.5" fill="none"/>',
        tabs: [{ id: 'imageedit-main', label: '편집', build: buildEditor }]
    });
})();
