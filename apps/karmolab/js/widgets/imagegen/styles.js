(function(){ 'use strict'; if(typeof Mdd!=='undefined') Mdd.injectCSS('imagegen', `
        /* ===== 생성 탭 ===== */
        .ig-layout { display:flex; gap:24px; flex:1; min-height:0; }
        .ig-sidebar {
            width:300px; flex-shrink:0; display:flex; flex-direction:column;
            gap:6px; overflow-y:auto; padding-right:4px; align-items:stretch;
            min-height:0;
        }
        .ig-sidebar .field-group { margin:0; padding:4px 0; flex-shrink:0; }
        .ig-canvas { flex:1; display:flex; flex-direction:column; min-width:0; }

        .ig-api-row { display:flex; gap:8px; align-items:center; }
        .ig-api-row input { flex:1; }

        .ig-preset-btns { display:flex; gap:6px; flex-wrap:wrap; }
        .ig-preset-btn {
            width:40px; height:40px; display:flex; align-items:center; justify-content:center;
            font-size:20px; background:var(--bg-tertiary); border:1px solid var(--border);
            border-radius:var(--radius-sm); cursor:pointer; transition:all var(--transition);
        }
        .ig-preset-btn:hover { border-color:var(--accent); background:var(--bg-hover); transform:scale(1.05); }
        .ig-preset-btn:active { transform:scale(0.98); }

        .ig-preset-popup { display:none; position:fixed; inset:0; z-index:9997; background:rgba(0,0,0,0.5); backdrop-filter:blur(4px); align-items:center; justify-content:center; padding:20px; }
        .ig-preset-popup.open { display:flex; }
        .ig-preset-panel {
            width:min(90vw, 420px); max-height:85vh;
            background:var(--bg-secondary); border:1px solid var(--border);
            border-radius:var(--radius-lg); overflow:hidden; display:flex; flex-direction:column;
        }
        .ig-preset-popup-header {
            display:flex; justify-content:space-between; align-items:center;
            padding:12px 16px; border-bottom:1px solid var(--border); flex-shrink:0;
        }
        .ig-preset-popup-header h3 { margin:0; font-size:14px; font-weight:600; }
        .ig-preset-popup-tabs { display:flex; flex-wrap:wrap; gap:4px; padding:8px 16px; border-bottom:1px solid var(--border); flex-shrink:0; }
        .ig-preset-tab-btn {
            padding:6px 10px; font-size:var(--font-size-2xs); font-weight:500;
            background:var(--bg-tertiary); border:1px solid var(--border);
            border-radius:var(--radius-sm); color:var(--text-secondary);
            cursor:pointer; transition:all var(--transition); font-family:inherit;
        }
        .ig-preset-tab-btn:hover { color:var(--text-primary); border-color:var(--accent); }
        .ig-preset-tab-btn.active { background:var(--accent); color:#fff; border-color:var(--accent); }
        .ig-preset-popup-body { overflow-y:auto; padding:12px; flex:1; min-height:0; }
        .ig-preset-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }
        .ig-card {
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            padding:10px 4px; border-radius:var(--radius-sm); border:1px solid var(--border);
            background:var(--bg-tertiary); cursor:pointer; transition:all var(--transition);
            gap:4px; text-align:center; min-height:68px;
        }
        .ig-card:hover { border-color:var(--accent); background:var(--bg-hover); }
        .ig-card.selected { border-color:var(--accent); background:var(--accent-subtle); }
        .ig-card-icon { font-size:20px; }
        .ig-card-label { font-size:var(--font-size-2xs); font-weight:500; color:var(--text-secondary); line-height:1.2; }

        .ig-preview {
            flex:1; display:flex; align-items:center; justify-content:center;
            background:var(--bg-tertiary); border:1px solid var(--border);
            border-radius:var(--radius-lg); min-height:0; position:relative; overflow:hidden;
        }
        .ig-preview img { max-width:100%; max-height:100%; border-radius:var(--radius-md); cursor:zoom-in; transition:transform 0.2s; }
        .ig-preview img:hover { transform:scale(1.02); }
        .ig-placeholder { color:var(--text-tertiary); font-size:var(--font-size-sm); text-align:center; }
        .ig-placeholder span { display:block; font-size:40px; margin-bottom:8px; opacity:0.3; }

        .ig-spinner { width:32px; height:32px; border:3px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.8s linear infinite; }
        .ig-loading-text { font-size:var(--font-size-xs); color:var(--text-tertiary); margin-top:8px; }

        .ig-input-area { margin-top:12px; }
        .ig-input-row { display:flex; gap:8px; }
        .ig-input-row textarea { flex:1; min-height:60px; resize:none; }
        .ig-enhance-btn { white-space:nowrap; font-size:var(--font-size-xs); transition:all 0.2s; }
        .ig-enhance-btn:disabled { opacity:0.6; cursor:wait; color:var(--accent); }
        .ig-enhancing {
            opacity:0.5; pointer-events:none;
            background:repeating-linear-gradient(
                -45deg, transparent, transparent 8px,
                var(--bg-hover) 8px, var(--bg-hover) 16px
            );
            background-size:200% 100%;
            animation:ig-enhance-stripe 1s linear infinite;
        }
        @keyframes ig-enhance-stripe { to { background-position:-32px 0; } }
        .ig-enhanced-flash { animation:ig-flash 0.6s ease; }
        @keyframes ig-flash {
            0% { box-shadow:0 0 0 0 var(--accent); }
            40% { box-shadow:0 0 0 4px var(--accent); }
            100% { box-shadow:0 0 0 0 transparent; }
        }
        .ig-vibe-info {
            margin-top:6px; padding:8px 10px; border-radius:var(--radius-sm);
            background:var(--bg-tertiary); border:1px solid var(--border);
            font-size:var(--font-size-2xs); line-height:1.5; color:var(--text-secondary);
            transition:all 0.2s;
        }
        .ig-vibe-info:empty { display:none; }
        .ig-ref-card {
            margin-top:12px; padding:8px 10px; border-radius:var(--radius-sm);
            background:var(--bg-tertiary); border:1px solid var(--border);
            font-size:var(--font-size-2xs); color:var(--text-tertiary);
        }
        .ig-ref-label { font-weight:600; margin-bottom:4px; color:var(--text-secondary); }
        .ig-ref-card a { color:var(--accent); text-decoration:none; margin-right:12px; }
        .ig-ref-card a:hover { text-decoration:underline; }
        .ig-vibe-info .ig-vibe-suffix {
            display:block; margin-top:4px; padding:4px 6px;
            background:var(--bg-primary); border-radius:3px;
            font-family:monospace; font-size:var(--font-size-2xs); color:var(--text-tertiary);
            word-break:break-all;
        }

        .ig-gen-btn { width:64px; flex-direction:column; gap:4px; font-size:var(--font-size-xs); }
        .ig-gen-btn span { font-size:18px; }
        .ig-gen-btn:disabled { opacity:0.5; cursor:not-allowed; }

        .ig-actions { display:flex; justify-content:flex-end; align-items:center; gap:8px; margin-top:6px; }
        .ig-token-display { font-size:var(--font-size-2xs); color:var(--text-tertiary); font-family:monospace; }
        .ig-meta-display { font-size:var(--font-size-2xs); color:var(--text-tertiary); flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

        .ig-gallery { display:flex; gap:6px; margin-top:10px; overflow-x:auto; padding:4px 0; }
        .ig-thumb {
            width:56px; height:56px; border-radius:var(--radius-sm); border:2px solid var(--border);
            cursor:pointer; object-fit:cover; transition:all var(--transition); flex-shrink:0; opacity:0.7;
        }
        .ig-thumb:hover { opacity:1; border-color:var(--accent); }
        .ig-thumb.active { opacity:1; border-color:var(--accent); box-shadow:0 0 8px rgba(139,124,246,0.4); }

        /* ===== 라이트박스 ===== */
        .ig-lightbox {
            display:none; position:fixed; inset:0; z-index:9999;
            background:rgba(0,0,0,0.92); backdrop-filter:blur(4px);
            align-items:center; justify-content:center; flex-direction:column;
        }
        .ig-lightbox.open { display:flex; }
        .ig-lightbox img { max-width:92vw; max-height:85vh; border-radius:var(--radius-md); }
        .ig-lightbox-actions { margin-top:16px; display:flex; gap:12px; }

        /* ===== 프롬프트 히스토리 ===== */
        .ig-slot-section { display:flex; flex-direction:column; gap:8px; margin-bottom:10px; padding:10px 12px; background:var(--bg-tertiary); border:1px solid var(--border); border-radius:var(--radius-sm); }
        .ig-slot-header { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px; }
        .ig-slot-context { font-size:var(--font-size-xs); font-weight:600; color:var(--text-secondary); }
        .ig-slot-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
        .ig-slot-label { font-size:var(--font-size-xs); font-weight:600; color:var(--text-secondary); min-width:70px; }
        .ig-slot-section input, .ig-slot-section select { flex:1; min-width:120px; padding:6px 10px; font-size:var(--font-size-xs); border-radius:var(--radius-sm); border:1px solid var(--border); background:var(--bg-primary); color:var(--text-primary); }
        .ig-slot-select-wrap { display:flex; align-items:center; gap:6px; flex:1; min-width:120px; }
        .ig-slot-select-wrap select { flex:1; }
        .ig-slot-custom-input { flex:1 1 100%; min-width:100%; margin-top:4px; }
        .ig-add-char-form { display:flex; flex-direction:column; gap:6px; margin:8px 0; padding:10px; background:var(--bg-primary); border:1px dashed var(--border); border-radius:var(--radius-sm); }
        .ig-add-char-row { display:flex; gap:6px; }
        .ig-add-char-form textarea { min-height:60px; resize:vertical; font-size:var(--font-size-xs); padding:6px 8px; }
        .ig-add-char-actions { display:flex; gap:8px; }
        .ig-history-dropdown { display:none; position:absolute; bottom:100%; left:0; right:0; max-height:200px; overflow-y:auto; background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-sm); box-shadow:0 -4px 16px rgba(0,0,0,0.3); z-index:10; margin-bottom:4px; }
        .ig-history-dropdown.open { display:block; }
        .ig-history-item { padding:8px 12px; font-size:var(--font-size-xs); color:var(--text-secondary); cursor:pointer; border-bottom:1px solid var(--border); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; transition:background 0.1s; }
        .ig-history-item:hover { background:var(--bg-hover); color:var(--text-primary); }
        .ig-history-item:last-child { border-bottom:none; }
        .ig-history-empty { padding:12px; text-align:center; font-size:var(--font-size-xs); color:var(--text-tertiary); }

        /* ===== 커스텀 프리셋 폼 ===== */
        .ig-custom-form { display:flex; flex-direction:column; gap:6px; margin-top:8px; padding:8px; background:var(--bg-primary); border:1px solid var(--border); border-radius:var(--radius-sm); }
        .ig-custom-form input, .ig-custom-form textarea { font-size:var(--font-size-xs); padding:6px 8px; }
        .ig-custom-form textarea { min-height:50px; resize:vertical; }
        .ig-custom-form-row { display:flex; gap:6px; }
        .ig-card-actions { display:flex; gap:4px; margin-top:4px; }

        /* ===== 큐 패널 ===== */
        .ig-queue-panel {
            margin-top:10px; border:1px solid var(--border); border-radius:var(--radius-md);
            background:var(--bg-secondary); overflow:hidden;
        }
        .ig-queue-header {
            display:flex; align-items:center; justify-content:space-between;
            padding:8px 12px; background:var(--bg-tertiary); border-bottom:1px solid var(--border);
        }
        .ig-queue-title { font-size:var(--font-size-xs); font-weight:600; color:var(--text-secondary); }
        .ig-queue-count { font-size:var(--font-size-2xs); color:var(--text-tertiary); margin-left:8px; }
        .ig-queue-clear { font-size:var(--font-size-2xs); color:var(--error); cursor:pointer; background:none; border:none; font-family:inherit; padding:2px 6px; border-radius:3px; }
        .ig-queue-clear:hover { background:var(--error-subtle); }
        .ig-queue-list { max-height:200px; overflow-y:auto; }
        .ig-q-item {
            display:flex; align-items:center; gap:8px; padding:8px 12px;
            border-bottom:1px solid var(--border); transition:background 0.15s;
        }
        .ig-q-item:last-child { border-bottom:none; }
        .ig-q-running { background:var(--accent-subtle); }
        .ig-q-done { opacity:0.6; }
        .ig-q-error { background:var(--error-subtle); }
        .ig-q-status { font-size:14px; flex-shrink:0; }
        .ig-q-body { flex:1; min-width:0; }
        .ig-q-prompt { font-size:var(--font-size-xs); color:var(--text-primary); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .ig-q-meta { font-size:var(--font-size-2xs); color:var(--text-tertiary); margin-top:2px; }
        .ig-q-cancel, .ig-q-remove {
            flex-shrink:0; width:22px; height:22px; border-radius:50%; border:1px solid var(--border);
            background:var(--bg-tertiary); color:var(--text-tertiary); font-size:var(--font-size-2xs);
            cursor:pointer; display:flex; align-items:center; justify-content:center;
            transition:all 0.15s; font-family:inherit; padding:0;
        }
        .ig-q-cancel:hover { border-color:var(--error); color:var(--error); background:var(--error-subtle); }
        .ig-q-remove:hover { border-color:var(--text-secondary); color:var(--text-primary); }

        @keyframes ig-q-pulse {
            0%, 100% { opacity:1; }
            50% { opacity:0.5; }
        }
        .ig-q-running .ig-q-status { animation:ig-q-pulse 1.5s ease-in-out infinite; }

        /* ===== API 이력 오버레이 ===== */
        .ig-api-history-overlay {
            display:none; position:fixed; inset:0; z-index:9998;
            background:rgba(0,0,0,0.6); backdrop-filter:blur(4px);
            align-items:center; justify-content:center; padding:20px;
        }
        .ig-api-history-overlay.open { display:flex; }
        .ig-api-history-panel {
            width:min(90vw, 720px); max-height:85vh;
            background:var(--bg-secondary); border:1px solid var(--border);
            border-radius:var(--radius-lg); overflow:hidden; display:flex; flex-direction:column;
        }
        .ig-api-history-header {
            display:flex; justify-content:space-between; align-items:center;
            padding:12px 16px; border-bottom:1px solid var(--border); flex-shrink:0;
        }
        .ig-api-history-header h3 { margin:0; font-size:14px; font-weight:600; }
        .ig-api-history-header div { display:flex; gap:8px; }
        .ig-api-history-list { overflow-y:auto; padding:12px; flex:1; min-height:0; }
        .ig-api-history-empty { padding:40px; text-align:center; color:var(--text-tertiary); font-size:13px; }
        .ig-api-history-card {
            border:1px solid var(--border); border-radius:var(--radius-sm);
            margin-bottom:8px; overflow:hidden;
        }
        .ig-api-history-card-head {
            display:flex; align-items:center; gap:8px; padding:8px 12px;
            cursor:pointer; background:var(--bg-tertiary); font-size:11px;
        }
        .ig-api-history-card-head:hover { background:var(--bg-hover); }
        .ig-api-history-badge { padding:2px 6px; border-radius:4px; font-weight:600; width:36px; text-align:center; }
        .ig-api-history-badge.ok { background:var(--success-subtle); color:var(--success); }
        .ig-api-history-badge.error { background:var(--error-subtle); color:var(--error); }
        .ig-api-history-type { font-size:10px; color:var(--text-tertiary); }
        .ig-api-history-ts { font-size:10px; color:var(--text-tertiary); margin-left:auto; }
        .ig-api-history-prompt { flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--text-secondary); }
        .ig-api-history-card-body { padding:12px; border-top:1px solid var(--border); background:var(--bg-primary); }
        .ig-api-history-section { margin-bottom:12px; }
        .ig-api-history-section:last-child { margin-bottom:0; }
        .ig-api-history-label { font-size:10px; font-weight:600; color:var(--text-tertiary); margin-bottom:4px; }
        .ig-api-history-pre {
            font-size:10px; font-family:monospace; background:var(--bg-tertiary);
            padding:8px; border-radius:4px; overflow-x:auto; max-height:200px; overflow-y:auto;
            white-space:pre-wrap; word-break:break-all; margin:0;
        }

        /* ===== 비교 모드 ===== */
        .ig-compare { display:flex; gap:8px; width:100%; height:100%; align-items:center; justify-content:center; padding:8px; }
        .ig-compare-pane { flex:1; display:flex; flex-direction:column; align-items:center; gap:4px; max-height:100%; overflow:hidden; }
        .ig-compare-pane img { max-width:100%; max-height:320px; border-radius:var(--radius-md); object-fit:contain; cursor:zoom-in; }
        .ig-compare-pane img:hover { transform:scale(1.02); }
        .ig-compare-label { font-size:var(--font-size-2xs); color:var(--text-tertiary); font-weight:600; }

        /* ===== 반응형 ===== */
        @media (max-width:768px) {
            .ig-layout { flex-direction:column; }
            .ig-sidebar { width:100%; max-height:none; }
            .ig-preview { min-height:300px; }
            .ig-preset-grid { grid-template-columns:repeat(4,1fr); }
        }
`); })();