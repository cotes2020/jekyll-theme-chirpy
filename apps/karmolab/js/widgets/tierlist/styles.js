(function () {
    const T = window.Tierlist = window.Tierlist || {};

    T.injectStyles = function injectStyles() {
        if (injectStyles._done) return;
        injectStyles._done = true;

        Mdd.injectCSS('tierlist', `
        .tl-wrap { display:flex; flex-direction:column; flex:1; min-height:0; gap:12px; position:relative; overflow:hidden; }
        .tl-wrap--toc-dock { padding-bottom: 96px; box-sizing: border-box; }
        .tl-wrap--embedded { border:1px solid rgba(198,40,40,.35); border-radius:var(--radius-lg); padding-top:2px; }
        .tl-wrap--embedded .tl-toolbar { padding-right:100px; }
        .tl-ribbon-embed {
            position:absolute; top:0; right:0; z-index:4;
            background:linear-gradient(135deg, #b71c1c 0%, #e53935 100%);
            color:#fff; font-size:10px; font-weight:800; letter-spacing:.04em;
            padding:5px 10px 5px 14px;
            border-radius:0 0 0 var(--radius-md);
            box-shadow:0 2px 8px rgba(0,0,0,.2);
            pointer-events:none;
        }
        .tl-toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding:0 2px; }
        .tl-toolbar select { min-width:160px; }
        .tl-toolbar select,
        .tl-toolbar input[type="text"] { background:var(--bg-secondary); border:1px solid var(--border); color:var(--text-primary); border-radius:var(--radius-sm); padding:6px 10px; font-size:var(--font-size-sm); }
        .tl-toolbar-spacer { flex:1; }
        .tl-btn { display:inline-flex; align-items:center; gap:6px; background:var(--bg-secondary); border:1px solid var(--border); color:var(--text-primary); border-radius:var(--radius-sm); padding:6px 12px; font-size:var(--font-size-xs); font-weight:600; cursor:pointer; transition:all var(--transition); white-space:nowrap; }
        .tl-btn:hover { background:var(--bg-hover); border-color:var(--text-tertiary); }
        .tl-btn-primary { background:var(--accent); border-color:var(--accent); color:#fff; }
        .tl-btn-primary:hover { background:var(--accent-hover); border-color:var(--accent-hover); }
        .tl-btn-toggle-on { border-color:rgba(229,57,53,.55); background:rgba(229,57,53,.1); color:#e53935; }
        .tl-btn-toggle-on:hover { border-color:rgba(229,57,53,.75); background:rgba(229,57,53,.16); }
        .tl-wrap.tl-delete-mode .tl-item:not(.tl-item--static) { box-shadow:inset 0 0 0 2px rgba(229,57,53,.35); cursor:crosshair; }
        .tl-badge { display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; font-size:12px; font-weight:700; border:1px solid var(--border); background:var(--glass); }
        .tl-badge.tl-badge-local { border-color: rgba(100,116,139,.5); color: var(--text-secondary); font-size:11px; }

        .tl-toc {
            display:flex; flex-wrap:wrap; align-items:center; gap:6px; padding:8px 10px;
            background:var(--bg-secondary);
            flex-shrink:0;
            border:1px solid var(--border);
        }
        /* 뷰포트 하단 리모컨 — 스크롤해도 미배치에서 티어로 드롭 가능 */
        .tl-toc--dock {
            position:fixed;
            left:50%;
            transform:translateX(-50%);
            bottom:max(10px, env(safe-area-inset-bottom, 0px));
            width:min(760px, calc(100vw - 20px));
            max-width:calc(100vw - 20px);
            z-index:9990;
            flex-direction:column;
            align-items:stretch;
            flex-wrap:nowrap;
            gap:6px;
            border-radius:var(--radius-lg);
            box-shadow:0 -4px 28px rgba(0,0,0,.2), 0 8px 32px rgba(0,0,0,.12);
            backdrop-filter:saturate(1.1) blur(8px);
        }
        .tl-toc--dock .tl-toc-hint { width:100%; margin:0; flex-shrink:0; }
        .tl-toc-chip-row {
            display:flex; flex-direction:row; flex-wrap:nowrap; gap:6px; align-items:center;
            overflow-x:auto; overflow-y:hidden; -webkit-overflow-scrolling:touch; scrollbar-width:thin;
            padding-bottom:2px;
        }
        .tl-wrap--embedded .tl-toc--dock {
            position:absolute;
            left:8px;
            right:8px;
            bottom:8px;
            transform:none;
            width:auto;
            max-width:none;
            z-index:10;
        }
        /* 편집 탭 + 하단 도크일 때만: 전역 토스트를 위로 (공용 toolbox.css 미변경) */
        body:has(#page-tierlist.active #panel-tl-edit.active .tl-toc--dock) #statusToast.status-toast {
            bottom: calc(96px + 28px + env(safe-area-inset-bottom, 0px));
        }
        .tl-toc-hint { font-size:11px; font-weight:700; color:var(--text-tertiary); margin-right:4px; width:100%; margin-bottom:2px; }
        @media (min-width:520px) {
            .tl-toc-hint { width:auto; margin-bottom:0; }
        }
        .tl-toc-chip {
            display:inline-flex; align-items:center; justify-content:center; min-height:36px; min-width:36px;
            padding:6px 12px; border-radius:var(--radius-md); border:2px solid rgba(0,0,0,.12);
            font-size:13px; font-weight:800; cursor:pointer; user-select:none; flex-shrink:0;
            color:#000; box-shadow:0 1px 2px rgba(0,0,0,.06); transition:transform var(--transition), box-shadow var(--transition);
        }
        .tl-toc-chip:hover { transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.1); }
        .tl-toc-chip.drag-over { outline:2px solid var(--accent); outline-offset:2px; box-shadow:0 0 0 3px var(--accent-subtle); }
        .tl-toc-pool { background:var(--bg-tertiary) !important; color:var(--text-primary) !important; border-color:var(--border) !important; font-weight:700 !important; font-size:12px !important; }

        .tl-board { flex:1; overflow:auto; display:flex; flex-direction:column; border:1px solid var(--border); border-radius:var(--radius-lg); background:var(--bg-tertiary); }
        .tl-row { display:flex; min-height:80px; border-bottom:1px solid var(--border); }
        .tl-row:last-child { border-bottom:none; }
        .tl-label { width:72px; min-width:72px; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:20px; user-select:none; }
        .tl-dropzone { flex:1; display:flex; flex-wrap:wrap; align-content:flex-start; gap:4px; padding:6px; min-height:72px; }
        .tl-dropzone.drag-over, .tl-pool.drag-over { background:var(--accent-subtle); }

        .tl-pool-section { border:1px solid var(--border); border-radius:var(--radius-lg); background:var(--bg-tertiary); }
        .tl-pool-header { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-bottom:1px solid var(--border); }
        .tl-pool { display:flex; flex-wrap:wrap; align-content:flex-start; gap:4px; padding:8px; min-height:80px; }

        .tl-item { width:68px; height:68px; border-radius:var(--radius-sm); overflow:hidden; position:relative; cursor:grab; user-select:none; background:var(--bg-secondary); border:2px solid transparent; transition:border-color var(--transition); touch-action:none; }
        .tl-item-badge { position:absolute; top:2px; left:2px; z-index:3; font-size:8px; font-weight:800; line-height:1.1; padding:2px 4px; border-radius:3px; pointer-events:none; letter-spacing:-0.02em; box-shadow:0 1px 2px rgba(0,0,0,.35); }
        .tl-item-badge--add { background:linear-gradient(135deg,#1565c0,#42a5f5); color:#fff; }
        .tl-item-badge--edit { background:linear-gradient(135deg,#e65100,#ff9800); color:#fff; }
        .tl-item-userlabels { position:absolute; top:2px; right:2px; left:22px; z-index:2; display:flex; flex-direction:column; align-items:flex-end; gap:2px; pointer-events:none; max-height:calc(100% - 6px); overflow:hidden; }
        .tl-item-userlabel { font-size:7px; font-weight:800; line-height:1.15; padding:2px 4px; border-radius:3px; color:#fff; text-shadow:0 1px 1px rgba(0,0,0,.45); max-width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; box-shadow:0 1px 2px rgba(0,0,0,.25); }
        .tl-item-userlabel--more { background:#37474f !important; font-size:7px; }
        .tl-item:hover { border-color:var(--accent); }
        .tl-item img { width:100%; height:100%; object-fit:cover; pointer-events:none; display:block; }
        .tl-item-text { width:100%; height:100%; display:flex; align-items:center; justify-content:center; text-align:center; font-size:12px; font-weight:600; color:var(--text-primary); padding:4px; word-break:break-word; line-height:1.2; }
        .tl-item-name { position:absolute; bottom:0; left:0; right:0; background:rgba(0,0,0,.7); color:#fff; font-size:10px; padding:2px 4px; text-align:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; opacity:0; transition:opacity var(--transition); pointer-events:none; }
        .tl-item:hover .tl-item-name { opacity:1; }
        .tl-item.dragging { opacity:.4; }
        .tl-drag-ghost { position:fixed; z-index:9999; pointer-events:none; opacity:.9; box-shadow:0 8px 24px rgba(0,0,0,.3); border-radius:var(--radius-sm); }
        .tl-placeholder { width:68px; height:68px; border:2px dashed var(--accent); border-radius:var(--radius-sm); background:var(--accent-subtle); flex-shrink:0; pointer-events:none; }

        .tl-ctx { position:fixed; z-index:10000; background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-md); padding:4px 0; min-width:140px; box-shadow:0 8px 24px rgba(0,0,0,.25); }
        .tl-ctx-item { display:block; width:100%; background:none; border:none; color:var(--text-primary); font-size:12px; padding:8px 14px; text-align:left; cursor:pointer; }
        .tl-ctx-item:hover { background:var(--bg-hover); }
        .tl-ctx-item.danger { color:var(--error); }
        .tl-ctx-sep { height:1px; background:var(--border); margin:4px 0; }

        .tl-dialog-overlay { position:fixed; inset:0; background:rgba(0,0,0,.5); z-index:9998; display:flex; align-items:center; justify-content:center; }
        .tl-dialog { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-lg); padding:24px; width:420px; max-width:92vw; max-height:80vh; overflow:auto; box-shadow:0 16px 48px rgba(0,0,0,.3); }
        .tl-dialog.tl-dialog-wide { width:min(900px, 92vw); }
        .tl-dialog h3 { margin:0 0 12px; font-size:16px; }
        .tl-dialog-wide { max-width:440px; }
        .tl-tier-hint { font-size:12px; color:var(--text-tertiary); margin:0 0 12px; line-height:1.45; }
        .tl-tier-rows { margin-bottom:12px; max-height:min(52vh, 420px); overflow-y:auto; }
        .tl-tier-row { display:flex; align-items:center; gap:8px; margin-bottom:8px; flex-wrap:wrap; }
        .tl-tier-row .tl-tier-label { width:72px; flex-shrink:0; margin-bottom:0 !important; padding:6px 8px; font-size:14px; }
        .tl-tier-color { width:44px; height:36px; padding:0; border:1px solid var(--border); border-radius:var(--radius-sm); cursor:pointer; flex-shrink:0; }
        .tl-tier-row-btns { display:flex; gap:4px; margin-left:auto; }
        .tl-tier-row-btns .tl-btn { padding:4px 10px; min-width:auto; font-size:12px; }
        .tl-tier-actions { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px; }
        .tl-ul-manager-body { margin-bottom:12px; max-height:min(50vh, 380px); overflow-y:auto; }
        .tl-ul-empty { font-size:12px; color:var(--text-tertiary); padding:12px 0; line-height:1.45; }
        .tl-ul-row { display:flex; align-items:center; gap:8px; margin-bottom:8px; flex-wrap:wrap; }
        .tl-ul-row .tl-ul-name { flex:1; min-width:120px; margin-bottom:0 !important; padding:6px 8px; font-size:14px; }
        .tl-ul-row .tl-ul-color { width:44px; height:36px; padding:0; border:1px solid var(--border); border-radius:var(--radius-sm); cursor:pointer; flex-shrink:0; }
        .tl-ul-usage { font-size:11px; color:var(--text-tertiary); white-space:nowrap; min-width:2.5em; text-align:right; }
        .tl-ul-row .tl-ul-del { padding:4px 10px; min-width:auto; font-size:12px; }
        .tl-ul-assign-list { max-height:min(45vh, 320px); overflow-y:auto; margin-bottom:12px; }
        .tl-ul-assign-row { display:flex; align-items:center; gap:8px; padding:6px 4px; border-radius:var(--radius-sm); cursor:pointer; font-size:13px; color:var(--text-primary); }
        .tl-ul-assign-row:hover { background:var(--bg-hover); }
        .tl-ul-assign-row input { flex-shrink:0; }
        .tl-ul-assign-swatch { width:18px; height:18px; border-radius:4px; flex-shrink:0; border:1px solid rgba(0,0,0,.2); }
        .tl-ul-assign-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .tl-dialog label { display:block; font-size:12px; font-weight:700; color:var(--text-secondary); margin-bottom:4px; }
        .tl-dialog input[type="text"], .tl-dialog textarea { width:100%; background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-primary); border-radius:var(--radius-sm); padding:8px 10px; font-size:14px; margin-bottom:12px; box-sizing:border-box; }
        .tl-dialog textarea { min-height:120px; font-family:var(--font-mono); }
        .tl-dialog-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:8px; }

        /* ── list tab ── */
        .tl-list-section { margin-bottom:28px; }
        .tl-list-section-title { margin:0 0 6px; font-size:15px; font-weight:700; color:var(--text-primary); }
        .tl-list-subsection-title { margin:12px 0 8px; font-size:13px; font-weight:700; color:var(--text-secondary); }
        .tl-embed-grids > .tl-list-subsection-title:first-of-type { margin-top:0; }
        .tl-list-section-desc { margin:0 0 12px; font-size:var(--font-size-xs); color:var(--text-tertiary); line-height:1.45; }
        .tl-list-section-local .tl-list-section-title { color:var(--text-secondary); }
        .tl-list-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); gap:12px; padding:4px 0; }
        .tl-list-card { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-md); padding:16px; cursor:pointer; transition:all var(--transition); position:relative; overflow:hidden; }
        .tl-list-card-embed { border-color:rgba(198,40,40,.35); }
        .tl-list-card-embed:hover { border-color:#e53935; box-shadow:0 2px 12px rgba(198,40,40,.12); }
        .tl-list-card-embed.active { border-color:#e53935; background:rgba(229,57,53,.08); }
        .tl-list-card-local { border-color:rgba(100,116,139,.35); }
        .tl-list-card-local:hover { border-color:rgba(100,116,139,.65); }
        .tl-list-card-local.active { border-color:var(--accent); background:var(--accent-subtle); }
        .tl-list-pill-row { display:flex; align-items:center; gap:6px; margin-bottom:8px; flex-wrap:wrap; }
        .tl-pill {
            display:inline-flex; align-items:center; font-size:10px; font-weight:800;
            letter-spacing:.03em; padding:3px 8px; border-radius:999px;
            line-height:1.2;
        }
        .tl-pill-blog, .tl-pill-catalog {
            background:linear-gradient(135deg, #b71c1c, #e53935); color:#fff;
            box-shadow:0 1px 4px rgba(198,40,40,.25);
        }
        .tl-pill-karmo {
            background:linear-gradient(135deg, #4527a0, #7e57c2); color:#fff;
            box-shadow:0 1px 4px rgba(69,39,160,.28);
        }
        .tl-list-card-embed--karmo { border-color:rgba(94,53,177,.42) !important; }
        .tl-list-card-embed--karmo:hover { border-color:#7e57c2 !important; box-shadow:0 2px 12px rgba(94,53,177,.14) !important; }
        .tl-list-card-embed--karmo.active { border-color:#7e57c2 !important; background:rgba(126,87,194,.1) !important; }
        .tl-pill-cache {
            background:rgba(100,116,139,.2); color:var(--text-secondary);
            border:1px solid rgba(100,116,139,.35);
        }
        .tl-list-card:hover { border-color:var(--accent); box-shadow:0 2px 8px rgba(0,0,0,.1); }
        .tl-list-card.active { border-color:var(--accent); background:var(--accent-subtle); }
        .tl-list-card-title { font-size:14px; font-weight:600; color:var(--text-primary); margin-bottom:4px; }
        .tl-list-card-meta { font-size:var(--font-size-xs); color:var(--text-tertiary); }
        .tl-list-card-cat { display:inline-block; font-size:var(--font-size-2xs); padding:2px 6px; border-radius:10px; background:var(--bg-hover); color:var(--text-secondary); margin-top:6px; }
        .tl-list-empty { text-align:center; padding:48px 16px; color:var(--text-tertiary); font-size:var(--font-size-sm); }

        /* ── stats tab ── */
        .tl-stats { display:flex; flex-direction:column; gap:16px; }
        .tl-stat-cards { display:grid; grid-template-columns:repeat(auto-fill, minmax(140px, 1fr)); gap:10px; }
        .tl-stat-card { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-md); padding:16px; text-align:center; }
        .tl-stat-card-value { font-size:28px; font-weight:700; color:var(--accent); }
        .tl-stat-card-label { font-size:var(--font-size-xs); color:var(--text-tertiary); margin-top:4px; }
        .tl-stat-section { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-md); padding:16px; }
        .tl-stat-section h4 { margin:0 0 12px; font-size:14px; color:var(--text-primary); }
        .tl-bar-row { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
        .tl-bar-label { width:32px; font-size:var(--font-size-sm); font-weight:700; text-align:center; flex-shrink:0; }
        .tl-bar-track { flex:1; height:24px; background:var(--bg-tertiary); border-radius:4px; overflow:hidden; position:relative; }
        .tl-bar-fill { height:100%; border-radius:4px; transition:width .4s ease; display:flex; align-items:center; padding-left:8px; }
        .tl-bar-count { font-size:var(--font-size-xs); font-weight:600; color:rgba(0,0,0,.7); }
        .tl-stat-table { width:100%; border-collapse:collapse; font-size:var(--font-size-xs); }
        .tl-stat-table th { text-align:left; padding:8px; color:var(--text-secondary); border-bottom:1px solid var(--border); font-weight:500; }
        .tl-stat-table td { padding:8px; color:var(--text-primary); border-bottom:1px solid var(--border); }

        @media (max-width:600px) {
            .tl-label { width:48px; min-width:48px; font-size:16px; }
            .tl-item { width:56px; height:56px; }
            .tl-placeholder { width:56px; height:56px; }
        }
        `);
    };
})();

