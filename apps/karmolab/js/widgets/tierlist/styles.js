(function () {
    const T = window.Tierlist = window.Tierlist || {};

    T.injectStyles = function injectStyles() {
        if (injectStyles._done) return;
        injectStyles._done = true;

        Mdd.injectCSS('tierlist', `
        .tl-wrap { display:flex; flex-direction:column; flex:1; min-height:0; gap:12px; position:relative; overflow:hidden; }
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
        .tl-badge { display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; font-size:12px; font-weight:700; border:1px solid var(--border); background:var(--glass); }
        .tl-badge.tl-badge-local { border-color: rgba(100,116,139,.5); color: var(--text-secondary); font-size:11px; }
        .tl-badge.tl-badge-dirty { border-color: rgba(99,102,241,.7); color: var(--accent); }

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
        .tl-item:hover { border-color:var(--accent); }
        .tl-item img { width:100%; height:100%; object-fit:cover; pointer-events:none; display:block; }
        .tl-item-text { width:100%; height:100%; display:flex; align-items:center; justify-content:center; text-align:center; font-size:12px; font-weight:600; color:var(--text-primary); padding:4px; word-break:break-word; line-height:1.2; }
        .tl-item-name { position:absolute; bottom:0; left:0; right:0; background:rgba(0,0,0,.7); color:#fff; font-size:10px; padding:2px 4px; text-align:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; opacity:0; transition:opacity var(--transition); pointer-events:none; }
        .tl-item:hover .tl-item-name { opacity:1; }
        .tl-item.dragging { opacity:.4; }
        .tl-drag-ghost { position:fixed; z-index:9999; pointer-events:none; opacity:.9; box-shadow:0 8px 24px rgba(0,0,0,.3); border-radius:var(--radius-sm); }
        .tl-placeholder { width:68px; height:68px; border:2px dashed var(--accent); border-radius:var(--radius-sm); background:var(--accent-subtle); flex-shrink:0; }

        .tl-ctx { position:fixed; z-index:10000; background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-md); padding:4px 0; min-width:140px; box-shadow:0 8px 24px rgba(0,0,0,.25); }
        .tl-ctx-item { display:block; width:100%; background:none; border:none; color:var(--text-primary); font-size:12px; padding:8px 14px; text-align:left; cursor:pointer; }
        .tl-ctx-item:hover { background:var(--bg-hover); }
        .tl-ctx-item.danger { color:var(--error); }
        .tl-ctx-sep { height:1px; background:var(--border); margin:4px 0; }

        .tl-dialog-overlay { position:fixed; inset:0; background:rgba(0,0,0,.5); z-index:9998; display:flex; align-items:center; justify-content:center; }
        .tl-dialog { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-lg); padding:24px; width:420px; max-width:92vw; max-height:80vh; overflow:auto; box-shadow:0 16px 48px rgba(0,0,0,.3); }
        .tl-dialog.tl-dialog-wide { width:min(900px, 92vw); }
        .tl-dialog h3 { margin:0 0 12px; font-size:16px; }
        .tl-dialog label { display:block; font-size:12px; font-weight:700; color:var(--text-secondary); margin-bottom:4px; }
        .tl-dialog input[type="text"], .tl-dialog textarea { width:100%; background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-primary); border-radius:var(--radius-sm); padding:8px 10px; font-size:14px; margin-bottom:12px; box-sizing:border-box; }
        .tl-dialog textarea { min-height:120px; font-family:var(--font-mono); }
        .tl-dialog-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:8px; }

        /* ── list tab ── */
        .tl-list-section { margin-bottom:28px; }
        .tl-list-section-title { margin:0 0 6px; font-size:15px; font-weight:700; color:var(--text-primary); }
        .tl-list-section-desc { margin:0 0 12px; font-size:var(--font-size-xs); color:var(--text-tertiary); line-height:1.45; }
        .tl-list-section-local .tl-list-section-title { color:var(--text-secondary); }
        .tl-list-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); gap:12px; padding:4px 0; }
        .tl-list-card { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-md); padding:16px; cursor:pointer; transition:all var(--transition); position:relative; overflow:hidden; }
        .tl-list-card-embed { border-color:rgba(198,40,40,.4); }
        .tl-list-card-embed:hover { border-color:#e53935; box-shadow:0 2px 12px rgba(198,40,40,.15); }
        .tl-list-card-embed.active { border-color:#e53935; background:rgba(229,57,53,.08); }
        .tl-list-card-ribbon {
            position:absolute; top:0; right:0; z-index:1;
            background:linear-gradient(135deg, #b71c1c, #e53935);
            color:#fff; font-size:9px; font-weight:800; padding:4px 8px;
            border-radius:0 0 0 var(--radius-sm);
            letter-spacing:.03em;
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

