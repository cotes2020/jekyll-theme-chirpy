(function () {

    /* ============================== CSS ============================== */

    Mdd.injectCSS('tierlist', `
        .tl-wrap { display:flex; flex-direction:column; flex:1; min-height:0; gap:12px; }

        /* ── toolbar ── */
        .tl-toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding:0 2px; }
        .tl-toolbar select,
        .tl-toolbar input[type="text"] { background:var(--bg-secondary); border:1px solid var(--border); color:var(--text-primary); border-radius:var(--radius-sm); padding:6px 10px; font-size:var(--font-size-sm); }
        .tl-toolbar select { min-width:160px; cursor:pointer; }
        .tl-toolbar input[type="text"] { flex:1; min-width:120px; max-width:260px; }
        .tl-toolbar-spacer { flex:1; }
        .tl-btn { display:inline-flex; align-items:center; gap:4px; background:var(--bg-secondary); border:1px solid var(--border); color:var(--text-primary); border-radius:var(--radius-sm); padding:6px 12px; font-size:var(--font-size-xs); font-weight:500; cursor:pointer; transition:all var(--transition); white-space:nowrap; }
        .tl-btn:hover { background:var(--bg-hover); border-color:var(--text-tertiary); }
        .tl-btn svg { width:14px; height:14px; flex-shrink:0; }
        .tl-btn-primary { background:var(--accent); border-color:var(--accent); color:#fff; }
        .tl-btn-primary:hover { background:var(--accent-hover); border-color:var(--accent-hover); }
        .tl-btn-danger { color:var(--error); }
        .tl-btn-danger:hover { background:var(--error-subtle); }

        /* ── tier board ── */
        .tl-board { flex:1; overflow-y:auto; display:flex; flex-direction:column; border:1px solid var(--border); border-radius:var(--radius-lg); background:var(--bg-tertiary); }
        .tl-row { display:flex; min-height:80px; border-bottom:1px solid var(--border); }
        .tl-row:last-child { border-bottom:none; }
        .tl-label { width:72px; min-width:72px; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:20px; cursor:pointer; user-select:none; transition:filter var(--transition); position:relative; }
        .tl-label:hover { filter:brightness(1.15); }
        .tl-label-edit { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:4px; background:inherit; z-index:5; padding:4px; border-radius:var(--radius-sm); }
        .tl-label-edit input[type="text"] { width:48px; text-align:center; background:rgba(0,0,0,.2); border:1px solid rgba(255,255,255,.3); color:inherit; border-radius:4px; padding:2px; font-size:14px; font-weight:700; }
        .tl-label-edit input[type="color"] { width:32px; height:24px; border:none; background:none; cursor:pointer; padding:0; }
        .tl-dropzone { flex:1; display:flex; flex-wrap:wrap; align-content:flex-start; gap:4px; padding:6px; min-height:72px; transition:background var(--transition); }
        .tl-dropzone.drag-over { background:var(--accent-subtle); }
        .tl-row-actions { display:flex; flex-direction:column; justify-content:center; gap:2px; padding:0 4px; opacity:0; transition:opacity var(--transition); }
        .tl-row:hover .tl-row-actions { opacity:1; }
        .tl-row-action-btn { background:none; border:none; color:var(--text-tertiary); cursor:pointer; padding:2px; border-radius:3px; font-size:14px; line-height:1; }
        .tl-row-action-btn:hover { background:var(--bg-hover); color:var(--text-primary); }

        /* ── pool ── */
        .tl-pool-section { border:1px solid var(--border); border-radius:var(--radius-lg); background:var(--bg-tertiary); }
        .tl-pool-header { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; border-bottom:1px solid var(--border); }
        .tl-pool-title { font-size:var(--font-size-sm); font-weight:600; color:var(--text-secondary); }
        .tl-pool { display:flex; flex-wrap:wrap; align-content:flex-start; gap:4px; padding:8px; min-height:80px; transition:background var(--transition); }
        .tl-pool.drag-over { background:var(--accent-subtle); }

        /* ── item card ── */
        .tl-item { width:68px; height:68px; border-radius:var(--radius-sm); overflow:hidden; position:relative; cursor:grab; user-select:none; background:var(--bg-secondary); border:2px solid transparent; transition:border-color var(--transition), box-shadow var(--transition); touch-action:none; }
        .tl-item:hover { border-color:var(--accent); }
        .tl-item img { width:100%; height:100%; object-fit:cover; pointer-events:none; display:block; }
        .tl-item-text { width:100%; height:100%; display:flex; align-items:center; justify-content:center; text-align:center; font-size:var(--font-size-xs); font-weight:500; color:var(--text-primary); padding:4px; word-break:break-word; line-height:1.2; }
        .tl-item-name { position:absolute; bottom:0; left:0; right:0; background:rgba(0,0,0,.7); color:#fff; font-size:var(--font-size-2xs); padding:2px 4px; text-align:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; opacity:0; transition:opacity var(--transition); pointer-events:none; }
        .tl-item:hover .tl-item-name { opacity:1; }
        .tl-item.dragging { opacity:.4; }
        .tl-drag-ghost { position:fixed; z-index:9999; pointer-events:none; opacity:.9; box-shadow:0 8px 24px rgba(0,0,0,.3); border-radius:var(--radius-sm); }
        .tl-placeholder { width:68px; height:68px; border:2px dashed var(--accent); border-radius:var(--radius-sm); background:var(--accent-subtle); flex-shrink:0; }

        /* ── context menu ── */
        .tl-ctx { position:fixed; z-index:10000; background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-md); padding:4px 0; min-width:140px; box-shadow:0 8px 24px rgba(0,0,0,.25); }
        .tl-ctx-item { display:block; width:100%; background:none; border:none; color:var(--text-primary); font-size:var(--font-size-xs); padding:8px 14px; text-align:left; cursor:pointer; transition:background var(--transition); }
        .tl-ctx-item:hover { background:var(--bg-hover); }
        .tl-ctx-item.danger { color:var(--error); }
        .tl-ctx-sep { height:1px; background:var(--border); margin:4px 0; }

        /* ── add dialog ── */
        .tl-dialog-overlay { position:fixed; inset:0; background:rgba(0,0,0,.5); z-index:9998; display:flex; align-items:center; justify-content:center; }
        .tl-dialog { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-lg); padding:24px; width:420px; max-width:90vw; max-height:80vh; overflow-y:auto; box-shadow:0 16px 48px rgba(0,0,0,.3); }
        .tl-dialog h3 { margin:0 0 16px; font-size:16px; color:var(--text-primary); }
        .tl-dialog label { display:block; font-size:var(--font-size-xs); font-weight:500; color:var(--text-secondary); margin-bottom:4px; }
        .tl-dialog input[type="text"],
        .tl-dialog textarea { width:100%; background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-primary); border-radius:var(--radius-sm); padding:8px 10px; font-size:var(--font-size-sm); margin-bottom:12px; box-sizing:border-box; }
        .tl-dialog textarea { min-height:60px; resize:vertical; font-family:inherit; }
        .tl-dialog-actions { display:flex; justify-content:flex-end; gap:8px; margin-top:8px; }
        .tl-drop-area { border:2px dashed var(--border); border-radius:var(--radius-md); padding:24px; text-align:center; color:var(--text-tertiary); font-size:var(--font-size-xs); margin-bottom:12px; cursor:pointer; transition:all var(--transition); }
        .tl-drop-area:hover,
        .tl-drop-area.drag-over { border-color:var(--accent); background:var(--accent-subtle); color:var(--text-primary); }
        .tl-drop-area img { max-width:120px; max-height:120px; object-fit:contain; border-radius:var(--radius-sm); margin-top:8px; }

        /* ── list tab ── */
        .tl-list-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); gap:12px; padding:4px 0; }
        .tl-list-card { background:var(--bg-secondary); border:1px solid var(--border); border-radius:var(--radius-md); padding:16px; cursor:pointer; transition:all var(--transition); }
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
            .tl-toolbar { gap:4px; }
            .tl-toolbar select { min-width:100px; }
        }
    `);

    /* ========================= TierDB (IndexedDB) ========================= */

    const TierDB = (() => {
        const DB_NAME = 'toolbox_tierlist_images';
        const STORE = 'images';
        const VER = 1;
        const cache = new Map();

        function open() {
            return new Promise((res, rej) => {
                const r = indexedDB.open(DB_NAME, VER);
                r.onupgradeneeded = e => {
                    const db = e.target.result;
                    if (!db.objectStoreNames.contains(STORE))
                        db.createObjectStore(STORE, { keyPath: 'id' });
                };
                r.onsuccess = () => res(r.result);
                r.onerror = () => rej(r.error);
            });
        }

        async function save(id, dataUrl) {
            const db = await open();
            try {
                await new Promise((res, rej) => {
                    const tx = db.transaction(STORE, 'readwrite');
                    tx.objectStore(STORE).put({ id, dataUrl });
                    tx.oncomplete = () => res();
                    tx.onerror = () => rej(tx.error);
                });
                cache.set(id, dataUrl);
            } finally { db.close(); }
        }

        async function get(id) {
            if (cache.has(id)) return cache.get(id);
            const db = await open();
            try {
                const item = await new Promise((res, rej) => {
                    const tx = db.transaction(STORE, 'readonly');
                    const req = tx.objectStore(STORE).get(id);
                    req.onsuccess = () => res(req.result);
                    req.onerror = () => rej(req.error);
                });
                if (item) { cache.set(id, item.dataUrl); return item.dataUrl; }
                return null;
            } finally { db.close(); }
        }

        async function remove(id) {
            cache.delete(id);
            const db = await open();
            try {
                await new Promise((res, rej) => {
                    const tx = db.transaction(STORE, 'readwrite');
                    tx.objectStore(STORE).delete(id);
                    tx.oncomplete = () => res();
                    tx.onerror = () => rej(tx.error);
                });
            } finally { db.close(); }
        }

        async function getAllKeys() {
            const db = await open();
            try {
                return await new Promise((res, rej) => {
                    const tx = db.transaction(STORE, 'readonly');
                    const req = tx.objectStore(STORE).getAllKeys();
                    req.onsuccess = () => res(req.result);
                    req.onerror = () => rej(req.error);
                });
            } finally { db.close(); }
        }

        async function getMany(ids) {
            const results = {};
            const missing = ids.filter(id => {
                if (cache.has(id)) { results[id] = cache.get(id); return false; }
                return true;
            });
            if (!missing.length) return results;
            const db = await open();
            try {
                const tx = db.transaction(STORE, 'readonly');
                const store = tx.objectStore(STORE);
                await Promise.all(missing.map(id => new Promise((res, rej) => {
                    const req = store.get(id);
                    req.onsuccess = () => {
                        if (req.result) { cache.set(id, req.result.dataUrl); results[id] = req.result.dataUrl; }
                        res();
                    };
                    req.onerror = () => rej(req.error);
                })));
            } finally { db.close(); }
            return results;
        }

        return { save, get, remove, getAllKeys, getMany };
    })();

    /* ========================= State Management ========================= */

    const STORAGE_KEY = 'toolbox_tierlists';
    const DEFAULT_TIERS = [
        { id: 's', label: 'S', color: '#ff7f7f' },
        { id: 'a', label: 'A', color: '#ffbf7f' },
        { id: 'b', label: 'B', color: '#ffdf7f' },
        { id: 'c', label: 'C', color: '#ffff7f' },
        { id: 'd', label: 'D', color: '#bfff7f' },
        { id: 'f', label: 'F', color: '#7fbfff' },
    ];

    let state = { lists: {}, currentListId: null };

    function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 8); }

    function loadState() {
        try { const raw = localStorage.getItem(STORAGE_KEY); if (raw) state = JSON.parse(raw); } catch (_) {}
    }

    function saveState() {
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (_) {}
    }

    function currentList() { return state.lists[state.currentListId] || null; }

    function createList(title, category) {
        const id = 'tl-' + uid();
        const now = Date.now();
        const tiers = DEFAULT_TIERS.map(t => ({ ...t }));
        const rankings = {};
        tiers.forEach(t => rankings[t.id] = []);
        rankings._pool = [];
        state.lists[id] = { id, title: title || '새 티어리스트', category: category || '', createdAt: now, updatedAt: now, tiers, rankings, items: {} };
        state.currentListId = id;
        saveState();
        return id;
    }

    function deleteList(id) {
        const list = state.lists[id];
        if (!list) return;
        const imgKeys = Object.values(list.items).filter(i => i.imageKey).map(i => i.imageKey);
        imgKeys.forEach(k => TierDB.remove(k));
        delete state.lists[id];
        if (state.currentListId === id) {
            const ids = Object.keys(state.lists);
            state.currentListId = ids.length ? ids[0] : null;
        }
        saveState();
    }

    function duplicateList(id) {
        const src = state.lists[id];
        if (!src) return null;
        const newId = 'tl-' + uid();
        const now = Date.now();
        const itemMap = {};
        const newItems = {};
        for (const [oldItemId, item] of Object.entries(src.items)) {
            const nid = 'ti-' + uid();
            itemMap[oldItemId] = nid;
            newItems[nid] = { ...item, id: nid };
            if (item.imageKey) {
                const newImgKey = 'tl-img-' + uid();
                TierDB.get(item.imageKey).then(data => { if (data) TierDB.save(newImgKey, data); });
                newItems[nid].imageKey = newImgKey;
            }
        }
        const newRankings = {};
        for (const [k, arr] of Object.entries(src.rankings)) {
            newRankings[k] = arr.map(oid => itemMap[oid] || oid);
        }
        state.lists[newId] = {
            id: newId, title: src.title + ' (복사)', category: src.category,
            createdAt: now, updatedAt: now,
            tiers: src.tiers.map(t => ({ ...t })), rankings: newRankings, items: newItems
        };
        state.currentListId = newId;
        saveState();
        return newId;
    }

    function addItem(name, imageKey) {
        const list = currentList();
        if (!list) return null;
        const id = 'ti-' + uid();
        list.items[id] = { id, name: name || '', imageKey: imageKey || null };
        list.rankings._pool.push(id);
        list.updatedAt = Date.now();
        saveState();
        return id;
    }

    function removeItem(itemId) {
        const list = currentList();
        if (!list) return;
        if (list.items[itemId]?.imageKey) TierDB.remove(list.items[itemId].imageKey);
        delete list.items[itemId];
        for (const arr of Object.values(list.rankings)) {
            const idx = arr.indexOf(itemId);
            if (idx !== -1) arr.splice(idx, 1);
        }
        list.updatedAt = Date.now();
        saveState();
    }

    function moveItem(itemId, targetTier, insertIdx) {
        const list = currentList();
        if (!list) return;
        for (const arr of Object.values(list.rankings)) {
            const idx = arr.indexOf(itemId);
            if (idx !== -1) arr.splice(idx, 1);
        }
        const target = list.rankings[targetTier];
        if (!target) return;
        if (insertIdx === undefined || insertIdx >= target.length) target.push(itemId);
        else target.splice(insertIdx, 0, itemId);
        list.updatedAt = Date.now();
        saveState();
    }

    /* ========================= File / Image Helpers ========================= */

    function fileToDataUrl(file) {
        return new Promise((res, rej) => {
            const reader = new FileReader();
            reader.onload = () => res(reader.result);
            reader.onerror = () => rej(reader.error);
            reader.readAsDataURL(file);
        });
    }

    async function processImageFile(file) {
        const dataUrl = await fileToDataUrl(file);
        const thumb = await createThumbnail(dataUrl, 200);
        const imgKey = 'tl-img-' + uid();
        await TierDB.save(imgKey, thumb);
        return imgKey;
    }

    function createThumbnail(dataUrl, maxSize) {
        return new Promise(res => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let w = img.width, h = img.height;
                if (w > maxSize || h > maxSize) {
                    const ratio = Math.min(maxSize / w, maxSize / h);
                    w = Math.round(w * ratio);
                    h = Math.round(h * ratio);
                }
                canvas.width = w;
                canvas.height = h;
                canvas.getContext('2d').drawImage(img, 0, 0, w, h);
                res(canvas.toDataURL('image/webp', 0.85));
            };
            img.onerror = () => res(dataUrl);
            img.src = dataUrl;
        });
    }

    /* ========================= Drag & Drop Engine ========================= */

    let dragState = null;

    function initDnD(boardEl, poolEl) {
        const root = boardEl.closest('.tl-wrap');

        function getDropTarget(x, y) {
            const zones = root.querySelectorAll('.tl-dropzone, .tl-pool');
            let best = null, bestDist = Infinity;
            for (const zone of zones) {
                const rect = zone.getBoundingClientRect();
                if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
                    const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
                    const dist = Math.hypot(x - cx, y - cy);
                    if (dist < bestDist) { bestDist = dist; best = zone; }
                }
            }
            return best;
        }

        function getInsertIndex(zone, x) {
            const cards = Array.from(zone.querySelectorAll('.tl-item:not(.dragging)'));
            for (let i = 0; i < cards.length; i++) {
                const rect = cards[i].getBoundingClientRect();
                if (x < rect.left + rect.width / 2) return i;
            }
            return cards.length;
        }

        function onPointerDown(e) {
            const itemEl = e.target.closest('.tl-item');
            if (!itemEl || e.button === 2) return;

            const itemId = itemEl.dataset.itemId;
            if (!itemId) return;

            e.preventDefault();
            itemEl.setPointerCapture(e.pointerId);

            const rect = itemEl.getBoundingClientRect();
            const offsetX = e.clientX - rect.left;
            const offsetY = e.clientY - rect.top;

            let moved = false;
            let ghost = null;
            let placeholder = null;
            let currentZone = null;

            const onMove = (ev) => {
                if (!moved) {
                    if (Math.abs(ev.clientX - e.clientX) < 4 && Math.abs(ev.clientY - e.clientY) < 4) return;
                    moved = true;
                    ghost = itemEl.cloneNode(true);
                    ghost.className = 'tl-drag-ghost';
                    ghost.style.width = rect.width + 'px';
                    ghost.style.height = rect.height + 'px';
                    document.body.appendChild(ghost);

                    placeholder = document.createElement('div');
                    placeholder.className = 'tl-placeholder';
                    itemEl.classList.add('dragging');

                    dragState = { itemId, ghost, placeholder, itemEl };
                }

                ghost.style.left = (ev.clientX - offsetX) + 'px';
                ghost.style.top = (ev.clientY - offsetY) + 'px';

                const zone = getDropTarget(ev.clientX, ev.clientY);
                if (currentZone && currentZone !== zone) currentZone.classList.remove('drag-over');
                if (zone) {
                    zone.classList.add('drag-over');
                    currentZone = zone;
                    const idx = getInsertIndex(zone, ev.clientX);
                    const children = Array.from(zone.querySelectorAll('.tl-item:not(.dragging), .tl-placeholder'));
                    const placeholderIdx = children.indexOf(placeholder);
                    if (placeholderIdx !== -1 && (placeholderIdx === idx || placeholderIdx === idx - 1)) return;
                    if (placeholder.parentNode) placeholder.remove();
                    const refChild = zone.querySelectorAll('.tl-item:not(.dragging)')[idx];
                    zone.insertBefore(placeholder, refChild || null);
                }
            };

            const onUp = (ev) => {
                itemEl.releasePointerCapture(ev.pointerId);
                itemEl.removeEventListener('pointermove', onMove);
                itemEl.removeEventListener('pointerup', onUp);

                if (!moved) return;

                if (ghost) ghost.remove();
                if (placeholder) placeholder.remove();
                itemEl.classList.remove('dragging');
                if (currentZone) currentZone.classList.remove('drag-over');

                const zone = getDropTarget(ev.clientX, ev.clientY);
                if (zone) {
                    const tierId = zone.dataset.tierId;
                    const idx = getInsertIndex(zone, ev.clientX);
                    moveItem(itemId, tierId, idx);
                }

                dragState = null;
                renderEditor();
            };

            itemEl.addEventListener('pointermove', onMove);
            itemEl.addEventListener('pointerup', onUp);
        }

        root.addEventListener('pointerdown', onPointerDown);
    }

    /* ========================= Context Menu ========================= */

    let ctxMenu = null;

    function showContextMenu(x, y, actions) {
        hideContextMenu();
        ctxMenu = document.createElement('div');
        ctxMenu.className = 'tl-ctx';
        actions.forEach(a => {
            if (a === 'sep') { const s = document.createElement('div'); s.className = 'tl-ctx-sep'; ctxMenu.appendChild(s); return; }
            const btn = document.createElement('button');
            btn.className = 'tl-ctx-item' + (a.danger ? ' danger' : '');
            btn.textContent = a.label;
            btn.onclick = () => { hideContextMenu(); a.action(); };
            ctxMenu.appendChild(btn);
        });
        document.body.appendChild(ctxMenu);

        const rect = ctxMenu.getBoundingClientRect();
        if (x + rect.width > window.innerWidth) x = window.innerWidth - rect.width - 8;
        if (y + rect.height > window.innerHeight) y = window.innerHeight - rect.height - 8;
        ctxMenu.style.left = x + 'px';
        ctxMenu.style.top = y + 'px';

        setTimeout(() => document.addEventListener('pointerdown', hideContextMenu, { once: true }), 0);
    }

    function hideContextMenu() {
        if (ctxMenu) { ctxMenu.remove(); ctxMenu = null; }
    }

    /* ========================= Dialogs ========================= */

    function showAddItemDialog() {
        const list = currentList();
        if (!list) { Toolbox.showToast('먼저 티어리스트를 선택하세요.', 'error'); return; }

        const overlay = document.createElement('div');
        overlay.className = 'tl-dialog-overlay';

        let pendingFiles = [];
        let previewUrl = null;

        const dialog = document.createElement('div');
        dialog.className = 'tl-dialog';
        dialog.innerHTML = `
            <h3>아이템 추가</h3>
            <div class="tl-drop-area" id="tl-add-drop">
                <div>이미지를 드래그하거나 클릭하여 업로드</div>
                <div style="margin-top:4px; font-size:var(--font-size-xs);">(여러 장 선택 가능)</div>
            </div>
            <input type="file" id="tl-add-file" accept="image/*" multiple style="display:none">
            <label>이름 (선택)</label>
            <input type="text" id="tl-add-name" placeholder="아이템 이름">
            <div class="tl-dialog-actions">
                <button class="tl-btn" id="tl-add-cancel">취소</button>
                <button class="tl-btn tl-btn-primary" id="tl-add-ok">추가</button>
            </div>
        `;
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        const dropArea = dialog.querySelector('#tl-add-drop');
        const fileInput = dialog.querySelector('#tl-add-file');
        const nameInput = dialog.querySelector('#tl-add-name');

        dropArea.onclick = () => fileInput.click();
        dropArea.ondragover = e => { e.preventDefault(); dropArea.classList.add('drag-over'); };
        dropArea.ondragleave = () => dropArea.classList.remove('drag-over');
        dropArea.ondrop = e => {
            e.preventDefault(); dropArea.classList.remove('drag-over');
            handleFiles(Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/')));
        };

        fileInput.onchange = () => handleFiles(Array.from(fileInput.files));

        function handleFiles(files) {
            if (!files.length) return;
            pendingFiles = files;
            if (files.length === 1) {
                const reader = new FileReader();
                reader.onload = () => {
                    previewUrl = reader.result;
                    dropArea.innerHTML = `<img src="${previewUrl}" alt="preview"><div style="margin-top:6px">${Toolbox.escapeHtml(files[0].name)}</div>`;
                };
                reader.readAsDataURL(files[0]);
            } else {
                dropArea.innerHTML = `<div>${files.length}장의 이미지가 선택됨</div>`;
            }
        }

        dialog.querySelector('#tl-add-cancel').onclick = () => overlay.remove();
        overlay.onclick = e => { if (e.target === overlay) overlay.remove(); };

        dialog.querySelector('#tl-add-ok').onclick = async () => {
            const name = nameInput.value.trim();
            if (!pendingFiles.length && !name) {
                Toolbox.showToast('이미지나 이름을 입력하세요.', 'error');
                return;
            }
            overlay.remove();

            if (pendingFiles.length > 1) {
                for (const file of pendingFiles) {
                    const imgKey = await processImageFile(file);
                    addItem(file.name.replace(/\.[^.]+$/, ''), imgKey);
                }
                Toolbox.showToast(`${pendingFiles.length}개 아이템 추가됨`);
                Mdd.setMood('happy'); Mdd.say('한꺼번에 추가했어요!');
            } else if (pendingFiles.length === 1) {
                const imgKey = await processImageFile(pendingFiles[0]);
                addItem(name || pendingFiles[0].name.replace(/\.[^.]+$/, ''), imgKey);
                Toolbox.showToast('아이템 추가됨');
                Mdd.setMood('happy'); Mdd.say('새 아이템이에요!');
            } else {
                addItem(name, null);
                Toolbox.showToast('아이템 추가됨');
                Mdd.setMood('happy'); Mdd.say('텍스트 아이템 추가했어요!');
            }
            renderEditor();
        };

        nameInput.focus();
    }

    function showEditItemDialog(itemId) {
        const list = currentList();
        if (!list || !list.items[itemId]) return;
        const item = list.items[itemId];

        const overlay = document.createElement('div');
        overlay.className = 'tl-dialog-overlay';

        let newImageKey = null;
        let removeImage = false;

        const dialog = document.createElement('div');
        dialog.className = 'tl-dialog';
        dialog.innerHTML = `
            <h3>아이템 편집</h3>
            <div class="tl-drop-area" id="tl-edit-drop">
                <div>새 이미지를 드래그하거나 클릭하여 변경</div>
            </div>
            <input type="file" id="tl-edit-file" accept="image/*" style="display:none">
            <label>이름</label>
            <input type="text" id="tl-edit-name" value="${Toolbox.escapeHtml(item.name || '')}">
            <div class="tl-dialog-actions">
                <button class="tl-btn tl-btn-danger" id="tl-edit-rmimg" style="${item.imageKey ? '' : 'display:none'}">이미지 제거</button>
                <div style="flex:1"></div>
                <button class="tl-btn" id="tl-edit-cancel">취소</button>
                <button class="tl-btn tl-btn-primary" id="tl-edit-ok">저장</button>
            </div>
        `;
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        const dropArea = dialog.querySelector('#tl-edit-drop');
        const fileInput = dialog.querySelector('#tl-edit-file');

        if (item.imageKey) {
            TierDB.get(item.imageKey).then(data => {
                if (data) dropArea.innerHTML = `<img src="${data}" alt="current"><div style="margin-top:4px">클릭하여 변경</div>`;
            });
        }

        dropArea.onclick = () => fileInput.click();
        dropArea.ondragover = e => { e.preventDefault(); dropArea.classList.add('drag-over'); };
        dropArea.ondragleave = () => dropArea.classList.remove('drag-over');
        dropArea.ondrop = e => { e.preventDefault(); dropArea.classList.remove('drag-over'); handleFile(e.dataTransfer.files[0]); };
        fileInput.onchange = () => handleFile(fileInput.files[0]);

        async function handleFile(file) {
            if (!file || !file.type.startsWith('image/')) return;
            newImageKey = await processImageFile(file);
            const data = await TierDB.get(newImageKey);
            if (data) dropArea.innerHTML = `<img src="${data}" alt="new">`;
            removeImage = false;
            dialog.querySelector('#tl-edit-rmimg').style.display = '';
        }

        dialog.querySelector('#tl-edit-rmimg').onclick = () => {
            removeImage = true;
            dropArea.innerHTML = '<div>이미지 제거됨</div>';
            dialog.querySelector('#tl-edit-rmimg').style.display = 'none';
        };

        dialog.querySelector('#tl-edit-cancel').onclick = () => overlay.remove();
        overlay.onclick = e => { if (e.target === overlay) overlay.remove(); };

        dialog.querySelector('#tl-edit-ok').onclick = () => {
            const name = dialog.querySelector('#tl-edit-name').value.trim();
            item.name = name;
            if (newImageKey) {
                if (item.imageKey) TierDB.remove(item.imageKey);
                item.imageKey = newImageKey;
            } else if (removeImage && item.imageKey) {
                TierDB.remove(item.imageKey);
                item.imageKey = null;
            }
            list.updatedAt = Date.now();
            saveState();
            overlay.remove();
            renderEditor();
            Toolbox.showToast('아이템 수정됨');
        };
    }

    function showNewListDialog() {
        const overlay = document.createElement('div');
        overlay.className = 'tl-dialog-overlay';
        const dialog = document.createElement('div');
        dialog.className = 'tl-dialog';
        dialog.innerHTML = `
            <h3>새 티어리스트</h3>
            <label>제목</label>
            <input type="text" id="tl-new-title" placeholder="예: 2024 베스트 애니">
            <label>카테고리</label>
            <input type="text" id="tl-new-cat" placeholder="예: 애니메이션">
            <div class="tl-dialog-actions">
                <button class="tl-btn" id="tl-new-cancel">취소</button>
                <button class="tl-btn tl-btn-primary" id="tl-new-ok">생성</button>
            </div>
        `;
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        dialog.querySelector('#tl-new-cancel').onclick = () => overlay.remove();
        overlay.onclick = e => { if (e.target === overlay) overlay.remove(); };
        dialog.querySelector('#tl-new-ok').onclick = () => {
            const title = dialog.querySelector('#tl-new-title').value.trim();
            const cat = dialog.querySelector('#tl-new-cat').value.trim();
            createList(title || '새 티어리스트', cat);
            overlay.remove();
            renderAll();
            Toolbox.showToast('새 티어리스트 생성됨');
            Mdd.setMood('cheer'); Mdd.say('새 리스트 시작이에요!');
        };
        dialog.querySelector('#tl-new-title').focus();
    }

    function showTierSettingsDialog() {
        const list = currentList();
        if (!list) return;

        const overlay = document.createElement('div');
        overlay.className = 'tl-dialog-overlay';
        const dialog = document.createElement('div');
        dialog.className = 'tl-dialog';

        function renderRows() {
            let html = '<h3>티어 설정</h3><div style="display:flex; flex-direction:column; gap:8px;">';
            list.tiers.forEach((t, i) => {
                html += `<div style="display:flex; gap:8px; align-items:center;">
                    <input type="color" value="${t.color}" data-idx="${i}" class="tl-tier-color" style="width:32px;height:28px;border:none;cursor:pointer;">
                    <input type="text" value="${Toolbox.escapeHtml(t.label)}" data-idx="${i}" class="tl-tier-label-input" style="width:60px;text-align:center;">
                    <button class="tl-btn" data-idx="${i}" data-dir="up" style="padding:4px 8px; ${i === 0 ? 'visibility:hidden' : ''}">↑</button>
                    <button class="tl-btn" data-idx="${i}" data-dir="down" style="padding:4px 8px; ${i === list.tiers.length - 1 ? 'visibility:hidden' : ''}">↓</button>
                    <button class="tl-btn tl-btn-danger" data-idx="${i}" data-action="del" style="padding:4px 8px;">✕</button>
                </div>`;
            });
            html += '</div>';
            html += `<div style="margin-top:12px;display:flex;gap:8px;">
                <button class="tl-btn" id="tl-tier-add">+ 티어 추가</button>
            </div>
            <div class="tl-dialog-actions">
                <button class="tl-btn" id="tl-tier-close">닫기</button>
            </div>`;
            dialog.innerHTML = html;

            dialog.querySelectorAll('.tl-tier-color').forEach(el => {
                el.oninput = () => {
                    list.tiers[+el.dataset.idx].color = el.value;
                    list.updatedAt = Date.now(); saveState();
                };
            });
            dialog.querySelectorAll('.tl-tier-label-input').forEach(el => {
                el.oninput = () => {
                    list.tiers[+el.dataset.idx].label = el.value;
                    list.updatedAt = Date.now(); saveState();
                };
            });
            dialog.querySelectorAll('[data-dir]').forEach(el => {
                el.onclick = () => {
                    const idx = +el.dataset.idx;
                    const dir = el.dataset.dir === 'up' ? -1 : 1;
                    const target = idx + dir;
                    if (target < 0 || target >= list.tiers.length) return;
                    [list.tiers[idx], list.tiers[target]] = [list.tiers[target], list.tiers[idx]];
                    list.updatedAt = Date.now(); saveState();
                    renderRows();
                };
            });
            dialog.querySelectorAll('[data-action="del"]').forEach(el => {
                el.onclick = () => {
                    const idx = +el.dataset.idx;
                    const tier = list.tiers[idx];
                    const items = list.rankings[tier.id] || [];
                    items.forEach(id => list.rankings._pool.push(id));
                    delete list.rankings[tier.id];
                    list.tiers.splice(idx, 1);
                    list.updatedAt = Date.now(); saveState();
                    renderRows();
                };
            });
            dialog.querySelector('#tl-tier-add').onclick = () => {
                const id = 't-' + uid();
                list.tiers.push({ id, label: '?', color: '#cccccc' });
                list.rankings[id] = [];
                list.updatedAt = Date.now(); saveState();
                renderRows();
            };
            dialog.querySelector('#tl-tier-close').onclick = () => { overlay.remove(); renderEditor(); };
        }

        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
        overlay.onclick = e => { if (e.target === overlay) { overlay.remove(); renderEditor(); } };
        renderRows();
    }

    /* ========================= Export / Import ========================= */

    async function exportAsImage() {
        const list = currentList();
        if (!list) return;

        const boardEl = document.querySelector('#tl-editor-board');
        if (!boardEl) return;

        Toolbox.showToast('이미지 생성 중...');

        if (!window.html2canvas) {
            await new Promise((res, rej) => {
                const s = document.createElement('script');
                s.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
                s.onload = res;
                s.onerror = rej;
                document.head.appendChild(s);
            });
        }

        try {
            const canvas = await window.html2canvas(boardEl, {
                backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--bg-tertiary').trim() || '#1a1a2e',
                scale: 2,
                useCORS: true,
            });
            const link = document.createElement('a');
            link.download = (list.title || 'tierlist') + '.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
            Toolbox.showToast('이미지 저장됨');
            Mdd.setMood('cheer'); Mdd.say('스크린샷 완료했어요!');
        } catch (err) {
            Toolbox.showToast('이미지 생성 실패', 'error', err);
        }
    }

    async function exportAsJSON() {
        const list = currentList();
        if (!list) return;

        const imageKeys = Object.values(list.items).filter(i => i.imageKey).map(i => i.imageKey);
        const imageData = await TierDB.getMany(imageKeys);

        const exportData = {
            version: 1,
            exportedAt: Date.now(),
            list: { ...list },
            images: imageData,
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const link = document.createElement('a');
        link.download = (list.title || 'tierlist') + '.json';
        link.href = URL.createObjectURL(blob);
        link.click();
        URL.revokeObjectURL(link.href);
        Toolbox.showToast('JSON 내보내기 완료');
    }

    async function importFromJSON() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = async () => {
            const file = input.files[0];
            if (!file) return;
            try {
                const text = await file.text();
                const data = JSON.parse(text);
                if (!data.list || !data.version) throw new Error('유효하지 않은 파일');

                const newId = 'tl-' + uid();
                const itemMap = {};
                const newItems = {};

                for (const [oldId, item] of Object.entries(data.list.items)) {
                    const nid = 'ti-' + uid();
                    itemMap[oldId] = nid;
                    newItems[nid] = { ...item, id: nid };
                    if (item.imageKey && data.images && data.images[item.imageKey]) {
                        const newImgKey = 'tl-img-' + uid();
                        await TierDB.save(newImgKey, data.images[item.imageKey]);
                        newItems[nid].imageKey = newImgKey;
                    } else {
                        newItems[nid].imageKey = null;
                    }
                }

                const newRankings = {};
                for (const [k, arr] of Object.entries(data.list.rankings)) {
                    newRankings[k] = arr.map(oid => itemMap[oid] || oid);
                }

                state.lists[newId] = {
                    ...data.list,
                    id: newId,
                    items: newItems,
                    rankings: newRankings,
                    createdAt: Date.now(),
                    updatedAt: Date.now(),
                };
                state.currentListId = newId;
                saveState();
                renderAll();
                Toolbox.showToast('가져오기 완료');
                Mdd.setMood('happy'); Mdd.say('데이터 불러왔어요!');
            } catch (err) {
                Toolbox.showToast('가져오기 실패', 'error', err);
            }
        };
        input.click();
    }

    /* ========================= Renderers ========================= */

    let editorContainer = null;
    let listContainer = null;
    let statsContainer = null;

    function renderAll() {
        renderEditor();
        renderListTab();
        renderStats();
    }

    /* ── Editor Tab ── */

    async function renderEditor() {
        if (!editorContainer) return;
        const list = currentList();

        if (!list) {
            editorContainer.innerHTML = `<div class="tl-wrap">
                <div class="tl-list-empty">
                    <div style="font-size:32px; margin-bottom:12px;">📋</div>
                    <div>티어리스트가 없습니다</div>
                    <div style="margin-top:12px;"><button class="tl-btn tl-btn-primary" id="tl-empty-create">새 티어리스트 만들기</button></div>
                </div>
            </div>`;
            editorContainer.querySelector('#tl-empty-create')?.addEventListener('click', showNewListDialog);
            return;
        }

        const allImageKeys = Object.values(list.items).filter(i => i.imageKey).map(i => i.imageKey);
        const imgMap = await TierDB.getMany(allImageKeys);

        function cardHtml(itemId) {
            const item = list.items[itemId];
            if (!item) return '';
            const imgData = item.imageKey ? imgMap[item.imageKey] : null;
            const inner = imgData
                ? `<img src="${imgData}" alt="${Toolbox.escapeHtml(item.name || '')}">`
                : `<div class="tl-item-text">${Toolbox.escapeHtml(item.name || '?')}</div>`;
            const nameTag = item.name ? `<div class="tl-item-name">${Toolbox.escapeHtml(item.name)}</div>` : '';
            return `<div class="tl-item" data-item-id="${itemId}" touch-action="none">${inner}${nameTag}</div>`;
        }

        const listSelector = Object.keys(state.lists).length > 1
            ? `<select id="tl-list-select">${Object.values(state.lists).map(l =>
                `<option value="${l.id}" ${l.id === state.currentListId ? 'selected' : ''}>${Toolbox.escapeHtml(l.title)}</option>`
            ).join('')}</select>` : `<span style="font-weight:600;color:var(--text-primary);font-size:14px;">${Toolbox.escapeHtml(list.title)}</span>`;

        let html = `<div class="tl-wrap">
            <div class="tl-toolbar">
                ${listSelector}
                <div class="tl-toolbar-spacer"></div>
                <button class="tl-btn" id="tl-btn-add"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg> 추가</button>
                <button class="tl-btn" id="tl-btn-settings"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg> 설정</button>
                <button class="tl-btn" id="tl-btn-export-img"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg> 캡처</button>
                <button class="tl-btn" id="tl-btn-export-json">JSON</button>
                <button class="tl-btn" id="tl-btn-import">가져오기</button>
            </div>
            <div class="tl-board" id="tl-editor-board">`;

        list.tiers.forEach(tier => {
            const items = (list.rankings[tier.id] || []).map(cardHtml).join('');
            html += `<div class="tl-row">
                <div class="tl-label" style="background:${tier.color};color:#000;" data-tier-id="${tier.id}" title="클릭하여 편집">${Toolbox.escapeHtml(tier.label)}</div>
                <div class="tl-dropzone" data-tier-id="${tier.id}">${items}</div>
            </div>`;
        });

        html += `</div>
            <div class="tl-pool-section">
                <div class="tl-pool-header">
                    <span class="tl-pool-title">미배치 아이템 (${(list.rankings._pool || []).length})</span>
                </div>
                <div class="tl-pool" data-tier-id="_pool">
                    ${(list.rankings._pool || []).map(cardHtml).join('')}
                </div>
            </div>
        </div>`;

        editorContainer.innerHTML = html;

        const wrap = editorContainer.querySelector('.tl-wrap');
        const boardEl = editorContainer.querySelector('.tl-board');
        const poolEl = editorContainer.querySelector('.tl-pool');

        initDnD(boardEl, poolEl);

        editorContainer.querySelector('#tl-list-select')?.addEventListener('change', e => {
            state.currentListId = e.target.value;
            saveState();
            renderAll();
        });

        editorContainer.querySelector('#tl-btn-add')?.addEventListener('click', showAddItemDialog);
        editorContainer.querySelector('#tl-btn-settings')?.addEventListener('click', showTierSettingsDialog);
        editorContainer.querySelector('#tl-btn-export-img')?.addEventListener('click', exportAsImage);
        editorContainer.querySelector('#tl-btn-export-json')?.addEventListener('click', exportAsJSON);
        editorContainer.querySelector('#tl-btn-import')?.addEventListener('click', importFromJSON);

        wrap.addEventListener('contextmenu', e => {
            const itemEl = e.target.closest('.tl-item');
            if (!itemEl) return;
            e.preventDefault();
            const itemId = itemEl.dataset.itemId;
            showContextMenu(e.clientX, e.clientY, [
                { label: '편집', action: () => showEditItemDialog(itemId) },
                'sep',
                { label: '삭제', danger: true, action: () => {
                    removeItem(itemId);
                    renderEditor();
                    Toolbox.showToast('아이템 삭제됨');
                    Mdd.setMood('shock'); Mdd.say('지워버렸어요...');
                }},
            ]);
        });

        wrap.addEventListener('paste', async e => {
            const items = Array.from(e.clipboardData?.items || []);
            const imageItem = items.find(i => i.type.startsWith('image/'));
            if (!imageItem) return;
            e.preventDefault();
            const file = imageItem.getAsFile();
            if (!file) return;
            const imgKey = await processImageFile(file);
            addItem('', imgKey);
            renderEditor();
            Toolbox.showToast('클립보드에서 추가됨');
            Mdd.setMood('happy'); Mdd.say('붙여넣기 성공이에요!');
        });

        wrap.setAttribute('tabindex', '0');
    }

    /* ── List Tab ── */

    function renderListTab() {
        if (!listContainer) return;
        const entries = Object.values(state.lists);

        let html = `<div style="display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap;">
            <button class="tl-btn tl-btn-primary" id="tl-list-new"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg> 새 리스트</button>
            <button class="tl-btn" id="tl-list-import">JSON 가져오기</button>
        </div>`;

        if (!entries.length) {
            html += '<div class="tl-list-empty"><div style="font-size:32px; margin-bottom:12px;">📋</div><div>아직 만든 티어리스트가 없어요</div></div>';
        } else {
            html += '<div class="tl-list-grid">';
            entries.sort((a, b) => b.updatedAt - a.updatedAt).forEach(l => {
                const itemCount = Object.keys(l.items).length;
                const ranked = Object.entries(l.rankings).filter(([k]) => k !== '_pool').reduce((s, [, arr]) => s + arr.length, 0);
                const date = new Date(l.updatedAt);
                html += `<div class="tl-list-card${l.id === state.currentListId ? ' active' : ''}" data-list-id="${l.id}">
                    <div class="tl-list-card-title">${Toolbox.escapeHtml(l.title)}</div>
                    <div class="tl-list-card-meta">아이템 ${itemCount}개 · 배치 ${ranked}개 · ${date.toLocaleDateString()}</div>
                    ${l.category ? `<div class="tl-list-card-cat">${Toolbox.escapeHtml(l.category)}</div>` : ''}
                </div>`;
            });
            html += '</div>';
        }

        listContainer.innerHTML = html;

        listContainer.querySelector('#tl-list-new')?.addEventListener('click', showNewListDialog);
        listContainer.querySelector('#tl-list-import')?.addEventListener('click', importFromJSON);

        listContainer.querySelectorAll('.tl-list-card').forEach(card => {
            card.addEventListener('click', () => {
                state.currentListId = card.dataset.listId;
                saveState();
                renderAll();
            });
            card.addEventListener('contextmenu', e => {
                e.preventDefault();
                const id = card.dataset.listId;
                showContextMenu(e.clientX, e.clientY, [
                    { label: '선택', action: () => { state.currentListId = id; saveState(); renderAll(); } },
                    { label: '복제', action: () => { duplicateList(id); renderAll(); Toolbox.showToast('복제됨'); } },
                    'sep',
                    { label: '삭제', danger: true, action: () => {
                        if (!confirm('이 티어리스트를 삭제하시겠습니까?')) return;
                        deleteList(id);
                        renderAll();
                        Toolbox.showToast('삭제됨');
                        Mdd.setMood('sad'); Mdd.say('지워버렸어요...');
                    }},
                ]);
            });
        });
    }

    /* ── Stats Tab ── */

    function renderStats() {
        if (!statsContainer) return;
        const entries = Object.values(state.lists);

        if (!entries.length) {
            statsContainer.innerHTML = '<div class="tl-list-empty"><div style="font-size:32px; margin-bottom:12px;">📊</div><div>통계를 표시할 티어리스트가 없습니다</div></div>';
            return;
        }

        let totalItems = 0, totalRanked = 0;
        const tierCounts = {};

        entries.forEach(l => {
            const itemCount = Object.keys(l.items).length;
            totalItems += itemCount;
            l.tiers.forEach(t => {
                const count = (l.rankings[t.id] || []).length;
                totalRanked += count;
                const key = t.label.toUpperCase();
                tierCounts[key] = (tierCounts[key] || { count: 0, color: t.color });
                tierCounts[key].count += count;
            });
        });

        const maxCount = Math.max(1, ...Object.values(tierCounts).map(v => v.count));

        let html = `<div class="tl-stats">
            <div class="tl-stat-cards">
                <div class="tl-stat-card"><div class="tl-stat-card-value">${entries.length}</div><div class="tl-stat-card-label">티어리스트</div></div>
                <div class="tl-stat-card"><div class="tl-stat-card-value">${totalItems}</div><div class="tl-stat-card-label">총 아이템</div></div>
                <div class="tl-stat-card"><div class="tl-stat-card-value">${totalRanked}</div><div class="tl-stat-card-label">배치 완료</div></div>
                <div class="tl-stat-card"><div class="tl-stat-card-value">${totalItems - totalRanked}</div><div class="tl-stat-card-label">미배치</div></div>
            </div>
            <div class="tl-stat-section">
                <h4>티어별 분포</h4>`;

        for (const [label, { count, color }] of Object.entries(tierCounts)) {
            const pct = Math.round((count / maxCount) * 100);
            html += `<div class="tl-bar-row">
                <div class="tl-bar-label" style="color:${color}">${Toolbox.escapeHtml(label)}</div>
                <div class="tl-bar-track"><div class="tl-bar-fill" style="width:${pct}%;background:${color};"><span class="tl-bar-count">${count}</span></div></div>
            </div>`;
        }

        html += `</div>
            <div class="tl-stat-section">
                <h4>리스트별 요약</h4>
                <table class="tl-stat-table">
                    <thead><tr><th>제목</th><th>카테고리</th><th>아이템</th><th>배치</th><th>최근 수정</th></tr></thead>
                    <tbody>`;

        entries.sort((a, b) => b.updatedAt - a.updatedAt).forEach(l => {
            const ic = Object.keys(l.items).length;
            const rc = Object.entries(l.rankings).filter(([k]) => k !== '_pool').reduce((s, [, arr]) => s + arr.length, 0);
            html += `<tr>
                <td>${Toolbox.escapeHtml(l.title)}</td>
                <td>${Toolbox.escapeHtml(l.category || '-')}</td>
                <td>${ic}</td>
                <td>${rc}</td>
                <td>${new Date(l.updatedAt).toLocaleDateString()}</td>
            </tr>`;
        });

        html += '</tbody></table></div></div>';
        statsContainer.innerHTML = html;
    }

    /* ========================= Widget Registration ========================= */

    loadState();

    Toolbox.register({
        id: 'tierlist',
        title: '티어리스트',
        category: 'tool',
        desc: '티어 리스트를 만들고 관리합니다',
        layout: 'form',
        icon: '<path d="M3 3h18v4H3zM3 9h14v4H3zM3 15h10v4H3z"/>',
        tabs: [
            {
                id: 'tl-edit',
                label: '편집',
                build(container) {
                    editorContainer = container;
                    renderEditor();
                }
            },
            {
                id: 'tl-list',
                label: '목록',
                build(container) {
                    listContainer = container;
                    renderListTab();
                }
            },
            {
                id: 'tl-stats',
                label: '통계',
                build(container) {
                    statsContainer = container;
                    renderStats();
                }
            },
        ]
    });

})();
