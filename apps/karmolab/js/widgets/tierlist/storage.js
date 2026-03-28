(function () {
    const T = window.Tierlist = window.Tierlist || {};

    const STORAGE_KEY = 'toolbox_tierlists';

    const DEFAULT_TIERS = [
        { id: 's', label: 'S', color: '#ff7f7f' },
        { id: 'a', label: 'A', color: '#ffbf7f' },
        { id: 'b', label: 'B', color: '#ffdf7f' },
        { id: 'c', label: 'C', color: '#ffff7f' },
        { id: 'd', label: 'D', color: '#bfff7f' },
        { id: 'f', label: 'F', color: '#7fbfff' },
    ];

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
                    if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: 'id' });
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
                if (item?.dataUrl) { cache.set(id, item.dataUrl); return item.dataUrl; }
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
                        if (req.result?.dataUrl) {
                            cache.set(id, req.result.dataUrl);
                            results[id] = req.result.dataUrl;
                        }
                        res();
                    };
                    req.onerror = () => rej(req.error);
                })));
            } finally { db.close(); }
            return results;
        }

        return { save, get, remove, getMany };
    })();

    let state = { lists: {}, currentListId: null };
    let publishedCurrent = null; // { meta, data }

    function uid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 8); }

    function loadState() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (raw) state = JSON.parse(raw);
        } catch (_) {}
    }

    function saveState() {
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (_) {}
    }

    function isPublishedMode() { return !!publishedCurrent; }

    function currentList() {
        if (publishedCurrent?.data?.list) return publishedCurrent.data.list;
        return state.lists[state.currentListId] || null;
    }

    function currentMeta() {
        const list = currentList();
        if (!list) return { title: '', id: '', source: 'none', dirty: false };
        if (publishedCurrent?.data?.list) {
            return {
                id: publishedCurrent.meta?.id || list.id,
                title: publishedCurrent.meta?.title || list.title,
                source: 'published',
                dirty: false,
                url: publishedCurrent.meta?.url || '',
            };
        }
        const local = state.lists[state.currentListId];
        return {
            id: local?.id || list.id,
            title: local?.title || list.title,
            source: local?.meta?.source || 'local',
            dirty: !!local?.meta?.dirty,
            url: local?.meta?.publishedUrl || '',
        };
    }

    function openPublished(meta, data) { publishedCurrent = { meta: meta || {}, data }; }
    function closePublished() { publishedCurrent = null; }

    function ensureWritableList(reason) {
        if (!isPublishedMode()) return currentList();
        const data = publishedCurrent?.data;
        if (!data?.list) return null;

        const src = data.list;
        const newId = 'draft-' + uid();
        const now = Date.now();
        state.lists[newId] = {
            ...src,
            id: newId,
            createdAt: now,
            updatedAt: now,
            meta: {
                source: 'published-draft',
                publishedId: publishedCurrent.meta?.id || '',
                publishedUrl: publishedCurrent.meta?.url || '',
                dirty: true,
                dirtyReason: reason || '',
            },
        };
        state.currentListId = newId;
        saveState();
        closePublished();
        return state.lists[newId];
    }

    function createList(title, category) {
        closePublished();
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

    function addItem(name, imageKey) {
        const list = ensureWritableList('addItem') || currentList();
        if (!list) return null;
        const id = 'ti-' + uid();
        list.items[id] = { id, name: name || '', imageKey: imageKey || null };
        list.rankings._pool = list.rankings._pool || [];
        list.rankings._pool.push(id);
        list.updatedAt = Date.now();
        saveState();
        return id;
    }

    function removeItem(itemId) {
        const list = ensureWritableList('removeItem') || currentList();
        if (!list) return;
        const imgKey = list.items?.[itemId]?.imageKey;
        if (imgKey) TierDB.remove(imgKey);
        delete list.items[itemId];
        Object.values(list.rankings || {}).forEach(arr => {
            const idx = arr.indexOf(itemId);
            if (idx !== -1) arr.splice(idx, 1);
        });
        list.updatedAt = Date.now();
        saveState();
    }

    function moveItem(itemId, targetTier, insertIdx) {
        const list = ensureWritableList('moveItem') || currentList();
        if (!list) return;
        Object.values(list.rankings || {}).forEach(arr => {
            const idx = arr.indexOf(itemId);
            if (idx !== -1) arr.splice(idx, 1);
        });
        const target = list.rankings[targetTier];
        if (!target) return;
        if (insertIdx === undefined || insertIdx >= target.length) target.push(itemId);
        else target.splice(insertIdx, 0, itemId);
        list.updatedAt = Date.now();
        saveState();
    }

    function duplicateList(listId) {
        closePublished();
        const src = state.lists[listId];
        if (!src) return null;
        const now = Date.now();
        const id = 'tl-' + uid();
        const copied = JSON.parse(JSON.stringify(src));
        copied.id = id;
        copied.title = (src.title || '티어리스트') + ' (복제)';
        copied.createdAt = now;
        copied.updatedAt = now;
        copied.meta = { ...(copied.meta || {}), source: 'duplicate', dirty: true };
        state.lists[id] = copied;
        state.currentListId = id;
        saveState();
        return id;
    }

    function deleteList(listId) {
        closePublished();
        const l = state.lists[listId];
        if (!l) return;

        // 이미지 정리
        try {
            Object.values(l.items || {}).forEach(it => {
                if (it?.imageKey) TierDB.remove(it.imageKey);
            });
        } catch (_) {}

        delete state.lists[listId];

        if (state.currentListId === listId) {
            const next = Object.values(state.lists).sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0))[0];
            state.currentListId = next?.id || null;
        }

        saveState();
    }

    function fileToDataUrl(file) {
        return new Promise((res, rej) => {
            const reader = new FileReader();
            reader.onload = () => res(reader.result);
            reader.onerror = () => rej(reader.error);
            reader.readAsDataURL(file);
        });
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

    async function processImageFile(file) {
        const dataUrl = await fileToDataUrl(file);
        const thumb = await createThumbnail(dataUrl, 200);
        const imgKey = 'tl-img-' + uid();
        await TierDB.save(imgKey, thumb);
        return imgKey;
    }

    T.db = TierDB;
    T.state = {
        uid,
        loadState,
        saveState,
        getState: () => state,
        currentList,
        currentMeta,
        isPublishedMode,
        openPublished,
        closePublished,
        ensureWritableList,
        createList,
        duplicateList,
        deleteList,
        addItem,
        removeItem,
        moveItem,
        processImageFile,
    };
})();

