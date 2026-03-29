(function () {
    const T = window.Tierlist = window.Tierlist || {};

    let publishedIndexCache = null;
    const publishedJsonCache = new Map();

    /**
     * 티어리스트 JSON(index·카탈로그)이 놓인 디렉터리 URL(끝에 슬래시 없음).
     * Jekyll: 앱은 permalink로 /karmolab/ 이지만 data 는 /apps/karmolab/data/ 에 그대로 출력됨.
     * 로컬 file://·/apps/karmolab/index.html 은 현재 문서 기준 디렉터리로 폴백.
     */
    function karmolabPublishedDataRoot() {
        const { origin, pathname } = location;
        const p = pathname.replace(/\/index\.html?$/i, '') || '/';
        if (p === '/karmolab' || p.startsWith('/karmolab/')) {
            return `${origin}/apps/karmolab`;
        }
        if (p.startsWith('/apps/karmolab')) {
            return `${origin}/apps/karmolab`;
        }
        return new URL('.', location.href).href.replace(/\/?$/, '');
    }

    function publishedIndexUrl() {
        const root = karmolabPublishedDataRoot();
        return new URL('data/tierlists/index.json', `${root}/`).toString();
    }

    function resolvePublishedUrl(url) {
        const s = String(url || '').trim();
        if (!s) return s;
        if (/^https?:\/\//i.test(s)) return s;
        const root = karmolabPublishedDataRoot();
        return new URL(s, `${root}/`).toString();
    }

    async function getPublishedIndex() {
        if (publishedIndexCache) return publishedIndexCache;
        const res = await fetch(publishedIndexUrl(), { cache: 'no-store' });
        if (!res.ok) throw new Error('index.json 로드 실패: ' + res.status);
        const items = await res.json();
        publishedIndexCache = Array.isArray(items) ? items : [];
        return publishedIndexCache;
    }

    async function getPublishedJsonByUrl(relUrl) {
        const url = resolvePublishedUrl(relUrl);
        if (publishedJsonCache.has(url)) return publishedJsonCache.get(url);
        const r = await fetch(url, { cache: 'no-store' });
        if (!r.ok) throw new Error('JSON 로드 실패: ' + r.status);
        const data = await r.json();
        publishedJsonCache.set(url, data);
        return data;
    }

    /** 가져오기·동기화 시 디스크에 갱신된 풀 JSON이 캐시에 남지 않게 */
    function clearPublishedFetchCache() {
        publishedIndexCache = null;
        publishedJsonCache.clear();
    }

    function collectRankingItemIds(list) {
        const s = new Set();
        Object.values(list.rankings || {}).forEach(arr => {
            (arr || []).forEach(id => { if (id) s.add(id); });
        });
        return s;
    }

    function mergeCatalogItemForHydrate(baseRaw, ovr, id) {
        const base = baseRaw
            ? { id, name: baseRaw.name || '', imageKey: baseRaw.imageKey ?? null }
            : null;
        const hasOvr = ovr && typeof ovr === 'object' && Object.keys(ovr).length > 0;
        if (!base && !hasOvr) return { id, name: '?', imageKey: null, tlOrigin: 'custom' };
        if (!base) {
            const m = { id, name: '', imageKey: null, ...ovr, id };
            m.tlOrigin = 'custom';
            delete m.tlEdited;
            return m;
        }
        const merged = { ...base, ...(hasOvr ? ovr : {}), id };
        if (merged.tlOrigin === 'custom') {
            delete merged.tlEdited;
            return merged;
        }
        merged.tlOrigin = 'catalog';
        const nameCh = hasOvr && Object.prototype.hasOwnProperty.call(ovr, 'name')
            && (merged.name || '') !== (base.name || '');
        const imgCh = hasOvr && Object.prototype.hasOwnProperty.call(ovr, 'imageKey')
            && (merged.imageKey ?? null) !== (base.imageKey ?? null);
        const baseForLabels = { id, name: base.name, imageKey: base.imageKey };
        const labelCh = hasOvr && normUserLabelIds(merged) !== normUserLabelIds(baseForLabels);
        /** 풀 대비 이름·이미지·라벨이 다르면 수정 뱃지(슬림에선 prune가 정본과 동일한 필드만 제거) */
        if (ovr?.tlEdited === true || nameCh || imgCh || labelCh) merged.tlEdited = true;
        else delete merged.tlEdited;
        return merged;
    }

    /**
     * 현재 카탈로그 기준으로 슬림 JSON을 정리.
     * 1) 풀에 없는 id의 override·순위 제거(custom 출처만 예외)
     * 2) tlEdited 없을 때: name·imageKey는 풀 정본과 같을 때만 제거(중복). 다르면 인스턴스 표시 유지.
     */
    function pruneSlimPayloadAgainstCatalog(data, catalogItems) {
        const catIds = new Set(Object.keys(catalogItems || {}));
        const ov = data.itemOverrides;
        if (ov && typeof ov === 'object') {
            Object.keys(ov).forEach(k => {
                if (catIds.has(k)) return;
                if (ov[k]?.tlOrigin === 'custom') return;
                delete ov[k];
            });
            Object.keys(ov).forEach(k => {
                if (!catIds.has(k)) return;
                const ovr = ov[k];
                if (!ovr || typeof ovr !== 'object') {
                    delete ov[k];
                    return;
                }
                if (ovr.tlEdited === true) return;
                const cat = catalogItems[k];
                const next = { ...ovr };
                if (Object.prototype.hasOwnProperty.call(next, 'name')) {
                    const cn = cat && typeof cat === 'object' ? String(cat.name ?? '') : '';
                    if (String(next.name ?? '') === cn) delete next.name;
                }
                if (Object.prototype.hasOwnProperty.call(next, 'imageKey')) {
                    const cik = cat && typeof cat === 'object' ? (cat.imageKey ?? null) : null;
                    if ((next.imageKey ?? null) === cik) delete next.imageKey;
                }
                const metaKeys = Object.keys(next).filter(x => x !== 'id');
                if (!metaKeys.length) delete ov[k];
                else ov[k] = next;
            });
        }
        const rk = data.list?.rankings;
        if (rk && typeof rk === 'object') {
            Object.keys(rk).forEach(tid => {
                const arr = rk[tid];
                if (!Array.isArray(arr)) return;
                rk[tid] = arr.filter(id => {
                    if (!id) return false;
                    if (catIds.has(id)) return true;
                    return !!(ov && ov[id]?.tlOrigin === 'custom');
                });
            });
        }
    }

    /** 슬림 순위 JSON(catalogRef + itemOverrides) → 풀과 합친 list + images */
    async function hydrateSlimInstance(data) {
        const rel = data.catalogRef?.url;
        if (!rel || typeof data.itemOverrides !== 'object') throw new Error('catalogRef.url 과 itemOverrides 가 필요합니다.');
        publishedJsonCache.delete(resolvePublishedUrl(rel));
        const cat = await getPublishedJsonByUrl(rel);
        if (!T.state.isCatalogPayload(cat)) throw new Error('catalogRef 가 가리키는 파일이 후보 풀이 아닙니다.');
        const catItems = cat.items || {};
        pruneSlimPayloadAgainstCatalog(data, catItems);
        const overrides = data.itemOverrides || {};
        const list = JSON.parse(JSON.stringify(data.list || {}));
        /** 순위에 없어도 itemOverrides(라벨 등)만 있는 항목은 복원해야 함 */
        const ids = new Set([
            ...collectRankingItemIds(list),
            ...Object.keys(overrides),
        ]);
        const items = {};
        ids.forEach(id => {
            const inCat = Object.prototype.hasOwnProperty.call(catItems, id);
            items[id] = mergeCatalogItemForHydrate(inCat ? catItems[id] : undefined, overrides[id], id);
            if (!inCat) {
                items[id].tlOrigin = 'custom';
                delete items[id].tlEdited;
            }
        });
        list.items = items;
        list.meta = { ...(list.meta || {}), catalogUrl: rel };
        T.state.reconcileListWithCatalogPayload(list, catItems);
        T.state.pruneStaleCatalogBindings(list, catItems);
        const images = { ...(cat.images || {}), ...(data.images || {}) };
        return { list, images };
    }

    function isSlimInstancePayload(data) {
        return data?.kind === 'instance' && data.catalogRef?.url && typeof data.itemOverrides === 'object' && data.list;
    }

    /** 목록 탭 임베드 카드용: JSON을 읽어 한 줄로 표시할 개수 문구 */
    async function getPublishedPreviewCountLine(relUrl) {
        if (!relUrl) return '';
        try {
            const data = await getPublishedJsonByUrl(relUrl);
            if (T.state.isCatalogPayload(data)) {
                const n = Object.keys(data.items || {}).length;
                return `총 ${n}개`;
            }
            if (isSlimInstancePayload(data)) {
                let poolN = null;
                try {
                    const cat = await getPublishedJsonByUrl(data.catalogRef.url);
                    if (T.state.isCatalogPayload(cat)) poolN = Object.keys(cat.items || {}).length;
                } catch (_) { /* 풀 URL 실패 */ }
                const ranked = collectRankingItemIds(data.list).size;
                if (poolN != null) return `후보 풀 총 ${poolN}개 · 판 ${ranked}개`;
                return ranked ? `판 ${ranked}개` : '';
            }
            if (data?.list) {
                const n = Object.keys(data.list.items || {}).length;
                if (n) return `총 ${n}개`;
                const ranked = collectRankingItemIds(data.list).size;
                return ranked ? `판 ${ranked}개` : '';
            }
        } catch (_) {}
        return '';
    }

    async function resolveCatalogIndexIdForUrl(catalogUrl) {
        try {
            const idx = await getPublishedIndex();
            const u = resolvePublishedUrl(catalogUrl);
            for (const row of idx) {
                if (!row?.url) continue;
                if (resolvePublishedUrl(row.url) === u) return row.id || '';
            }
        } catch (_) {}
        return '';
    }

    function normUserLabelIds(it) {
        return JSON.stringify([...(it?.userLabelIds || [])].filter(Boolean).sort());
    }

    function diffItemOverride(base, item) {
        if (!base) {
            const o = {
                id: item.id,
                name: item.name || '',
                imageKey: item.imageKey ?? null,
                tlOrigin: 'custom',
            };
            if ((item.userLabelIds || []).length) o.userLabelIds = [...item.userLabelIds];
            return o;
        }
        const o = {};
        const nameCh = (item.name || '') !== (base.name || '');
        const ik = item.imageKey ?? null;
        const bk = base.imageKey ?? null;
        const imgCh = ik !== bk;
        if (nameCh) o.name = item.name;
        if (imgCh) o.imageKey = item.imageKey ?? null;
        if (nameCh || imgCh) o.tlEdited = true;
        if (item.tlEdited === true) o.tlEdited = true;
        if (item.tlOrigin === 'custom') o.tlOrigin = 'custom';
        if (normUserLabelIds(item) !== normUserLabelIds(base)) {
            o.userLabelIds = [...(item.userLabelIds || [])];
            o.tlEdited = true;
        }
        return Object.keys(o).length ? o : null;
    }

    /**
     * 로컬 인스턴스: 풀 출처 카드 표시가 풀 정본과 다르면 tlEdited 맞춤(새로고침 뒤 뱃지 누락 방지).
     * 정본과 완전히 같으면 tlEdited 제거.
     */
    function syncCatalogItemEditedFlags(list, catItems) {
        if (!list?.items || !catItems || typeof catItems !== 'object') return false;
        let changed = false;
        Object.keys(list.items).forEach(id => {
            const it = list.items[id];
            if (!it || it.tlOrigin === 'custom') return;
            const base = catItems[id];
            if (!base || typeof base !== 'object') return;
            const nameDiff = (it.name || '') !== (base.name || '');
            const imgDiff = (it.imageKey ?? null) !== (base.imageKey ?? null);
            const labelDiff = normUserLabelIds(it) !== normUserLabelIds(base);
            const shouldMark = nameDiff || imgDiff || labelDiff;
            if (shouldMark) {
                if (!it.tlEdited) {
                    it.tlEdited = true;
                    changed = true;
                }
            } else if (it.tlEdited) {
                delete it.tlEdited;
                changed = true;
            }
        });
        return changed;
    }

    /** meta.catalogUrl 이 있고 풀을 fetch 할 수 있으면 슬림 페이로드, 아니면 null */
    async function tryBuildSlimInstanceExport(list, meta) {
        const catalogUrl = String(list.meta?.catalogUrl || '').trim();
        if (!catalogUrl) return null;
        let cat;
        try {
            cat = await getPublishedJsonByUrl(catalogUrl);
        } catch (_) {
            return null;
        }
        if (!T.state.isCatalogPayload(cat)) return null;
        const catalogItems = cat.items || {};
        const overrides = {};
        for (const [id, item] of Object.entries(list.items || {})) {
            const diff = diffItemOverride(catalogItems[id], item);
            if (diff) overrides[id] = diff;
        }
        const imageKeys = new Set();
        Object.values(overrides).forEach(o => {
            if (o && o.imageKey) imageKeys.add(o.imageKey);
        });
        const images = await T.db.getMany([...imageKeys]);
        const listCopy = JSON.parse(JSON.stringify(list));
        delete listCopy.items;
        const refId = await resolveCatalogIndexIdForUrl(catalogUrl);
        return {
            version: 2,
            kind: 'instance',
            title: meta.title || list.title,
            catalogRef: { id: refId, url: catalogUrl },
            itemOverrides: overrides,
            list: listCopy,
            images,
        };
    }

    async function openPublishedDirect(relUrl, meta) {
        const data = await getPublishedJsonByUrl(relUrl);
        if (T.state.isCatalogPayload(data)) {
            T.state.openPublished({ ...(meta || {}), url: relUrl }, data);
        } else if (isSlimInstancePayload(data)) {
            const h = await hydrateSlimInstance(data);
            T.state.openPublished({ ...(meta || {}), url: relUrl }, {
                version: 2,
                kind: 'instance',
                list: h.list,
                images: h.images,
            });
        } else if (data?.list) {
            T.state.openPublished({ ...(meta || {}), url: relUrl }, data);
            const plist = data.list;
            const curl = String(plist?.meta?.catalogUrl || '').trim();
            if (curl) {
                try {
                    const cat = await getPublishedJsonByUrl(curl);
                    if (T.state.isCatalogPayload(cat)) {
                        const ci = cat.items || {};
                        T.state.reconcileListWithCatalogPayload(plist, ci);
                        T.state.pruneStaleCatalogBindings(plist, ci);
                    }
                } catch (_) { /* 오프라인 */ }
            }
        } else {
            throw new Error('유효하지 않은 JSON(후보 풀·순위 중 하나여야 해요)');
        }
        await T.render.renderAll();
    }

    async function buildExportPayload() {
        if (T.state.isPublishedCatalogMode()) {
            const snap = T.state.getPublishedCatalogSnapshot();
            const d = snap?.data;
            if (!d) return null;
            const imageKeys = Object.values(d.items || {}).filter(i => i.imageKey).map(i => i.imageKey);
            const images = await T.db.getMany(imageKeys);
            return {
                version: 2,
                kind: 'catalog',
                exportedAt: Date.now(),
                title: snap.meta?.title || d.title || '후보 풀',
                items: JSON.parse(JSON.stringify(d.items || {})),
                images,
            };
        }
        const list = T.state.currentList();
        if (!list) return null;
        const meta = T.state.currentMeta();
        try {
            const slim = await tryBuildSlimInstanceExport(list, meta);
            if (slim) {
                return { ...slim, exportedAt: Date.now() };
            }
        } catch (_) { /* 풀 fetch 실패 시 아래 전체보내기 */ }
        const imageKeys = Object.values(list.items || {}).filter(i => i.imageKey).map(i => i.imageKey);
        const images = await T.db.getMany(imageKeys);
        return {
            version: 2,
            kind: 'instance',
            exportedAt: Date.now(),
            title: meta.title || list.title,
            list: JSON.parse(JSON.stringify(list)),
            images,
        };
    }

    function downloadJson(filename, obj) {
        const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
        const link = document.createElement('a');
        link.download = filename;
        link.href = URL.createObjectURL(blob);
        link.click();
        URL.revokeObjectURL(link.href);
    }

    async function showJsonPreview() {
        const data = await buildExportPayload();
        if (!data) {
            Toolbox.showToast('보낼 데이터가 없습니다.', 'error');
            return;
        }
        const jsonText = JSON.stringify(data, null, 2);

        T.ui.openDialog({
            title: data.kind === 'catalog' ? '후보 풀 JSON' : '순위 JSON',
            wide: true,
            bodyHtml: `
                <div style="font-size:12px; color:var(--text-tertiary); margin-bottom:10px;">
                    ${data.kind === 'catalog'
                        ? '열어 둔 후보 풀(요소만)을 JSON으로 보냅니다.'
                        : data.catalogRef
                            ? '<strong>슬림 순위</strong>: 베이스 후보 풀 URL(<code>catalogRef.url</code>) + 풀과 다른 항목만 <code>itemOverrides</code>에 넣습니다. 풀 JSON은 그대로 두고 순위만 가볍게 올릴 때 쓰세요.'
                            : '순위를 이 브라우저에서 연 데가 후보 풀 URL과 연결되지 않았거나 풀을 불러올 수 없어, <strong>전체 항목</strong>이 들어간 순위 JSON입니다.'}
                </div>
                <textarea id="tl-json-preview" spellcheck="false" readonly style="min-height:360px; white-space:pre; overflow:auto;"></textarea>
                <div class="tl-dialog-actions">
                    <button class="tl-btn" id="tl-json-copy">복사</button>
                    <button class="tl-btn tl-btn-primary" id="tl-json-download">다운로드</button>
                    <button class="tl-btn" id="tl-json-close">닫기</button>
                </div>
            `,
            onMount: ({ dialog, close }) => {
                const ta = dialog.querySelector('#tl-json-preview');
                ta.value = jsonText;
                dialog.querySelector('#tl-json-close').onclick = close;
                dialog.querySelector('#tl-json-copy').onclick = async () => {
                    try {
                        await navigator.clipboard.writeText(jsonText);
                        Toolbox.showToast('클립보드에 복사됨');
                    } catch (_) {
                        ta.focus(); ta.select(); document.execCommand('copy');
                        Toolbox.showToast('클립보드에 복사됨');
                    }
                };
                dialog.querySelector('#tl-json-download').onclick = () => {
                    const base = (data.title || 'tierlist').replace(/[\\/:*?"<>|]+/g, '-');
                    downloadJson(base + '.json', data);
                    Toolbox.showToast('JSON 다운로드 시작');
                };
            }
        });
    }

    async function exportAsImage() {
        const list = T.state.currentList();
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
        } catch (err) {
            Toolbox.showToast('이미지 생성 실패', 'error', err);
        }
    }

    async function saveImagesMap(images) {
        for (const [k, url] of Object.entries(images || {})) {
            if (url) await T.db.save(k, url);
        }
    }

    async function forkPublishedCatalogToLocal() {
        const snap = T.state.getPublishedCatalogSnapshot();
        if (!snap?.data) return null;
        await saveImagesMap(snap.data.images);
        const inst = T.state.forkInstanceFromCatalogData(snap.data, snap.meta);
        return inst;
    }

    async function reconcileImportedInstanceList(list) {
        if (!list) return;
        const st = T.state.getState();
        if (list.catalogId && st.catalogs[list.catalogId]) {
            const ci = st.catalogs[list.catalogId].items || {};
            T.state.reconcileListWithCatalogPayload(list, ci);
            T.state.pruneStaleCatalogBindings(list, ci);
            return;
        }
        const u = String(list.meta?.catalogUrl || '').trim();
        if (!u) return;
        try {
            const cat = await getPublishedJsonByUrl(u);
            if (T.state.isCatalogPayload(cat)) {
                const ci = cat.items || {};
                T.state.reconcileListWithCatalogPayload(list, ci);
                T.state.pruneStaleCatalogBindings(list, ci);
            }
        } catch (_) { /* 로컬 파일/오프라인 */ }
    }

    async function importFromDataObject(data) {
        clearPublishedFetchCache();
        const st = T.state.getState();

        if (T.state.isCatalogPayload(data)) {
            const id = 'cat-' + T.state.uid();
            st.catalogs[id] = {
                id,
                title: data.title || '가져온 후보 풀',
                category: data.category || '',
                updatedAt: Date.now(),
                items: JSON.parse(JSON.stringify(data.items || {})),
            };
            await saveImagesMap(data.images);
            T.state.closePublished?.();
            T.state.saveState();
            await T.render.renderAll();
            return;
        }

        if (data.version >= 2 && data.lists && typeof data.lists === 'object' && Object.keys(data.lists).length) {
            const newIds = [];
            for (const [, list] of Object.entries(data.lists)) {
                const copy = JSON.parse(JSON.stringify(list));
                const newId = 'tl-' + T.state.uid();
                copy.id = newId;
                copy.createdAt = Date.now();
                copy.updatedAt = Date.now();
                copy.meta = { ...(copy.meta || {}), source: 'import', dirty: true };
                st.instances[newId] = copy;
                newIds.push(newId);
            }
            for (const nid of newIds) {
                await reconcileImportedInstanceList(st.instances[nid]);
            }
            await saveImagesMap(data.images);
            T.state.closePublished?.();
            st.currentInstanceId = newIds[0] || st.currentInstanceId;
            T.state.saveState();
            await T.render.renderAll();
            return;
        }

        if (isSlimInstancePayload(data)) {
            const h = await hydrateSlimInstance(data);
            const newLid = 'tl-' + T.state.uid();
            h.list.id = newLid;
            h.list.createdAt = Date.now();
            h.list.updatedAt = Date.now();
            h.list.meta = {
                ...(h.list.meta || {}),
                source: 'import',
                dirty: true,
                catalogUrl: data.catalogRef.url,
            };
            st.instances[newLid] = h.list;
            await saveImagesMap(h.images);
            st.currentInstanceId = newLid;
            T.state.closePublished?.();
            T.state.saveState();
            await T.render.renderAll();
            return;
        }

        if (data.kind === 'instance' && data.list) {
            const newLid = 'tl-' + T.state.uid();
            const list = {
                ...data.list,
                id: newLid,
                createdAt: Date.now(),
                updatedAt: Date.now(),
                meta: { ...(data.list.meta || {}), source: 'import', dirty: true },
            };
            st.instances[newLid] = list;
            await reconcileImportedInstanceList(list);
            await saveImagesMap(data.images);
            st.currentInstanceId = newLid;
            T.state.closePublished?.();
            T.state.saveState();
            await T.render.renderAll();
            return;
        }

        if (data.list) {
            const newLid = 'tl-' + T.state.uid();
            const list = {
                ...data.list,
                id: newLid,
                createdAt: Date.now(),
                updatedAt: Date.now(),
                meta: { ...(data.list.meta || {}), source: 'import', dirty: true },
            };
            st.instances[newLid] = list;
            await reconcileImportedInstanceList(list);
            await saveImagesMap(data.images);
            st.currentInstanceId = newLid;
            T.state.closePublished?.();
            T.state.saveState();
            await T.render.renderAll();
            return;
        }

        throw new Error('유효하지 않은 데이터');
    }

    async function importFromJSONFilePicker() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = async () => {
            const file = input.files[0];
            if (!file) return;
            try {
                const text = await file.text();
                const data = JSON.parse(text);
                await importFromDataObject(data);
                Toolbox.showToast('가져오기 완료');
            } catch (err) {
                Toolbox.showToast('가져오기 실패', 'error', err);
            }
        };
        input.click();
    }

    /**
     * 로컬 순위 편집 시: 연결된 카탈로그와 동기화하고,
     * 풀에서 사라진 id는 후보 출처 카드·override 잔여를 걷어낸다(직접 추가 custom 만 유지).
     */
    async function syncInstanceItemOriginsWithCatalogIfNeeded() {
        if (T.state.isPublishedMode()) return false;
        const list = T.state.currentList();
        if (!list) return false;
        let changed = false;
        const st = T.state.getState();
        if (list.catalogId && st.catalogs[list.catalogId]) {
            const catItems = st.catalogs[list.catalogId].items || {};
            if (T.state.reconcileListWithCatalogPayload(list, catItems)) changed = true;
            if (T.state.pruneStaleCatalogBindings(list, catItems)) changed = true;
            if (syncCatalogItemEditedFlags(list, catItems)) changed = true;
        }
        const url = String(list.meta?.catalogUrl || '').trim();
        if (url && !list.catalogId) {
            try {
                const cat = await getPublishedJsonByUrl(url);
                if (T.state.isCatalogPayload(cat)) {
                    const catItems = cat.items || {};
                    if (T.state.reconcileListWithCatalogPayload(list, catItems)) changed = true;
                    if (T.state.pruneStaleCatalogBindings(list, catItems)) changed = true;
                    if (syncCatalogItemEditedFlags(list, catItems)) changed = true;
                }
            } catch (_) {
                /* 오프라인 등 — 건너뜀 */
            }
        }
        if (changed) T.state.persistList(list);
        return changed;
    }

    async function resetItemToCatalogDefault(itemId) {
        if (!confirm('이름·이미지·라벨을 후보 풀 기준으로 되돌릴까요?')) return false;
        T.state.ensureWritableList?.('resetItem');
        const list = T.state.currentList();
        if (!list?.items?.[itemId]) {
            Toolbox.showToast('항목을 찾을 수 없어요.', 'error');
            return false;
        }
        const st = T.state.getState();
        let entry = null;
        if (list.catalogId && st.catalogs[list.catalogId]?.items?.[itemId]) {
            entry = st.catalogs[list.catalogId].items[itemId];
        }
        if (!entry) {
            const url = String(list.meta?.catalogUrl || '').trim();
            if (url) {
                try {
                    const cat = await getPublishedJsonByUrl(url);
                    if (T.state.isCatalogPayload(cat)) entry = cat.items?.[itemId] || null;
                } catch (_) { /* 오프라인 */ }
            }
        }
        if (!entry) {
            Toolbox.showToast('후보 풀에 없는 항목이거나 풀을 불러올 수 없어요.', 'error');
            return false;
        }
        if (!T.state.applyCatalogEntryToItem(list, itemId, entry)) {
            Toolbox.showToast('되돌리기에 실패했어요.', 'error');
            return false;
        }
        Toolbox.showToast('후보 풀 기준으로 되돌렸어요.');
        return true;
    }

    T.publish = {
        getPublishedIndex,
        getPublishedJsonByUrl,
        getPublishedPreviewCountLine,
        openPublishedDirect,
        buildExportPayload,
        exportAsImage,
        showJsonPreview,
        forkPublishedCatalogToLocal,
        importFromJSONFilePicker,
        syncInstanceItemOriginsWithCatalogIfNeeded,
        resetItemToCatalogDefault,
    };
})();
