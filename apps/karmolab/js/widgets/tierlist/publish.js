(function () {
    const T = window.Tierlist = window.Tierlist || {};

    let publishedIndexCache = null;
    const publishedJsonCache = new Map();

    function publishedIndexUrl() {
        return new URL('data/tierlists/index.json', location.href).toString();
    }

    function resolvePublishedUrl(url) {
        return new URL(url, location.href).toString();
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

    async function openPublishedDirect(relUrl, meta) {
        const data = await getPublishedJsonByUrl(relUrl);
        if (!data?.list) throw new Error('유효하지 않은 JSON');
        T.state.openPublished({ ...(meta || {}), url: relUrl }, data);
        T.render.renderAll();
    }

    async function buildExportDataForCurrentList() {
        const list = T.state.currentList();
        if (!list) return null;
        const imageKeys = Object.values(list.items || {}).filter(i => i.imageKey).map(i => i.imageKey);
        const imageData = await T.db.getMany(imageKeys);
        return { version: 1, exportedAt: Date.now(), list: { ...list }, images: imageData };
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
        const list = T.state.currentList();
        if (!list) return;
        const data = await buildExportDataForCurrentList();
        if (!data) return;
        const jsonText = JSON.stringify(data, null, 2);

        T.ui.openDialog({
            title: 'JSON보내기',
            wide: true,
            bodyHtml: `
                <div style="font-size:12px; color:var(--text-tertiary); margin-bottom:10px;">
                    아래 내용을 복사하거나 파일로 다운로드할 수 있어요.
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
                    const safeTitle = (list.title || 'tierlist').replace(/[\\/:*?"<>|]+/g, '-');
                    downloadJson(safeTitle + '.json', data);
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

    async function importFromDataObject(data) {
        if (!data?.list) throw new Error('유효하지 않은 데이터');
        const st = T.state.getState();
        const nid = 'tl-' + T.state.uid();
        st.lists[nid] = { ...data.list, id: nid, createdAt: Date.now(), updatedAt: Date.now(), meta: { source: 'import', dirty: true } };
        st.currentListId = nid;
        T.state.closePublished?.();
        T.state.saveState();
        T.render.renderAll();
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

    T.publish = {
        getPublishedIndex,
        openPublishedDirect,
        exportAsImage,
        showJsonPreview,
        importFromJSONFilePicker,
    };
})();
