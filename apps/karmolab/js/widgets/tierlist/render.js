(function () {
    const T = window.Tierlist = window.Tierlist || {};

    let editorContainer = null;
    let listContainer = null;
    let statsContainer = null;
    /** 편집기: 카드 클릭으로 삭제 */
    let editorDeleteMode = false;
    let editorDeleteModeEscBound = false;
    let lastQuickDeleteBlockedToastAt = 0;
    function setContainers({ editor, list, stats }) {
        if (editor !== undefined) editorContainer = editor;
        if (list !== undefined) listContainer = list;
        if (stats !== undefined) statsContainer = stats;
    }

    /** index.json 행: 후보 풀(catalog) vs 사이트에 올린 순위판(karmo). 미표기는 카탈로그. */
    function publishedIndexGroup(it) {
        const g = String(it?.tierlistGroup ?? it?.group ?? '')
            .toLowerCase()
            .trim();
        if (g === 'karmo' || g === 'ranking' || g === 'instance' || g === '순위') return 'karmo';
        return 'catalog';
    }

    /** 목록에서 항목 선택 후 편집 탭으로 전환 */
    function goToTierlistEditTab() {
        try {
            if (typeof Toolbox !== 'undefined' && Toolbox.switchPage && Toolbox.switchTab) {
                Toolbox.switchPage('tierlist', { pushHistory: false });
                Toolbox.switchTab('tl-edit');
            }
        } catch (_) {}
    }

    function optionsFromPublishedRows(rows, meta) {
        if (!rows.length) return '<option disabled>(항목 없음)</option>';
        return [...rows]
            .sort((a, b) => String(a.title || '').localeCompare(String(b.title || ''), 'ko-KR'))
            .map(it => {
                const selected = T.state.isPublishedMode() && meta.url === it.url;
                return `<option value="blog:${Toolbox.escapeHtml(it.url || '')}" ${selected ? 'selected' : ''}>${Toolbox.escapeHtml(it.title || it.id || 'tierlist')}</option>`;
            })
            .join('');
    }

    async function buildListSelectorHtml(st, meta) {
        const localInst = Object.values(st.instances || {})
            .sort((a, b) => String(a.title || '').localeCompare(String(b.title || ''), 'ko-KR'))
            .map(l => {
                const val = `local:${l.id}`;
                const sel = !T.state.isPublishedMode() && l.id === st.currentInstanceId;
                return `<option value="${Toolbox.escapeHtml(val)}" ${sel ? 'selected' : ''}>${Toolbox.escapeHtml(l.title || '(제목 없음)')}</option>`;
            }).join('');
        const localOptgroup = localInst
            ? `<optgroup label="로컬 순위">${localInst}</optgroup>`
            : '<optgroup label="로컬 순위"><option disabled>(순위 없음)</option></optgroup>';

        let catalogOpts = '<option disabled>(항목 없음)</option>';
        let karmoOpts = '<option disabled>(항목 없음)</option>';
        try {
            const publishedRows = await T.publish.getPublishedIndex();
            const catalogRows = [];
            const karmoRows = [];
            publishedRows.forEach(it => {
                (publishedIndexGroup(it) === 'karmo' ? karmoRows : catalogRows).push(it);
            });
            catalogOpts = optionsFromPublishedRows(catalogRows, meta);
            karmoOpts = optionsFromPublishedRows(karmoRows, meta);
        } catch (_) {
            catalogOpts = '<option disabled>(index.json 로드 실패)</option>';
            karmoOpts = '<option disabled>(index.json 로드 실패)</option>';
        }

        return `<select id="tl-list-select">
            <optgroup label="카탈로그">${catalogOpts}</optgroup>
            <optgroup label="Karmo 순위">${karmoOpts}</optgroup>
            ${localOptgroup}
        </select>`;
    }

    function bindListSelectChange() {
        editorContainer.querySelector('#tl-list-select')?.addEventListener('change', async (e) => {
            const v = String(e.target.value || '');
            if (v.startsWith('local:')) {
                const id = v.slice('local:'.length);
                T.state.switchToInstance(id);
                await renderAll();
                return;
            }
            if (v.startsWith('blog:')) {
                const rel = v.slice('blog:'.length);
                try {
                    const items = await T.publish.getPublishedIndex();
                    const row = items.find(x => x.url === rel);
                    const g = row ? publishedIndexGroup(row) : 'catalog';
                    await T.publish.openPublishedDirect(rel, {
                        id: row?.id || '',
                        title: row?.title || '',
                        url: rel,
                        tierlistGroup: g === 'karmo' ? 'karmo' : 'catalog',
                    });
                } catch (err) { Toolbox.showToast('사이트 데이터 열기 실패', 'error', err); }
            }
        });
    }

    async function renderEditor() {
        if (!editorContainer) return;

        const st = T.state.getState();
        const meta = T.state.currentMeta();

        if (T.state.isPublishedCatalogMode()) {
            const snap = T.state.getPublishedCatalogSnapshot();
            const d = snap?.data || {};
            const items = d.items || {};
            const fileImages = d.images || {};
            const keys = Object.values(items).filter(i => i.imageKey).map(i => i.imageKey);
            const dbMap = await T.db.getMany(keys);

            function cardHtmlCatalog(itemId) {
                const item = items[itemId];
                if (!item) return '';
                const imgData = item.imageKey ? (fileImages[item.imageKey] || dbMap[item.imageKey]) : null;
                const inner = imgData
                    ? `<img src="${imgData}" alt="${Toolbox.escapeHtml(item.name || '')}">`
                    : `<div class="tl-item-text">${Toolbox.escapeHtml(item.name || '?')}</div>`;
                const nameTag = item.name ? `<div class="tl-item-name">${Toolbox.escapeHtml(item.name)}</div>` : '';
                return `<div class="tl-item tl-item--static" data-item-id="${itemId}">${inner}${nameTag}</div>`;
            }

            const selector = await buildListSelectorHtml(st, meta);
            const ids = Object.keys(items);
            const grid = ids.length
                ? ids.map(cardHtmlCatalog).join('')
                : '<div style="color:var(--text-tertiary);padding:24px;">이 풀에 항목이 없습니다.</div>';

            editorContainer.innerHTML = `<div class="tl-wrap tl-wrap--embedded">
                <div class="tl-ribbon-embed" aria-hidden="true">카탈로그 · 후보 풀</div>
                <div class="tl-toolbar">
                    ${selector}
                    <div class="tl-toolbar-spacer"></div>
                    <button class="tl-btn tl-btn-primary" id="tl-fork-catalog">이 풀로 순위 만들기</button>
                    <button class="tl-btn" id="tl-btn-export-json">JSON</button>
                </div>
                <p style="font-size:13px;color:var(--text-tertiary);margin:0 0 12px;line-height:1.45;">
                    여기는 <strong>순위 전</strong> 후보만 있어요. 아래를 확인한 뒤 <strong>이 풀로 순위 만들기</strong>를 누르면 로컬에 새 순위 보드가 생깁니다.
                </p>
                <div class="tl-pool" style="min-height:120px;">${grid}</div>
            </div>`;

            bindListSelectChange();
            editorContainer.querySelector('#tl-fork-catalog')?.addEventListener('click', async () => {
                try {
                    await T.publish.forkPublishedCatalogToLocal();
                    Toolbox.showToast('새 순위 인스턴스를 만들었어요');
                    await renderAll();
                } catch (err) {
                    Toolbox.showToast('만들기 실패', 'error', err);
                }
            });
            editorContainer.querySelector('#tl-btn-export-json')?.addEventListener('click', () => T.publish.showJsonPreview());
            return;
        }

        const list = T.state.currentList();

        if (!list) {
            editorContainer.innerHTML = `<div class="tl-wrap"><div style="text-align:center; padding:48px 16px; color:var(--text-tertiary);">
                <div style="font-size:32px; margin-bottom:12px;">📋</div>
                <div>열린 순위 인스턴스가 없습니다</div>
                <div style="margin-top:12px;"><button class="tl-btn tl-btn-primary" id="tl-empty-create">새 순위 만들기</button></div>
            </div></div>`;
            editorContainer.querySelector('#tl-empty-create')?.addEventListener('click', T.dialogs.showNewListDialog);
            return;
        }

        T.state.ensureListUserLabels(list);

        const syncListId = list.id;
        queueMicrotask(async () => {
            try {
                if (T.state.currentList()?.id !== syncListId) return;
                const changed = await T.publish.syncInstanceItemOriginsWithCatalogIfNeeded();
                if (changed && T.state.currentList()?.id === syncListId) await renderAll();
            } catch (_) {}
        });

        const allImageKeys = Object.values(list.items || {}).filter(i => i.imageKey).map(i => i.imageKey);
        const imgMap = await T.db.getMany(allImageKeys);
        const embeddedImages = T.state.getPublishedEmbeddedImages();

        function itemUserLabelsHtml(item) {
            const defs = list.userLabels || {};
            const ids = item.userLabelIds || [];
            if (!ids.length) return '';
            const maxShow = 3;
            const shown = ids.slice(0, maxShow);
            const more = ids.length - shown.length;
            const pills = [];
            shown.forEach(lid => {
                const d = defs[lid];
                if (!d) return;
                const bg = Toolbox.escapeHtml(d.color || '#666');
                const nm = Toolbox.escapeHtml(d.name || '');
                pills.push(`<span class="tl-item-userlabel" style="background:${bg}" title="${nm}">${nm}</span>`);
            });
            if (more > 0) {
                const rest = ids.slice(maxShow).map(lid => defs[lid]?.name || lid).join(', ');
                pills.push(`<span class="tl-item-userlabel tl-item-userlabel--more" title="${Toolbox.escapeHtml(rest)}">+${more}</span>`);
            }
            return pills.length ? `<div class="tl-item-userlabels">${pills.join('')}</div>` : '';
        }

        function itemOriginBadge(item) {
            if (!item || (item.tlOrigin !== 'custom' && !item.tlEdited)) return '';
            const isAdd = item.tlOrigin === 'custom';
            const label = isAdd ? '추가' : '수정';
            const tip = isAdd ? '직접 추가한 항목입니다.' : '후보 풀에서 가져온 뒤 이름 등을 바꾼 항목입니다.';
            const cls = isAdd ? 'tl-item-badge tl-item-badge--add' : 'tl-item-badge tl-item-badge--edit';
            return `<span class="${cls}" title="${Toolbox.escapeHtml(tip)}">${Toolbox.escapeHtml(label)}</span>`;
        }

        function cardHtml(itemId) {
            const item = list.items[itemId];
            if (!item) return '';
            const imgData = item.imageKey
                ? (embeddedImages[item.imageKey] || imgMap[item.imageKey])
                : null;
            const inner = imgData
                ? `<img src="${imgData}" alt="${Toolbox.escapeHtml(item.name || '')}">`
                : `<div class="tl-item-text">${Toolbox.escapeHtml(item.name || '?')}</div>`;
            const nameTag = item.name ? `<div class="tl-item-name">${Toolbox.escapeHtml(item.name)}</div>` : '';
            return `<div class="tl-item" data-item-id="${itemId}">${itemOriginBadge(item)}${itemUserLabelsHtml(item)}${inner}${nameTag}</div>`;
        }

        const selector = await buildListSelectorHtml(st, meta);

        const isPublished = meta.source === 'published';
        const ribbon = isPublished
            ? `<div class="tl-ribbon-embed" aria-hidden="true">${meta.tierlistGroup === 'karmo' ? 'Karmo · 순위' : '카탈로그 · 순위'}</div>`
            : '';
        const localBadge = !isPublished && list ? '<span class="tl-badge tl-badge-local">로컬 순위</span>' : '';
        const wrapClass = `${isPublished ? 'tl-wrap tl-wrap--embedded' : 'tl-wrap'} tl-wrap--toc-dock`;

        const tocChips = `${(list.tiers || []).map(tier => {
            const col = Toolbox.escapeHtml(tier.color || '#ccc');
            const lab = Toolbox.escapeHtml(tier.label || '?');
            const tid = Toolbox.escapeHtml(tier.id);
            return `<button type="button" class="tl-dropzone tl-toc-chip" data-tier-id="${tid}" data-toc-drop="1" style="background:${col}" title="${lab} 티어 맨 뒤로">${lab}</button>`;
        }).join('')}
                <button type="button" class="tl-dropzone tl-toc-chip tl-toc-pool" data-tier-id="_pool" data-toc-drop="1" title="미배치 풀 맨 뒤로">미배치</button>`;
        const tocNav = `<nav class="tl-toc tl-toc--dock" aria-label="티어 빠른 이동 (화면 고정)">
                <span class="tl-toc-hint">티어로 드롭 · 클릭 시 줄 이동 · 스크롤해도 이 바는 화면에 고정</span>
                <div class="tl-toc-chip-row">${tocChips}</div>
            </nav>`;

        let html = `<div class="${wrapClass}">
            ${ribbon}
            <div class="tl-toolbar">
                ${selector}
                <div class="tl-toolbar-spacer"></div>
                ${localBadge}
                <button class="tl-btn" id="tl-btn-tiers" title="티어 이름·색·줄 순서">티어</button>
                <button class="tl-btn" id="tl-btn-userlabels" title="카드에 달 색 라벨 만들기·정리">라벨</button>
                <button class="tl-btn" id="tl-btn-add">추가</button>
                <button type="button" class="tl-btn${editorDeleteMode ? ' tl-btn-toggle-on' : ''}" id="tl-btn-delete-mode" aria-pressed="${editorDeleteMode ? 'true' : 'false'}" title="직접 추가한 카드만 삭제. 풀 출처 카드는 삭제할 수 없어요. Ctrl+클릭·⌘+클릭 동일. Esc로 모드 해제.">삭제 모드</button>
                <button class="tl-btn" id="tl-btn-export-img">캡처</button>
                <button class="tl-btn" id="tl-btn-export-json" title="현재 순위·또는 후보 풀">JSON</button>
            </div>
            <div class="tl-board" id="tl-editor-board">`;

        (list.tiers || []).forEach(tier => {
            const rowItems = (list.rankings?.[tier.id] || []).map(cardHtml).join('');
            html += `<div class="tl-row">
                <div class="tl-label" style="background:${tier.color};color:#000;">${Toolbox.escapeHtml(tier.label)}</div>
                <div class="tl-dropzone" data-tier-id="${tier.id}">${rowItems}</div>
            </div>`;
        });

        html += `</div>
            <div class="tl-pool-section">
                <div class="tl-pool-header"><span class="tl-pool-title">미배치</span></div>
                <div class="tl-pool" data-tier-id="_pool">${(list.rankings?._pool || []).map(cardHtml).join('')}</div>
            </div>
            ${tocNav}
        </div>`;

        editorContainer.innerHTML = html;

        const wrap = editorContainer.querySelector('.tl-wrap');
        if (editorDeleteMode) wrap.classList.add('tl-delete-mode');

        function toastTierlistDrop(itemId, tierId, insertIdx) {
            const item = list.items[itemId];
            const raw = String(item?.name || '').trim();
            const disp = raw.length > 30 ? raw.slice(0, 27) + '…' : (raw || '이름 없음');
            const tocAppend = Number(insertIdx) >= 999999;
            let dest;
            if (tierId === '_pool') {
                dest = tocAppend ? '미배치 풀 맨 뒤' : '미배치';
            } else {
                const t = (list.tiers || []).find(x => x.id === tierId);
                const lab = String(t?.label || '').trim() || '티어';
                dest = tocAppend ? `${lab} 맨 뒤` : lab;
            }
            Toolbox.showToast(`「${disp}」→ ${dest}`);
        }

        T.dnd.initDnD(wrap, {
            onDrop: ({ itemId, tierId, insertIdx }) => {
                if (T.state.moveItem(itemId, tierId, insertIdx)) toastTierlistDrop(itemId, tierId, insertIdx);
                renderAll();
            },
            shouldBlockDragStart(e) {
                return editorDeleteMode || !!(e.ctrlKey || e.metaKey);
            },
        });

        wrap.querySelector('.tl-toc')?.addEventListener('click', e => {
            const chip = e.target.closest('.tl-toc-chip');
            if (!chip || e.button !== 0) return;
            const tid = chip.getAttribute('data-tier-id');
            if (!tid) return;
            if (tid === '_pool') {
                wrap.querySelector('.tl-pool-section')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                return;
            }
            const board = wrap.querySelector('#tl-editor-board');
            const dz = board && [...board.querySelectorAll('.tl-dropzone[data-tier-id]')].find(z => z.getAttribute('data-tier-id') === tid && !z.classList.contains('tl-toc-chip'));
            dz?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        });

        wrap.addEventListener(
            'click',
            e => {
                const itemEl = e.target.closest('.tl-item');
                if (!itemEl || itemEl.classList.contains('tl-item--static')) return;
                const itemId = itemEl.dataset.itemId;
                if (!itemId) return;
                const useDelete = editorDeleteMode || e.ctrlKey || e.metaKey;
                if (!useDelete) return;
                const item = list.items[itemId];
                if (item?.tlOrigin !== 'custom') {
                    const now = Date.now();
                    if (now - lastQuickDeleteBlockedToastAt > 2000) {
                        lastQuickDeleteBlockedToastAt = now;
                        Toolbox.showToast('풀 출처 카드는 삭제할 수 없어요. 직접 추가한 카드만 삭제 모드·Ctrl+클릭으로 지울 수 있어요.', 'error');
                    }
                    return;
                }
                e.preventDefault();
                e.stopPropagation();
                if (!T.state.removeItem(itemId)) {
                    Toolbox.showToast('후보 풀 카드는 삭제할 수 없어요.', 'error');
                    return;
                }
                renderAll();
            },
            true,
        );

        bindListSelectChange();

        editorContainer.querySelector('#tl-btn-tiers')?.addEventListener('click', T.dialogs.showTierSettingsDialog);
        editorContainer.querySelector('#tl-btn-userlabels')?.addEventListener('click', T.dialogs.showUserLabelsManagerDialog);
        editorContainer.querySelector('#tl-btn-add')?.addEventListener('click', T.dialogs.showAddItemDialog);
        editorContainer.querySelector('#tl-btn-delete-mode')?.addEventListener('click', () => {
            editorDeleteMode = !editorDeleteMode;
            renderAll();
        });
        editorContainer.querySelector('#tl-btn-export-img')?.addEventListener('click', T.publish.exportAsImage);
        editorContainer.querySelector('#tl-btn-export-json')?.addEventListener('click', () => T.publish.showJsonPreview());

        wrap.addEventListener('contextmenu', e => {
            const itemEl = e.target.closest('.tl-item');
            if (!itemEl) return;
            e.preventDefault();
            const itemId = itemEl.dataset.itemId;
            const menu = [
                { label: '편집', action: () => T.dialogs.showEditItemDialog(itemId) },
                { label: '라벨…', action: () => T.dialogs.showAssignUserLabelsDialog(itemId) },
            ];
            if (T.state.canResetItemFromPool(list, itemId)) {
                menu.push({
                    label: '수정 초기화',
                    action: () => {
                        T.publish.resetItemToCatalogDefault(itemId).then(ok => { if (ok) renderAll(); });
                    },
                });
            }
            if (T.state.isItemRemovable(list, itemId)) {
                menu.push('sep');
                menu.push({
                    label: '삭제',
                    danger: true,
                    action: () => {
                        if (!T.state.removeItem(itemId)) {
                            Toolbox.showToast('삭제할 수 없어요.', 'error');
                            return;
                        }
                        renderAll();
                    },
                });
            }
            T.ui.showContextMenu(e.clientX, e.clientY, menu);
        });

        if (!editorDeleteModeEscBound) {
            editorDeleteModeEscBound = true;
            document.addEventListener('keydown', ev => {
                if (ev.key !== 'Escape' || !editorDeleteMode) return;
                editorDeleteMode = false;
                renderAll();
            });
        }
    }

    function renderListTab() {
        if (!listContainer) return;
        const st = T.state.getState();
        const meta = T.state.currentMeta();

        let html = `<div style="display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap;">
            <button class="tl-btn tl-btn-primary" id="tl-list-new-cat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg> 새 후보 풀</button>
            <button class="tl-btn" id="tl-list-new">빈 순위 만들기</button>
            <button class="tl-btn" id="tl-list-import">JSON 가져오기</button>
        </div>
        <div class="tl-list-section">
            <h3 class="tl-list-section-title">후보 풀</h3>
            <p class="tl-list-section-desc">
                <strong>카탈로그</strong>와 <strong>Karmo 순위</strong>는 저장소 기준 <code>apps/karmolab/data/tierlists/index.json</code>에 등록된 JSON입니다(배포 시 자동으로 불러옵니다).
                <strong>로컬 후보 풀</strong> 카드를 누르면 그 주제로 <strong>새 순위 인스턴스</strong>가 만들어져 편집 탭으로 이동합니다.
            </p>
            <div class="tl-embed-grids">
                <h4 class="tl-list-subsection-title">카탈로그</h4>
                <div id="tl-grid-embed-catalog" class="tl-list-grid">불러오는 중…</div>
                <h4 class="tl-list-subsection-title">Karmo 순위</h4>
                <div id="tl-grid-embed-karmo" class="tl-list-grid">불러오는 중…</div>
                <h4 class="tl-list-subsection-title">로컬 후보 풀</h4>
                <div id="tl-grid-local-pools" class="tl-list-grid">불러오는 중…</div>
            </div>
        </div>
        <div class="tl-list-section" style="margin-top:28px;">
            <h3 class="tl-list-section-title">로컬 순위 인스턴스</h3>
            <p class="tl-list-section-desc">실제 티어판(S/A/… 배치)입니다. 카드를 누르면 그 순위를 이어서 편집합니다.</p>
            <div id="tl-grid-instances" class="tl-list-grid">불러오는 중…</div>
        </div>`;

        listContainer.innerHTML = html;

        listContainer.querySelector('#tl-list-new-cat')?.addEventListener('click', T.dialogs.showNewCatalogDialog);
        listContainer.querySelector('#tl-list-new')?.addEventListener('click', T.dialogs.showNewListDialog);
        listContainer.querySelector('#tl-list-import')?.addEventListener('click', T.publish.importFromJSONFilePicker);

        const gridEmbedCatalog = listContainer.querySelector('#tl-grid-embed-catalog');
        const gridEmbedKarmo = listContainer.querySelector('#tl-grid-embed-karmo');
        const gridLocalPools = listContainer.querySelector('#tl-grid-local-pools');
        const instGrid = listContainer.querySelector('#tl-grid-instances');

        async function embedIndexCardHtml(it) {
            const title = it.title || it.id || 'tierlist';
            const rel = it.url || '';
            const grp = publishedIndexGroup(it);
            const activeEmbed = T.state.isPublishedMode() && meta.url === rel;
            let countLine = '';
            if (rel) {
                try { countLine = await T.publish.getPublishedPreviewCountLine(rel); } catch (_) { /* 무시 */ }
            }
            const metaLine = countLine
                ? `${Toolbox.escapeHtml(countLine)} · ${Toolbox.escapeHtml(it.updatedAt || '—')}`
                : `index.json · ${Toolbox.escapeHtml(it.updatedAt || '—')}`;
            const pillClass = grp === 'karmo' ? 'tl-pill-karmo' : 'tl-pill-catalog';
            const pillLabel = grp === 'karmo' ? 'Karmo 순위' : '카탈로그';
            const karmoCls = grp === 'karmo' ? ' tl-list-card-embed--karmo' : '';
            return `<div class="tl-list-card tl-list-card-embed${karmoCls}${activeEmbed ? ' active' : ''}" data-embed-url="${Toolbox.escapeHtml(rel)}" data-embed-title="${Toolbox.escapeHtml(title)}" data-embed-id="${Toolbox.escapeHtml(it.id || '')}" data-embed-group="${Toolbox.escapeHtml(grp)}">
                <div class="tl-list-pill-row"><span class="tl-pill ${pillClass}">${pillLabel}</span></div>
                <div class="tl-list-card-title">${Toolbox.escapeHtml(title)}</div>
                <div class="tl-list-card-meta">${metaLine}</div>
            </div>`;
        }

        (async () => {
            let blogErr = false;
            let blogItems = [];
            try {
                blogItems = await T.publish.getPublishedIndex();
            } catch (_) {
                blogErr = true;
            }

            const catalogItems = [];
            const karmoItems = [];
            blogItems.forEach(it => {
                (publishedIndexGroup(it) === 'karmo' ? karmoItems : catalogItems).push(it);
            });

            let catalogEmbedHtml = '';
            let karmoEmbedHtml = '';
            if (blogErr) {
                const errCell =
                    '<div class="tl-list-empty" style="grid-column:1/-1;padding:20px;">index.json을 불러오지 못했습니다.</div>';
                catalogEmbedHtml = errCell;
                karmoEmbedHtml = errCell;
            } else {
                if (catalogItems.length) {
                    catalogEmbedHtml = (await Promise.all(catalogItems.map(embedIndexCardHtml))).join('');
                }
                if (karmoItems.length) {
                    karmoEmbedHtml = (await Promise.all(karmoItems.map(embedIndexCardHtml))).join('');
                }
            }

            const catalogs = Object.values(st.catalogs || {}).sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
            let catHtml = '';
            catalogs.forEach(c => {
                const n = Object.keys(c.items || {}).length;
                const date = new Date(c.updatedAt || Date.now());
                catHtml += `<div class="tl-list-card tl-list-card-catalog" data-catalog-id="${Toolbox.escapeHtml(c.id)}">
                    <div class="tl-list-pill-row"><span class="tl-pill tl-pill-cache">로컬 후보 풀</span></div>
                    <div class="tl-list-card-title">${Toolbox.escapeHtml(c.title || '(이름 없음)')}</div>
                    <div class="tl-list-card-meta">총 ${n}개 · ${date.toLocaleDateString()} · 클릭 시 새 순위 생성</div>
                </div>`;
            });

            const emptyCat = '<div class="tl-list-empty" style="grid-column:1/-1;padding:24px;">등록된 카탈로그 항목이 없습니다.</div>';
            const emptyKarmo = '<div class="tl-list-empty" style="grid-column:1/-1;padding:24px;">등록된 Karmo 순위가 없습니다.</div>';
            const emptyLocal = '<div class="tl-list-empty" style="grid-column:1/-1;padding:24px;">로컬 후보 풀이 없습니다.</div>';

            if (gridEmbedCatalog) {
                gridEmbedCatalog.innerHTML = blogErr ? catalogEmbedHtml : catalogEmbedHtml || emptyCat;
            }
            if (gridEmbedKarmo) {
                gridEmbedKarmo.innerHTML = blogErr ? karmoEmbedHtml : karmoEmbedHtml || emptyKarmo;
            }
            if (gridLocalPools) {
                gridLocalPools.innerHTML = catHtml || emptyLocal;
            }

            const instances = Object.values(st.instances || {}).sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
            let instHtml = '';
            instances.forEach(inst => {
                const ic = Object.keys(inst.items || {}).length;
                const rc = Object.entries(inst.rankings || {}).filter(([k]) => k !== '_pool').reduce((s, [, arr]) => s + (Array.isArray(arr) ? arr.length : 0), 0);
                const date = new Date(inst.updatedAt || Date.now());
                const active = !T.state.isPublishedMode() && inst.id === st.currentInstanceId;
                instHtml += `<div class="tl-list-card tl-list-card-instance${active ? ' active' : ''}" data-instance-id="${Toolbox.escapeHtml(inst.id)}">
                    <div class="tl-list-pill-row"><span class="tl-pill tl-pill-cache">순위</span></div>
                    <div class="tl-list-card-title">${Toolbox.escapeHtml(inst.title || '(제목 없음)')}</div>
                    <div class="tl-list-card-meta">아이템 ${ic} · 배치 ${rc} · ${date.toLocaleDateString()}</div>
                </div>`;
            });

            if (instGrid) {
                instHtml = instHtml || '<div class="tl-list-empty" style="grid-column:1/-1;"><div style="font-size:28px;margin-bottom:8px;">📋</div><div>로컬 순위가 없습니다. 후보 풀 카드를 누르거나 「빈 순위 만들기」를 쓰세요.</div></div>';
                instGrid.innerHTML = instHtml;
            }

            listContainer.querySelectorAll('.tl-list-card-embed[data-embed-url]').forEach(card => {
                card.addEventListener('click', async () => {
                    const rel = card.getAttribute('data-embed-url');
                    if (!rel) return;
                    const tit = card.getAttribute('data-embed-title') || '';
                    const eid = card.getAttribute('data-embed-id') || '';
                    const grp = card.getAttribute('data-embed-group') || 'catalog';
                    try {
                        await T.publish.openPublishedDirect(rel, {
                            id: eid,
                            title: tit,
                            url: rel,
                            tierlistGroup: grp === 'karmo' ? 'karmo' : 'catalog',
                        });
                        Toolbox.showToast('사이트 JSON을 열었어요');
                        await renderAll();
                        goToTierlistEditTab();
                    } catch (err) {
                        Toolbox.showToast('불러오기 실패', 'error', err);
                    }
                });
            });

            gridLocalPools?.querySelectorAll('.tl-list-card-catalog[data-catalog-id]').forEach(card => {
                card.addEventListener('click', async () => {
                    const cid = card.dataset.catalogId;
                    if (!cid) return;
                    const c = T.state.getState().catalogs[cid];
                    if (!c || !Object.keys(c.items || {}).length) {
                        Toolbox.showToast('후보가 비어 있어요. 우클릭으로 항목을 추가하세요.', 'error');
                        return;
                    }
                    T.state.createInstanceFromLocalCatalog(cid);
                    Toolbox.showToast('새 순위 인스턴스를 만들었어요');
                    await renderAll();
                    goToTierlistEditTab();
                });
                card.addEventListener('contextmenu', e => {
                    e.preventDefault();
                    const cid = card.dataset.catalogId;
                    T.ui.showContextMenu(e.clientX, e.clientY, [
                        { label: '항목 추가', action: () => T.dialogs.showAddCatalogItemDialog(cid) },
                        {
                            label: '삭제',
                            danger: true,
                            action: () => {
                                if (!confirm('이 후보 풀과 이미지를 삭제할까요?')) return;
                                T.state.deleteCatalog(cid);
                                void renderAll();
                                Toolbox.showToast('후보 풀 삭제됨');
                            },
                        },
                    ]);
                });
            });

            instGrid?.querySelectorAll('.tl-list-card-instance[data-instance-id]').forEach(card => {
                card.addEventListener('click', async () => {
                    const iid = card.dataset.instanceId;
                    if (!iid) return;
                    T.state.switchToInstance(iid);
                    await renderAll();
                    goToTierlistEditTab();
                });
                card.addEventListener('contextmenu', e => {
                    e.preventDefault();
                    const iid = card.dataset.instanceId;
                    T.ui.showContextMenu(e.clientX, e.clientY, [
                        {
                            label: '복제',
                            action: () => {
                                T.state.duplicateList(iid);
                                void renderAll().then(() => goToTierlistEditTab());
                                Toolbox.showToast('복제됨');
                            },
                        },
                        {
                            label: '삭제',
                            danger: true,
                            action: () => {
                                if (!confirm('이 순위와 이미지를 삭제할까요?')) return;
                                T.state.deleteList(iid);
                                void renderAll().then(() => goToTierlistEditTab());
                                Toolbox.showToast('삭제됨');
                            },
                        },
                    ]);
                });
            });
        })();
    }

    function renderStats() {
        if (!statsContainer) return;
        const bundles = T.state.iterAllInstances();

        if (!bundles.length) {
            statsContainer.innerHTML = '<div class="tl-list-empty"><div style="font-size:32px; margin-bottom:12px;">📊</div><div>통계를 표시할 순위 인스턴스가 없습니다</div></div>';
            return;
        }

        let totalItems = 0, totalRanked = 0;
        const tierCounts = {};

        bundles.forEach(({ list: l }) => {
            const itemCount = Object.keys(l.items || {}).length;
            totalItems += itemCount;
            (l.tiers || []).forEach(t => {
                const count = (l.rankings?.[t.id] || []).length;
                totalRanked += count;
                const key = String(t.label || t.id || '?').toUpperCase();
                tierCounts[key] = (tierCounts[key] || { count: 0, color: t.color || '#999' });
                tierCounts[key].count += count;
            });
        });

        const maxCount = Math.max(1, ...Object.values(tierCounts).map(v => v.count));

        let html = `<div class="tl-stats">
            <div class="tl-stat-cards">
                <div class="tl-stat-card"><div class="tl-stat-card-value">${bundles.length}</div><div class="tl-stat-card-label">순위 인스턴스</div></div>
                <div class="tl-stat-card"><div class="tl-stat-card-value">${totalItems}</div><div class="tl-stat-card-label">총 아이템</div></div>
                <div class="tl-stat-card"><div class="tl-stat-card-value">${totalRanked}</div><div class="tl-stat-card-label">배치 완료</div></div>
                <div class="tl-stat-card"><div class="tl-stat-card-value">${totalItems - totalRanked}</div><div class="tl-stat-card-label">미배치</div></div>
            </div>
            <div class="tl-stat-section">
                <h4>티어별 분포</h4>`;

        for (const [label, { count, color }] of Object.entries(tierCounts)) {
            const pct = Math.round((count / maxCount) * 100);
            html += `<div class="tl-bar-row">
                <div class="tl-bar-label" style="color:${Toolbox.escapeHtml(color)}">${Toolbox.escapeHtml(label)}</div>
                <div class="tl-bar-track"><div class="tl-bar-fill" style="width:${pct}%;background:${Toolbox.escapeHtml(color)};"><span class="tl-bar-count">${count}</span></div></div>
            </div>`;
        }

        html += `</div>
            <div class="tl-stat-section">
                <h4>순위별 요약</h4>
                <table class="tl-stat-table">
                    <thead><tr><th>출처</th><th>제목</th><th>카테고리</th><th>아이템</th><th>배치</th><th>최근 수정</th></tr></thead>
                    <tbody>`;

        function sourceLabel(l) {
            const s = l.meta?.source || 'local';
            if (s === 'from-local-catalog') return '로컬 후보 풀';
            if (s === 'from-catalog') return '사이트 후보 풀';
            if (s === 'published-draft') return '블로그 초안';
            if (s === 'import') return '가져옴';
            if (s === 'duplicate') return '복제';
            return s;
        }

        bundles.sort((a, b) => (b.list.updatedAt || 0) - (a.list.updatedAt || 0)).forEach(({ list: l }) => {
            const ic = Object.keys(l.items || {}).length;
            const rc = Object.entries(l.rankings || {}).filter(([k]) => k !== '_pool').reduce((s, [, arr]) => s + (Array.isArray(arr) ? arr.length : 0), 0);
            html += `<tr>
                <td>${Toolbox.escapeHtml(sourceLabel(l))}</td>
                <td>${Toolbox.escapeHtml(l.title || '(제목 없음)')}</td>
                <td>${Toolbox.escapeHtml(l.category || '-')}</td>
                <td>${ic}</td>
                <td>${rc}</td>
                <td>${new Date(l.updatedAt || Date.now()).toLocaleDateString()}</td>
            </tr>`;
        });

        html += '</tbody></table></div></div>';
        statsContainer.innerHTML = html;
    }

    async function renderAll() {
        await renderEditor();
        renderListTab();
        renderStats();
    }

    T.render = {
        setContainers,
        renderEditor,
        renderAll,
        renderListTab,
        renderStats,
        publishedIndexGroup,
    };
})();

