(function () {
    const T = window.Tierlist = window.Tierlist || {};

    let editorContainer = null;
    let listContainer = null;
    let statsContainer = null;

    function setContainers({ editor, list, stats }) {
        if (editor !== undefined) editorContainer = editor;
        if (list !== undefined) listContainer = list;
        if (stats !== undefined) statsContainer = stats;
    }

    async function renderEditor() {
        if (!editorContainer) return;

        const list = T.state.currentList();
        const st = T.state.getState();
        const meta = T.state.currentMeta();

        if (!list) {
            editorContainer.innerHTML = `<div class="tl-wrap"><div style="text-align:center; padding:48px 16px; color:var(--text-tertiary);">
                <div style="font-size:32px; margin-bottom:12px;">📋</div>
                <div>티어리스트가 없습니다</div>
                <div style="margin-top:12px;"><button class="tl-btn tl-btn-primary" id="tl-empty-create">새 티어리스트 만들기</button></div>
            </div></div>`;
            editorContainer.querySelector('#tl-empty-create')?.addEventListener('click', T.dialogs.showNewListDialog);
            return;
        }

        const allImageKeys = Object.values(list.items || {}).filter(i => i.imageKey).map(i => i.imageKey);
        const imgMap = await T.db.getMany(allImageKeys);

        function cardHtml(itemId) {
            const item = list.items[itemId];
            if (!item) return '';
            const imgData = item.imageKey ? imgMap[item.imageKey] : null;
            const inner = imgData
                ? `<img src="${imgData}" alt="${Toolbox.escapeHtml(item.name || '')}">`
                : `<div class="tl-item-text">${Toolbox.escapeHtml(item.name || '?')}</div>`;
            const nameTag = item.name ? `<div class="tl-item-name">${Toolbox.escapeHtml(item.name)}</div>` : '';
            return `<div class="tl-item" data-item-id="${itemId}">${inner}${nameTag}</div>`;
        }

        const localOptions = Object.values(st.lists || {})
            .sort((a, b) => String(a.title || '').localeCompare(String(b.title || ''), 'ko-KR'))
            .map(l => `<option value="local:${l.id}" ${(!T.state.isPublishedMode() && l.id === st.currentListId) ? 'selected' : ''}>${Toolbox.escapeHtml(l.title)}${l.meta?.dirty ? ' *' : ''}</option>`)
            .join('');

        let blogOptions = '';
        try {
            const blogItems = await T.publish.getPublishedIndex();
            if (blogItems.length) {
                blogOptions = blogItems.map(it => {
                    const selected = T.state.isPublishedMode() && (meta.url === it.url);
                    return `<option value="blog:${Toolbox.escapeHtml(it.url || '')}" ${selected ? 'selected' : ''}>${Toolbox.escapeHtml(it.title || it.id || 'tierlist')}</option>`;
                }).join('');
            } else blogOptions = `<option disabled>(사이트 데이터 없음)</option>`;
        } catch (_) {
            blogOptions = `<option disabled>(index.json 로드 실패)</option>`;
        }

        const selector = `<select id="tl-list-select">
            ${blogOptions ? `<optgroup label="사이트 데이터">${blogOptions}</optgroup>` : ''}
            ${localOptions ? `<optgroup label="로컬 캐시">${localOptions}</optgroup>` : ''}
        </select>`;

        const isPublished = meta.source === 'published';
        const ribbon = isPublished ? '<div class="tl-ribbon-embed" aria-hidden="true">블로그 데이터</div>' : '';
        const dirtyBadge = !isPublished && meta.dirty ? `<span class="tl-badge tl-badge-dirty">🎀 수정됨 *</span>` : '';
        const localBadge = !isPublished && list ? `<span class="tl-badge tl-badge-local">로컬 캐시</span>` : '';
        const wrapClass = isPublished ? 'tl-wrap tl-wrap--embedded' : 'tl-wrap';

        let html = `<div class="${wrapClass}">
            ${ribbon}
            <div class="tl-toolbar">
                ${selector}
                <div class="tl-toolbar-spacer"></div>
                ${localBadge}${dirtyBadge}
                <button class="tl-btn" id="tl-btn-add">추가</button>
                <button class="tl-btn" id="tl-btn-export-img">캡처</button>
                <button class="tl-btn" id="tl-btn-export-json">JSON</button>
            </div>
            <div class="tl-board" id="tl-editor-board">`;

        (list.tiers || []).forEach(tier => {
            const items = (list.rankings?.[tier.id] || []).map(cardHtml).join('');
            html += `<div class="tl-row">
                <div class="tl-label" style="background:${tier.color};color:#000;">${Toolbox.escapeHtml(tier.label)}</div>
                <div class="tl-dropzone" data-tier-id="${tier.id}">${items}</div>
            </div>`;
        });

        html += `</div>
            <div class="tl-pool-section">
                <div class="tl-pool-header"><span class="tl-pool-title">미배치</span></div>
                <div class="tl-pool" data-tier-id="_pool">${(list.rankings?._pool || []).map(cardHtml).join('')}</div>
            </div>
        </div>`;

        editorContainer.innerHTML = html;

        const wrap = editorContainer.querySelector('.tl-wrap');
        T.dnd.initDnD(wrap, {
            onDrop: ({ itemId, tierId, insertIdx }) => {
                T.state.moveItem(itemId, tierId, insertIdx);
                renderAll();
            }
        });

        editorContainer.querySelector('#tl-list-select')?.addEventListener('change', async (e) => {
            const v = String(e.target.value || '');
            if (v.startsWith('local:')) {
                T.state.closePublished();
                st.currentListId = v.slice('local:'.length);
                T.state.saveState();
                renderAll();
                return;
            }
            if (v.startsWith('blog:')) {
                const rel = v.slice('blog:'.length);
                try { await T.publish.openPublishedDirect(rel, { url: rel }); }
                catch (err) { Toolbox.showToast('사이트 데이터 열기 실패', 'error', err); }
            }
        });

        editorContainer.querySelector('#tl-btn-add')?.addEventListener('click', T.dialogs.showAddItemDialog);
        editorContainer.querySelector('#tl-btn-export-img')?.addEventListener('click', T.publish.exportAsImage);
        editorContainer.querySelector('#tl-btn-export-json')?.addEventListener('click', () => T.publish.showJsonPreview());

        wrap.addEventListener('contextmenu', e => {
            const itemEl = e.target.closest('.tl-item');
            if (!itemEl) return;
            e.preventDefault();
            const itemId = itemEl.dataset.itemId;
            T.ui.showContextMenu(e.clientX, e.clientY, [
                { label: '편집', action: () => T.dialogs.showEditItemDialog(itemId) },
                'sep',
                { label: '삭제', danger: true, action: () => { T.state.removeItem(itemId); renderAll(); } },
            ]);
        });
    }

    function renderListTab() {
        if (!listContainer) return;
        const st = T.state.getState();
        const entries = Object.values(st.lists || {});
        const meta = T.state.currentMeta();

        let html = `<div style="display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap;">
            <button class="tl-btn tl-btn-primary" id="tl-list-new"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg> 새 리스트</button>
            <button class="tl-btn" id="tl-list-import">가져오기</button>
        </div>
        <div class="tl-list-section">
            <h3 class="tl-list-section-title">사이트에 포함된 데이터</h3>
            <p class="tl-list-section-desc"><code>data/tierlists/index.json</code>에 등록된 목록이 여기에 표시됩니다.</p>
            <div id="tl-embed-cards" class="tl-list-grid">불러오는 중…</div>
        </div>
        <div class="tl-list-section tl-list-section-local">
            <h3 class="tl-list-section-title">로컬 캐시</h3>
            <p class="tl-list-section-desc">이 브라우저에만 저장된 목록입니다.</p>`;

        if (!entries.length) {
            html += '<div class="tl-list-empty"><div style="font-size:32px; margin-bottom:12px;">📋</div><div>아직 만든 티어리스트가 없어요</div></div>';
        } else {
            html += '<div class="tl-list-grid">';
            entries.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)).forEach(l => {
                const itemCount = Object.keys(l.items || {}).length;
                const ranked = Object.entries(l.rankings || {}).filter(([k]) => k !== '_pool').reduce((s, [, arr]) => s + (Array.isArray(arr) ? arr.length : 0), 0);
                const date = new Date(l.updatedAt || Date.now());
                const activeLocal = !T.state.isPublishedMode() && l.id === st.currentListId;
                html += `<div class="tl-list-card tl-list-card-local${activeLocal ? ' active' : ''}" data-list-id="${Toolbox.escapeHtml(l.id)}">
                    <div class="tl-list-card-title">${Toolbox.escapeHtml(l.title || '(제목 없음)')}</div>
                    <div class="tl-list-card-meta">아이템 ${itemCount}개 · 배치 ${ranked}개 · ${date.toLocaleDateString()}</div>
                    ${l.category ? `<div class="tl-list-card-cat">${Toolbox.escapeHtml(l.category)}</div>` : ''}
                </div>`;
            });
            html += '</div>';
        }
        html += '</div>';

        listContainer.innerHTML = html;

        listContainer.querySelector('#tl-list-new')?.addEventListener('click', T.dialogs.showNewListDialog);
        listContainer.querySelector('#tl-list-import')?.addEventListener('click', T.publish.importFromJSONFilePicker);

        const embedHost = listContainer.querySelector('#tl-embed-cards');
        (async () => {
            if (!embedHost) return;
            try {
                const items = await T.publish.getPublishedIndex();
                if (!items.length) {
                    embedHost.innerHTML = '<div class="tl-list-empty" style="padding:24px;">등록된 사이트 데이터가 없습니다.</div>';
                    return;
                }
                embedHost.innerHTML = items.map(it => {
                    const title = it.title || it.id || 'tierlist';
                    const rel = it.url || '';
                    const activeEmbed = T.state.isPublishedMode() && meta.url === rel;
                    return `<div class="tl-list-card tl-list-card-embed${activeEmbed ? ' active' : ''}" data-embed-url="${Toolbox.escapeHtml(rel)}" data-embed-title="${Toolbox.escapeHtml(title)}" data-embed-id="${Toolbox.escapeHtml(it.id || '')}">
                        <div class="tl-list-card-ribbon">블로그 데이터</div>
                        <div class="tl-list-card-title">${Toolbox.escapeHtml(title)}</div>
                        <div class="tl-list-card-meta">${Toolbox.escapeHtml(it.updatedAt || '')}</div>
                    </div>`;
                }).join('');
                embedHost.querySelectorAll('.tl-list-card-embed[data-embed-url]').forEach(card => {
                    card.addEventListener('click', async () => {
                        const rel = card.getAttribute('data-embed-url');
                        if (!rel) return;
                        const tit = card.getAttribute('data-embed-title') || '';
                        const eid = card.getAttribute('data-embed-id') || '';
                        try {
                            await T.publish.openPublishedDirect(rel, { id: eid, title: tit, url: rel });
                            Toolbox.showToast('사이트 데이터를 불러왔어요');
                        } catch (err) {
                            Toolbox.showToast('불러오기 실패', 'error', err);
                        }
                    });
                });
            } catch (_) {
                embedHost.innerHTML = '<div class="tl-list-empty" style="padding:24px;">index.json을 불러오지 못했습니다.</div>';
            }
        })();

        listContainer.querySelectorAll('.tl-list-card-local[data-list-id]').forEach(card => {
            card.addEventListener('click', () => {
                T.state.closePublished?.();
                st.currentListId = card.dataset.listId;
                T.state.saveState();
                renderAll();
            });
            card.addEventListener('contextmenu', e => {
                e.preventDefault();
                const id = card.dataset.listId;
                T.ui.showContextMenu(e.clientX, e.clientY, [
                    { label: '선택', action: () => { T.state.closePublished?.(); st.currentListId = id; T.state.saveState(); renderAll(); } },
                    { label: '복제', action: () => { T.state.duplicateList(id); renderAll(); Toolbox.showToast('복제됨'); } },
                    'sep',
                    {
                        label: '삭제',
                        danger: true,
                        action: () => {
                            if (!confirm('이 티어리스트를 삭제하시겠습니까?')) return;
                            T.state.deleteList(id);
                            renderAll();
                            Toolbox.showToast('삭제됨');
                            try { Mdd.setMood('sad'); Mdd.say('지워버렸어요...'); } catch (_) {}
                        }
                    },
                ]);
            });
        });
    }

    function renderStats() {
        if (!statsContainer) return;
        const st = T.state.getState();
        const entries = Object.values(st.lists || {});

        if (!entries.length) {
            statsContainer.innerHTML = '<div class="tl-list-empty"><div style="font-size:32px; margin-bottom:12px;">📊</div><div>통계를 표시할 티어리스트가 없습니다</div></div>';
            return;
        }

        let totalItems = 0, totalRanked = 0;
        const tierCounts = {};

        entries.forEach(l => {
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
                <div class="tl-bar-label" style="color:${Toolbox.escapeHtml(color)}">${Toolbox.escapeHtml(label)}</div>
                <div class="tl-bar-track"><div class="tl-bar-fill" style="width:${pct}%;background:${Toolbox.escapeHtml(color)};"><span class="tl-bar-count">${count}</span></div></div>
            </div>`;
        }

        html += `</div>
            <div class="tl-stat-section">
                <h4>리스트별 요약</h4>
                <table class="tl-stat-table">
                    <thead><tr><th>제목</th><th>카테고리</th><th>아이템</th><th>배치</th><th>최근 수정</th></tr></thead>
                    <tbody>`;

        entries.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)).forEach(l => {
            const ic = Object.keys(l.items || {}).length;
            const rc = Object.entries(l.rankings || {}).filter(([k]) => k !== '_pool').reduce((s, [, arr]) => s + (Array.isArray(arr) ? arr.length : 0), 0);
            html += `<tr>
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

    function renderAll() {
        renderEditor();
        renderListTab();
        renderStats();
    }

    T.render = { setContainers, renderEditor, renderAll, renderListTab, renderStats };
})();

