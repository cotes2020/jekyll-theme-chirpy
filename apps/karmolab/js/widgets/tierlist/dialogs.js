(function () {
    const T = window.Tierlist = window.Tierlist || {};

    function showAddItemDialog() {
        const list = T.state.currentList();
        if (!list) { Toolbox.showToast('먼저 티어리스트를 선택하세요.', 'error'); return; }

        let pendingFiles = [];
        T.ui.openDialog({
            title: '아이템 추가',
            wide: false,
            bodyHtml: `
                <input type="file" id="tl-add-file" accept="image/*" multiple>
                <label>이름(선택)</label>
                <input type="text" id="tl-add-name" placeholder="아이템 이름">
                <div class="tl-dialog-actions">
                    <button class="tl-btn" id="tl-add-cancel">취소</button>
                    <button class="tl-btn tl-btn-primary" id="tl-add-ok">추가</button>
                </div>
            `,
            onMount: ({ dialog, close }) => {
                const fileInput = dialog.querySelector('#tl-add-file');
                const nameInput = dialog.querySelector('#tl-add-name');
                fileInput.onchange = () => pendingFiles = Array.from(fileInput.files || []).filter(f => f.type.startsWith('image/'));
                dialog.querySelector('#tl-add-cancel').onclick = close;
                dialog.querySelector('#tl-add-ok').onclick = async () => {
                    const name = nameInput.value.trim();
                    if (!pendingFiles.length && !name) { Toolbox.showToast('이미지나 이름을 입력하세요.', 'error'); return; }
                    close();
                    if (pendingFiles.length) {
                        for (const f of pendingFiles) {
                            const imgKey = await T.state.processImageFile(f);
                            T.state.addItem(name || f.name.replace(/\.[^.]+$/, ''), imgKey);
                        }
                    } else {
                        T.state.addItem(name, null);
                    }
                    T.render.renderAll();
                };
            }
        });
    }

    function showEditItemDialog(itemId) {
        const list = T.state.currentList();
        if (!list?.items?.[itemId]) return;
        const item = list.items[itemId];

        T.ui.openDialog({
            title: '아이템 편집',
            wide: false,
            bodyHtml: `
                <label>이름</label>
                <input type="text" id="tl-edit-name" value="${Toolbox.escapeHtml(item.name || '')}">
                <div class="tl-dialog-actions">
                    <button class="tl-btn" id="tl-edit-cancel">취소</button>
                    <button class="tl-btn tl-btn-primary" id="tl-edit-ok">저장</button>
                </div>
            `,
            onMount: ({ dialog, close }) => {
                dialog.querySelector('#tl-edit-cancel').onclick = close;
                dialog.querySelector('#tl-edit-ok').onclick = () => {
                    T.state.ensureWritableList?.('editItem');
                    const cur = T.state.currentList();
                    if (!cur?.items?.[itemId]) { close(); return; }
                    cur.items[itemId].name = dialog.querySelector('#tl-edit-name').value.trim();
                    cur.updatedAt = Date.now();
                    T.state.saveState();
                    close();
                    T.render.renderAll();
                };
            }
        });
    }

    function showNewListDialog() {
        T.ui.openDialog({
            title: '새 티어리스트',
            wide: false,
            bodyHtml: `
                <label>제목</label>
                <input type="text" id="tl-new-title" placeholder="예: 2026 애니">
                <label>카테고리</label>
                <input type="text" id="tl-new-cat" placeholder="예: 애니">
                <div class="tl-dialog-actions">
                    <button class="tl-btn" id="tl-new-cancel">취소</button>
                    <button class="tl-btn tl-btn-primary" id="tl-new-ok">생성</button>
                </div>
            `,
            onMount: ({ dialog, close }) => {
                dialog.querySelector('#tl-new-cancel').onclick = close;
                dialog.querySelector('#tl-new-ok').onclick = () => {
                    const t = dialog.querySelector('#tl-new-title').value.trim();
                    const c = dialog.querySelector('#tl-new-cat').value.trim();
                    T.state.createList(t || '새 티어리스트', c);
                    close();
                    T.render.renderAll();
                };
            }
        });
    }

    T.dialogs = { showAddItemDialog, showEditItemDialog, showNewListDialog };
})();

