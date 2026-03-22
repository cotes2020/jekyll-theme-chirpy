(function() {
    Mdd.injectCSS('memo', `
        .memo-container { display:flex; flex:1; min-height:400px; background:var(--bg-tertiary); border:1px solid var(--border); border-radius:var(--radius-lg); overflow:hidden; }
        .memo-sidebar { width:250px; background:var(--bg-secondary); border-right:1px solid var(--border); display:flex; flex-direction:column; }
        .memo-sidebar-header { padding:16px; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }
        .memo-sidebar-title { font-size:var(--font-size-sm); font-weight:600; color:var(--text-primary); }
        .memo-add-btn { padding:6px; }
        .memo-list { flex:1; overflow-y:auto; padding:8px; display:flex; flex-direction:column; gap:4px; }
        .memo-item { padding:12px; border-radius:var(--radius-sm); cursor:pointer; transition:background var(--transition); border:1px solid transparent; }
        .memo-item:hover { background:var(--bg-hover); }
        .memo-item.active { background:var(--bg-hover); border-color:var(--border); }
        .memo-item-title { font-size:var(--font-size-sm); font-weight:500; color:var(--text-primary); margin-bottom:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .memo-item-date { font-size:var(--font-size-xs); color:var(--text-tertiary); }
        .memo-empty-state { padding:24px 16px; text-align:center; font-size:var(--font-size-xs); color:var(--text-tertiary); }
        .memo-editor { flex:1; display:flex; flex-direction:column; background:var(--bg-tertiary); }
        .memo-editor-header { padding:16px 24px; border-bottom:1px solid var(--border); display:flex; gap:12px; align-items:center; }
        .memo-title-input { flex:1; background:transparent; border:none; font-size:18px; font-weight:600; color:var(--text-primary); padding:0; outline:none; }
        .memo-title-input::placeholder { color:var(--text-tertiary); font-weight:500; }
        .memo-body-input { flex:1; background:transparent; border:none; resize:none; padding:24px; font-size:14px; line-height:1.7; color:var(--text-primary); outline:none; font-family:inherit; }
        .memo-body-input::placeholder { color:var(--text-tertiary); }
        .memo-status-indicator { padding:8px 24px; font-size:var(--font-size-xs); color:var(--text-tertiary); text-align:right; border-top:1px solid var(--border); background:var(--bg-secondary); }
        @media (max-width:768px) { .memo-container { flex-direction:column; min-height:500px; } .memo-sidebar { width:100%; height:200px; flex:none; border-right:none; border-bottom:1px solid var(--border); } }
    `);

    const MemoApp = (() => {
        function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;'); }
        const STORAGE_KEY = 'toolbox_memos';
        let memos = [];
        let currentId = null;

        function loadMemos() {
            try { memos = JSON.parse(localStorage.getItem(STORAGE_KEY)) || []; } catch(e) { memos = []; }
        }
        function saveMemos() { localStorage.setItem(STORAGE_KEY, JSON.stringify(memos)); }

        function createMemo() {
            const newMemo = { id: Date.now().toString(), title: '새 메모', body: '', updatedAt: Date.now() };
            memos.unshift(newMemo); saveMemos(); currentId = newMemo.id; render();
            Mdd.setMood('happy'); Mdd.say('새 메모 생성이다냥!');
        }

        function deleteMemo(id) {
            if (!confirm('이 메모를 삭제하시겠습니까?')) return;
            memos = memos.filter(m => m.id !== id);
            if (currentId === id) currentId = memos.length > 0 ? memos[0].id : null;
            saveMemos(); render();
            Toolbox.showToast('삭제되었습니다.');
            Mdd.setMood('shock'); Mdd.say('지워버렸다냥... 안녕...');
        }

        function updateMemo(updates) {
            const msg = memos.find(m => m.id === currentId);
            if (!msg) return;
            Object.assign(msg, updates); msg.updatedAt = Date.now(); saveMemos(); renderList();
            const status = document.getElementById('memoStatus');
            if(status) status.textContent = '저장됨 (' + new Date().toLocaleTimeString() + ')';
        }

        let saveTimeout = null;
        function handleInput() {
            const status = document.getElementById('memoStatus');
            if (status) status.textContent = '저장 중...';
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                updateMemo({ title: document.getElementById('memoTitleInput').value.trim() || '제목 없음', body: document.getElementById('memoBodyInput').value });
                Mdd.setMood('happy'); Mdd.say('기억해둘게냥!');
            }, 500);
        }

        function renderList() {
            const list = document.getElementById('memoList');
            if (!list) return;
            if (memos.length === 0) { list.innerHTML = '<div class="memo-empty-state">메모가 없습니다.<br>새 메모를 추가해보세요.</div>'; return; }
            list.innerHTML = '';
            memos.forEach(m => {
                const d = new Date(m.updatedAt);
                const dateStr = d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
                const item = document.createElement('div');
                item.className = 'memo-item' + (m.id === currentId ? ' active' : '');
                item.onclick = () => { currentId = m.id; render(); };
                item.innerHTML = `<div class="memo-item-title">${esc(m.title)}</div><div class="memo-item-date">${dateStr}</div>`;
                list.appendChild(item);
            });
        }

        function renderEditor() {
            const editor = document.getElementById('memoEditor');
            if (!editor) return;
            if (!currentId || memos.length === 0) { editor.innerHTML = '<div style="margin:auto; color:var(--text-tertiary); font-size:var(--font-size-sm);">리스트에서 메모를 선택하거나 새 메모를 만드세요.</div>'; return; }
            const m = memos.find(x => x.id === currentId);
            if (!m) return;
            editor.innerHTML = `
                <div class="memo-editor-header">
                    <input type="text" id="memoTitleInput" class="memo-title-input" placeholder="메모 제목" value="${esc(m.title)}">
                    <button class="btn btn-danger" id="memoDeleteBtn">삭제</button>
                </div>
                <textarea id="memoBodyInput" class="memo-body-input" placeholder="여기에 내용을 입력하세요...">${esc(m.body)}</textarea>
                <div class="memo-status-indicator" id="memoStatus">자동 저장 대기 중</div>
            `;
            document.getElementById('memoTitleInput').oninput = handleInput;
            document.getElementById('memoBodyInput').oninput = handleInput;
            document.getElementById('memoDeleteBtn').onclick = () => deleteMemo(m.id);
        }

        function render() { renderList(); renderEditor(); }

        function exportMemos() {
            if (memos.length === 0) { Toolbox.showToast('내보낼 메모가 없습니다.', 'error'); return; }
            const a = document.createElement('a');
            a.href = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(memos, null, 2));
            a.download = `toolbox_memos_${Date.now()}.json`;
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
            Toolbox.showToast('메모를 내보냈습니다.');
        }

        function importMemos() {
            const input = document.createElement('input'); input.type = 'file'; input.accept = 'application/json';
            input.onchange = e => {
                const file = e.target.files[0]; if (!file) return;
                const reader = new FileReader();
                reader.onload = e => {
                    try {
                        const imported = JSON.parse(e.target.result);
                        if (!Array.isArray(imported)) throw new Error('Invalid format');
                        if (confirm(`기존 메모에 덮어씌워집니다. 진행하시겠습니까? (불러올 메모 수: ${imported.length}개)`)) {
                            memos = imported; saveMemos(); currentId = memos.length > 0 ? memos[0].id : null; render();
                            Toolbox.showToast('메모를 불러왔습니다.');
                        }
                    } catch(err) { Toolbox.showToast('올바르지 않은 백업 파일입니다.', 'error'); }
                };
                reader.readAsText(file);
            };
            input.click();
        }

        return {
            build(container) {
                loadMemos();
                if (memos.length > 0 && !currentId) currentId = memos[0].id;
                container.innerHTML = `
                    <div class="memo-container">
                        <div class="memo-sidebar">
                            <div class="memo-sidebar-header">
                                <span class="memo-sidebar-title">전체 메모</span>
                                <div style="display:flex; gap:4px;">
                                    <button class="btn btn-ghost memo-add-btn" id="memoImportBtn" title="불러오기"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg></button>
                                    <button class="btn btn-ghost memo-add-btn" id="memoExportBtn" title="저장하기"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg></button>
                                    <button class="btn btn-ghost memo-add-btn" id="memoAddBtn" title="새 메모"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg></button>
                                </div>
                            </div>
                            <div class="memo-list" id="memoList"></div>
                        </div>
                        <div class="memo-editor" id="memoEditor"></div>
                    </div>
                `;
                Mdd.setMood('idle'); Mdd.say('메모할 거 있냥?');
                requestAnimationFrame(() => {
                    const addBtn = document.getElementById('memoAddBtn'); if(addBtn) addBtn.onclick = createMemo;
                    const exportBtn = document.getElementById('memoExportBtn'); if(exportBtn) exportBtn.onclick = exportMemos;
                    const importBtn = document.getElementById('memoImportBtn'); if(importBtn) importBtn.onclick = importMemos;
                    render();
                });
            }
        };
    })();

    Toolbox.register({
        id: 'memo', title: '메모장',
        category: 'feature',
        desc: '로컬 메모를 저장하고 관리합니다',
        layout: 'full',
        icon: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline>',
        tabs: [{ id: 'editor', label: '에디터', build: MemoApp.build }]
    });
})();
