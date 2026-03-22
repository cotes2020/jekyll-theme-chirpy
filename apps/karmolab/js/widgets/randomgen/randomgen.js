/**
 * 랜덤 생성기 — 창작용 키워드·주제 뽑기
 * randomgen-topics.js에 정의된 주제를 기반으로 동작
 *
 * 참고: [니힐 랜덤 키워드](https://nihilapp.github.io/keyword) — 창작자용 랜덤 키워드 사이트
 */
(function () {
    const topics = window.RANDOMGEN_TOPICS || [];

    function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

    function generate(topic, count) {
        const results = [];
        for (let i = 0; i < count; i++) {
            let name, sub;
            if (topic.items) {
                name = pick(topic.items);
                sub = topic.label;
            } else if (topic.generator) {
                const r = topic.generator();
                if (typeof r === 'object' && r !== null && 'name' in r) {
                    name = r.name;
                    sub = r.sub != null ? r.sub : topic.label;
                } else {
                    name = String(r);
                    sub = topic.label;
                }
            } else {
                continue;
            }
            results.push({ name, sub });
        }
        return results;
    }

    function getGroups() {
        const seen = new Set();
        const groups = [];
        topics.forEach(t => {
            const g = t.group || '기타';
            if (!seen.has(g)) {
                seen.add(g);
                groups.push({ id: g, label: g });
            }
        });
        return groups;
    }

    function getTopicsByGroup() {
        const byGroup = {};
        topics.forEach(t => {
            const g = t.group || '기타';
            if (!byGroup[g]) byGroup[g] = [];
            byGroup[g].push(t);
        });
        return byGroup;
    }

    function getTopicItems(topic) {
        if (topic.items) return topic.items;
        if (topic.generator) {
            const samples = [];
            const seen = new Set();
            for (let i = 0; i < 20 && samples.length < 10; i++) {
                const r = topic.generator();
                const name = (typeof r === 'object' && r !== null && 'name' in r) ? r.name : String(r);
                if (!seen.has(name)) { seen.add(name); samples.push(name); }
            }
            return samples;
        }
        return [];
    }

    Toolbox.register({
        id: 'randomgen',
        title: '랜덤 생성기',
        category: 'tool',
        desc: '창작용 키워드·주제를 랜덤으로 뽑습니다',
        layout: 'wide',
        icon: '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/><line x1="16" y1="3" x2="22" y2="3"/><line x1="19" y1="0" x2="19" y2="6"/>',
        tabs: [{
            id: 'app',
            label: '생성',
            build: function (container) {
                Mdd.setMood('think');
                Mdd.say('어떤 주제를 뽑을까냥?');

                function buildTopicCardsHtml() {
                    var html = '';
                    var byGroup = getTopicsByGroup();
                    var gids = getGroups().map(function (g) { return g.id; });
                    gids.forEach(function (gid) {
                        var list = byGroup[gid] || [];
                        html += '<div class="randomgen-topic-group"><div class="randomgen-topic-group-title">' + Toolbox.escapeHtml(gid) + '</div><div class="randomgen-topic-cards">';
                        list.forEach(function (t) {
                            html += '<div class="randomgen-topic-card randomgen-topic-custom" data-value="' + Toolbox.escapeHtml(t.id) + '"><span class="randomgen-topic-card-label">' + Toolbox.escapeHtml(t.label) + '</span><button type="button" class="randomgen-topic-info" data-topic-id="' + Toolbox.escapeHtml(t.id) + '" title="항목 보기">i</button></div>';
                        });
                        html += '</div></div>';
                    });
                    return html;
                }

                container.innerHTML = '<div class="randomgen-wide">' +
                    '<main class="randomgen-display"><div id="randomResults" class="randomgen-results"></div></main>' +
                    '<footer class="randomgen-footer">' +
                    '<div class="randomgen-bottom-bar">' +
                    '<div class="randomgen-row randomgen-row-main"><button class="btn btn-primary randomgen-gen-btn" id="randomGenBtn">뽑기</button></div>' +
                    '<div class="randomgen-row randomgen-row-options">' +
                    '<button type="button" class="randomgen-icon-btn" id="randomTopicBtn" title="주제 선택"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg><span class="randomgen-topic-label" id="randomTopicLabel">전체</span></button>' +
                    '<div class="randomgen-count-wrap"><input type="number" id="randomCountInput" min="1" max="99" value="5" class="field-input"><div class="randomgen-presets"><button type="button" class="btn btn-ghost random-count-preset" data-value="1">1</button><button type="button" class="btn btn-ghost random-count-preset" data-value="3">3</button><button type="button" class="btn btn-ghost random-count-preset" data-value="5">5</button><button type="button" class="btn btn-ghost random-count-preset" data-value="10">10</button></div></div>' +
                    '<button type="button" class="randomgen-icon-btn" id="randomNoDupBtn" title="중복 없음"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h10M4 18h6"/></svg></button>' +
                    '<button type="button" class="randomgen-icon-btn" id="randomStoryBtn" title="이야기 만들기" style="display:none;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/><path d="M8 7h8"/><path d="M8 11h8"/></svg></button>' +
                    '<button type="button" class="randomgen-icon-btn" id="randomCopyBtn" title="전체 키워드 복사" style="display:none;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>' +
                    '<button type="button" class="randomgen-icon-btn" id="randomInfoBtn" title="출처"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg></button>' +
                    '</div></div>' +
                    '</footer>' +
                    '</div>' +
                    '<div class="randomgen-popup-backdrop" id="randomTopicPopup"><div class="randomgen-popup randomgen-topic-popup-large" onclick="event.stopPropagation()"><div class="randomgen-topic-popup-tabs"><button type="button" class="randomgen-topic-tab active" data-tab="select">주제 선택</button><button type="button" class="randomgen-topic-tab" data-tab="add">주제 추가</button></div><div class="randomgen-topic-tab-panel active" id="randomTopicSelectPanel" data-tab="select"><div class="randomgen-topic-actions"><button type="button" class="btn btn-ghost" id="randomTopicSelectAll">전체 선택</button><button type="button" class="btn btn-ghost" id="randomTopicDeselectAll">전체 해제</button></div><div id="randomTopicList" class="randomgen-topic-grid">' + buildTopicCardsHtml() + '</div><div id="randomTopicOptionsWrap" class="randomgen-topic-options-wrap"><div class="randomgen-popup-title">선택된 주제 <span class="randomgen-topic-hint">(핀: 고정 1개 포함)</span></div><div id="randomTopicSelectedList" class="randomgen-topic-selected-list"></div></div></div><div class="randomgen-topic-tab-panel" id="randomTopicAddPanel" data-tab="add"><div class="randomgen-add-section"><div class="randomgen-add-header"><h3 class="randomgen-add-title">새 주제 추가</h3><p class="randomgen-add-desc">쉼표로 구분해 항목을 입력하세요.</p></div><div id="randomTopicAddForm" class="randomgen-add-form"><div class="randomgen-add-row"><div class="randomgen-add-field"><label class="randomgen-add-label">이름</label><input type="text" id="addTopicLabel" placeholder="내 주제" class="field-input"></div><div class="randomgen-add-field"><label class="randomgen-add-label">그룹</label><input type="text" id="addTopicGroup" placeholder="기타" class="field-input" value="기타"></div></div><div class="randomgen-add-field randomgen-add-field-id"><label class="randomgen-add-label">ID <span class="randomgen-add-hint">(비워두면 이름에서 자동 생성)</span></label><input type="text" id="addTopicId" placeholder="my_topic" class="field-input"></div><div class="randomgen-add-field"><label class="randomgen-add-label">항목</label><textarea id="addTopicItems" class="field-input" rows="4" placeholder="항목1, 항목2, 항목3"></textarea></div><div class="randomgen-add-actions"><button class="btn btn-primary" id="addTopicBtn">추가</button><span id="addTopicMsg" class="randomgen-add-msg"></span></div></div></div></div><button type="button" class="randomgen-topic-close-fab" id="randomTopicClose" title="닫기">×</button></div></div>' +
                    '<div class="randomgen-popup-backdrop" id="randomTablePopup"><div class="randomgen-popup randomgen-table-popup" onclick="event.stopPropagation()"><div class="randomgen-popup-title" id="randomTableTitle">항목</div><div id="randomTableContent" class="randomgen-table-content"></div><button type="button" class="btn btn-ghost" style="margin-top:12px;" id="randomTableClose">닫기</button></div></div>' +
                    '<div class="randomgen-popup-backdrop" id="randomInfoPopup"><div class="randomgen-popup" onclick="event.stopPropagation()"><div class="randomgen-popup-title">출처</div><div class="randomgen-info-content"><p>주제·키워드 참고:</p><p><a href="https://nihilapp.github.io/keyword" target="_blank" rel="noopener" style="color:var(--accent);">니힐 랜덤 키워드</a> · <a href="https://github.com/nihilapp/random-keyword-code" target="_blank" rel="noopener" style="color:var(--accent);">GitHub</a></p></div><button type="button" class="btn btn-ghost" style="margin-top:12px;" id="randomInfoClose">닫기</button></div></div>';

                var selectedTopicIds = new Set();
                var fixedTopicIds = new Set();
                var topicLabels = {};
                topics.forEach(function (t) { topicLabels[t.id] = t.label; });
                window.RANDOMGEN_TOPIC_LABELS = topicLabels;

                const topicBtn = container.querySelector('#randomTopicBtn');
                const topicLabel = container.querySelector('#randomTopicLabel');
                const countInput = container.querySelector('#randomCountInput');
                const noDupBtn = container.querySelector('#randomNoDupBtn');
                var noDuplicate = false;
                noDupBtn.onclick = function () {
                    noDuplicate = !noDuplicate;
                    noDupBtn.classList.toggle('active', noDuplicate);
                    noDupBtn.title = noDuplicate ? '중복 없음 (켜짐)' : '중복 없음';
                };
                const genBtn = container.querySelector('#randomGenBtn');

                function getCount() {
                    var v = parseInt(countInput.value, 10);
                    return (isNaN(v) || v < 1) ? 1 : Math.min(99, v);
                }
                function setCount(n) {
                    n = Math.max(1, Math.min(99, n));
                    countInput.value = n;
                    container.querySelectorAll('.random-count-preset').forEach(function (btn) {
                        btn.classList.toggle('active', parseInt(btn.dataset.value, 10) === n);
                    });
                }
                function updateCountPresetStyle() {
                    var n = getCount();
                    container.querySelectorAll('.random-count-preset').forEach(function (btn) {
                        btn.classList.toggle('active', parseInt(btn.dataset.value, 10) === n);
                    });
                }
                countInput.addEventListener('input', function () {
                    updateCountPresetStyle();
                });
                countInput.addEventListener('change', function () {
                    setCount(getCount());
                });
                container.querySelectorAll('.random-count-preset').forEach(function (btn) {
                    btn.onclick = function () { setCount(parseInt(btn.dataset.value, 10)); };
                });
                setCount(5);

                function updateTopicLabel() {
                    topicLabel.textContent = selectedTopicIds.size ? selectedTopicIds.size + '개 선택' : '전체';
                }
                function updateOptionsWrap() {
                    var wrap = container.querySelector('#randomTopicOptionsWrap');
                    var selectedList = container.querySelector('#randomTopicSelectedList');
                    container.querySelectorAll('.randomgen-topic-custom').forEach(function (card) {
                        var id = card.dataset.value;
                        card.classList.toggle('selected', selectedTopicIds.has(id));
                        card.classList.toggle('pinned', fixedTopicIds.has(id));
                    });
                    wrap.style.display = '';
                    if (selectedTopicIds.size > 0) {
                        selectedList.innerHTML = Array.from(selectedTopicIds).map(function (id) {
                            var label = Toolbox.escapeHtml(topicLabels[id] || id);
                            var pinned = fixedTopicIds.has(id);
                            var pinSvg = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v10m0 0l-3-3m3 3l3-3M5 12v7a1 1 0 001 1h12a1 1 0 001-1v-7"/></svg>';
                            var pinTitle = pinned ? '고정 해제' : '고정 (항상 1개 포함)';
                            return '<span class="randomgen-topic-selected-chip' + (pinned ? ' pinned' : '') + '" data-topic-id="' + Toolbox.escapeHtml(id) + '"><span class="randomgen-chip-label">' + label + '</span><button type="button" class="randomgen-topic-pin" data-topic-id="' + Toolbox.escapeHtml(id) + '" title="' + pinTitle + '">' + pinSvg + '</button></span>';
                        }).join('');
                        selectedList.querySelectorAll('.randomgen-topic-pin').forEach(function (btn) {
                            btn.onclick = function (e) {
                                e.stopPropagation();
                                var id = btn.dataset.topicId;
                                if (fixedTopicIds.has(id)) fixedTopicIds.delete(id); else fixedTopicIds.add(id);
                                updateOptionsWrap();
                            };
                        });
                        selectedList.querySelectorAll('.randomgen-topic-selected-chip').forEach(function (chip) {
                            chip.onclick = function (e) {
                                if (e.target.closest('.randomgen-topic-pin')) return;
                                var id = chip.dataset.topicId;
                                selectedTopicIds.delete(id);
                                fixedTopicIds.delete(id);
                                updateTopicLabel();
                                updateOptionsWrap();
                            };
                        });
                    } else {
                        selectedList.innerHTML = '<span class="randomgen-topic-selected-empty">카드를 클릭해 주제를 선택하세요 (비어 있으면 전체)</span>';
                    }
                }

                var topicPopup = container.querySelector('#randomTopicPopup');
                topicBtn.onclick = function () { switchTopicTab('select'); updateOptionsWrap(); topicPopup.classList.add('open'); };
                container.querySelector('#randomTopicClose').onclick = function () { topicPopup.classList.remove('open'); };
                container.querySelector('#randomTopicSelectAll').onclick = function () {
                    topics.forEach(function (t) { selectedTopicIds.add(t.id); });
                    updateTopicLabel();
                    updateOptionsWrap();
                };
                container.querySelector('#randomTopicDeselectAll').onclick = function () {
                    selectedTopicIds.clear();
                    fixedTopicIds.clear();
                    updateTopicLabel();
                    updateOptionsWrap();
                };
                topicPopup.onclick = function (e) { if (e.target === topicPopup) topicPopup.classList.remove('open'); };
                container.querySelector('#randomTopicList').addEventListener('click', function (e) {
                    if (e.target.closest('.randomgen-topic-info')) return;
                    var customCard = e.target.closest('.randomgen-topic-custom');
                    if (customCard && !e.target.closest('.randomgen-topic-info')) {
                        var id = customCard.dataset.value;
                        if (selectedTopicIds.has(id)) {
                            selectedTopicIds.delete(id);
                            fixedTopicIds.delete(id);
                        } else {
                            selectedTopicIds.add(id);
                        }
                        updateTopicLabel();
                        updateOptionsWrap();
                    }
                });
                container.querySelector('#randomTopicList').addEventListener('click', function (e) {
                    var infoBtn = e.target.closest('.randomgen-topic-info');
                    if (!infoBtn) return;
                    e.stopPropagation();
                    var tid = infoBtn.dataset.topicId;
                    var t = topics.find(function (x) { return x.id === tid; });
                    if (!t) return;
                    var items = getTopicItems(t);
                    var isGen = !!t.generator && !t.items;
                    var title = t.label + (isGen ? ' (샘플)' : '');
                    var cellHtml = items.map(function (it) { return '<span class="randomgen-table-tag">' + Toolbox.escapeHtml(it) + '</span>'; }).join('');
                    container.querySelector('#randomTableTitle').textContent = title;
                    container.querySelector('#randomTableContent').innerHTML = cellHtml || '<span style="color:var(--text-tertiary);">항목 없음</span>';
                    container.querySelector('#randomTablePopup').classList.add('open');
                });
                container.querySelector('#randomTableClose').onclick = function () { container.querySelector('#randomTablePopup').classList.remove('open'); };
                container.querySelector('#randomTablePopup').onclick = function (e) { if (e.target === this) this.classList.remove('open'); };

                var infoPopup = container.querySelector('#randomInfoPopup');
                container.querySelector('#randomInfoBtn').onclick = function () { infoPopup.classList.add('open'); };
                container.querySelector('#randomInfoClose').onclick = function () { infoPopup.classList.remove('open'); };
                infoPopup.onclick = function (e) { if (e.target === infoPopup) infoPopup.classList.remove('open'); };

                function switchTopicTab(tabId) {
                    container.querySelectorAll('.randomgen-topic-tab').forEach(function (t) { t.classList.toggle('active', t.dataset.tab === tabId); });
                    container.querySelectorAll('.randomgen-topic-tab-panel').forEach(function (p) { p.classList.toggle('active', p.dataset.tab === tabId); });
                }
                container.querySelectorAll('.randomgen-topic-tab').forEach(function (tab) {
                    tab.onclick = function () { switchTopicTab(tab.dataset.tab); };
                });
                container.querySelector('#addTopicBtn').onclick = function () {
                    var label = (container.querySelector('#addTopicLabel').value || '').trim();
                    var idRaw = (container.querySelector('#addTopicId').value || '').trim();
                    var id = idRaw ? idRaw.replace(/\s+/g, '_') : (label ? label.replace(/\s+/g, '_') : '');
                    var group = (container.querySelector('#addTopicGroup').value || '기타').trim();
                    var itemsStr = (container.querySelector('#addTopicItems').value || '').trim();
                    var items = itemsStr.split(/[,，]/).map(function (s) { return s.trim(); }).filter(Boolean);
                    var msgEl = container.querySelector('#addTopicMsg');
                    if (!id || !label || items.length === 0) {
                        msgEl.textContent = '이름과 항목을 입력해 주세요.';
                        msgEl.style.color = 'var(--accent)';
                        return;
                    }
                    if (topics.some(function (t) { return t.id === id; })) {
                        msgEl.textContent = '이미 존재하는 ID입니다.';
                        msgEl.style.color = 'var(--accent)';
                        return;
                    }
                    topics.push({ id: id, label: label, group: group, items: items });
                    if (window.RANDOMGEN_TOPIC_LABELS) window.RANDOMGEN_TOPIC_LABELS[id] = label;
                    var list = container.querySelector('#randomTopicList');
                    var grp = Array.from(list.querySelectorAll('.randomgen-topic-group')).find(function (g) { return g.querySelector('.randomgen-topic-group-title').textContent === group; });
                    if (!grp) {
                        grp = document.createElement('div');
                        grp.className = 'randomgen-topic-group';
                        grp.innerHTML = '<div class="randomgen-topic-group-title">' + Toolbox.escapeHtml(group) + '</div><div class="randomgen-topic-cards"></div>';
                        list.appendChild(grp);
                    }
                    var cards = grp.querySelector('.randomgen-topic-cards');
                    var card = document.createElement('div');
                    card.className = 'randomgen-topic-card randomgen-topic-custom';
                    card.dataset.value = id;
                    card.innerHTML = '<span class="randomgen-topic-card-label">' + Toolbox.escapeHtml(label) + '</span><button type="button" class="randomgen-topic-info" data-topic-id="' + Toolbox.escapeHtml(id) + '" title="항목 보기">i</button>';
                    cards.appendChild(card);
                    container.querySelector('#addTopicId').value = '';
                    container.querySelector('#addTopicLabel').value = '';
                    container.querySelector('#addTopicItems').value = '';
                    msgEl.textContent = '"' + label + '" 추가됨.';
                    msgEl.style.color = 'var(--text-secondary)';
                    Mdd.setMood('happy');
                    Mdd.say('추가됐다냥!');
                    switchTopicTab('select');
                    updateOptionsWrap();
                };
                const storyBtn = container.querySelector('#randomStoryBtn');
                const resultsEl = container.querySelector('#randomResults');
                let lastBatchResults = [];

                const copyBtn = container.querySelector('#randomCopyBtn');
                copyBtn.onclick = function () {
                    var text = lastBatchResults.map(function (r) { return r.name; }).join(', ');
                    if (!text) return;
                    if (navigator.clipboard) {
                        navigator.clipboard.writeText(text);
                        Toolbox.showToast('전체 키워드 복사됨 (' + lastBatchResults.length + '개)');
                    }
                };

                function generateBatch() {
                    resultsEl.innerHTML = '';
                    if (copyBtn) copyBtn.style.display = 'none';
                    const count = getCount();

                    var targetTopics = [];
                    if (selectedTopicIds.size > 0) {
                        selectedTopicIds.forEach(function (id) {
                            var t = topics.find(function (x) { return x.id === id; });
                            if (t) targetTopics.push(t);
                        });
                    } else {
                        targetTopics = topics.slice();
                    }

                    if (!targetTopics.length) {
                        resultsEl.innerHTML = '<div class="randomgen-empty">주제를 선택해 주세요.</div>';
                        return;
                    }

                    const allResults = [];
                    var seenNames = new Set();
                    fixedTopicIds.forEach(function (fixedId) {
                        if (!selectedTopicIds.has(fixedId)) return;
                        var t = topics.find(function (x) { return x.id === fixedId; });
                        if (t && targetTopics.some(function (x) { return x.id === fixedId; })) {
                            var r = generate(t, 1)[0];
                            if (r) {
                                allResults.push(r);
                                seenNames.add(r.name);
                            }
                        }
                    });
                    var remain = count - allResults.length;
                    for (var i = 0; i < remain; i++) {
                        var r = null;
                        if (noDuplicate) {
                            for (var retries = 0; retries < 100; retries++) {
                                var topic = pick(targetTopics);
                                var candidate = generate(topic, 1)[0];
                                if (candidate && !seenNames.has(candidate.name)) {
                                    r = candidate;
                                    seenNames.add(candidate.name);
                                    break;
                                }
                            }
                        } else {
                            var topic = pick(targetTopics);
                            r = generate(topic, 1)[0];
                            if (r) seenNames.add(r.name);
                        }
                        if (r) allResults.push(r);
                    }
                    for (var j = allResults.length - 1; j > 0; j--) {
                        var k = Math.floor(Math.random() * (j + 1));
                        var tmp = allResults[j];
                        allResults[j] = allResults[k];
                        allResults[k] = tmp;
                    }

                    lastBatchResults = allResults.slice();

                    function isHexColor(s) { return /^#[0-9a-fA-F]{6}$/.test(String(s)); }
                    function hexLuminance(hex) {
                        var m = hex.match(/^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$/);
                        if (!m) return 0;
                        var r = parseInt(m[1], 16) / 255, g = parseInt(m[2], 16) / 255, b = parseInt(m[3], 16) / 255;
                        return 0.299 * r + 0.587 * g + 0.114 * b;
                    }
                    var colorNameToHex = { '빨강':'#dc2626','주황':'#ea580c','노랑':'#eab308','초록':'#22c55e','파랑':'#3b82f6','남색':'#1e40af','보라':'#8b5cf6','분홍':'#ec4899','흰색':'#f8fafc','검정':'#1e293b','회색':'#64748b','베이지':'#d4a574','골드':'#eab308','실버':'#94a3b8','청록':'#14b8a6','마룬':'#881337' };
                    function getCardBgColor(result) {
                        if (isHexColor(result.name)) return result.name;
                        if (result.sub === '색' && colorNameToHex[result.name]) return colorNameToHex[result.name];
                        return null;
                    }
                    var ccgColors = ['blue', 'purple', 'gold', 'green', 'red'];
                    allResults.forEach(function (result, idx) {
                        const card = document.createElement('div');
                        var ccgClass = 'randomgen-ccg-' + ccgColors[Math.floor(Math.random() * ccgColors.length)];
                        var bgHex = getCardBgColor(result);
                        var isColor = !!bgHex;
                        if (isColor) ccgClass = 'randomgen-ccg-hex';
                        card.className = 'randomgen-result-card randomgen-ccg ' + ccgClass;
                        card.style.animationDelay = (idx * 120) + 'ms';
                        card.title = '클릭하면 복사';
                        const nameEsc = Toolbox.escapeHtml(result.name);
                        const subEsc = Toolbox.escapeHtml(result.sub);
                        var frameStyle = '';
                        var titleClass = 'randomgen-ccg-title';
                        if (isColor) {
                            frameStyle = ' style="background:' + bgHex + '!important;border-color:' + bgHex + ';"';
                            if (hexLuminance(bgHex) > 0.6) titleClass += ' randomgen-ccg-title-dark';
                        }
                        card.innerHTML = '<div class="randomgen-card-inner"><div class="randomgen-card-back"><span class="randomgen-card-question">?</span></div><div class="randomgen-card-front"><div class="randomgen-ccg-frame"' + frameStyle + '><div class="randomgen-ccg-title-area"><div class="' + titleClass + '">' + nameEsc + '</div></div><div class="randomgen-ccg-type">' + subEsc + '</div></div></div></div>';
                        resultsEl.appendChild(card);
                        (function (c) {
                            setTimeout(function () { c.classList.add('revealed'); }, 350 + idx * 100);
                        })(card);

                        (function (cardEl, text) {
                            var inner = cardEl.querySelector('.randomgen-card-inner');
                            var maxTilt = 22;
                            cardEl.addEventListener('click', function () {
                                if (!cardEl.classList.contains('revealed')) return;
                                if (navigator.clipboard) {
                                    navigator.clipboard.writeText(text).then(function () {
                                        Toolbox.showToast('복사됨');
                                    }).catch(function () {});
                                }
                            });
                            cardEl.addEventListener('mousemove', function (e) {
                                if (!cardEl.classList.contains('revealed')) return;
                                var rect = cardEl.getBoundingClientRect();
                                var x = (e.clientX - rect.left) / rect.width - 0.5;
                                var y = (e.clientY - rect.top) / rect.height - 0.5;
                                var rotY = x * maxTilt * 2;
                                var rotX = -y * maxTilt * 2;
                                cardEl.classList.add('randomgen-tilt');
                                inner.style.transform = 'rotateY(180deg) rotateX(' + rotX + 'deg) rotateY(' + rotY + 'deg)';
                            });
                            cardEl.addEventListener('mouseleave', function () {
                                cardEl.classList.remove('randomgen-tilt');
                                inner.style.transform = cardEl.classList.contains('revealed') ? 'rotateY(180deg)' : '';
                            });
                        })(card, result.name);
                    });

                    if (storyBtn) storyBtn.style.display = '';
                    if (copyBtn) copyBtn.style.display = '';

                    if (noDuplicate && allResults.length < count) {
                        Toolbox.showToast('고유 항목이 부족해 ' + allResults.length + '개만 뽑았어요.', 'warning');
                        Mdd.setMood('think');
                        Mdd.say('중복 없이 뽑을 수 있는 게 이거까지만 있대요.');
                    } else {
                        Mdd.setMood('happy');
                        Mdd.say('이거 어떠냥?');
                    }
                    Mdd.addAffection(1);
                }

                if (storyBtn) {
                    storyBtn.onclick = function () {
                        var kw = lastBatchResults.map(function (r) { return r.name; });
                        if (kw.length === 0) {
                            Toolbox.showToast('먼저 키워드를 뽑아주세요.', 'error');
                            return;
                        }
                        try {
                            sessionStorage.setItem('toolbox_chatbot_story_keywords', JSON.stringify(kw));
                        } catch (e) {}
                        Toolbox.switchPage('chatbot');
                        Mdd.say('챗봇에서 이야기 만들어보라냥!');
                    };
                }

                genBtn.onclick = generateBatch;
                generateBatch();
            }
        }]
    });
})();
