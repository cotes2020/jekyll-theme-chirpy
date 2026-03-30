// @ts-nocheck
/** 캐릭터 저장소·폼·모달 (chatbot.js에서 세션·전송과 연동) */
(function () {
    let cbCharModalEscBound = false;
    let cbCharModalTabBound = false;
    let charModalPreviousFocus = null;

    const CHARACTERS_KEY = 'karmolab_chatbot_characters_v1';
    function defaultCharacterSeed() {
        return {
            id: 'c_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8),
            name: '새 캐릭터',
            userName: '사용자',
            userNote: '',
            visualDescription: '',
            description: '',
            personality: '',
            scenario: '',
            firstMes: '',
            referenceImageDataUrl: ''
        };
    }

    /** SillyTavern Character Card V2/V3 `data` 또는 이 위젯 내보내기 JSON → 내부 캐릭터 객체 (항상 새 id) */
    function mapImportedJsonToCharacter(obj) {
        if (!obj || typeof obj !== 'object') throw new Error('JSON이 아닙니다.');
        if (obj.spec === 'karmochat_character_v1' && obj.data && typeof obj.data === 'object') {
            const d = obj.data;
            const str = x => (x == null ? '' : String(x)).trim();
            return {
                id: 'c_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8),
                name: str(d.name) || '가져온 캐릭터',
                userName: str(d.userName) || '사용자',
                userNote: str(d.userNote),
                visualDescription: str(d.visualDescription),
                description: str(d.description),
                personality: str(d.personality),
                scenario: str(d.scenario),
                firstMes: str(d.firstMes),
                referenceImageDataUrl: typeof d.referenceImageDataUrl === 'string' && d.referenceImageDataUrl.startsWith('data:')
                    ? d.referenceImageDataUrl
                    : ''
            };
        }
        let d = obj.data;
        if (!d || typeof d !== 'object') {
            if (obj.name != null || obj.description != null || obj.personality != null) d = obj;
        }
        if (!d || typeof d !== 'object') throw new Error('캐릭터 data 블록을 찾을 수 없습니다. (SillyTavern V2 JSON인지 확인)');
        const str = x => (x == null ? '' : String(x)).trim();
        const name = str(d.name) || '가져온 캐릭터';
        let description = str(d.description);
        const sp = str(d.system_prompt);
        if (sp) description += (description ? '\n\n' : '') + '[시스템 프롬프트]\n' + sp;
        let scenario = str(d.scenario);
        const mesEx = str(d.mes_example);
        if (mesEx) scenario += (scenario ? '\n\n' : '') + '[대화 예시]\n' + mesEx;
        const notes = [str(d.creator_notes), str(d.post_history_instructions)].filter(Boolean).join('\n\n');
        let firstMes = str(d.first_mes || d.firstMes);
        if (!firstMes && Array.isArray(d.alternate_greetings) && d.alternate_greetings.length) {
            firstMes = str(d.alternate_greetings[0]);
        }
        let visualDescription = str(d.appearance);
        if (!visualDescription && d.extensions && typeof d.extensions === 'object') {
            visualDescription = str(d.extensions.portrait || d.extensions.face || '');
        }
        return {
            id: 'c_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8),
            name,
            userName: '사용자',
            userNote: notes,
            visualDescription,
            description,
            personality: str(d.personality),
            scenario,
            firstMes,
            referenceImageDataUrl: ''
        };
    }

    function exportCurrentCharacterToJsonFile() {
        const ch = readCharacterFromForm();
        if (!ch) {
            Toolbox.showToast('내보낼 캐릭터가 없습니다.', 'error');
            return;
        }
        const exportObj = {
            spec: 'karmochat_character_v1',
            exportedAt: new Date().toISOString(),
            data: {
                name: ch.name,
                userName: ch.userName,
                userNote: ch.userNote,
                visualDescription: ch.visualDescription,
                description: ch.description,
                personality: ch.personality,
                scenario: ch.scenario,
                firstMes: ch.firstMes,
                referenceImageDataUrl: ch.referenceImageDataUrl || ''
            }
        };
        const safe = (ch.name || 'character').replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_').slice(0, 60);
        const blob = new Blob([JSON.stringify(exportObj, null, 2)], { type: 'application/json;charset=utf-8' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `karmochat-${safe}-${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        URL.revokeObjectURL(a.href);
        Toolbox.showToast('캐릭터 JSON을 내보냈습니다.');
    }

    async function zlibInflateZtxChunk(compressed) {
        if (typeof DecompressionStream === 'undefined') throw new Error('이 브라우저는 PNG zTXt 압축 해제를 지원하지 않습니다.');
        const ds = new DecompressionStream('deflate');
        const buf = await new Response(new Blob([compressed]).stream().pipeThrough(ds)).arrayBuffer();
        return new Uint8Array(buf);
    }

    /** SillyTavern 등 PNG의 tEXt/zTXt `chara` 청크 → 원본 JSON 객체 */
    async function extractCharaObjectFromPngBuffer(buffer) {
        const view = new DataView(buffer);
        if (buffer.byteLength < 24) return null;
        if (view.getUint32(0) !== 0x89504E47 || view.getUint32(4) !== 0x0D0A1A0A) return null;
        let offset = 8;
        const decoder = new TextDecoder();
        while (offset + 12 <= buffer.byteLength) {
            const len = view.getUint32(offset);
            const type = String.fromCharCode(
                view.getUint8(offset + 4), view.getUint8(offset + 5),
                view.getUint8(offset + 6), view.getUint8(offset + 7)
            );
            const dataOffset = offset + 8;
            if (len < 0 || len > buffer.byteLength || dataOffset + len > buffer.byteLength) break;
            if (type === 'tEXt' && len > 0) {
                const chunk = new Uint8Array(buffer, dataOffset, len);
                let i = 0;
                while (i < chunk.length && chunk[i] !== 0) i++;
                const keyword = decoder.decode(chunk.slice(0, i));
                const text = decoder.decode(chunk.slice(i + 1));
                if (keyword.toLowerCase() === 'chara') {
                    try {
                        const jsonStr = atob(text.replace(/\s/g, ''));
                        return JSON.parse(jsonStr);
                    } catch (_) {}
                }
            }
            if (type === 'zTXt' && len > 2) {
                const chunk = new Uint8Array(buffer, dataOffset, len);
                let i = 0;
                while (i < chunk.length && chunk[i] !== 0) i++;
                const keyword = decoder.decode(chunk.slice(0, i));
                const compMethod = chunk[i + 1];
                const compressed = chunk.slice(i + 2);
                if (keyword.toLowerCase() === 'chara' && compMethod === 0 && compressed.length) {
                    try {
                        const inflated = await zlibInflateZtxChunk(compressed);
                        const jsonStr = decoder.decode(inflated);
                        return JSON.parse(jsonStr);
                    } catch (_) {}
                }
            }
            offset += 12 + len;
        }
        return null;
    }

    async function parseCharacterImportFile(buffer) {
        const u8 = new Uint8Array(buffer);
        if (u8.length >= 8 && u8[0] === 0x89 && u8[1] === 0x50 && u8[2] === 0x4E && u8[3] === 0x47) {
            const obj = await extractCharaObjectFromPngBuffer(buffer);
            if (obj) return obj;
            throw new Error('PNG에 chara 메타데이터가 없습니다. (SillyTavern에서 내보낸 카드 PNG인지 확인)');
        }
        const text = new TextDecoder('utf-8').decode(buffer);
        return JSON.parse(text);
    }

    function loadCharacterList() {
        try {
            const raw = localStorage.getItem(CHARACTERS_KEY);
            const arr = raw ? JSON.parse(raw) : [];
            return Array.isArray(arr) ? arr : [];
        } catch (_) {
            return [];
        }
    }

    function saveCharacterList(list) {
        try {
            localStorage.setItem(CHARACTERS_KEY, JSON.stringify(list));
        } catch (e) {
            console.warn('saveCharacterList', e);
            Toolbox.showToast('캐릭터 저장 실패(용량 초과 등). 참조 이미지를 줄여 보세요.', 'error');
        }
    }

    /** imagegen CHARACTER_PRESETS(witch / alisa / ling)와 동일 컨셉 — id 기준으로 없을 때만 병합 */
    function getBuiltinMascotCharacters() {
        try {
            const b = window.KarmoWorld?.bindings?.chatbot?.characters;
            if (Array.isArray(b) && b.length) {
                const out = b.map(x => ({
                    id: x.chatbotId,
                    name: x.name,
                    userName: x.userName,
                    userNote: x.userNote,
                    visualDescription: x.visualDescription,
                    description: x.description,
                    personality: x.personality,
                    scenario: x.scenario,
                    firstMes: x.firstMes,
                    referenceImageDataUrl: ''
                })).filter(x => x.id && x.name);
                if (out.length) return out;
            }
        } catch (_) {}
        return [
            {
                id: 'c_mascot_yon',
                name: '욘 (Yawn)',
                userName: '조수님',
                userNote: '이미지 생성 캐릭터 프리셋「마녀 욘」과 같은 설정.',
                visualDescription: 'Young adult witch Yawn, very slender petite, messy orange hair, half-lidded sleepy eyes, short thick eyebrows (maro-mayu), round glasses, slight blush, drooping nightcap, large fluffy sleeping earmuffs with orange spiral pattern, oversized loose witch robe, introverted cute atmosphere, soft colors, anime style',
                description: '나무 마법 저택에 사는 잠 많은 마녀. 카레·알리사·링과 같은 저택 세계관.',
                personality: '늘어지고 하품이 많지만 속은 따뜻하다. 귀찮은 듯 말하지만 챙겨 준다. 한국어로 말한다.',
                scenario: '따뜻한 마법 저택 거실이나 실험실에서 조수와 이야기 중.',
                firstMes: '…응, 조수님. 나 아직 살아 있어. 오늘은 뭐 할 거야? 나는… 일단 소파.',
                referenceImageDataUrl: ''
            },
            {
                id: 'c_mascot_alisa',
                name: '알리사',
                userName: '조수님',
                userNote: '이미지 생성 캐릭터 프리셋「메이드 알리사」와 같은 설정.',
                visualDescription: 'Cute maid Alisa, sharp intellectual eyes, stylish glasses (megane), stoic cool beauty expression, black ponytail, classic black and white maid outfit, large magical broomstick, dynamic posing, anime style, detailed',
                description: '저택을 돌보는 메이드. 냉정하고 똑똑해 보이지만 직무에는 성실하다.',
                personality: '차분하고 간결한 말투. 감정 표현은 적지만 툭툭 챙겨 준다. 한국어로 말한다.',
                scenario: '마법 저택에서 청소·정리·조수의 실험 보조를 맡고 있다.',
                firstMes: '조수님, 오늘 할 일 목록입니다. …빵 부스러기는 치울 테니 책상은 비워 주세요.',
                referenceImageDataUrl: ''
            },
            {
                id: 'c_mascot_ling',
                name: '링 (Ling)',
                userName: '조수님',
                userNote: '이미지 생성 캐릭터 프리셋「강시 링」과 같은 설정.',
                visualDescription: 'Beautiful Jiangshi Chinese vampire maid girl Ling, innocent baby face, mischievous smile, glamorous curvy body, dark brown hair in cute twin buns, black Qipao-Maid fusion dress form-fitting with frills, yellow paper talisman on forehead, floating pose, anime style, white background friendly',
                description: '강시(殭屍) 혈통의 메이드. 이마 부적과 장난기 많은 성격이 특징.',
                personality: '애교와 장난이 많고, 가끔 수줍은 척한다. 한국어로 말한다.',
                scenario: '저택에서 알리사와 함께 일하며 조수를 골탕 먹이기도 한다.',
                firstMes: '조수님 왔어요~? 오늘 저랑 놀아줄 거죠? …농담이에요. 아마도.',
                referenceImageDataUrl: ''
            }
        ];
    }

    function mergeBuiltinMascotCharactersIfMissing() {
        const list = loadCharacterList();
        const existing = new Set(list.map(c => c.id));
        let changed = false;
        for (const ch of getBuiltinMascotCharacters()) {
            if (!existing.has(ch.id)) {
                list.push(ch);
                changed = true;
            }
        }
        if (changed) saveCharacterList(list);
    }

    function ensureDefaultCharacters() {
        let list = loadCharacterList();
        if (list.length === 0) {
            const ch = defaultCharacterSeed();
            ch.id = 'c_default_secretary';
            ch.name = '카레 (비서)';
            ch.userName = '조수님';
            ch.userNote = '실험과 기록을 맡은 연구 조수.';
            ch.visualDescription = 'Anime chibi mascot, short hair, white lab coat, friendly eyes, soft warm lighting, clean illustration, single character';
            ch.description = 'KarmoLab에서 조수를 돕는 전문 비서이자 실험 조수.';
            ch.personality = '친절하고 효율적이며, 가끔 가벼운 유머를 섞는다. 한국어로 말한다.';
            ch.scenario = 'KarmoLab 연구실에서 조수와 대화하고 있다.';
            ch.firstMes = '어서 오세요, 조수님! 오늘은 무엇부터 정리할까요?';
            list = [ch];
            saveCharacterList(list);
        }
        mergeBuiltinMascotCharactersIfMissing();
        return loadCharacterList();
    }

    function getCharacterById(id) {
        return loadCharacterList().find(x => x.id === id) || null;
    }
    function buildCharacterSystemBlock(char) {
        if (!char) return '';
        const user = char.userName || '사용자';
        const charName = char.name || '캐릭터';
        const parts = [
            '[역할]',
            `당신은 아래 캐릭터 "${charName}"으로 롤플레이합니다. 대사와 서술에 몰입하되, 사용자의 실제 질문·업무 요청에는 정확히 대답합니다.`,
            '',
            '[플레이어 / {{user}}]',
            `이름: ${user}`,
            char.userNote ? `설명: ${char.userNote}` : '',
            '',
            `[캐릭터 / {{char}} — ${charName}]`,
            char.description ? `설정: ${char.description}` : '',
            char.personality ? `성격: ${char.personality}` : '',
            char.scenario ? `상황: ${char.scenario}` : '',
            char.visualDescription ? `외형(이미지·일관성용, 대사에 그대로 읊지 말 것): ${char.visualDescription}` : '',
            '',
            '캐릭터의 말투를 유지하세요. 한국어를 기본으로 사용합니다.'
        ];
        return parts.filter(Boolean).join('\n');
    }
    function syncAfterSessionLoad(sessionCharacterId) {
        const charSel = document.getElementById('cbCharacterSelect');
        if (!charSel) return;
        if (sessionCharacterId && getCharacterById(sessionCharacterId)) {
            charSel.value = sessionCharacterId;
        }
        applyCharacterFormFromSelection();
        updateChatHeaderTitle();
    }

    function populateCharacterSelectOptions() {
        const charSel = document.getElementById('cbCharacterSelect');
        if (!charSel) return;
        const cur = charSel.value;
        const list = ensureDefaultCharacters();
        charSel.innerHTML = list.map(c => `<option value="${Toolbox.escapeHtml(c.id)}">${Toolbox.escapeHtml(c.name || c.id)}</option>`).join('');
        if (cur && getCharacterById(cur)) charSel.value = cur;
        else if (list[0]) charSel.value = list[0].id;
    }

    function applyCharacterFormFromSelection() {
        const charSel = document.getElementById('cbCharacterSelect');
        const ch = charSel && getCharacterById(charSel.value);
        if (!ch) {
            updateCharProfilePreview();
            return;
        }
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.value = v ?? ''; };
        set('cbCharName', ch.name);
        set('cbCharUserName', ch.userName);
        set('cbCharUserNote', ch.userNote);
        set('cbCharVisual', ch.visualDescription);
        set('cbCharDesc', ch.description);
        set('cbCharPersonality', ch.personality);
        set('cbCharScenario', ch.scenario);
        set('cbCharFirstMes', ch.firstMes);
        const thumb = document.getElementById('cbCharRefThumb');
        if (thumb) {
            if (ch.referenceImageDataUrl) {
                thumb.src = ch.referenceImageDataUrl;
                thumb.style.display = '';
            } else {
                thumb.removeAttribute('src');
                thumb.style.display = 'none';
            }
        }
        updateCharProfilePreview();
    }

    function updateCharProfilePreview() {
        const charSel = document.getElementById('cbCharacterSelect');
        const ch = charSel && getCharacterById(charSel.value);
        const refThumb = document.getElementById('cbCharRefThumb');
        const av = document.getElementById('cbCharProfileAvatar');
        const ph = document.getElementById('cbCharProfilePlaceholder');
        const nameEl = document.getElementById('cbCharProfileName');
        let refUrl = ch?.referenceImageDataUrl || '';
        if (refThumb && refThumb.getAttribute('src') && refThumb.style.display !== 'none') refUrl = refThumb.src || refUrl;
        if (nameEl) nameEl.textContent = ch ? (ch.name || '이름 없음') : '—';
        if (av && ph) {
            if (refUrl) {
                av.src = refUrl;
                av.style.display = '';
                ph.style.display = 'none';
            } else {
                av.removeAttribute('src');
                av.style.display = 'none';
                ph.style.display = '';
            }
        }
    }

    function readCharacterFromForm() {
        const g = id => document.getElementById(id)?.value?.trim() ?? '';
        const charSel = document.getElementById('cbCharacterSelect');
        const id = charSel?.value;
        if (!id) return null;
        const ch = getCharacterById(id) || { id };
        ch.name = g('cbCharName') || ch.name || '이름 없음';
        ch.userName = g('cbCharUserName') || '사용자';
        ch.userNote = g('cbCharUserNote');
        ch.visualDescription = g('cbCharVisual');
        ch.description = g('cbCharDesc');
        ch.personality = g('cbCharPersonality');
        ch.scenario = g('cbCharScenario');
        ch.firstMes = g('cbCharFirstMes');
        const thumb = document.getElementById('cbCharRefThumb');
        if (thumb && thumb.src && thumb.style.display !== 'none') ch.referenceImageDataUrl = thumb.src;
        else ch.referenceImageDataUrl = '';
        return ch;
    }

    function persistCharacterFromForm() {
        const ch = readCharacterFromForm();
        if (!ch) return;
        const list = loadCharacterList();
        const idx = list.findIndex(c => c.id === ch.id);
        if (idx >= 0) list[idx] = ch;
        else list.push(ch);
        saveCharacterList(list);
        populateCharacterSelectOptions();
        const selAfter = document.getElementById('cbCharacterSelect');
        if (selAfter) selAfter.value = ch.id;
        Toolbox.showToast('캐릭터 저장됨');
        updateChatHeaderTitle();
        updateCharProfilePreview();
    }

    function updateChatHeaderTitle() {
        const el = document.getElementById('cbChatTitle');
        const charSel = document.getElementById('cbCharacterSelect');
        if (!el || !charSel) return;
        const ch = getCharacterById(charSel.value);
        el.textContent = ch ? `💬 ${ch.name}` : '💬 챗봇';
    }

    function openCharEditModal() {
        const modal = document.getElementById('cbCharEditModal');
        if (!modal) return;
        charModalPreviousFocus = document.activeElement;
        applyCharacterFormFromSelection();
        modal.hidden = false;
        modal.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        document.body.dataset.cbCharModalOpen = '1';
        document.getElementById('cbCharEditClose')?.focus();
    }

    function closeCharEditModal() {
        const modal = document.getElementById('cbCharEditModal');
        if (!modal) return;
        modal.hidden = true;
        modal.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        delete document.body.dataset.cbCharModalOpen;
        updateCharProfilePreview();
        const prev = charModalPreviousFocus;
        charModalPreviousFocus = null;
        if (prev && typeof prev.focus === 'function' && document.body.contains(prev)) prev.focus();
        else document.getElementById('cbCharProfileOpen')?.focus();
    }

    function initCharacterUi(deps) {
        const saveSession = deps && deps.saveSession;
        const getChatHistoryLength = deps && deps.getChatHistoryLength;
        const appendBotFirstMes = deps && deps.appendBotFirstMes;
        const getLastLoadedSessionCharacterId = deps && deps.getLastLoadedSessionCharacterId;
        const block = document.getElementById('cbCharacterBlock');
        if (block?.dataset.inited === '1') {
            populateCharacterSelectOptions();
            applyCharacterFormFromSelection();
            updateChatHeaderTitle();
            updateCharProfilePreview();
            return;
        }
        if (block) block.dataset.inited = '1';

        populateCharacterSelectOptions();
        const pref = Toolbox.getPref('cb_active_character');
        const charSel = document.getElementById('cbCharacterSelect');
        if (charSel && pref && getCharacterById(pref)) charSel.value = pref;
        if (charSel && getLastLoadedSessionCharacterId) {
            const sid = getLastLoadedSessionCharacterId();
            if (sid && getCharacterById(sid)) charSel.value = sid;
        }
        applyCharacterFormFromSelection();

        const imgSel = document.getElementById('cbCharImageModel');
        if (imgSel && typeof Gemini !== 'undefined' && Gemini.MODELS?.geminiImage) {
            imgSel.innerHTML = '';
            Gemini.MODELS.geminiImage.forEach(m => {
                const o = document.createElement('option');
                o.value = m.id;
                o.textContent = m.name;
                if (m.isDefault) o.selected = true;
                imgSel.appendChild(o);
            });
            const saved = Toolbox.getPref('cb_char_image_model');
            if (saved) imgSel.value = saved;
            imgSel.addEventListener('change', () => Toolbox.setPref('cb_char_image_model', imgSel.value));
        }

        charSel?.addEventListener('change', () => {
            Toolbox.setPref('cb_active_character', charSel.value);
            applyCharacterFormFromSelection();
            updateChatHeaderTitle();
            saveSession();
        });

        document.getElementById('cbCharSave')?.addEventListener('click', () => persistCharacterFromForm());
        document.getElementById('cbCharNew')?.addEventListener('click', () => {
            const n = defaultCharacterSeed();
            const list = loadCharacterList();
            list.push(n);
            saveCharacterList(list);
            populateCharacterSelectOptions();
            charSel.value = n.id;
            applyCharacterFormFromSelection();
            updateChatHeaderTitle();
            Toolbox.showToast('새 캐릭터 — 내용을 채운 뒤 저장하세요');
        });
        document.getElementById('cbCharDel')?.addEventListener('click', () => {
            if (!charSel?.value) return;
            if (!confirm('이 캐릭터를 삭제할까요?')) return;
            const list = loadCharacterList().filter(c => c.id !== charSel.value);
            if (!list.length) {
                Toolbox.showToast('마지막 캐릭터는 삭제할 수 없습니다.', 'error');
                return;
            }
            saveCharacterList(list);
            populateCharacterSelectOptions();
            applyCharacterFormFromSelection();
            updateChatHeaderTitle();
            saveSession();
        });
        document.getElementById('cbCharRefFile')?.addEventListener('change', async e => {
            const f = e.target.files?.[0];
            const thumb = document.getElementById('cbCharRefThumb');
            if (!f || !f.type.startsWith('image/') || !thumb) return;
            const reader = new FileReader();
            reader.onload = () => {
                thumb.src = reader.result;
                thumb.style.display = '';
                updateCharProfilePreview();
            };
            reader.readAsDataURL(f);
            e.target.value = '';
        });
        document.getElementById('cbCharRefClear')?.addEventListener('click', () => {
            const thumb = document.getElementById('cbCharRefThumb');
            if (thumb) { thumb.removeAttribute('src'); thumb.style.display = 'none'; }
            updateCharProfilePreview();
        });
        document.getElementById('cbCharFirstBtn')?.addEventListener('click', () => {
            const ch = getCharacterById(charSel?.value);
            const fm = ch?.firstMes?.trim();
            if (!fm) { Toolbox.showToast('첫 인사가 비어 있습니다.', 'error'); return; }
            if (getChatHistoryLength && getChatHistoryLength() > 0) { Toolbox.showToast('대화가 이미 있을 때는 넣지 않습니다.', 'error'); return; }
            if (appendBotFirstMes) appendBotFirstMes(fm);
        });

        const importOverwriteEl = document.getElementById('cbCharImportOverwrite');
        if (importOverwriteEl) {
            const savedOw = Toolbox.getPref('cb_char_import_overwrite');
            if (savedOw === '1') importOverwriteEl.checked = true;
            importOverwriteEl.addEventListener('change', () => {
                Toolbox.setPref('cb_char_import_overwrite', importOverwriteEl.checked ? '1' : '0');
            });
        }

        document.getElementById('cbCharImportBtn')?.addEventListener('click', () => document.getElementById('cbCharImportFile')?.click());
        document.getElementById('cbCharImportFile')?.addEventListener('change', e => {
            const f = e.target.files?.[0];
            e.target.value = '';
            if (!f) return;
            const reader = new FileReader();
            reader.onload = async () => {
                try {
                    const buffer = reader.result;
                    if (!buffer || !(buffer instanceof ArrayBuffer)) throw new Error('파일을 읽지 못했습니다.');
                    const obj = await parseCharacterImportFile(buffer);
                    let ch = mapImportedJsonToCharacter(obj);
                    const overwrite = document.getElementById('cbCharImportOverwrite')?.checked;
                    const curId = charSel?.value;
                    const list = loadCharacterList();
                    if (overwrite && curId && getCharacterById(curId)) {
                        ch = Object.assign({}, ch, { id: curId });
                        const idx = list.findIndex(c => c.id === curId);
                        if (idx >= 0) list[idx] = ch;
                        else list.push(ch);
                        saveCharacterList(list);
                        Toolbox.showToast(`캐릭터를 덮어썼습니다: ${ch.name}`);
                    } else {
                        list.push(ch);
                        saveCharacterList(list);
                        Toolbox.showToast(`캐릭터 카드를 불러왔습니다: ${ch.name}`);
                    }
                    populateCharacterSelectOptions();
                    if (charSel) charSel.value = ch.id;
                    Toolbox.setPref('cb_active_character', ch.id);
                    applyCharacterFormFromSelection();
                    updateChatHeaderTitle();
                    updateCharProfilePreview();
                } catch (err) {
                    Toolbox.showToast('가져오기 실패: ' + (err && err.message ? err.message : err), 'error');
                }
            };
            reader.onerror = () => Toolbox.showToast('파일을 읽지 못했습니다.', 'error');
            reader.readAsArrayBuffer(f);
        });
        document.getElementById('cbCharExportBtn')?.addEventListener('click', () => exportCurrentCharacterToJsonFile());

        const syncAuto = () => {
            const use = document.getElementById('cbCharUse')?.checked;
            const auto = document.getElementById('cbCharAutoImage');
            if (auto) auto.disabled = !use;
            if (!use && auto) auto.checked = false;
        };
        document.getElementById('cbCharUse')?.addEventListener('change', () => { syncAuto(); saveSession(); });
        document.getElementById('cbCharAutoImage')?.addEventListener('change', () => saveSession());
        syncAuto();

        document.getElementById('cbCharProfileOpen')?.addEventListener('click', () => openCharEditModal());
        document.getElementById('cbCharEditBackdrop')?.addEventListener('click', closeCharEditModal);
        document.getElementById('cbCharEditClose')?.addEventListener('click', closeCharEditModal);
        if (!cbCharModalEscBound) {
            cbCharModalEscBound = true;
            document.addEventListener('keydown', e => {
                if (e.key !== 'Escape') return;
                const modal = document.getElementById('cbCharEditModal');
                if (!modal || modal.hidden) return;
                e.preventDefault();
                e.stopPropagation();
                closeCharEditModal();
            }, true);
        }
        if (!cbCharModalTabBound) {
            cbCharModalTabBound = true;
            document.addEventListener('keydown', e => {
                if (e.key !== 'Tab' || !document.body.dataset.cbCharModalOpen) return;
                const modal = document.getElementById('cbCharEditModal');
                if (!modal || modal.hidden) return;
                const dialog = modal.querySelector('.cb-modal-dialog');
                if (!dialog) return;
                const nodes = dialog.querySelectorAll('button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled])');
                const list = Array.from(nodes).filter(el => {
                    if (el.disabled) return false;
                    const st = window.getComputedStyle(el);
                    return st.display !== 'none' && st.visibility !== 'hidden';
                });
                if (list.length === 0) return;
                const first = list[0];
                const last = list[list.length - 1];
                if (!dialog.contains(document.activeElement)) {
                    first.focus();
                    e.preventDefault();
                    return;
                }
                if (e.shiftKey) {
                    if (document.activeElement === first) {
                        last.focus();
                        e.preventDefault();
                    }
                } else if (document.activeElement === last) {
                    first.focus();
                    e.preventDefault();
                }
            }, true);
        }

        updateChatHeaderTitle();
    }

    window.ChatbotCharacters = {
        getCharacterById,
        buildCharacterSystemBlock,
        syncAfterSessionLoad,
        initCharacterUi
    };
})();
