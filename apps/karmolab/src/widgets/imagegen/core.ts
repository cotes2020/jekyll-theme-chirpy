// @ts-nocheck
/**
 * imagegen - 큐 시스템, 유틸, 히스토리
 */
(function () {
    'use strict';
    const IG = window.ImageGen;
    if (!IG) return;

    const { VIBE_OPTIONS } = IG;
    const { GALLERY_SESSION_KEY, GALLERY_SESSION_MAX, PROMPT_HISTORY_KEY, PROMPT_HISTORY_MAX } = IG;

    const escapeHtml = Toolbox.escapeHtml;

    function getModelDisplayName(modelId) {
        const all = [
            ...(Gemini.MODELS.gemini || []),
            ...(Gemini.MODELS.geminiImage || []),
            ...(Gemini.MODELS.imagen || [])
        ];
        const found = all.find(m => m.id === modelId);
        return found ? found.name : modelId;
    }

    function showLightbox(imageUrl) {
        let lb = document.getElementById('igLightbox');
        if (!lb) {
            lb = document.createElement('div');
            lb.id = 'igLightbox';
            lb.className = 'ig-lightbox';
            lb.innerHTML = `
                <img id="igLightboxImg" src="" alt="Full Size">
                <div class="ig-lightbox-actions">
                    <button class="btn btn-accent" id="igLightboxDl">⬇️ 다운로드</button>
                    <button class="btn btn-ghost" id="igLightboxClose">닫기</button>
                </div>`;
            lb.onclick = (e) => { if (e.target === lb) lb.classList.remove('open'); };
            document.body.appendChild(lb);
        }
        document.getElementById('igLightboxImg').src = imageUrl;
        document.getElementById('igLightboxDl').onclick = () => IG.downloadImage(imageUrl);
        document.getElementById('igLightboxClose').onclick = () => lb.classList.remove('open');
        lb.classList.add('open');
    }

    function downloadImage(url) {
        const a = document.createElement('a');
        a.href = url;
        a.download = `ai-image-${Date.now()}.png`;
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        Toolbox.showToast('다운로드 시작');
    }

    function getPromptHistory() {
        try { return JSON.parse(localStorage.getItem(PROMPT_HISTORY_KEY)) || []; }
        catch (_) { return []; }
    }

    function addPromptHistory(text) {
        if (!text || text.length < 5) return;
        let history = getPromptHistory().filter(h => h !== text);
        history.unshift(text);
        if (history.length > PROMPT_HISTORY_MAX) history = history.slice(0, PROMPT_HISTORY_MAX);
        localStorage.setItem(PROMPT_HISTORY_KEY, JSON.stringify(history));
    }

    const queue = [];
    let queueProcessing = false;
    let queueIdCounter = 0;
    let _deps = null;

    function collectCurrentOptions() {
        const modelSel = document.getElementById('igModelSelect');
        const ratioSel = document.getElementById('igAspectRatio');
        const safetySel = document.getElementById('igSafety');
        const vibeSel = document.getElementById('igVibe');
        const apiSel = document.getElementById('igApiRoute');
        const vibeId = vibeSel?.value || 'none';
        const vibeSuffix = (VIBE_OPTIONS.find(v => v.id === vibeId) || VIBE_OPTIONS[0]).suffix;
        const modelId = modelSel?.value || Gemini.getDefaultModel('geminiImage');
        const opts = {
            apiRoute: apiSel?.value || Toolbox.getPref('ig_api_route') || 'aiStudio',
            modelId,
            aspectRatio: ratioSel?.value || '16:9',
            safetyThreshold: safetySel?.value || 'BLOCK_ONLY_HIGH',
            vibeSuffix,
            vertexProjectId: (document.getElementById('igVertexProjectId')?.value || Toolbox.getPref('ig_vertex_project_id') || '').trim(),
            vertexLocation: (document.getElementById('igVertexLocation')?.value || Toolbox.getPref('ig_vertex_location') || 'us-central1').trim() || 'us-central1'
        };
        if (modelId.startsWith('imagen')) {
            opts.negativePrompt = document.getElementById('igNegPrompt')?.value.trim() || '';
            opts.personGeneration = document.getElementById('igPersonGen')?.value || 'allow_adult';
        }
        return opts;
    }

    function enqueue(promptText, isEmoji, emojiChar, isMascot) {
        const opts = collectCurrentOptions();
        if (isEmoji || isMascot) opts.aspectRatio = '1:1';
        const id = ++queueIdCounter;
        queue.push({
            id, prompt: promptText, finalPrompt: promptText + opts.vibeSuffix, options: opts,
            status: 'pending', abortController: null, elapsed: null, error: null,
            emojiChar: isEmoji ? emojiChar : undefined
        });
        if (_deps) _deps.renderQueue();
        processQueue();
    }

    async function processQueue() {
        if (queueProcessing || !_deps) return;
        const next = queue.find(q => q.status === 'pending');
        if (!next) return;

        queueProcessing = true;
        next.status = 'running';
        next.abortController = new AbortController();
        _deps.renderQueue();
        _deps.updateMainPreview();

        const start = Date.now();
        const timerId = setInterval(() => {
            next.elapsed = ((Date.now() - start) / 1000).toFixed(0);
            _deps.renderQueueItem(next);
            const lt = document.getElementById('igLoadingText');
            if (lt) lt.textContent = `Dreaming... ${next.elapsed}s`;
        }, 1000);

        try {
            let imageUrl, usage = null;
            const { modelId, aspectRatio, safetyThreshold } = next.options;
            const signal = next.abortController.signal;

            if (modelId.startsWith('gemini')) {
                const route = next.options.apiRoute || 'aiStudio';
                if (route === 'vertex') {
                    if (!Gemini.requireVertexApiKey()) throw new Error('Vertex API 키가 필요합니다.');
                    const result = await Gemini.callVertexGeminiImage(next.finalPrompt, modelId, {
                        signal,
                        aspectRatio,
                        safetyThreshold,
                        projectId: next.options.vertexProjectId,
                        location: next.options.vertexLocation
                    });
                    imageUrl = result.dataUrl;
                    usage = result.usage;
                } else {
                    const result = await Gemini.callGeminiImage(next.finalPrompt, modelId, { signal, aspectRatio, safetyThreshold });
                    imageUrl = result.dataUrl;
                    usage = result.usage;
                }
            } else {
                const imgSafety = safetyThreshold === 'OFF' ? 'block_none' : safetyThreshold.toLowerCase();
                const route = next.options.apiRoute || 'aiStudio';
                if (route === 'vertex') {
                    if (!Gemini.requireVertexApiKey()) throw new Error('Vertex API 키가 필요합니다.');
                    const images = await Gemini.callVertexImagen(next.finalPrompt, modelId, 1, {
                        signal,
                        aspectRatio,
                        safetyFilterLevel: imgSafety,
                        negativePrompt: next.options.negativePrompt || undefined,
                        personGeneration: next.options.personGeneration,
                        projectId: next.options.vertexProjectId,
                        location: next.options.vertexLocation
                    });
                    imageUrl = images[0];
                } else {
                    const images = await Gemini.callImagen(next.finalPrompt, modelId, 1, {
                        signal, aspectRatio, safetyFilterLevel: imgSafety,
                        negativePrompt: next.options.negativePrompt || undefined,
                        personGeneration: next.options.personGeneration
                    });
                    imageUrl = images[0];
                }
            }

            const elapsed = ((Date.now() - start) / 1000).toFixed(1);
            const tokens = usage?.totalTokenCount || null;
            const item = {
                id: `img_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
                url: imageUrl, prompt: next.prompt, model: modelId,
                modelName: getModelDisplayName(modelId), timestamp: Date.now(), tokens, elapsed
            };

            _deps.state.sessionGallery.push(item);
            _deps.state.currentItem = item;
            next.status = 'done';
            next.resultItem = item;

            _deps.showResultInPreview(item);
            _deps.renderSessionGallery();
            _deps.saveGallerySession();
            ImageDB.save(item).catch(e => console.warn('Library save failed:', e));
            addPromptHistory(next.emojiChar || next.prompt);
            Toolbox.recordUsage('image', tokens || 0);

            if (_deps.state.compareMode) {
                _deps.state.compareMode = false;
                const cmp = document.getElementById('igPreview')?.querySelector('.ig-compare');
                if (cmp) cmp.style.display = 'none';
            }
            const compareBtn = document.getElementById('igCompareBtn');
            if (compareBtn && _deps.state.sessionGallery.length >= 2) compareBtn.style.display = '';
        } catch (e) {
            next.status = 'error';
            next.error = e.message || '생성 실패';
            if (e.message !== '요청이 취소되었습니다.') {
                const is429 = /429|Too Many/i.test(e.message || '');
                if (is429) {
                    clearQueue();
                    Toolbox.showToast('API 요청 한도 초과. 큐를 비웠습니다. 잠시 후 다시 시도해주세요.', 'error');
                } else {
                    Toolbox.showToast(e.message || '이미지 생성 실패', 'error', e);
                }
            }
        } finally {
            clearInterval(timerId);
            queueProcessing = false;
            _deps.renderQueue();

            const remaining = queue.filter(q => q.status === 'pending').length;
            const doneCount = queue.filter(q => q.status === 'done').length;
            if (remaining > 0) {
                Mdd.linePreset('tool_run', { msg: `다음 이미지 시작! (${remaining}개 남음)` });
                processQueue();
            } else if (doneCount > 0) {
                Mdd.linePreset('success', { msg: '큐 작업 모두 완료!' });
                Toolbox.showToast(`큐 완료: ${doneCount}장 생성됨`);
                _deps.hideMainLoading();
            }
        }
    }

    function cancelQueueItem(queueId) {
        const item = queue.find(q => q.id === queueId);
        if (!item) return;
        if (item.status === 'running' && item.abortController) {
            item.abortController.abort();
            item.status = 'cancelled';
        } else if (item.status === 'pending') {
            item.status = 'cancelled';
        }
        if (_deps) _deps.renderQueue();
    }

    function removeQueueItem(queueId) {
        const idx = queue.findIndex(q => q.id === queueId);
        if (idx >= 0) queue.splice(idx, 1);
        if (_deps) _deps.renderQueue();
    }

    function clearQueue() {
        queue.forEach(q => {
            if (q.status === 'running' && q.abortController) q.abortController.abort();
            if (q.status === 'pending') q.status = 'cancelled';
        });
        queue.length = 0;
        if (_deps) { _deps.renderQueue(); _deps.hideMainLoading(); }
    }

    function saveGallerySession() {
        if (!_deps) return;
        try {
            const { sessionGallery, currentItem } = _deps.state;
            const toSave = sessionGallery.slice(-GALLERY_SESSION_MAX);
            sessionStorage.setItem(GALLERY_SESSION_KEY, JSON.stringify({
                items: toSave,
                currentId: currentItem?.id
            }));
        } catch (e) { console.warn('Gallery session save failed', e); }
    }

    function loadGallerySession() {
        if (!_deps) return false;
        try {
            const raw = sessionStorage.getItem(GALLERY_SESSION_KEY);
            if (!raw) return false;
            const data = JSON.parse(raw);
            const { state } = _deps;
            if (data.items && Array.isArray(data.items) && data.items.length > 0) {
                state.sessionGallery = data.items;
                state.currentItem = state.sessionGallery.find(i => i.id === data.currentId) || state.sessionGallery[state.sessionGallery.length - 1];
                return true;
            }
            if (data.urls && Array.isArray(data.urls) && data.urls.length > 0) {
                state.sessionGallery = data.urls.map((url, i) => ({
                    id: `legacy_${i}`, url, prompt: '', model: '', modelName: '',
                    timestamp: Date.now(), tokens: null, elapsed: null
                }));
                state.currentItem = state.sessionGallery[state.sessionGallery.length - 1];
                return true;
            }
        } catch (e) { console.warn('Gallery session load failed', e); }
        return false;
    }

    function initCore(deps) {
        _deps = deps;
    }

    Object.assign(IG, {
        queue,
        queueProcessing: () => queueProcessing,
        getModelDisplayName,
        showLightbox,
        downloadImage,
        getPromptHistory,
        addPromptHistory,
        collectCurrentOptions,
        enqueue,
        processQueue,
        cancelQueueItem,
        removeQueueItem,
        clearQueue,
        saveGallerySession,
        loadGallerySession,
        initCore,
        escapeHtml
    });
})();
