// @ts-nocheck
import { DEFAULT_VERTEX_LOCATION } from 'karmolab-ai';
import { chatbotUiSurfaceToPackage, getChatbotApiSurfaceUi } from './api-surface';

/** 스트리밍 표시용 KARMO_IMAGE 태그 제거·파싱·캐릭터 이미지 생성 */
(function () {
    const KARMO_IMAGE_RE = /\[\[KARMO_IMAGE:(\{[\s\S]*?\})\]\]/;

    function displayTextForStream(s) {
        const start = s.indexOf('[[KARMO_IMAGE:');
        if (start === -1) return s;
        const tail = s.slice(start);
        const m = tail.match(/^\[\[KARMO_IMAGE:\{[\s\S]*?\}\]\]/);
        if (m) return (s.slice(0, start) + s.slice(start + m[0].length)).trimEnd();
        return s.slice(0, start).trimEnd();
    }

    function extractKarmoImage(text) {
        const m = text.match(KARMO_IMAGE_RE);
        if (!m) return { cleanText: text.trim(), spec: null };
        try {
            const spec = JSON.parse(m[1]);
            const cleanText = text.replace(KARMO_IMAGE_RE, '').trim();
            return { cleanText, spec };
        } catch (_) {
            return { cleanText: text.replace(KARMO_IMAGE_RE, '').trim(), spec: null };
        }
    }

    async function appendCharacterImageAfterMessage(wrap, char, spec) {
        if (!wrap?.parentNode || !spec?.show || !spec.prompt) return;
        const loading = document.createElement('div');
        loading.className = 'cb-msg cb-msg-bot cb-msg-image cb-msg-image-loading';
        loading.textContent = '🖼 이미지 생성 중…';
        wrap.parentNode.insertBefore(loading, wrap.nextSibling);

        const sceneEn = String(spec.prompt).slice(0, 800);
        const vis = char?.visualDescription ? String(char.visualDescription).slice(0, 600) : '';
        const fullPrompt = [
            'High quality illustration, single clear subject, no text in image.',
            vis ? `Character appearance (keep consistent): ${vis}` : '',
            `Scene and mood: ${sceneEn}`
        ].filter(Boolean).join('\n');

        const imgModelSel = document.getElementById('cbCharImageModel');
        const imgModel = imgModelSel?.value || (typeof Gemini !== 'undefined' && Gemini.getDefaultModel ? Gemini.getDefaultModel('geminiImage') : 'gemini-2.5-flash-image');

        const ref = char?.referenceImageDataUrl;
        const opt = {};
        if (ref && ref.startsWith('data:')) opt.referenceImage = ref;

        try {
            let res;
            if (chatbotUiSurfaceToPackage(getChatbotApiSurfaceUi()) === 'vertex') {
                if (!Gemini.requireVertexApiKey()) {
                    loading.className = 'cb-msg cb-msg-bot cb-msg-error cb-msg-image';
                    loading.textContent = '이미지 생성: Vertex API 키가 설정되지 않았습니다.';
                    return;
                }
                const projectId = (Toolbox.getPref('ig_vertex_project_id') || '').trim();
                if (!projectId) {
                    loading.className = 'cb-msg cb-msg-bot cb-msg-error cb-msg-image';
                    loading.textContent = '이미지 생성: Vertex 사용 시 설정에 GCP 프로젝트 ID가 필요합니다.';
                    Toolbox.showToast('Vertex: 프로젝트 ID를 설정하세요.', 'error');
                    return;
                }
                const locationRaw = (Toolbox.getPref('ig_vertex_location') || '').trim();
                const location = locationRaw || DEFAULT_VERTEX_LOCATION;
                res = await Gemini.callVertexGeminiImage(fullPrompt, imgModel, {
                    ...opt,
                    projectId,
                    location,
                });
            } else {
                res = await Gemini.callGeminiImage(fullPrompt, imgModel, opt);
            }
            loading.remove();
            const box = document.createElement('div');
            box.className = 'cb-msg cb-msg-bot cb-msg-image';
            const img = document.createElement('img');
            img.src = res.dataUrl;
            img.alt = '캐릭터 이미지';
            box.appendChild(img);
            wrap.parentNode.insertBefore(box, wrap.nextSibling);
            const msgs = document.getElementById('cbMessages');
            if (msgs) msgs.scrollTop = msgs.scrollHeight;
            if (res.usage?.totalTokenCount) Toolbox.recordUsage('image', res.usage.totalTokenCount);
        } catch (e) {
            loading.className = 'cb-msg cb-msg-bot cb-msg-error cb-msg-image';
            loading.textContent = '이미지 생성 실패: ' + (e.message || e);
            Toolbox.showToast(e.message || '이미지 오류', 'error', e);
        }
    }

    window.ChatbotKarmoImage = {
        KARMO_IMAGE_RE,
        displayTextForStream,
        extractKarmoImage,
        appendCharacterImageAfterMessage
    };
})();
