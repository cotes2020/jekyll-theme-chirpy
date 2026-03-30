// @ts-nocheck
/**
 * wiki/entities/characters/*.md 를 fetch → 파싱 → KarmoWorld.entities + bindings 채움
 * imagegen/chatbot 스크립트보다 먼저 실행되며, Toolbox 지연 로더가 Promise 완료까지 대기합니다.
 */
(function () {
    const SLUGS = ['yon', 'alisa', 'ling'];

    function worldBaseUrl() {
        const s = document.currentScript;
        if (s && s.src) {
            try {
                const u = new URL(s.src);
                return u.origin + u.pathname.replace(/\/[^/]+$/, '/');
            } catch (_) {}
        }
        return (location.origin || '') + '/apps/karmolab/world/';
    }

    function wikiUrl(base, slug) {
        return base + 'wiki/entities/characters/' + slug + '.md';
    }

    function metaToImagegen(m) {
        return {
            entityId: m.entityId || '',
            imagegenPresetId: m.imagegen_presetId || '',
            icon: m.imagegen_icon || '📷',
            label: m.imagegen_label || '',
            shortLabel: m.imagegen_shortLabel || '',
            prompt: m.imagegen_prompt || ''
        };
    }

    function metaToChatbot(m) {
        return {
            entityId: m.entityId || '',
            chatbotId: m.chatbot_id || '',
            name: m.chatbot_name || '',
            userName: m.chatbot_userName || '사용자',
            userNote: m.chatbot_userNote || '',
            visualDescription: m.chatbot_visualDescription || '',
            description: m.chatbot_description || '',
            personality: m.chatbot_personality || '',
            scenario: m.chatbot_scenario || '',
            firstMes: m.chatbot_firstMes || ''
        };
    }

    function metaToEntity(slug, m) {
        const tags = Array.isArray(m.tags) ? m.tags : typeof m.tags === 'string' ? [m.tags] : [];
        return {
            id: m.entityId || '',
            slug: m.slug || slug,
            nameKo: m.nameKo || '',
            nameEn: m.nameEn || '',
            aliases: Array.isArray(m.aliases) ? m.aliases : [],
            oneLine: m.oneLine || '',
            tags
        };
    }

    async function loadAll() {
        const parse = window.KarmoWorld?.parseMd?.parseCharacterWikiMarkdown;
        if (typeof parse !== 'function') throw new Error('KarmoWorld.parseMd.parseCharacterWikiMarkdown 없음');

        const base = worldBaseUrl();
        const imagegen = [];
        const chatbot = [];
        const characters = {};

        for (const slug of SLUGS) {
            const url = wikiUrl(base, slug);
            const r = await fetch(url);
            if (!r.ok) throw new Error('wiki 로드 실패: ' + url);
            const text = await r.text();
            const { meta } = parse(text);
            if (!meta.entityId) throw new Error('entityId 없음: ' + slug);
            characters[slug] = metaToEntity(slug, meta);
            imagegen.push(metaToImagegen(meta));
            chatbot.push(metaToChatbot(meta));
        }

        window.KarmoWorld = window.KarmoWorld || {};
        window.KarmoWorld.entities = window.KarmoWorld.entities || {};
        window.KarmoWorld.entities.characters = characters;

        window.KarmoWorld.bindings = window.KarmoWorld.bindings || {};
        window.KarmoWorld.bindings.imagegen = window.KarmoWorld.bindings.imagegen || {};
        window.KarmoWorld.bindings.chatbot = window.KarmoWorld.bindings.chatbot || {};
        window.KarmoWorld.bindings.imagegen.characters = imagegen;
        window.KarmoWorld.bindings.chatbot.characters = chatbot;
    }

    window.KARMOLAB_WIDGET_LOADER_WAIT = window.KARMOLAB_WIDGET_LOADER_WAIT || [];
    const p = loadAll().catch(err => {
        try { console.error('[KarmoWorld] wiki 캐릭터 로드 실패', err); } catch (_) {}
        try {
            if (typeof Toolbox !== 'undefined' && Toolbox.showToast) {
                Toolbox.showToast('세계관 캐릭터 데이터 로드 실패', 'error');
            }
        } catch (_) {}
    });
    window.KARMOLAB_WIDGET_LOADER_WAIT.push(p);
})();
