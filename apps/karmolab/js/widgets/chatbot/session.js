/** Chatbot 세션 관리 */
window.Chatbot = window.Chatbot || {};
window.Chatbot.Session = (function () {
    var INDEX_KEY = 'toolbox_chatbot_sessions_index';
    var PREFIX = 'toolbox_chatbot_session_';
    var MAX = 10;

    function genId() { return 's_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6); }

    function getIndex() {
        try { return JSON.parse(sessionStorage.getItem(INDEX_KEY)) || []; }
        catch (_) { return []; }
    }

    function saveIndex(idx) {
        sessionStorage.setItem(INDEX_KEY, JSON.stringify(idx));
    }

    function create(name) {
        var id = genId();
        var idx = getIndex();
        idx.push({ id: id, name: name || ('대화 ' + (idx.length + 1)), createdAt: Date.now() });
        if (idx.length > MAX) {
            var r = idx.shift();
            sessionStorage.removeItem(PREFIX + r.id);
        }
        saveIndex(idx);
        return id;
    }

    function remove(id) {
        saveIndex(getIndex().filter(function (s) { return s.id !== id; }));
        sessionStorage.removeItem(PREFIX + id);
    }

    function save(id, data) {
        if (!id) return;
        try {
            sessionStorage.setItem(PREFIX + id, JSON.stringify(Object.assign({}, data, { savedAt: Date.now() })));
        } catch (e) { console.warn('Chatbot session save failed', e); }
    }

    function load(id) {
        try {
            var raw = sessionStorage.getItem(PREFIX + id);
            if (!raw) return null;
            var d = JSON.parse(raw);
            return d.chatHistory && Array.isArray(d.chatHistory) ? d : null;
        } catch (e) { console.warn('Chatbot session load failed', e); return null; }
    }

    return {
        getSessionsIndex: getIndex,
        saveSessionsIndex: saveIndex,
        createNewSession: create,
        deleteSession: remove,
        saveSession: function (id, data) { save(id, data); },
        loadSession: load,
        MAX_SESSIONS: MAX,
        PREFIX: PREFIX
    };
})();
