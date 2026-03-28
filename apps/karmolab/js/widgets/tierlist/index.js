(function () {
    const T = window.Tierlist = window.Tierlist || {};

    T.injectStyles?.();
    T.state.loadState();

    /** publish.js 캐시 구버전이어도 동작하도록 index에서 직접 부트스트랩 */
    (async function bootstrapDefaultEmbedded() {
        const pub = T.publish;
        if (!pub || typeof pub.getPublishedIndex !== 'function' || typeof pub.openPublishedDirect !== 'function') return;
        const st = T.state.getState();
        if (T.state.isPublishedMode()) return;
        if (st.currentListId && st.lists[st.currentListId]) return;
        let items;
        try { items = await pub.getPublishedIndex(); } catch (_) { return; }
        if (!items.length) return;
        const first = items[0];
        if (!first?.url) return;
        try {
            await pub.openPublishedDirect(first.url, { id: first.id, title: first.title, url: first.url });
        } catch (_) { /* 네트워크/JSON 오류 시 무시 */ }
    })();

    Toolbox.register({
        ...Toolbox.getLazyWidgetPublicMeta('tierlist'),
        tabs: [
            {
                id: 'tl-edit',
                label: '편집',
                build(container) {
                    T.render.setContainers({ editor: container });
                    T.render.renderEditor();
                }
            },
            {
                id: 'tl-list',
                label: '목록',
                build(container) {
                    T.render.setContainers({ list: container });
                    T.render.renderListTab();
                }
            },
            {
                id: 'tl-stats',
                label: '통계',
                build(container) {
                    T.render.setContainers({ stats: container });
                    T.render.renderStats();
                }
            },
        ]
    });
})();

