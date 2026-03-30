// @ts-nocheck
/* global Toolbox, Mdd */
(function () {
  let lastDestroy = null;

  function karmoPalette() {
    const theme = document.documentElement.getAttribute('data-theme');
    const dark = theme !== 'light';
    return {
      link: dark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.16)',
      linkDim: dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
      linkHi: dark ? 'rgba(147,197,253,0.95)' : 'rgba(37,99,235,0.85)',
      text: dark ? '#e5e7eb' : '#1f2937',
      nodeStroke: dark ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.2)',
      nodeFillDefault: dark ? '#6366f1' : '#4f46e5'
    };
  }

  const PostGraph = {
    build(container) {
      if (typeof lastDestroy === 'function') {
        try {
          lastDestroy();
        } catch {
          /* ignore teardown errors */
        }
        lastDestroy = null;
      }

      container.innerHTML = '';
      const wrap = document.createElement('div');
      wrap.className = 'postgraph-wrap';
      wrap.style.cssText =
        'width:100%;min-height:min(70vh,640px);height:70vh;border:1px solid var(--border);border-radius:var(--radius-lg);background:var(--bg-tertiary);overflow:hidden;';
      container.appendChild(wrap);

      const origin = location.origin || '';
      const dataUrl = new URL('/assets/js/data/post-graph.json', origin || 'http://localhost').href;
      const moduleUrl = new URL('/assets/js/graph-view/graph-view.js', origin || 'http://localhost').href;

      (async () => {
        try {
          const { createGraphView } = await import(moduleUrl);
          const api = await createGraphView({
            container: wrap,
            dataUrl,
            getPalette: karmoPalette,
            onNodeOpen(node) {
              if (node.href) {
                window.open(new URL(node.href, origin || 'http://localhost').href, '_blank', 'noopener,noreferrer');
              }
            }
          });
          lastDestroy = typeof api.destroy === 'function' ? api.destroy.bind(api) : null;
        } catch (e) {
          console.error(e);
          wrap.textContent = '그래프를 불러오지 못했습니다. (배포된 사이트에서 /assets/js/data/post-graph.json 확인)';
        }
      })();

      if (typeof Mdd !== 'undefined') {
        Mdd.linePreset('tool_run', { mood: 'idle', msg: '포스트 링크 관계예요. 노드를 누르면 글이 새 탭에서 열려요.' });
      }
    }
  };

  Toolbox.register({
    ...Toolbox.getLazyWidgetPublicMeta('postgraph'),
    tabs: [{ id: 'graph', label: '그래프', build: PostGraph.build }]
  });
})();
