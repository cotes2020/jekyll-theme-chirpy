/**
 * 위젯 로더 — boot 위젯만 즉시 로드, 나머지는 지연(Toolbox.kickLazyLoad)
 */
(function () {
  if (!window.KARMOLAB_WIDGET_LOADER_WAIT) window.KARMOLAB_WIDGET_LOADER_WAIT = [];

  const base = (function () {
    const s = (document.currentScript || [].slice.call(document.scripts).pop()) as HTMLScriptElement | null;
    if (s && s.src) {
      try {
        const u = new URL(s.src);
        return u.origin + u.pathname.replace(/\/[^/]+$/, '/') + 'widgets/';
      } catch {
        /* noop */
      }
    }
    return (location.origin || '') + '/apps/karmolab/js/widgets/';
  })();

  window.KARMOLAB_WIDGET_SCRIPT_BASE = base;

  window.KARMOLAB_LAZY_META_BY_ID = {};
  const registerDeferred = typeof Toolbox !== 'undefined' ? Toolbox.registerDeferred : undefined;
  if (registerDeferred) {
    (window.KARMOLAB_LAZY_META || []).forEach(function (stub) {
      if (stub && stub.id) window.KARMOLAB_LAZY_META_BY_ID![stub.id] = stub;
      registerDeferred(stub);
    });
  }

  const list = window.KARMOLAB_WIDGETS_BOOT || [];
  let pending = list.length;

  function done() {
    if (--pending === 0) {
      const waits = window.KARMOLAB_WIDGET_LOADER_WAIT || [];
      Promise.allSettled(waits).then(function () {
        Toolbox.initTheme();
        Toolbox.init();
        const lastPage = (function () {
          try {
            return localStorage.getItem('toolbox_last_page');
          } catch {
            return null;
          }
        })();
        const tools = Toolbox.getTools();
        const showHome =
          !lastPage || lastPage === 'home' || !tools.some(function (t) {
            return t.id === lastPage;
          });
        const intro = document.getElementById('introOverlay');
        if (intro && showHome) {
          intro.classList.remove('hidden');
          setTimeout(function () {
            intro.classList.add('done');
            setTimeout(function () {
              intro.classList.add('hidden');
              intro.classList.remove('done');
            }, 320);
          }, 700);
        } else if (intro) {
          intro.classList.add('hidden');
        }
      });
    }
  }

  if (pending === 0) {
    done();
    return;
  }

  list.forEach(function (path) {
    const s = document.createElement('script');
    s.async = false;
    s.src = base + path + '.js';
    s.onload = done;
    s.onerror = done;
    document.body.appendChild(s);
  });
})();
