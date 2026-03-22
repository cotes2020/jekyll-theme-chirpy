/**
 * 위젯 로더 — 매니페스트 기반 동적 로드
 * 모든 위젯 로드 후 Toolbox.init() 실행
 */
(function () {
    var base = (function () {
        var s = document.currentScript || [].slice.call(document.scripts).pop();
        if (s && s.src) {
            try {
                var u = new URL(s.src);
                return u.origin + u.pathname.replace(/\/[^/]+$/, '/') + 'widgets/';
            } catch (_) {}
        }
        return (location.origin || '') + '/apps/karmolab/js/widgets/';
    })();
    var list = window.KARMOLAB_WIDGETS || [];
    var pending = list.length;

    function done() {
        if (--pending === 0) {
            Toolbox.initTheme();
            Toolbox.init();
            var lastPage = (function () {
                try { return localStorage.getItem('toolbox_last_page'); } catch (_) { return null; }
            })();
            var tools = Toolbox.getTools();
            var showHome = !lastPage || lastPage === 'home' || !tools.some(function (t) { return t.id === lastPage; });
            var intro = document.getElementById('introOverlay');
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
        }
    }

    if (pending === 0) {
        done();
        return;
    }

    list.forEach(function (path) {
        var s = document.createElement('script');
        s.src = base + path + '.js';
        s.onload = done;
        s.onerror = done;
        document.body.appendChild(s);
    });
})();
