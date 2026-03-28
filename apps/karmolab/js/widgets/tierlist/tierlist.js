(function () {
    // tierlist 위젯 엔트리포인트(매니페스트용 단일 파일)
    // 이 파일은 widgets/tierlist/ 폴더 안에 있으므로, baseUrl은 "현재 파일의 폴더"면 충분합니다.

    function baseUrl() {
        var s = document.currentScript || [].slice.call(document.scripts).pop();
        if (s && s.src) {
            try {
                var u = new URL(s.src);
                // .../widgets/tierlist/tierlist.js → .../widgets/tierlist/
                return u.origin + u.pathname.replace(/\/[^/]+$/, '/');
            } catch (_) {}
        }
        return (location.origin || '') + '/apps/karmolab/js/widgets/tierlist/';
    }

    function loadSeq(urls) {
        return urls.reduce(function (p, url) {
            return p.then(function () {
                return new Promise(function (res, rej) {
                    var s = document.createElement('script');
                    s.src = url;
                    s.onload = res;
                    s.onerror = function () { rej(new Error('failed to load: ' + url)); };
                    document.body.appendChild(s);
                });
            });
        }, Promise.resolve());
    }

    var base = baseUrl();
    var files = [
        'namespace.js',
        'styles.js',
        'storage.js',
        'ui.js',
        'dnd.js',
        'publish.js',
        'dialogs.js',
        'render.js',
        'index.js',
    ].map(function (f) { return base + f; });

    var p = loadSeq(files);

    // widgets-loader가 init 전에 기다릴 수 있도록 Promise를 공유
    try {
        window.KARMOLAB_WIDGET_LOADER_WAIT = window.KARMOLAB_WIDGET_LOADER_WAIT || [];
        window.KARMOLAB_WIDGET_LOADER_WAIT.push(p);
    } catch (_) {}

    p.catch(function (err) {
        try { Toolbox.showToast('tierlist 로드 실패', 'error', err); } catch (_) {}
        console.error(err);
    });
})();

