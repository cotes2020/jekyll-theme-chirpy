// @ts-nocheck
(function () {
    function baseUrl() {
        var s = document.currentScript || [].slice.call(document.scripts).pop();
        if (s && s.src) {
            try {
                var u = new URL(s.src);
                return u.origin + u.pathname.replace(/\/[^/]+$/, '/');
            } catch (_) {}
        }
        return (location.origin || '') + '/apps/karmolab/js/widgets/imageconvert/';
    }

    function loadSeq(urls) {
        return urls.reduce(function (p, url) {
            return p.then(function () {
                return new Promise(function (res, rej) {
                    var el = document.createElement('script');
                    el.src = url;
                    el.onload = res;
                    el.onerror = function () {
                        rej(new Error('failed to load: ' + url));
                    };
                    document.body.appendChild(el);
                });
            });
        }, Promise.resolve());
    }

    var base = baseUrl();
    var p = loadSeq([base + 'core.js', base + 'batch-pipeline.js', base + 'widget.js']);

    try {
        window.KARMOLAB_WIDGET_LOADER_WAIT = window.KARMOLAB_WIDGET_LOADER_WAIT || [];
        window.KARMOLAB_WIDGET_LOADER_WAIT.push(p);
    } catch (_) {}

    p.catch(function (err) {
        try {
            Toolbox.showToast('이미지 변환 로드 실패', 'error', err);
        } catch (_) {}
        console.error(err);
    });
})();
