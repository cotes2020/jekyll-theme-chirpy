/**
 * 랜덤 생성기 — 주제 데이터 로드
 *
 * 단순 주제( items 기반)는 topics.json에서 로드.
 * generator 기반 주제는 randomgen-number.js, randomgen-time.js, randomgen-color.js, randomgen-name.js에서 추가.
 *
 * 참고: [니힐 랜덤 키워드](https://nihilapp.github.io/keyword) / [nihilapp/random-keyword-code](https://github.com/nihilapp/random-keyword-code)
 *       창작자용 랜덤 키워드 사이트를 참고하여 주제·키워드를 보강했습니다. (MIT License)
 *
 * 새 주제 추가:
 * - 단순: topics.json에 { id, label, group, items: ["a","b",...] } 추가
 * - 커스텀: 각 모듈에서 topics.push({ id, label, group, generator: () => string | { name, sub } })
 */
(function () {
    var base = (function () {
        var s = document.currentScript;
        if (s && s.src) {
            try {
                return s.src.replace(/[^/]+$/, '');
            } catch (_) {}
        }
        return (location.origin || '') + '/apps/karmolab/js/widgets/randomgen/';
    })();

    var data = [];
    try {
        var xhr = new XMLHttpRequest();
        xhr.open('GET', base + 'topics.json', false);
        xhr.send(null);
        if (xhr.status === 200) {
            data = JSON.parse(xhr.responseText);
        }
    } catch (e) {
        console.warn('randomgen: topics.json 로드 실패', e);
    }

    window.RANDOMGEN_TOPICS = data;
})();
