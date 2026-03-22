/**
 * 랜덤 생성기 — 색상 hex 코드 (generator 전용)
 * randomgen-topics.js 로드 후 RANDOMGEN_TOPICS에 추가됨
 */
(function () {
    var topics = window.RANDOMGEN_TOPICS;
    if (!Array.isArray(topics)) return;

    function randHex() {
        return '#' + Array.from({ length: 6 }, function () {
            return '0123456789abcdef'[Math.floor(Math.random() * 16)];
        }).join('');
    }

    topics.push({
        id: 'color_hex',
        label: '색 (hex)',
        group: '자연',
        generator: randHex
    });
})();
