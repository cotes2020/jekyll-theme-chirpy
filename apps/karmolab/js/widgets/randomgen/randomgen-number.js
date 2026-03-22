/**
 * 랜덤 생성기 — 숫자·나이 (generator 전용)
 * randomgen-topics.js 로드 후 RANDOMGEN_TOPICS에 추가됨
 */
(function () {
    var topics = window.RANDOMGEN_TOPICS;
    if (!Array.isArray(topics)) return;

    topics.push(
        { id: 'number_1_10', label: '숫자 (1~10)', group: '기본', generator: function () { return String(Math.floor(Math.random() * 10) + 1); } },
        { id: 'number_1_100', label: '숫자 (1~100)', group: '기본', generator: function () { return String(Math.floor(Math.random() * 100) + 1); } },
        { id: 'age', label: '나이', group: '캐릭터', generator: function () { return String(Math.floor(Math.random() * 100) + 1) + '세'; } }
    );
})();
