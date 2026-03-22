/**
 * 랜덤 생성기 — 이름 (generator 전용)
 * randomgen-topics.js 로드 후 RANDOMGEN_TOPICS에 추가됨
 */
(function () {
    function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

    var topics = window.RANDOMGEN_TOPICS;
    if (!Array.isArray(topics)) return;

    var family = ['김', '이', '박', '최', '정', '강', '조', '윤', '장', '임', '한', '오', '서', '신', '권', '황', '안', '송', '류', '전', '홍', '문', '배', '백', '남궁', '제갈'];
    var given = ['현', '지', '민', '수', '하', '윤', '서', '준', '도', '은', '시', '우', '진', '율', '아', '연', '채', '영', '성', '태', '혁', '빈', '유', '린', '솔', '단', '별', '온', '해', '달'];

    topics.push({
        id: 'kr',
        label: '한국 이름',
        group: '이름',
        generator: function () { return pick(family) + pick(given) + pick(given); }
    });
})();
