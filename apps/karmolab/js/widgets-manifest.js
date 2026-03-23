/**
 * 위젯 매니페스트 — 포함할 위젯 목록
 * 위젯 추가/제거 시 이 배열만 수정하면 됨
 *
 * 경로 규칙: widgets/{path}.js
 *   - 단일 파일: 'crypto' → widgets/crypto.js
 *   - 하위 폴더: 'docs/docs' → widgets/docs/docs.js
 *
 * 정렬 기준: 카테고리 → 용도별 → id 가나다순
 */
window.KARMOLAB_WIDGETS = [
    // ── 기능 (feature): 랜딩 → 유틸 → AI → 미디어 → 문서 → 사용자
    'favorites',
    'linktree/linktree',
    'crypto', 'memo', // 'morse', 'password',
    'chatbot/chatbot',
    'imagegen/presets', 'imagegen/config', 'imagegen/styles', 'imagegen/core', 'imagegen/imagegen',
    'imageedit', 'imagelib',
    // 'youtubedl',
    'docs/docs',

    'user',
    'dashboard',

    // ── 도구 (tool): 시간 → 정리 → 생성 → 시스템
    // 'countdown', 'hourglass', 'moon',
    'randomgen/randomgen',// 'tierlist',
    'servermonitor',

    // ── 미니게임 (play): id 가나다순
    'conch', 'fortune', 'gacha',
    // 'bounce', 'bubble', 'darkroom', 'eyes', 'folder', 'font',
    // 'hacker', 'news', 'particle', 'pet', 'reaction',
    // 'shylink', 'speed', 'stone', 'toast',
];
