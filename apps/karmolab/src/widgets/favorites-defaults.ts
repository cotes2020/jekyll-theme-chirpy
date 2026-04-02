/**
 * 기본 즐겨찾기 데이터 — URL·그룹만 편집하면 됨 (위젯 로직은 favorites.ts)
 */
export type FavoriteItem = {
    url?: string;
    label: string;
    icon?: string | null;
    type?: 'tool';
    toolId?: string;
    isCustom?: boolean;
};

export type FavoriteGroup = {
    group: string;
    items: FavoriteItem[];
};

/** Google favicon API가 404만 줄 때 수동 지정 (favorites.ts의 폴백과 동일 문자열) */
export const FAVICON_FALLBACK = 'data:image/svg+xml,' + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#666">' +
    '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>' +
    '</svg>'
);

export const DEFAULT_ITEMS: FavoriteGroup[] = [
    { group: '개발', items: [
        { url: 'https://github.com', label: 'GitHub', icon: 'https://cdn.simpleicons.org/github' },
        { url: 'https://www.postman.com/', label: 'Postman', icon: 'https://cdn.simpleicons.org/postman' },
        { url: 'https://discord.com/developers/applications/', label: 'Discord Developer', icon: 'https://cdn.simpleicons.org/discord' },
        { url: 'https://my.vultr.com', label: 'Vultr', icon: 'https://cdn.simpleicons.org/vultr' },
        { url: 'https://solved.ac/class?class=5', label: 'solved.ac CLASS', icon: null },
        { url: 'https://solved.ac', label: 'solved.ac', icon: null },
        { url: 'https://assetstore.unity.com/ko-KR/publisher-sale', label: 'Unity Asset Store - Publisher Sale', icon: null },
        { url: 'http://127.0.0.1:4000/', label: 'Local', icon: null },
        { url: 'http://localhost:8899/apps/karmolab/index.html', label: 'Karmolab', icon: null },
        { url: 'https://wrchat.github.io/Woodon/', label: 'WRChat VCC Listing', icon: FAVICON_FALLBACK },
        { url: 'https://wrchat.github.io/Woodon/index.json', label: 'WRChat index.json', icon: FAVICON_FALLBACK },
        { url: 'https://status.vrchat.com/', label: 'VRChat Status', icon: 'https://cdn.simpleicons.org/vrchat' },
    ]},
    { group: '채용·커리어', items: [
        { url: 'https://blog.maplestory.nexon.com/Employment', label: '메이플 채용', icon: 'https://cdn.simpleicons.org/nexon' },
        { url: 'https://maplecareer.stibee.com', label: '메이플 커리어 레터', icon: FAVICON_FALLBACK },
        { url: 'https://careers.nexon.com', label: '넥슨 채용', icon: 'https://cdn.simpleicons.org/nexon' },
        { url: 'https://www.nexon-tutorial.com', label: '넥토리얼', icon: 'https://cdn.simpleicons.org/nexon' },
        { url: 'https://www.gamejob.co.kr/User/resumemng/portfolio', label: '게임잡 포트폴리오', icon: null },
        { url: 'https://inditor.co.kr', label: '인디터웹', icon: null },
    ]},
    { group: '메이플', items: [
        { url: 'https://blog.maplestory.nexon.com/Tech', label: '메이플 테크 블로그', icon: 'https://cdn.simpleicons.org/nexon' },
        { url: 'https://blog.maplestory.nexon.com/Tech/Content/10', label: '메이플 테크 (1)', icon: 'https://cdn.simpleicons.org/nexon' },
        { url: 'https://blog.maplestory.nexon.com/Tech/Content/2', label: '메이플 테크 (2)', icon: 'https://cdn.simpleicons.org/nexon' },
        { url: 'https://maplescouter.com/ko', label: '환산주스탯', icon: null },
    ]},
    { group: '검색·AI', items: [
        { url: 'https://www.google.com', label: 'Google', icon: 'https://cdn.simpleicons.org/google' },
        { url: 'https://www.naver.com/', label: 'Naver', icon: null },
        { url: 'https://feedly.com/i/my', label: 'Feedly', icon: null },
        { url: 'https://chat.openai.com', label: 'ChatGPT', icon: null },
        { url: 'https://claude.ai', label: 'Claude', icon: 'https://cdn.simpleicons.org/anthropic' },
        { url: 'https://gemini.google.com', label: 'Gemini', icon: 'https://cdn.simpleicons.org/google' },
        { url: 'https://aistudio.google.com', label: 'AI Studio', icon: 'https://cdn.simpleicons.org/google' },
        { url: 'https://notebooklm.google.com/', label: 'NotebookLM', icon: 'https://cdn.simpleicons.org/google' },
        { url: 'https://cursor.com/dashboard/spending', label: 'Cursor Spending', icon: 'https://cdn.simpleicons.org/cursor' },
    ]},
    { group: 'AI 아트', items: [
        { url: 'https://pixai.art/', label: 'PixAI', icon: null },
        { url: 'https://tensor.art/', label: 'Tensor.art', icon: null },
        { url: 'https://novelai.net/', label: 'NovelAI', icon: null },
        { url: 'https://www.seaart.ai/', label: 'SeaArt', icon: null },
    ]},
    { group: '소셜·미디어', items: [
        { url: 'https://www.netflix.com', label: '넷플릭스', icon: 'https://cdn.simpleicons.org/netflix' },
        { url: 'https://laftel.net', label: '라프텔', icon: null },
        { url: 'https://www.youtube.com', label: 'YouTube', icon: 'https://cdn.simpleicons.org/youtube' },
        { url: 'https://music.youtube.com', label: 'YouTube Music', icon: 'https://cdn.simpleicons.org/youtube' },
        { url: 'https://kr.pinterest.com', label: 'Pinterest', icon: 'https://cdn.simpleicons.org/pinterest' },
        { url: 'https://chzzk.naver.com', label: '치지직', icon: null },
        { url: 'https://sooplive.co.kr', label: '숲 (SOOP)', icon: null },
        { url: 'https://www.twitch.tv', label: 'Twitch', icon: 'https://cdn.simpleicons.org/twitch' },
        { url: 'https://x.com', label: 'X (Twitter)', icon: 'https://cdn.simpleicons.org/x' },
        { url: 'https://www.reddit.com', label: 'Reddit', icon: 'https://cdn.simpleicons.org/reddit' },
        { url: 'https://discord.com', label: 'Discord', icon: 'https://cdn.simpleicons.org/discord' },
        { url: 'https://lolesports.com/', label: 'LoL Esports', icon: null },
    ]},
    { group: '서로이웃', items: [
        { url: 'https://orbit3230.github.io', label: 'orbit3230', icon: 'https://orbit3230.github.io/favicon.ico' },
    ]},
    { group: '짝이웃', items: [
        { url: 'https://hyngng.github.io', label: 'HYNGNG', icon: null },
        { url: 'https://blog.naver.com/tigermon', label: '윤농 - 윤농의 작업실', icon: FAVICON_FALLBACK },
        { url: 'https://shoark7.github.io/', label: 'Parkito - Faster, Faster', icon: null },
        { url: 'https://blog.naver.com/blancleo/', label: '블랑레오', icon: FAVICON_FALLBACK },
        { url: 'https://blog.naver.com/hugspa', label: '그대만을(hugspa) - 득행하자!', icon: FAVICON_FALLBACK },
    ]},
    { group: '도구', items: [
        { url: 'https://www.dhlottery.co.kr/main', label: '동행복권', icon: null },
        { url: 'https://www.notion.so', label: 'Notion', icon: 'https://cdn.simpleicons.org/notion' },
        { url: 'https://figma.com', label: 'Figma', icon: 'https://cdn.simpleicons.org/figma' },
        { url: 'https://excalidraw.com', label: 'Excalidraw', icon: 'https://cdn.simpleicons.org/excalidraw' },
    ]},
    { group: '이끔', items: [
        { url: '/posts/year-2025/', label: '2025', icon: null },
        { url: '/posts/music-playlist/', label: '플레이리스트', icon: null },
        { url: '/posts/strategy/', label: '전략', icon: null },
        { url: '/posts/strategy-tech/', label: '전략: 테크', icon: null },
        { url: '/posts/strategy-tech-career/', label: '전략: 테크: 커리어', icon: null },
        { url: '/posts/advice/', label: '조언', icon: null },
        { url: '/posts/cold/', label: '찬 바람', icon: null },
        { url: '/posts/ps-algorithm/', label: '코딩테스트', icon: null },
    ]},
    { group: '계속', items: [
        { url: 'https://brunch.co.kr/@dangkunlove/21', label: '시시콜콜한 이야기의 위로', icon: null },
        { url: 'https://brunch.co.kr/@064040503a2242a/42', label: '내 작업물에 대한 공격은 나에 대한 공격으로 간주한다', icon: null },
        { url: 'https://blog.naver.com/jysa000/223676533324', label: '나는 어떤 경험을 하고 싶을까?', icon: FAVICON_FALLBACK },
        { url: 'https://brunch.co.kr/@whizzer4/79', label: '우리 엄마도 한때는 소녀였으니까', icon: null },
    ]},
];
