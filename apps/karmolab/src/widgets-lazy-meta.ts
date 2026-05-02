/**
 * 지연 로드 위젯 공통 메타 (단일 출처)
 * - 지연 등록 stub + 각 위젯 Toolbox.register 시 ...getLazyWidgetPublicMeta(id) 로 재사용
 * - lazyScriptPaths: 로더가 순서대로 불러올 스크립트 경로(widgets/ 기준, .js 제외)
 */
import type { KarmoLabLazyWidgetStub } from '../types/karmolab';

window.KARMOLAB_LAZY_META = [
  {
    id: 'crypto',
    title: '암호화 / 복호화',
    category: 'tool',
    desc: '텍스트를 AES, Base64, URL 인코딩으로 암호화·복호화합니다',
    layout: 'form',
    icon: '<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/>',
    lazyScriptPaths: ['crypto']
  },
  {
    id: 'memo',
    title: '메모장',
    category: 'tool',
    desc: '로컬 메모를 저장하고 관리합니다',
    layout: 'full',
    icon: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline>',
    lazyScriptPaths: ['memo']
  },
  {
    id: 'chatbot',
    title: '챗봇',
    category: 'tool',
    desc: 'AI와 대화합니다',
    layout: 'full',
    icon: '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>',
    /** load order matters; see widgets/chatbot/README.md */
    lazyScriptPaths: [
      'world/world',
      'world/parse-md',
      'world/load-characters-from-wiki',
      'chatbot/styles',
      'chatbot/markdown',
      'chatbot/characters',
      'chatbot/karmo-image',
      'chatbot/prompt',
      'chatbot/chatbot'
    ]
  },
  {
    id: 'imagegen',
    title: '이미지 생성',
    category: 'tool',
    desc: 'AI로 이미지를 생성합니다',
    layout: 'full',
    icon: '<circle cx="12" cy="12" r="10"/><line x1="14.31" y1="8" x2="20.05" y2="17.94"/><line x1="9.69" y1="8" x2="21.17" y2="8"/><line x1="7.38" y1="12" x2="13.12" y2="2.06"/><line x1="9.69" y1="16" x2="3.95" y2="6.06"/><line x1="14.31" y1="16" x2="2.83" y2="16"/><line x1="16.62" y1="12" x2="10.88" y2="21.94"/>',
    lazyScriptPaths: [
      'world/world',
      'world/parse-md',
      'world/load-characters-from-wiki',
      'imagegen/presets',
      'imagegen/config',
      'imagegen/styles',
      'imagegen/core',
      'imagegen/imagegen'
    ]
  },
  {
    id: 'worldwiki',
    title: '세계관 위키',
    category: 'lab',
    desc: '세계관(캐릭터·아티팩트) 문서를 앱 안에서 봅니다 (개발 중)',
    layout: 'full',
    icon: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    lazyScriptPaths: [
      'world/world',
      'world/parse-md',
      'world/load-characters-from-wiki',
      'chatbot/markdown',
      'worldwiki/worldwiki'
    ]
  },
  {
    id: 'imageedit',
    title: '이미지 편집',
    category: 'tool',
    desc: '편집·형식·해상도 변환(PNG·JPEG·WebP 등)을 한 화면에서',
    layout: 'full',
    icon: '<rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M9 3v18" stroke="currentColor" stroke-width="1.5"/><path d="M3 15h18" stroke="currentColor" stroke-width="1.5"/><circle cx="15" cy="9" r="2" stroke="currentColor" stroke-width="1.5" fill="none"/>',
    lazyScriptPaths: ['imageconvert/imageconvert', 'imageedit']
  },
  {
    id: 'imagelib',
    title: '이미지 라이브러리',
    category: 'tool',
    desc: '생성한 이미지를 저장하고 관리합니다',
    layout: 'full',
    icon: '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>',
    lazyScriptPaths: ['imagelib']
  },
  {
    id: 'tierlist',
    title: '티어리스트',
    category: 'lab',
    desc: '후보 풀(주제별 요소)에서 순위 인스턴스를 만들고, 블로그·로컬 JSON으로 주고받기 (개발 중)',
    layout: 'form',
    icon: '<path d="M3 3h18v4H3zM3 9h14v4H3zM3 15h10v4H3z"/>',
    lazyScriptPaths: ['tierlist/tierlist']
  },
  {
    id: 'postgraph',
    title: '글 그래프',
    category: 'lab',
    desc: '블로그 포스트 간 내부 링크 관계를 그래프로 봅니다 (개발 중)',
    layout: 'full',
    icon: '<circle cx="8" cy="8" r="3" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="16" cy="16" r="3" fill="none" stroke="currentColor" stroke-width="1.5"/><line x1="10.2" y1="10.2" x2="13.8" y2="13.8" stroke="currentColor" stroke-width="1.5"/>',
    lazyScriptPaths: ['postgraph']
  },
  {
    id: 'conch',
    title: '소라고동',
    category: 'play',
    desc: '소라고동에게 질문합니다',
    layout: 'form',
    icon: '<path d="M12 2A10 10 0 0 0 2 12a10 10 0 0 0 10 10 10 10 0 0 0 10-10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 0-16 8 8 0 0 1 0 16z M12 6c-3.31 0-6 2.69-6 6 M12 8c-2.21 0-4 1.79-4 4" stroke="currentColor" stroke-width="1.5" fill="none"/>',
    lazyScriptPaths: ['conch']
  },
  {
    id: 'planner',
    title: '플래너',
    category: 'lab',
    desc: '나만의 일정 동기화 및 스트릭 칸반 보드 (개발 중)',
    layout: 'full',
    icon: '<rect x="3" y="4" width="18" height="18" rx="2" ry="2" fill="none" stroke="currentColor" stroke-width="2"/><line x1="16" y1="2" x2="16" y2="6" stroke="currentColor" stroke-width="2"/><line x1="8" y1="2" x2="8" y2="6" stroke="currentColor" stroke-width="2"/><line x1="3" y1="10" x2="21" y2="10" stroke="currentColor" stroke-width="2"/>',
    lazyScriptPaths: ['planner/planner']
  },
  {
    id: 'quest-log',
    title: 'Quest Log',
    category: 'tool',
    desc: '관측실 — 프로젝트·인생 항목 트리, 진행도, 봉인',
    layout: 'full',
    noHero: true,
    icon: '<path d="M12 2l2.9 6.95 7.6.6-5.75 4.95L18.4 22 12 17.9 5.6 22l1.65-7.5L1.5 9.55l7.6-.6z" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linejoin="round"/>',
    lazyScriptPaths: ['quest-log/quest-log']
  },
  {
    id: 'karmoddrine-dashboard',
    title: 'karmoddrine 대시보드',
    category: 'desktop',
    desc: 'umbrella 활성 세션 / commit / 도구 / 룰 단일 출처를 카드 + 그래프로 (10s 폴링)',
    layout: 'full',
    noHero: true,
    icon: '<rect x="3" y="3" width="7" height="9" rx="1" stroke="currentColor" stroke-width="1.5" fill="none"/><rect x="14" y="3" width="7" height="5" rx="1" stroke="currentColor" stroke-width="1.5" fill="none"/><rect x="14" y="12" width="7" height="9" rx="1" stroke="currentColor" stroke-width="1.5" fill="none"/><rect x="3" y="16" width="7" height="5" rx="1" stroke="currentColor" stroke-width="1.5" fill="none"/>',
    lazyScriptPaths: ['karmoddrine-dashboard/karmoddrine-dashboard']
  }
] as KarmoLabLazyWidgetStub[];
