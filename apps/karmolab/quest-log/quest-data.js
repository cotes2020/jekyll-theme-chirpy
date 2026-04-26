// Quest Board data.
// Structure: Projects → Areas → Tasks (tasks can nest infinitely).
// Progress is auto-calculated from checklist completion on leaf nodes.
// Status: seed 🌱 / fire 🔥 / sleep 💤 / sealed 🏆
//
// Every node has:
//   id, title, note?, status, children?, checks?
// A node with `checks` is a leaf; its progress = checked/total.
// A node with `children` rolls up the average progress of its children.

window.QUEST_DATA = {
  projects: [
    {
      id: 'wm',
      title: 'WitchMendokusai',
      subtitle: '메인 프로젝트 · 주황머리 마녀와 인형들',
      kind: 'main',
      icon: '🔮',
      children: [
        {
          id: 'wm-foundation',
          title: '기반 시스템',
          note: '3D 공간, 카메라, 렌더링 파이프라인',
          children: [
            { id: 'wm-f-scene', title: '씬 구조', status: 'fire',
              checks: [
                { t: '레이어 분리 정리', done: true },
                { t: '카메라 릭 베이스', done: true },
                { t: '라이팅 프리셋 3종', done: true },
                { t: '포스트프로세스 스택', done: false },
                { t: '성능 프로파일링', done: false },
              ]},
            { id: 'wm-f-input', title: '인풋 시스템', status: 'seed',
              checks: [
                { t: '게임패드 매핑', done: false },
                { t: '액션 바인딩 스키마', done: false },
                { t: '입력 버퍼링', done: false },
              ]},
          ],
        },
        {
          id: 'wm-mining',
          title: '채광',
          note: '광맥 탐색, 도구, 드롭',
          children: [
            { id: 'wm-m-system', title: '드롭 테이블 설계', status: 'fire',
              checks: [
                { t: '광석 종류 10개 정의', done: true },
                { t: '희귀도 곡선', done: true },
                { t: '도구별 보정치', done: false },
              ]},
            { id: 'wm-m-art', title: '광석 스프라이트', status: 'seed',
              checks: [
                { t: '기본 4종 스프라이트', done: false },
                { t: '반짝임 이펙트', done: false },
              ]},
            { id: 'wm-m-feedback', title: '채굴 피드백', status: 'sleep',
              checks: [
                { t: '타격감 사운드', done: false },
                { t: '파티클', done: false },
                { t: '진동/쉐이크', done: false },
              ]},
          ],
        },
        {
          id: 'wm-combat',
          title: '전투',
          note: '로그라이크? 실시간? — 실험 중',
          children: [
            { id: 'wm-c-explore', title: '전투 방향 결정', status: 'fire',
              checks: [
                { t: '덱빌딩 프로토', done: true },
                { t: '실시간 액션 프로토', done: false },
                { t: '방향성 확정', done: false },
              ]},
            { id: 'wm-c-enemies', title: '적 설계', status: 'seed',
              checks: [
                { t: '기본 적 3종', done: false },
                { t: '보스 1종', done: false },
              ]},
          ],
        },
        {
          id: 'wm-farming',
          title: '농사',
          note: '성장 단계, 계절, 씨앗',
          children: [
            { id: 'wm-fa-growth', title: '작물 성장 로직', status: 'seed',
              checks: [
                { t: '단계 정의', done: false },
                { t: '시간/날씨 연동', done: false },
                { t: '수확 로직', done: false },
              ]},
            { id: 'wm-fa-seeds', title: '씨앗 종류', status: 'seed',
              checks: [
                { t: '기본 씨앗 5종', done: false },
                { t: '희귀 씨앗 3종', done: false },
              ]},
          ],
        },
        {
          id: 'wm-village',
          title: '마을 경영',
          note: 'NPC 배치, 건물, 자원 순환',
          children: [
            { id: 'wm-v-build', title: '건축 시스템', status: 'seed',
              checks: [
                { t: '그리드 vs 자유 배치 결정', done: false },
                { t: '기본 건물 3종', done: false },
                { t: '설치/철거 UI', done: false },
              ]},
            { id: 'wm-v-npc', title: 'NPC 기본 루틴', status: 'seed',
              checks: [
                { t: '스케줄 시스템', done: false },
                { t: 'NPC 3명 프로토', done: false },
              ]},
          ],
        },
        {
          id: 'wm-dolls',
          title: '인형 부리기',
          note: '핵심 판타지 — AI 루틴, 지시',
          children: [
            { id: 'wm-d-ai', title: '인형 AI 베이스', status: 'seed',
              checks: [
                { t: '상태 머신', done: false },
                { t: '지시 큐', done: false },
                { t: '우선순위', done: false },
              ]},
            { id: 'wm-d-command', title: '지시 UI', status: 'seed',
              checks: [
                { t: '컨텍스트 메뉴', done: false },
                { t: '다중 선택', done: false },
              ]},
          ],
        },
        {
          id: 'wm-fishing',
          title: '낚시 · 미니게임',
          note: '휴식용 루프',
          children: [
            { id: 'wm-fi-basic', title: '낚시 기본 루프', status: 'sleep',
              checks: [
                { t: '던지기/걸기', done: false },
                { t: '물고기 5종', done: false },
                { t: '미끼 시스템', done: false },
              ]},
          ],
        },
        {
          id: 'wm-collect',
          title: '수집 · 도감',
          children: [
            { id: 'wm-co-book', title: '도감 UI', status: 'seed',
              checks: [
                { t: '카테고리 구조', done: false },
                { t: '미수집 플레이스홀더', done: false },
              ]},
          ],
        },
        {
          id: 'wm-story',
          title: '스토리 · 톤',
          note: '주황머리 마녀, 인형, 방황',
          children: [
            { id: 'wm-s-bible', title: '세계관 바이블', status: 'fire',
              checks: [
                { t: '주인공 배경', done: true },
                { t: '인형들의 기원', done: true },
                { t: '지역 3곳 설정', done: false },
                { t: '주요 NPC 5명', done: false },
                { t: '엔딩 방향', done: false },
              ]},
            { id: 'wm-s-tone', title: '톤 레퍼런스', status: 'fire',
              checks: [
                { t: '비주얼 무드보드', done: true },
                { t: '사운드 레퍼런스', done: false },
                { t: '대사 톤 샘플', done: false },
              ]},
          ],
        },
        {
          id: 'wm-identity',
          title: '게임 정체성',
          note: '만들고 싶은 게임이 뭔지 결정하기',
          children: [
            { id: 'wm-i-pitch', title: '한 줄 피치', status: 'seed',
              checks: [
                { t: '5가지 버전 쓰기', done: false },
                { t: '주변에 테스트', done: false },
                { t: '하나로 수렴', done: false },
              ]},
          ],
        },
      ],
    },

    // ───────── 개인 성장 프로젝트 ─────────
    {
      id: 'blog',
      title: '블로그',
      subtitle: '쓰고 공유하기',
      kind: 'growth',
      icon: '✒️',
      children: [
        { id: 'blog-setup', title: '블로그 세팅', status: 'fire',
          checks: [
            { t: '도메인 연결', done: true },
            { t: '테마 커스텀', done: true },
            { t: 'RSS 정리', done: false },
            { t: 'About 페이지', done: false },
          ]},
        { id: 'blog-write', title: '글 쓰기 습관',
          children: [
            { id: 'blog-w-dev', title: '개발 일지', status: 'fire',
              checks: [
                { t: 'WM 개발 회고 1편', done: true },
                { t: 'WM 개발 회고 2편', done: false },
                { t: '셰이더 실험 노트', done: false },
              ]},
            { id: 'blog-w-essay', title: '에세이', status: 'sleep',
              checks: [
                { t: '주제 3개 후보', done: true },
                { t: '초안 작성', done: false },
              ]},
          ]},
      ],
    },
    {
      id: 'learn',
      title: '공부',
      subtitle: '손에 익히는 중',
      kind: 'growth',
      icon: '📖',
      children: [
        { id: 'learn-rust', title: '러스트',
          children: [
            { id: 'learn-r-book', title: 'The Rust Book', status: 'fire',
              checks: [
                { t: 'Ch.1 시작하기', done: true },
                { t: 'Ch.2 추측 게임', done: true },
                { t: 'Ch.3 일반 개념', done: true },
                { t: 'Ch.4 소유권', done: true },
                { t: 'Ch.5 구조체', done: false },
                { t: 'Ch.6 열거형', done: false },
                { t: 'Ch.7~', done: false },
              ]},
            { id: 'learn-r-mini', title: '토이 프로젝트', status: 'seed',
              checks: [
                { t: 'CLI 계산기', done: false },
                { t: '작은 웹서버', done: false },
              ]},
          ]},
        { id: 'learn-shader', title: '셰이더', status: 'seed',
          checks: [
            { t: 'Book of Shaders Ch.1~5', done: false },
            { t: 'WM에 활용 가능한 것 1개', done: false },
          ]},
        { id: 'learn-game-design', title: '게임 디자인 읽기', status: 'sleep',
          checks: [
            { t: 'The Art of Game Design', done: false },
            { t: 'A Theory of Fun', done: false },
          ]},
      ],
    },
    {
      id: 'travel',
      title: '여행',
      subtitle: '가고 싶은 곳들',
      kind: 'growth',
      icon: '🧭',
      children: [
        { id: 'travel-jp', title: '일본 재방문', status: 'fire',
          checks: [
            { t: '항공권 알림 설정', done: true },
            { t: '숙소 후보 정리', done: false },
            { t: '루트 초안', done: false },
            { t: '예약', done: false },
          ]},
        { id: 'travel-iceland', title: '아이슬란드', status: 'seed',
          checks: [
            { t: '시즌 조사', done: false },
            { t: '예산 계산', done: false },
          ]},
        { id: 'travel-local', title: '국내 근거리', status: 'seed',
          checks: [
            { t: '강릉', done: false },
            { t: '통영', done: false },
            { t: '제주 한달살이', done: false },
          ]},
      ],
    },
    {
      id: 'body',
      title: '몸 돌보기',
      subtitle: '지속가능한 개발자 몸',
      kind: 'growth',
      icon: '🌿',
      children: [
        { id: 'body-habit', title: '운동 습관', status: 'fire',
          checks: [
            { t: '주 2회 유지', done: true },
            { t: '주 3회로 올리기', done: false },
            { t: '루틴 안정화', done: false },
          ]},
        { id: 'body-posture', title: '자세/거북목 교정', status: 'seed',
          checks: [
            { t: '모니터 높이', done: false },
            { t: '스트레칭 루틴', done: false },
          ]},
      ],
    },
  ],

  // Sealed (trophy room) — completed meaningful chunks
  sealed: [
    { id: 's-1', title: 'WM 초기 프로토타입', project: 'WitchMendokusai', note: '움직이고 때리는 30초', sealedNote: '방향성 확인' },
    { id: 's-2', title: 'WM 메인 캐릭터 컨셉 확정', project: 'WitchMendokusai', note: '주황머리, 인형 동반', sealedNote: '흔들리지 않는 코어' },
    { id: 's-3', title: '블로그 1호 글', project: '블로그', note: '첫 발행', sealedNote: '시작이 절반' },
    { id: 's-4', title: '러스트 소유권 이해', project: '공부', note: 'Ch.4 완주', sealedNote: '고비 넘김' },
    { id: 's-5', title: 'WM 로고 드래프트', project: 'WitchMendokusai', note: '3가지 방향 → 1개 수렴', sealedNote: '' },
    { id: 's-6', title: '운동 주 2회 3개월', project: '몸 돌보기', note: '습관 정착', sealedNote: '이제 늘릴 차례' },
  ],
};
