# KarmoLab — Agent 할 일 목록

다른 에이전트/기여자가 바로 이어서 작업할 수 있도록 **구체 파일 경로**와 **완료 조건**을 적었습니다.  
제품 전반 기획은 [`../js/widgets/docs/roadmap.md`](../js/widgets/docs/roadmap.md)를 참고하세요.

---

## 빠른 맥락 (2026 기준)

| 영역 | 단일 출처 / 핵심 파일 |
|------|----------------------|
| 세계관 캐릭터 | [`world/wiki/entities/characters/*.md`](../world/wiki/entities/characters/) — YAML frontmatter + 본문 |
| 파싱·로드 | [`world/parse-md.js`](../world/parse-md.js), [`world/load-characters-from-wiki.js`](../world/load-characters-from-wiki.js) |
| 위키 UI | [`js/widgets/worldwiki/worldwiki.js`](../js/widgets/worldwiki/worldwiki.js) |
| 지연 로드 메타 | [`js/widgets-lazy-meta.js`](../js/widgets-lazy-meta.js) |
| 사이드바 그룹 | `lab` = **실험실 · 개발중** — [`js/toolbox.js`](../js/toolbox.js) 의 `CATEGORIES` |

---

## 이미 반영된 것 (중복 작업 방지)

- [x] 캐릭터: 위키 MD frontmatter → `KarmoWorld.entities` / `bindings.imagegen` / `bindings.chatbot`
- [x] `imagegen/presets.js`: 위키 바인딩은 **preset id로만 덮어쓰기** (`timeto` 등 유지)
- [x] 사이드바: 티어리스트·플래너·글 그래프·세계관 위키 → **`lab`** 그룹
- [x] 글 그래프 `postgraph`: `category: 'lab'` ([`widgets-lazy-meta.js`](../js/widgets-lazy-meta.js))

---

## 우선순위 높음

### 세계관 위키 (World Wiki)

- [ ] **목록 소스**: 캐릭터 슬ugs가 [`load-characters-from-wiki.js`](../world/load-characters-from-wiki.js)의 `SLUGS`에 하드코딩됨 → `wiki/`에 `index.json` 또는 frontmatter 스캔으로 **확장 시 한 곳만 수정**되게.
- [ ] **위키 링크**: 본문에서 `[[slug]]` 또는 내부 링크 규칙 → 클릭 시 같은 위젯 내 문서 전환.
- [ ] **검색**: 제목·본문·별칭(aliases) 기준 클라이언트 검색(또는 사전 빌드 인덱스 JSON).
- [ ] **아티팩트**: `world/wiki/artifacts/` + frontmatter 스키마 초안 + 위키 목록에 kind 필터(이미 UI 여지 있음).

### KarmoWorld 데이터

- [ ] **티메토(`timeto`)**: 이미지 프리셋에는 있으나 위키 MD 단일 출처 없음 → 캐릭터와 동일 패턴으로 `.md` 추가 여부 결정 후 `SLUGS`/로더 반영.
- [ ] **`parse-md.js`**: 필요 시 nested YAML(`imagegen:` 블록 등) 지원 — 키가 늘어날 때 중복 접두사 정리.

### 실험실 위젯 (`lab`)

- [ ] **플래너 / 티어리스트 / 글 그래프**: 각 README 또는 이 문서에 **알려진 이슈·제약**(CORS, `file://`, 데이터 경로) 명시.
- [ ] **글 그래프**: `/assets/js/data/post-graph.json` 생성 파이프라인이 레포에 있으면 링크; 없으면 “배포 시 수동 갱신” 등 한 줄이라도 [`postgraph.js`](../js/widgets/postgraph.js) 상단 주석에.

---

## 우선순위 중간

- [ ] 위키 → 이미지 생성 / 챗봇 **딥링크** 또는 “이 캐릭터로 열기” 액션(쿼리스트링 또는 `localStorage` 한 번 쓰기).
- [ ] `file://` 로컬 오프라인: fetch 실패 시 사용자 메시지 개선(이미 [`world/README.md`](../world/README.md)에 메모 있음).
- [ ] 접근성: 위키 TOC·헤딩 앵커 키보드/스크린리더 점검.

---

## 우선순위 낮음 · 조사만

- [ ] 위키 본문에 Mermaid/수식 — 번들 크기·보안 정책 확인 후.
- [ ] CI: `parse-md` 단위 테스트(Node에서 frontmatter 샘플 파싱) — 선택.

---

## 작업 시 주의

- **`widgets-lazy-meta.js`**에서 `world/*` 스크립트 순서 바꾸지 말 것: `load-characters-from-wiki`가 이미지/챗봇/위키보다 먼저 완료되어야 함.
- 커밋 시 `_posts/` 등 블로그 초안은 **사용자가 명시하지 않으면 포함하지 않기** (저장소 정책).

---

## 완료 체크 (PR 전)

1. KarmoLab을 **HTTP로** 연 뒤(또는 GitHub Pages) 챗봇·이미지 생성·세계관 위키에서 캐릭터 데이터가 일치하는지 확인.
2. 사이드바에서 **실험실 · 개발중** 그룹에 해당 위젯이 모두 있는지 확인.
