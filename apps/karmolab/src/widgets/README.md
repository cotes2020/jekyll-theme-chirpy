# KarmoLab 위젯 작성 가이드

> 새 위젯을 만들기 전 / 기존 위젯을 수정할 때 반드시 본다. 같은 일을 두 번 구현하지 않기 위해.

## 시작 전 체크리스트

새 위젯·기능을 *코드로 옮기기 전* 다음 순서로 정독한다. 정독 결과는 TASK 문서 「관련 파일 / 읽기 (참고 패턴)」 에 명시 — *어떤 파일을 보고 어떤 결론* 인지.

- [ ] **같은 카테고리·layout 위젯 1~2개 정독** — `widgets-lazy-meta.ts` 에서 같은 `layout` (`'full'` / `'form'`) 또는 `category` (`'desktop'` / `'lab'` / `'tool'` / `'play'`) 위젯 골라 *컨테이너 / 디자인 토큰 / 폴링·라이프사이클 / 외부 lib 처리* 패턴 확인.
- [ ] **공통 helper 모듈 검토** — `Toolbox`, `chatbot/markdown.ts` (마크다운→HTML), `world/parse-md.ts` (frontmatter), `widgets/docs/docs.ts` (마크다운+mermaid+Prism+동일 출처 lib), `karmolab-ai` 패키지. 같은 기능 거의 다 *이미 있음*. 새로 만들기 전 grep.
- [ ] **외부 lib 동일 출처** — mermaid·marked·prism 등은 `assets/lib/<lib>/<lib>.min.js` 에 동일 출처로 박혀있음 (Tauri webview 의 Tracking Prevention 회피). CDN (`cdn.jsdelivr.net` 등) 우선 안 쓴다.
- [ ] **글로벌 디자인 토큰 사용** — 자체 색·폰트·spacing 박지 말 것. CSS 변수: `--bg-primary` / `--bg-secondary` / `--bg-tertiary` / `--text-primary` / `--text-tertiary` / `--accent` / `--border` / `--border-color` / `--radius-sm` / `--radius-md` / `--font-mono`. 다른 위젯이 쓰는 패턴 그대로.
- [ ] **자체 CSS injection 최소화** — `injectStyles()` 패턴은 *위젯 한정 클래스* 만. 컨테이너 / 카드 / 버튼 / 표 / 코드블록은 글로벌 스타일에 맡긴다. 다른 위젯과 외관 톤 어긋나면 사용자가 *이질감* 느낀다.

체크 안 하고 자체 구현하면 *사용자 부정적 경험* (재구현, 디자인 일관성 깨짐, 유지보수 부담 증가) — 룰 단일 출처: `memo/CLAUDE-karmoddrine.md` § 새 기능·위젯·모듈 — 기존 정독 우선.

## 결정 — 신규 vs 흡수 vs helper 재사용

정독 후 셋 중 하나:

| 결과 | 의미 |
| --- | --- |
| **기존 흡수** | 같은 일 하는 위젯이 있다 → 그 위젯에 카테고리·모드·탭 추가. 별도 위젯 X. |
| **helper 재사용** | 일부 변환·유틸 (마크다운, mermaid, frontmatter, escapeHtml 등) 만 공유 → 공용 모듈로 빼고 우리 위젯이 호출. |
| **진짜 신규** | (a)·(b) 다 안 맞는 *진짜* 새 도메인 → 그 *진짜 신규임* 의 근거 1줄 TASK 에 명시. |

의심되면 사용자에게 묻기.

## 위젯 등록 흐름 — 신규일 때만

진짜 신규 결정 후:

1. `apps/karmolab/src/widgets/<slug>/<slug>.ts` (또는 단일 파일 `<slug>.ts`) — IIFE + `Toolbox.register({ ...Toolbox.getLazyWidgetPublicMeta(slug), tabs: [...] })`.
2. `apps/karmolab/src/widgets-lazy-meta.ts` — 메타 entry 추가 (id / title / category / desc / layout / icon / lazyScriptPaths).
3. `apps/karmolab/build.mjs` — `entryPoints` 배열에 `src/widgets/<slug>/<slug>.ts` 추가.
4. Tauri 명령이 필요하면 `apps/karmolab-tauri/src-tauri/src/<feature>.rs` (신규) + `lib.rs` mod 등록 + `permissions/<feature>.toml` (신규) + `capabilities/default.json` allow.

## 외부 lib 동일 출처 패턴 (mermaid 예)

`widgets/docs/docs.ts` 가 정본 패턴:
- `assets/lib/mermaid/mermaid.min.js` → `<script src=...>` 동적 삽입
- `widgets-lazy-meta.ts` 의 `widget script base` URL 로 상대 경로 해결
- *fallback* — 그것도 없으면 `location.origin + '/assets/lib/...'`

CDN 직접 import 는 *Tauri webview Tracking Prevention* 에서 사용자에 따라 차단됨. 동일 출처 우선.

## CSS 토큰 reference

다른 위젯의 실제 사용 예:
- `widgets/docs/docs.ts` — `.docs-body .mermaid` 가 `var(--bg-tertiary)`, `var(--border)`, `var(--radius-md)` 사용.
- `widgets/karmoddrine-dashboard/karmoddrine-dashboard.ts` — `.kd-card` 가 `var(--accent, #d4a849)` (fallback), `var(--text-primary, #e8e8e8)` 사용.
- `widgets/quest-log/quest-log.ts` — 같은 패턴.

자체 hex (`#d4a849`, `#e8e8e8` 등) 박지 말고 토큰 + fallback hex 패턴 따름.

## 관련

- 룰 정본: `memo/CLAUDE-karmoddrine.md` § 새 기능·위젯·모듈 — 기존 정독 우선
- TASK 작성: `memo/TASK-SCHEMA.md` 본문 포맷 (특히 「목표」 첫 줄 사용자 원본 발화 인용 + 「관련 파일 / 읽기」 필수)
- 옵션 제안: `memo/CLAUDE-karmoddrine.md` § 옵션 제안 시 추천 + 근본 수정 평가 명시
