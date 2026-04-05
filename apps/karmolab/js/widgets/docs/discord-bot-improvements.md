# Discord 봇(욘 / yawnbot) 개선 · 할 일

레포: [`apps/discord-bots/apps/yawnbot/`](https://github.com/mascari4615/mascari4615.github.io/tree/master/apps/discord-bots/apps/yawnbot)

---

## TODO는 어디에 둘까요?

| 방식 | 용도 |
|------|------|
| **이 파일 (`discord-bot-improvements.md`)** | KarmoLab **문서 → Discord·개선** 탭에서 바로 보고, Git으로 이력 관리. `- [ ]` / `- [x]` 만 바꾸면 됨. |
| **GitHub Issues** | 마일스톤·담당·토론이 필요할 때. 이 문서 항목과 이슈 번호를 상호 링크하면 됨. |
| **레포 `docs/TODO.md` 등** | 블로그·전체 레포 할 일과 합치고 싶을 때 (지금은 Discord만 이 페이지에 모음). |

**권장:** 일상은 **이 Markdown 체크리스트**로 진행하고, 큰 덩어리만 Issue로 쪼개기.

---

## 1. `/music play` · 지금 재생 · 대기열 편집

- [x] **지금 재생 표시** — `/music queue` 상단에 **재생 중** 제목 표시 (`currentTrack`)
- [x] **상태 보관** — `GuildMusicState.currentTrack` (재생 성공 시 설정, 대기열 비움·정지 시 해제)
- [ ] **대기열 삭제** — `/music queue remove` 또는 `/remove` 로 번호·범위 삭제
- [ ] **대기열 이동** — 항목 순서 바꾸기 (옵션)
- [ ] **`/shuffle`** — 대기열 무작위 섞기 (재생 중 트랙은 그대로 두고 뒤만)
- [ ] **`/loop`** — 한 곡 / 전체 루프 모드 (플래그 + `playNext` 동작 연동)

---

## 2. 대기열 UX — 긴 플레이리스트 · 응답 정책

- [ ] **긴 플레이리스트 요약** — `/music play` 완료 메시지에 “추가 N곡, 스킵 M곡” 등 embed 한 줄 요약
- [ ] **실패 곡 사용자 안내** — 스트림 실패 시 로그 + 채널/ephemeral 한 줄 (스팸 방지 규칙과 함께)
- [x] **`/music queue` 페이지** — 12곡/페이지, `page` 옵션 + **이전·다음** 버튼
- [x] **응답 정책 (음악 일부)** — `music.ts` 주석 + `/music queue`·`/music skip`·`/music stop` 성공은 채널 공개, 무동작·길드 밖은 ephemeral
- [ ] **DJ 권한 (옵션)** — 역할 기반 `/music skip`·`/music stop` 제한

---

## 3. 음성 연결 — DAVE · 재연결 · 스테이지

- [ ] **DAVE·4017** — **「Discord·음성」** 탭과 README 유지 보수, 환경 변수 설명 최신화
- [ ] **진단** — (선택) `/voice-status` 또는 로그에 마지막 voice close code 요약
- [ ] **재연결 정책** — 끊김 시 1회 재입장 등 상한·백오프 설계 후 `voice-connection`·music 경로에 반영
- [ ] **스테이지** — 발언 요청 흐름·봇 Speak 타이밍·문서화 (실패 시 사용자 메시지)

---

## 4. 슬래시 · 명령 · 배포

- [x] **명령 그룹** — `/music play|speak|sound|skip|stop|queue` (`deploy-commands` + 라우터; 구 `/play` 등 최상위 명령 제거)
- [ ] **한영 병기** — 한글 표시명 유지 + `setDescription` / 옵션 설명에 짧은 영문 보조
- [ ] **`/help` 세분화** — 음성·게임·AI 등 카테고리별 ephemeral (또는 embed 필드)
- [ ] **등록 문서** — 글로벌 vs 길드 배포, `npm run deploy` 절차를 README에 한 블록으로 고정

---

## 5. 데이터 · `/yawn` · AI

- [ ] **`gamedata.json`** — 백업 주기, git 추적 여부, 민감 필드(유저 ID 등) 운영 정책 README에 명시
- [ ] **남용·쿨다운** — 게임/도박/용돈 등 서버별 쿨다운 표시·관리자 예외(필요 시)
- [ ] **`/yawn`** — 컨텍스트/토큰 길이 제한 안내, 비용·레이트 리밋 시 사용자 메시지
- [ ] **거절 톤** — 민감 주제·빈 프롬프트 등 시스템·에러 문구 톤 통일

---

## 6. 운영 · 품질

- [ ] **로그** — `[play]`·`[music]`·`[voice]` 접두 유지 + 심각도·인터랙션 id 일부(추적용) 규칙
- [ ] **테스트** — URL 정규화·플레이리스트 ID 추출·`canonicalYoutubeWatchUrl` 등 단위 테스트 (예: `node:test` 또는 기존 도구에 맞춤)
- [ ] **CI** — `apps/discord-bots/apps/yawnbot` `npm run build` (및 테스트 스크립트 추가 시 함께 실행)
- [ ] **헬스 (선택)** — 업타임·메모리·길드별 음성 연결 수 HTTP `/health` 또는 로그 한 줄

---

## 아이디어 풀 (참고 · 우선순위 낮음)

| 구역 | 메모 |
|------|------|
| 자동 연속 재생 | 관련 영상 등 — YouTube·Discord 정책 확인 후 |
| Spotify 등 외부 URL | ToS·매칭 품질·유지비 |
| `yt-dlp` 포맷 env | 호스트 튜닝용 프리셋 노출 |
| Lavalink·대기열 영속화 | 트래픽·샤딩 커지면 검토 |

---

## 참고 링크

- 워크스페이스: [`apps/discord-bots/README.md`](https://github.com/mascari4615/mascari4615.github.io/blob/master/apps/discord-bots/README.md)
- 봇 README: [`apps/discord-bots/apps/yawnbot/README.md`](https://github.com/mascari4615/mascari4615.github.io/blob/master/apps/discord-bots/apps/yawnbot/README.md)
- 음성·DAVE: **「Discord·음성」** 탭
