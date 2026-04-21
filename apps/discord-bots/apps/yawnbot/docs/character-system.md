# AI 비서 & 캐릭터 시스템

YawnBot의 AI 비서(DM/전용 채널 자유 대화)와 캐릭터 카드 시스템 사용 설명서.
설계 문서는 [`memo/CLAUDE-karmoddrine.md`](https://github.com/mascari4615/memo/blob/master/CLAUDE-karmoddrine.md), [`memo/assistant/design.md`](https://github.com/mascari4615/memo/blob/master/assistant/design.md) 참조.

---

## 개요

봇은 페르소나를 **하드코딩하지 않는다**. 실행 시 `memo/characters/<slug>/card.md` 본문을 시스템 프롬프트로 주입한다. 같은 봇 프로세스·토큰으로 **DM/채널 단위**로 다른 캐릭터를 붙일 수 있다.

기억은 **캐릭터별로 격리** — 티메토의 대화 로그와 욘의 대화 로그는 서로 보지 않는다.

---

## 데이터 위치 (memo 레포)

```
memo/characters/
├── .active.json              # { default: slug, channels: { <channelKey>: slug } }
├── <slug>/
│   ├── card.md               # frontmatter + 시스템 프롬프트 본문 (사람이 편집)
│   ├── appearance.md         # 외형·이미지 프롬프트 (사람이 편집)
│   ├── reference/            # (선택) 이미지 레퍼런스
│   └── memory/               # 봇 자동 관리 — 직접 편집 금지
│       ├── user.md
│       ├── self.md
│       ├── logs/YYYY-MM-DD.md
│       ├── daily/YYYY-MM-DD.md
│       └── weekly/YYYY-WNN.md
└── ...
```

`channelKey`: DM은 `dm:<userId>`, 그 외는 Discord 채널 ID.

---

## 슬래시 명령어

### `/character` — 캐릭터 관리

| 명령 | 설명 |
|------|------|
| `/character list` | 등록된 캐릭터 목록 + 이 DM/채널의 활성 캐릭터 |
| `/character switch <slug>` | 이 DM/채널의 활성 캐릭터 전환. 카드 재로드까지 수행 |
| `/character info [slug]` | frontmatter + card.md 본문 프리뷰. `slug` 비우면 현재 활성 |
| `/character reset` | 이 DM/채널 매핑 제거 → default 로 복귀 |

**카드 편집 반영 타이밍**: `CharacterService`가 카드를 캐시하므로 `card.md` 수정 후 `/character switch <same-slug>` 한 번 더 치거나 봇을 재시작해야 반영된다.

### `/기억` — 활성 캐릭터 메모리

모든 서브커맨드는 **현재 DM/채널의 활성 슬러그** 메모리 기준으로 동작한다.

| 명령 | 설명 |
|------|------|
| `/기억 확인` | user.md / self.md 조회 |
| `/기억 저장` | 즉시 memo 레포에 git commit |
| `/기억 수정 <내용>` | AI 도움으로 user.md 업데이트 |
| `/기억 핫로그` | 최근 중요 기억 최대 20개 |

---

## 컨텍스트 주입 순서

AI 호출 시 프롬프트는 다음 순서로 조립된다 (`ASSISTANT_MAX_PROMPT_CHARS` 한도 내):

1. `characters/<slug>/card.md` **본문** → 시스템 프롬프트
2. 채널 타입 한 줄 (DM인지 채널인지)
3. `memory/user.md`
4. `memory/self.md`
5. `memory/weekly/` 최신 파일
6. `memory/daily/어제.md`
7. `memory/logs/오늘.md`
8. 사용자 메시지

---

## 능동 인사 (proactive)

- **기상 인사** (봇 시작 직후): 대상 유저 DM의 활성 카드로 짧은 인사
- **아침 인사** (매일 `ASSISTANT_MORNING_HOUR` 시 KST): 동일

활성 카드가 없거나 `GEMINI_API_KEY` 가 비어 있으면 스킵.

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MEMO_REPO_PATH` | (필수) | `memo` 레포 로컬 경로. 비우면 AI 비서·캐릭터 시스템 전부 비활성화 |
| `ASSISTANT_USER_ID` | (필수) | 봇이 DM·지정 채널에서 응답할 유저 ID |
| `ASSISTANT_PUBLIC_CHANNEL_ID` | — | (선택) DM 외 응답할 채널 ID |
| `ASSISTANT_AI_PROVIDER` | `gemini` | `gemini` 또는 `claude-cli` |
| `ASSISTANT_AGENT_REPO_PATH` | — | (`claude-cli`일 때) Claude가 작업할 cwd. 비우면 텍스트 생성만 |
| `ASSISTANT_DEFAULT_CHARACTER` | `yawn` | `.active.json.default` 가 없을 때 폴백 슬러그 |
| `ASSISTANT_MORNING_HOUR` | `8` | 아침 인사 시각 (KST, 0–23) |
| `ASSISTANT_MEMORY_COMMIT_INTERVAL_MS` | `3600000` | 기억 자동 커밋 주기(ms). 기본 1시간 |
| `ASSISTANT_MAX_PROMPT_CHARS` | `12000` | AI 프롬프트 상한 (시스템+컨텍스트+질문 포함) |
| `CLAUDE_CLI_COMMAND` | `claude` | `claude-cli` 프로바이더 실행 파일 이름 |
| `CLAUDE_CLI_TIMEOUT_MS` | `60000` | Claude CLI 타임아웃(ms) |

`yawnbot-defaults.txt` 에 커밋된 기본값은 `apps/yawnbot/.env` 로 덮어쓸 수 있다.

---

## 새 캐릭터 추가

1. `memo/characters/<slug>/card.md` 작성 — frontmatter + 본문. 본문이 비면 로드 실패(스킵).
2. (선택) `appearance.md`, `reference/*.png`
3. Discord에서 `/character switch <slug>` — 해당 DM/채널에서 즉시 활성화 + 카드 재로드
4. `/character list` 로 목록 확인

frontmatter 권장 키 (전부 선택, 있으면 `/character list·info` 요약에 사용):

```yaml
---
slug: yawn
name: 욘 (YawnBot)
display_name: 욘
tone: nonchalant, warm, slightly sultry
speech_style: 반말·편안한 말투
image_style: "anime, soft lighting, warm tones"
relationship: 연인 같은 비서
---
```

---

## 코드 포인터

```
apps/discord-bots/apps/yawnbot/src/
├── services/
│   ├── character-service.ts     ← 카드 로드·캐시, .active.json, channelKey helper
│   └── memory-service.ts        ← 슬러그별 계층형 기억, git commit
└── bot/
    ├── assistant-handler.ts     ← DM/채널 메시지 → 활성 카드 + 메모리로 AI 대화
    ├── proactive.ts             ← 기상·아침 인사 (기본 캐릭터 기준)
    └── slash/
        ├── character.ts         ← /character list·switch·info·reset
        └── router.ts            ← /character, /기억 디스패치
```

`main.ts` 에서 `characterService.initialize()` → default 슬러그 `getMemory(...)` 선-초기화 순으로 호출한다. 종료(SIGINT/SIGTERM) 시 `characterService.commitIfDirty()` 로 `.active.json` 변경분을 자동 커밋한다.
