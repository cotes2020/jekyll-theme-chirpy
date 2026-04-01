# YawnBot - Node.js Server

> **Deprecated (migration):** 새 코드는 `apps/discord-bots/` 워크스페이스로 옮겼습니다. 게임 봇은 `apps/discord-bots/apps/yawnbot`, Unity 무료 에셋 봇은 `apps/discord-bots/apps/unityfree-bot`에서 빌드·실행하세요. 이 폴더(`apps/yawnbot-server`)는 호환을 위해 남겨 둘 수 있으나, 변경은 위 경로를 기준으로 합니다.

이 프로젝트는 KarmoLab에 있던 C# Discord 봇(YawnBot)의 핵심 게임/상호작용 기능들을 Node.js 환경(`discord.js v14`)으로 완벽하게 이식한 서버 애플리케이션입니다.

## 🚀 시작하기

이 봇은 Node.js `v18.0.0` 이상 버전을 권장합니다.

### 1️⃣ 패키지 설치

봇을 실행하기 전, 필요한 패키지를 설치해야 합니다. CMD나 터미널에서 다음 명령어를 실행하세요:

```bash
npm install
```

### 2️⃣ 환경 변수 설정

최상위 폴더에 있는 `.env.template` 파일을 복사하여 `.env` 파일을 생성합니다.

```bash
cp .env.template .env
```

생성된 `.env` 파일을 열고 본인의 Discord 봇 토큰과 Client ID 등을 알맞게 입력하세요.

- `DISCORD_TOKEN`: 디스코드 개발자 포털에서 발급받은 봇 토큰
- `CLIENT_ID`: 봇 애플리케이션 클라이언트 ID
- `/cursor-edit`(관리자 전용): 로컬에서 Cursor CLI `agent acp`로 코드 작업을 시키려면 `CURSOR_LOCAL_REPO_DIR` 등을 설정합니다. 자세한 키는 `.env.template` 참고.
- 기타 항목 (AI, Webhook 포트 등)은 선택 사항입니다.

### 3️⃣ 커맨드 배포 (필수, 최초 1회)

Discord 서버에 `/강화`, `/주식` 등 슬래시 커맨드를 처음으로 등록하거나 추가/수정이 있었을 때 실행해야 합니다. `.env`에 올바른 값이 있어야 봇 서버에 정상적으로 등록됩니다.

```bash
npm run deploy
```

### 4️⃣ 봇 실행

커맨드 배포가 완료되었다면 다음 명령어로 봇 서버를 구동합니다:

```bash
npm start
```

서버 로그에 `"⚔️ YawnBot (Node.js) - 로그인: YawnBot#1234"` 메시지가 나타나면 성공입니다.

### `/cursor-edit` 사용법

- 관리자(`ADMIN_IDS`)만 사용할 수 있습니다.
- `.env`에 `CURSOR_LOCAL_REPO_DIR`(로컬 git 저장소 절대 경로)을 설정하고, 봇을 실행하는 PC에 Cursor CLI `agent`가 PATH에 있어야 합니다.
- Discord에서 `/cursor-edit`를 실행해 `prompt`(필수)에 지시를 입력하고, 선택적으로 `mode`를 `agent`(기본)·`ask`·`plan` 중 하나로 지정합니다.
- 동시에 하나의 작업만 실행되며, 대기 시간은 `CURSOR_TIMEOUT_MS` 등으로 조정할 수 있습니다.
- 완료되면 에이전트 응답 요약과 변경 파일·`diff --stat` 요약이 임베드로 표시됩니다.

---

## 📂 주요 구조

- `data/`: 확률, 텍스트(chat, bot_messages), 그리고 유저 정보(`gamedata.json`) 저장
- `resources/img/`: 강화, 미니게임, 밈(meme) 컨텐츠 등에 쓰이는 이미지 파일 전체 (100% 이식됨)
- `src/`: 봇의 모든 비즈니스 로직(강화, 레이드, 주식 시뮬레이션 등)
- `cli/cursor-local-runner.js`: Discord `/cursor-edit`에서 호출하는 Cursor `agent acp` 래퍼(작업 후 `git status` / `git diff` 요약을 JSON으로 출력). 봇을 돌리는 PC에 Cursor CLI(`agent`)가 PATH에 있어야 합니다.
