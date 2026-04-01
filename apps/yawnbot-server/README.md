# YawnBot - Node.js Server

> **Deprecated (migration):** 새 코드는 `apps/discord-bots/` 워크스페이스로 옮겼습니다. 게임 봇은 `apps/discord-bots/apps/yawnbot`, Unity 무료 에셋 봇은 `apps/discord-bots/apps/unityfree-bot`에서 빌드·실행하세요.

이 폴더(`apps/yawnbot-server`)는 **레거시 호환(shim)** 용도로만 남겨 둔 것이며, `npm start` / `npm run deploy` 같은 명령은 모두 `apps/discord-bots`로 **위임**됩니다. (이 폴더의 소스는 더 이상 기준이 아닙니다.)

이 프로젝트는 KarmoLab에 있던 C# Discord 봇(YawnBot)의 핵심 게임/상호작용 기능들을 Node.js 환경(`discord.js v14`)으로 완벽하게 이식한 서버 애플리케이션입니다.

## 🚀 시작하기

이 봇은 Node.js `v18.0.0` 이상 버전을 권장합니다.

### 1️⃣ 패키지 설치

실제 코드는 `apps/discord-bots` 워크스페이스에 있습니다. 이 폴더에서 `npm run start` 등은 그쪽으로 **위임**되므로, **먼저** 아래를 실행하세요:

```bash
cd ../discord-bots
npm install
```

(이후 `apps/yawnbot-server`에서 `npm run build` / `npm start`를 써도 됩니다.)

### 2️⃣ 환경 변수 설정

YawnBot용 `.env`는 **`apps/discord-bots/apps/yawnbot`** 기준입니다. 해당 폴더의 `.env.template`을 복사해 `.env`를 만듭니다.

```bash
cd apps/discord-bots/apps/yawnbot
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

자세한 사용법(명령, `.env`, `/cursor-edit` 포함)은 `apps/discord-bots/README.md`를 참고하세요.
