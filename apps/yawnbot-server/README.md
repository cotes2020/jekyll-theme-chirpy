# YawnBot — 레거시 경로(shim)

> **Deprecated:** 구현과 에셋은 `apps/discord-bots/apps/yawnbot`에 있습니다. Unity 무료 에셋 봇은 `apps/discord-bots/apps/unityfree-bot`입니다.

이 디렉터리(`apps/yawnbot-server`)에는 **npm 스크립트 위임용 `package.json`만** 남아 있습니다. 예전에 여기 있던 `src` / `data` / `Resources` 등은 제거되었고, 저장소의 단일 기준은 위 워크스페이스 경로입니다.

## 사용법

### 1. 설치

```bash
cd ../discord-bots
npm install
```

### 2. 환경 변수

`apps/discord-bots/apps/yawnbot/.env.template` → 같은 폴더에 `.env`로 복사합니다.

### 3. 배포 / 실행

이 폴더에서 아래를 실행하면 `apps/discord-bots`의 해당 스크립트로 **위임**됩니다.

```bash
npm run deploy
npm start
```

UnityFree 봇만: `npm run deploy:unityfree` / `npm run start:unityfree`  
두 봇 동시: `npm run start:all`

자세한 내용은 `apps/discord-bots/README.md`, YawnBot 전용 요약은 `apps/discord-bots/apps/yawnbot/README.md`를 참고하세요.
