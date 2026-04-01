# Discord Bots Workspace

이 디렉터리는 디스코드 봇들을 위한 독립 워크스페이스입니다.

## 앱 목록

- `apps/yawnbot`: 게임/슬래시 커맨드 봇 (기존 YawnBot)
- `apps/unityfree-bot`: Unity 무료 에셋 알림 봇 (별도 토큰/프로세스)

## 설치

```bash
cd apps/discord-bots
npm install
```

## 실행

### YawnBot

```bash
npm -w apps/yawnbot run build
npm -w apps/yawnbot run start
```

### UnityFree Bot

```bash
npm -w apps/unityfree-bot run build
npm -w apps/unityfree-bot run start
```

## 커맨드 배포

### YawnBot

```bash
npm -w apps/yawnbot run deploy
```

### UnityFree Bot

```bash
npm -w apps/unityfree-bot run deploy
```

