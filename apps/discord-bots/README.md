# Discord Bots Workspace

이 디렉터리는 디스코드 봇들을 위한 독립 워크스페이스입니다.

## 앱 목록

- `apps/yawnbot`: 게임/슬래시 커맨드 봇 (기존 YawnBot) — 앱별 요약은 [`apps/yawnbot/README.md`](apps/yawnbot/README.md)
- `apps/atkup-bot`: **ATKUp** (Unity 무료·긱 뉴스 알림, 별도 토큰/프로세스) — [`apps/atkup-bot/README.md`](apps/atkup-bot/README.md)

## 설치

```bash
cd apps/discord-bots
npm install
```

## 실행

루트(`apps/discord-bots`)에서 워크스페이스 스크립트를 쓰면 한 곳에서 둘 다 제어할 수 있습니다.

### 한 번에 두 봇 실행

```bash
npm run start
```

### 봇별로만 실행

```bash
npm run start:yawnbot
npm run start:atkup
```

### 앱 단위(직접 `-w`)

#### YawnBot

```bash
npm -w apps/yawnbot run build
npm -w apps/yawnbot run start
```

#### ATKUp (`atkup-bot`)

```bash
npm -w apps/atkup-bot run build
npm -w apps/atkup-bot run start
```

## 커맨드 배포

### 한 번에 둘 다 배포

```bash
npm run deploy
```

### 봇별 배포

```bash
npm run deploy:yawnbot
npm run deploy:atkup
```

### 앱 단위(직접 `-w`)

#### YawnBot

```bash
npm -w apps/yawnbot run deploy
```

#### ATKUp (`atkup-bot`)

```bash
npm -w apps/atkup-bot run deploy
```

## 레거시 `apps/yawnbot-server`

이 폴더의 `npm run start` / `build` / `deploy` 는 위 `apps/discord-bots` 워크스페이스로 **위임**됩니다. 최초 1회는 `apps/discord-bots`에서 `npm install`이 필요합니다.
