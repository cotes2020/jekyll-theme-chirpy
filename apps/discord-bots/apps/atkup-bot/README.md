# ATKUp (Discord 봇)

**ATKUp** — Unity Publisher Sale 무료 에셋 알림과 Hacker News 긱 뉴스를 **설정한 알림 채널**로 보냅니다. **별도 토큰·별도 프로세스**로 운영합니다.

패키지/폴더명: `atkup-bot` (`apps/discord-bots/apps/atkup-bot`)

## 빠른 시작

워크스페이스 루트에서:

```bash
cd apps/discord-bots
npm install
npm run build:atkup
```

`.env`는 `apps/discord-bots/apps/atkup-bot/.env.template`을 복사해 같은 폴더에 `.env`로 만듭니다.

필수 키:

- `ATKUP_DISCORD_TOKEN`
- `ATKUP_CLIENT_ID`
- `ATKUP_TARGET_CHANNEL_ID`

선택:

- `ATKUP_HEALTH_PORT` — 봇 기동 시 `127.0.0.1`에서 `GET /health` 로 `ok` 응답(기본 **4616**). YawnBot 웹훅(4615)과 겹치지 않게 둡니다. 끄려면 `0`.

실행/배포:

```bash
npm run start:atkup
npm run deploy:atkup
```

## 슬래시 커맨드

- `/atkup unity` — 무료 에셋 정보를 가져와 알림 채널에 전송 (`force` 옵션)
- `/atkup news` — Hacker News 상위 글 목록을 알림 채널에 전송 (`count` 5~15, 기본 10)

커맨드 등록/갱신은 `npm run deploy:atkup` (또는 해당 패키지의 `npm run deploy`)를 실행하세요.

**이전 환경 변수** `UNITYFREE_*` 를 쓰던 경우 `.env` 키를 `ATKUP_*` 로 바꿔 주세요.

전체 워크스페이스 설명은 `apps/discord-bots/README.md`를 참고하세요.
