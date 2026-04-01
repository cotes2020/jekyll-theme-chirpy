# UnityFree Bot

Unity 무료 에셋(쿠폰/프로모) 정보를 주기적으로 확인해서 지정한 채널에 알리는 Discord 봇입니다. **별도 토큰/별도 프로세스**로 운영합니다.

## 빠른 시작

워크스페이스 루트에서:

```bash
cd apps/discord-bots
npm install
npm run build:unityfree
```

`.env`는 `apps/discord-bots/apps/unityfree-bot/.env.template`을 복사해 같은 폴더에 `.env`로 만듭니다.

필수 키:

- `UNITYFREE_DISCORD_TOKEN`
- `UNITYFREE_CLIENT_ID`
- `UNITYFREE_TARGET_CHANNEL_ID`

실행/배포:

```bash
npm run start:unityfree
npm run deploy:unityfree
```

전체 워크스페이스 설명은 `apps/discord-bots/README.md`를 참고하세요.

