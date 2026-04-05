# 욘 (yawnbot)

Discord 슬래시 봇입니다. **표시 이름은「욘」**, npm·폴더명은 `yawnbot` 입니다. 소스·데이터·이미지 기준 경로는 이 디렉터리입니다.

개선·할 일·아이디어 백로그는 KarmoLab **문서** 위젯의 **「Discord·개선」** 탭(`apps/karmolab/js/widgets/docs/discord-bot-improvements.md`)에서 봅니다.

## 빠른 시작

워크스페이스 루트에서:

```bash
cd apps/discord-bots
npm install
npm run build:yawnbot
```

환경 변수는 **레이어드 dotenv**를 지원합니다. 요약은 [.env.template](./.env.template)를 보고, 키 목록은 `.yawnbot.env.template`·`.yawnbot.kakao.env.template` 등을 각각 `.yawnbot.env`·`.yawnbot.kakao.env` 로 복사해 쓰면 됩니다. **`.env` 하나만** 있어도 마지막에 로드되어 동작합니다.

```bash
npm run start:yawnbot
npm run deploy:yawnbot
```

전체 워크스페이스 설명은 [apps/discord-bots/README.md](../../README.md)를 참고하세요.

로그인 후 **프로필 활동(Playing …)** 은 기본으로 **약 3초마다** 순환합니다. 끄려면 `.yawnbot.env` 또는 `.env`에 `BOT_PRESENCE_INTERVAL_SEC=0`, 문구·간격은 `BOT_PRESENCE_LINES`·`BOT_PRESENCE_INTERVAL_SEC`로 바꿀 수 있습니다(`.yawnbot.env.template` 주석).

---

## 슬래시 명령어 요약

### 음성·미디어 (같은 재생 대기열)

YouTube·TTS·클립·대기열은 **`/music`** 아래 **서브커맨드**로만 등록됩니다.

| 명령 | 설명 |
|------|------|
| `/music` `play` `query` | YouTube **동영상**·**플레이리스트** URL 또는 검색어. `playlist?list=` / `watch?…&list=` 지원. 플레이리스트 곡 수 상한은 환경 변수로 조절(기본 40, `0`이면 끝까지). 음성 채널 필수. |
| `/music` `skip` | 현재 재생 건너뛰기 |
| `/music` `stop` | 재생 중지·대기열 비우기 |
| `/music` `shuffle` | **대기 중인 곡만** 순서 무작위 섞기 (지금 재생 곡은 그대로) |
| `/music` `remove` `index` | 대기열 **번호** 곡 제거 (`/music queue`에 나온 1·2·3…과 동일. **지금 재생 중**인 곡은 `skip`으로만 건너뜀) |
| `/music` `loop` `mode` | `off` 끔 · `track` 지금 곡만 반복 · `queue` 대기열 끝나면 같은 목록 다시 (빈 대기열이면 `queue` 불가) |
| `/music` `queue` `page?` | 대기열·**재생 중**(경과/전체 길이·남은 시간, 알 수 있을 때)·**반복** 표시, `page` 옵션·**이전/다음** 버튼 (12곡/페이지) |
| `/음성입장` / `/voice-join` | 봇을 음성·스테이지 채널에 연결 |
| `/음성퇴장` / `/voice-leave` | 음성 연결 해제 |
| `/music` `speak` `text?` | **Edge 온라인 TTS**로 읽기 (디스코드 내장 TTS 아님). `text` 비우면 데모 문장. |
| `/music` `sound` | **오디오 파일 재생**. `file`·`url`·`clip` 중 **하나만** 지정. |

`/music sound` 옵션:

| 옵션 | 설명 |
|------|------|
| `file` | 명령에 오디오 첨부 (mp3, wav, ogg 등) |
| `url` | 직접 다운로드 가능한 `http(s)` 오디오 주소 (YouTube **페이지** URL은 `/music play` 사용) |
| `clip` | 봇 패키지 내 `resources/audio/`에 넣은 파일의 **파일명만** (기본 샘플: `demo.wav`) |

### AI·관리

| 명령 | 설명 |
|------|------|
| `/yawn` `질문` | Gemini / Vertex (`.yawnbot.env` / `.env` — AI Studio 또는 Vertex). 채널 **최근 메시지**를 맥락으로 붙일 수 있음(`YAWN_CONTEXT_MESSAGES`). 시스템 톤은 `YAWN_SYSTEM_PROMPT`로 조절 |
| `/cursor-edit` | [관리자] 로컬 Cursor 에이전트 |
| `/admin-reload` / `/admin-save` | [관리자] 데이터 리로드·저장 |

### 게임·주식·레이드·기타

`/강화`, `/판매`, `/정보`, `/돈`, `/랭킹`, `/출첵`, `/돈내놔`, `/배틀`, `/슬롯`, `/홀짝`, `/가위바위보`, `/주식목록`, `/주식차트`, `/매수`, `/매도`, `/내주식`, `/레이드정보`, `/공격`, `/레이드소환`, `/ping`, `/도움말`

---

### 운영·제한 (선택)

| 변수 | 설명 |
|------|------|
| `YAWNBOT_ALLOWED_GUILD_IDS` | (선택) 허용 길드 ID만 슬래시 사용. 비우면 제한 없음. 예전 이름 `YAWNBOT_SLASH_GUILD_IDS`도 동일 |
| `YAWNBOT_ALLOWED_SLASH_CHANNEL_IDS` | (선택) 슬래시를 칠 수 있는 텍스트 채널 ID만 허용 |
| `YAWNBOT_SLASH_USAGE_LOG` | `1` 등 — 슬래시 사용을 콘솔에 한 줄로 기록 |
| `YAWN_CONTEXT_MESSAGES` | `/yawn`에 붙일 최근 **사용자** 메시지 개수(0~30, 기본 10). `Message Content` 인텐트·채널 읽기 권한 필요 |
| `YAWN_SYSTEM_PROMPT` | `/yawn` 시스템 역할(비우면 기본 톤). `\n` 이스케이프 가능 |
| `YAWN_MAX_QUESTION_CHARS` / `YAWN_MAX_PROMPT_CHARS` | 질문 길이·전체 프롬프트 상한(비용·토큰 완화) |
| `YAWNBOT_NOW_PLAYING_MESSAGE` | `0` 등으로 끄면 `/music play` 알림 채널의「지금 재생」임베드를 보내지 않음 |
| `YAWNBOT_NOW_PLAYING_REFRESH_SEC` | 위 임베드 자동 수정 주기(초). `0`이면 첫 메시지만 |

플레이리스트를 큐에 넣는 동안 `/music play` 응답에 **몇 곡/전체** 진행이 간헐적으로 갱신됩니다.

---

## 환경 변수 (음성·TTS·YouTube)

Discord·Gemini·Cursor 등 키 목록은 [.yawnbot.env.template](./.yawnbot.env.template) 주석을 참고하세요. 로드 순서·파일 이름은 [.env.template](./.env.template)에 정리되어 있습니다.

### `/music speak` (Edge TTS)

| 변수 | 설명 |
|------|------|
| `SPEAK_VOICE` | 기본 `ko-KR-SunHiNeural`. 다른 보이스는 [Azure Speech — 언어 및 음성 지원 (TTS)](https://learn.microsoft.com/azure/ai-services/speech-service/language-support?tabs=tts) 목록의 **신경망 보이스 이름**을 참고. |
| `SPEAK_LANG` | 기본 `ko-KR` |
| `SPEAK_TTS_PROXY` | TTS 요청만 프록시할 때. 비우면 `HTTPS_PROXY`를 사용. |
| `SPEAK_TTS_TIMEOUT_MS` | Edge TTS WebSocket 타임아웃(ms). 코드에서 **최소 5초·최대 120초**로 잘림. |

구현은 npm 패키지 [node-edge-tts](https://www.npmjs.com/package/node-edge-tts) (Microsoft Edge 온라인 TTS)를 사용합니다. 서비스 변경 시 동작이 깨질 수 있습니다.

### `/music play` (YouTube)

| 변수 | 설명 |
|------|------|
| `YT_DLP_PATH` / `YAWNBOT_YT_DLP_PATH` | `yt-dlp` 실행 파일 직접 지정 |
| `YT_DLP_COOKIES_PATH` / `YAWNBOT_YOUTUBE_COOKIES_PATH` | Netscape `cookies.txt` (연령·로그인 제한 완화) |
| `YAWNBOT_PLAYLIST_MAX_TRACKS` | 플레이리스트에서 가져올 **최대 곡 수**. 기본 `40`. **`0` 이하**면 **한도 없음**(페이지 끝까지; 대형 목록은 로드·디스코드 응답 시간이 길어질 수 있음). 양수면 그 개수만큼만. |

내장 `ffmpeg-static`이 `FFMPEG_PATH`를 잡습니다. 무음이면 봇이 음성 채널에서 음소거되지 않았는지 확인하세요.

### 음성 연결·DAVE(E2EE)

일부 음성·스테이지 채널은 Discord 쪽에서 **DAVE(E2EE)를 지원하는 클라이언트만** 허용합니다. 봇은 **기본으로 DAVE를 켠 상태**로 접속합니다. DAVE를 꺼 둔 경우(`DISCORD_VOICE_DAVE=0` 등)에는 Hello 직후 끊기며 로그에 **close code `4017`**이 찍힐 수 있습니다([Voice Close Event Codes](https://discord.com/developers/docs/topics/opcodes-and-status-codes#voice-voice-close-event-codes)).

| 변수 | 설명 |
|------|------|
| `DISCORD_VOICE_DAVE` | **기본값: DAVE 켬**(`daveEncryption: true`). E2EE 필수 채널(종료 코드 4017)에 맞춤. 끄려면 `0`·`false`·`off`·`no` 중 하나. 특정 환경에서만 문제가 나면 끄고 `@discordjs/voice`/Discord 쪽 이슈를 확인하세요. |
| `VOICE_DEBUG` | `1`이면 `@discordjs/voice` 네트워킹 디버그와 상태 전이 로그가 나옵니다. `4014`가 아닌 close는 라이브러리가 곧바로 `signalling`으로 돌리므로, 원인 파악 시 **`[voice] [NW] Discord 음성 WebSocket close code:`** 줄을 보면 됩니다. 민감 정보(세션·키·IP)가 섞이므로 상시 켜두지 말 것. |

---

## 자주 쓰는 링크

| 용도 | URL |
|------|-----|
| Discord 앱·봇 토큰 | https://discord.com/developers/applications |
| Google AI Studio (Gemini 키) | https://aistudio.google.com/app/apikey |
| Azure / Speech — TTS 보이스 목록 | https://learn.microsoft.com/azure/ai-services/speech-service/language-support?tabs=tts |
| node-edge-tts (npm) | https://www.npmjs.com/package/node-edge-tts |
