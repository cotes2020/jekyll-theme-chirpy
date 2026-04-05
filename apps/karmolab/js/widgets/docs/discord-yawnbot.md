# Discord 봇 (욘 / yawnbot)

KarmoLab **문서** 위젯 **「Discord·욘봇」** 탭 — 음성·DAVE 참고 자료, 구현된 기능 요약, **미완 작업(TODO)** 만 정리합니다.

| 무엇 | 어디 |
|------|------|
| **이 탭** | **음성·DAVE** · **기능 요약** · **TODO** · (짧은) **향후 검토** |
| **슬래시·`.env`·사용자 안내** | 레포 [yawnbot README.md](https://github.com/mascari4615/mascari4615.github.io/blob/master/apps/discord-bots/apps/yawnbot/README.md) |
| **워크스페이스 빌드·`npm run`** | 문서 위젯 **「discord-bots · README」** 탭 또는 [discord-bots/README.md](https://github.com/mascari4615/mascari4615.github.io/blob/master/apps/discord-bots/README.md) |

레포: [`apps/discord-bots/apps/yawnbot/`](https://github.com/mascari4615/mascari4615.github.io/tree/master/apps/discord-bots/apps/yawnbot)

---

## 음성 연결과 DAVE 트러블슈팅

레포의 Discord 슬래시 봇 **「욘」**은 npm·폴더명 `yawnbot`으로 `apps/discord-bots/apps/yawnbot/`에 있습니다. 음성(`/music play`·`/music speak`·`/music sound`, 음성 입장 등)은 [`@discordjs/voice`](https://github.com/discordjs/voice)와 Discord 음성 게이트웨이·UDP를 사용합니다.

### DAVE가 무엇인가요?

**DAVE**(Discord Audio & Video Encryption)는 Discord가 도입한 **음성·영상 E2EE(종단 간 암호화)** 프로토콜입니다. 이름 그대로 오디오/비디오 미디어에 추가 암호화 계층을 두고, **MLS(Message Layer Security)** 등으로 참가자 간 키를 맞춥니다. 클라이언트(공식 앱·지원 봇 라이브러리)가 DAVE를 구현해야 해당 방식의 채널에 참여할 수 있습니다.

- Discord 개발자 문서에서는 음성 연결 종료 코드 **`4017`**을 *「E2EE/DAVE protocol required — This channel requires a client supporting E2EE via the DAVE Protocol」*로 정의합니다.  
  [Voice Close Event Codes](https://discord.com/developers/docs/topics/opcodes-and-status-codes#voice-voice-close-event-codes)
- Node 봇 쪽에서는 `@discordjs/voice`의 `joinVoiceChannel({ daveEncryption: true })`와 **`@snazzah/davey`** 등 의존성이 DAVE 경로를 담당합니다.

**정리:** 예전에는 `daveEncryption: false`로도 일반 음성 채널에 붙을 수 있었지만, **서버·채널 설정에 따라 DAVE가 필수**인 경우가 있고, 그때는 DAVE를 켜지 않으면 **Hello 직후 WebSocket이 닫히고 UDP(미디어) 단계까지 가지 못합니다.**

### Discord 음성 연결이 이루어지는 순서 (아주 짧게)

1. 메인 **Gateway**로 `VOICE_STATE_UPDATE` / `VOICE_SERVER_UPDATE`를 받은 뒤, 음성 서버(예: `*.discord.media`)로 **별도의 WebSocket**을 엽니다.
2. 클라이언트가 **Identify** 등을 내면, 서버가 **Hello**(heartbeat 간격 등)를 보냅니다.
3. 서버가 **Ready**(`op:2`)로 **UDP IP·포트·SSRC·암호화 모드 목록**을 주면, 클라이언트가 UDP 소켓을 열고 IP discovery·**Select Protocol**을 진행합니다.
4. **Session Description** 등으로 세션 키를 맞춘 뒤 상태가 **Ready**가 되면 실제 오디오 패킷을 주고받습니다.
5. DAVE가 켜진 연결에서는 추가로 **DAVE/MLS** 메시지(opcode 21–30대 등)가 오가며 암호화 그룹을 맞춥니다.

`@discordjs/voice` 내부 **Networking** 상태는 대략 `OpeningWs(0) → Identifying(1) → UdpHandshaking(2) → SelectingProtocol(3) → Ready(4)` 순이고, 끊기면 `Closed(6)`입니다.

### `/music sound` 슬래시 명령 (첨부·URL·패키지 클립)

YouTube 검색/재생용 **`/music play`** 와 **같은 음성 연결·재생 대기열**(`@discordjs/voice` + 내부 뮤직 플레이어)을 사용합니다. 채널이 **DAVE(E2EE) 필수**이면 봇이 DAVE로 음성에 붙어 있어야 `/music sound`도 재생됩니다.

#### 사용 조건

- **길드(서버)에서만** 사용할 수 있습니다. DM에서는 불가입니다.
- 명령을 친 사용자가 **음성 또는 스테이지 채널**에 들어가 있어야 합니다.
- 봇에게 해당 채널 **보기(View Channel)·연결(Connect)·말하기(Speak)** 권한이 필요합니다.

#### 옵션 — `file` / `url` / `clip` 중 **정확히 하나**

| 옵션 | 설명 |
|------|------|
| `file` | 슬래시 커맨드 입력창에 **오디오 파일을 첨부**합니다. `Content-Type`이 `audio/*`이거나, 파일명 확장자가 아래 목록에 있으면 통과합니다. |
| `url` | 브라우저나 `curl`로 **직접 받을 수 있는** `http(s)` 오디오 주소입니다. **YouTube 영상 페이지 URL은 `/music play`** 를 사용하세요. URL 길이는 대략 2000자까지입니다. |
| `clip` | 봇 패키지의 **`resources/audio/`** 아래에 넣은 파일의 **파일명만** 적습니다. `foo/bar.mp3` 같은 경로나 `..` 는 허용하지 않아 **경로 조작을 막습니다**. |

**`clip`에서 허용하는 확장자:** `.mp3` `.ogg` `.wav` `.m4a` `.opus` `.flac` `.webm`  

레포 기준 폴더: [`apps/discord-bots/apps/yawnbot/resources/audio/`](https://github.com/mascari4615/mascari4615.github.io/tree/master/apps/discord-bots/apps/yawnbot/resources/audio) — 여기에 파일을 두고 `/music sound clip:파일명.wav` 형태로 호출합니다. 기본 샘플로 `demo.wav` 등을 둘 수 있습니다.

#### 동작 요약

- 오디오 스트림을 만든 뒤 **`enqueueCustomTrack`** 으로 대기열에 넣습니다. 이미 재생 중이면 **대기열에만 추가**되고, 응답에 순번이 표시됩니다.
- 준비·재생 시도 전체에 **120초 타임아웃**이 있습니다. 네트워크·포맷 문제 시 `/music sound` 응답에 에러 문구가 나갑니다.

### 이번에 겪은 증상과 좁혀 간 과정

#### 증상

- 음성 입장 시 **`signalling` ↔ `connecting`만 반복**하거나, 곧바로 실패.
- `VOICE_DEBUG=1` 로그에서 음성 WebSocket은 잠깐 붙은 것 같지만 **`udp:false`가 길게 유지**되거나, **Ready(`op:2`) 전에 끊김**.
- Windows **리소스 모니터**에서 `node.exe`에 **UDP가 안 보이고 TCP만 보임** — 처음에는 “UDP가 막혔나?” 쪽으로 의심.

#### 정리된 오해·주의점

- **리소스 모니터의 「연결」** 목록은 TCP 중심이라, UDP는 **같은 방식으로 잘 안 보이거나** 짧게만 잡힐 수 있습니다. **UDP가 없다고 해서 곧바로 OS가 UDP를 막는다고 결론 내리기 어렵습니다.**
- **Windows Defender만 쓰는 경우** 서드파티 백신 이슈보다는 **방화벽·실제 패킷(Wireshark)** 쪽이 진단에 유리한 경우가 많습니다.
- 진짜로 “UDP가 안 나간다”를 보려면 **Wireshark**로 Discord 미디어 IP 기준 `udp` 필터를 보는 편이 확실합니다.

#### 원인 파악의 전환점

- 로그상 흐름은 **Identify → Hello(`op:8`)까지는 성공**하지만, 그 직후 **Ready(`op:2`) 전에 WebSocket이 닫힘** → **UDP 소켓을 만들 타이밍 자체에 도달하지 못한 것**과 일치합니다.
- `@discordjs/voice`는 음성 WebSocket **close code가 `4014`일 때만** `VoiceConnection`을 **`Disconnected` 상태로 두고 `closeCode`를 남깁니다.**  
  **`4014`가 아니면** 라이브러리가 **자동 재접속**을 위해 곧바로 **`signalling`으로 되돌리기 때문에**, `Disconnected`·`reason` 로그가 **안 찍힐 수 있습니다.**
- 그래서 **`[voice] disconnected, reason:`** 만으로는 원인이 안 보였고, 내부 **`Networking`의 `close` 이벤트**에 훅을 걸어 **Discord가 준 종료 코드 숫자**를 로그로 남기는 쪽이 필요했습니다.

#### 확인된 종료 코드: **4017**

- 로그에 **`Discord 음성 WebSocket close code: 4017`**이 찍힘.
- 문서상 **해당 채널은 DAVE(E2EE)를 지원하는 클라이언트만 허용** — 봇에서 **`daveEncryption`을 켜야** 합니다.

#### 해결

- **`DISCORD_VOICE_DAVE=1`**(또는 이후 코드 변경으로 **기본값을 DAVE 켬**) 후 재시도.
- 성공 시 로그에는 **`op:2` Ready**, **`udp:true`**, **`[DAVE] Session initialized`**, MLS 관련 송수신 등이 이어지고, 상위 상태가 **`connecting → ready`**로 바뀝니다.

### 환경 변수 (yawnbot)

| 변수 | 역할 |
|------|------|
| `VOICE_DEBUG` | `1`이면 `@discordjs/voice` 네트워킹 디버그·상태 로그. **토큰·키·IP 등 민감 정보가 섞일 수 있어** 상시 켜두지 말 것. |
| `DISCORD_VOICE_DAVE` | **기본값: DAVE 사용(켬).** 끄려면 `0` / `false` / `off` / `no`. 예전에 4017이 났던 채널은 DAVE가 꺼져 있으면 다시 실패합니다. |

자세한 명령·경로는 레포 [yawnbot README.md](https://github.com/mascari4615/mascari4615.github.io/blob/master/apps/discord-bots/apps/yawnbot/README.md)를 참고하세요.

---

## 구현된 기능 요약

운영·디버깅용으로 **이미 들어가 있는 동작**만 짧게 묶었습니다. 세부 env·명령은 README가 기준입니다.

- **`/yawn`**: 채널 최근 메시지 맥락(`YAWN_CONTEXT_MESSAGES`), 시스템 프롬프트(`YAWN_SYSTEM_PROMPT`), 질문·전체 프롬프트 길이 상한, API 오류 시 짧은 사용자 안내와 `[yawn]` 콘솔 로그.
- **슬래시 가드·로그**: 허용 길드·채널 ID(`YAWNBOT_ALLOWED_*`), 선택적 사용 로그(`YAWNBOT_SLASH_USAGE_LOG`).
- **`/music`**: 지금 재생 임베드(경과·길이·주기 갱신, env로 끄기/주기 조절), 플레이리스트를 큐에 넣을 때 **n/N** 진행 표시, `shuffle` / `loop` / `remove`, `queue` 페이지·버튼, 긴 플레이리스트 한 줄 요약, 실패 시 텍스트 채널 알림·쿨다운, 응답 공개/ephemeral 정책.
- **음성·DAVE**: 위 절 및 `DISCORD_VOICE_DAVE` / `VOICE_DEBUG`.

---

## 남은 작업 (TODO)

아래만 체크리스트로 둡니다. 완료되면 항목을 지우거나 README/코드 주석으로 옮기면 됩니다.

- [ ] **대기열** — 항목 순서 이동(옵션)
- [ ] **음성** — (선택) `/voice-status` 또는 close code 요약 로그
- [ ] **음성** — 끊김 시 재입장 상한·백오프 정책 정리·구현
- [ ] **스테이지** — 발언 요청·봇 Speak 타이밍 문서화
- [ ] **슬래시** — `setDescription`·옵션에 짧은 영문 병기
- [ ] **슬래시** — `/help` 카테고리별 정리(ephemeral / embed)
- [ ] **배포** — 글로벌 vs 길드 `deploy` 절차를 README 한 블록으로
- [ ] **데이터** — `gamedata.json` 백업·git 추적·민감 필드 운영 정책
- [ ] **남용 방지** — 게임/도박 등 서버별 쿨다운·관리자 예외
- [ ] **`/yawn`** — 민감 주제·빈 입력 등 거절 문구 통일
- [ ] **AI** — `maxOutputTokens` 등을 env로 (`karmolab-ai` 연동 검토)
- [ ] **운영** — 로그 접두·심각도·인터랙션 id 일부 규칙
- [ ] **테스트** — URL 정규화·가드 파싱·에러 메시지 매핑 등 단위 테스트
- [ ] **CI** — `yawnbot` `npm run build`
- [ ] **헬스** — ping·음성 세션 수·HTTP `/health`(웹훅 서버 등과 연계)

**TODO 위치:** 평소엔 이 Markdown으로 두고, 큰 덩어리는 GitHub Issues로 쪼개도 됩니다.

---

## 향후 검토 (우선순위 낮음)

한 번에 구현하기보다 필요할 때 이슈로 빼기 좋은 아이디어입니다.

- 운영: 관리자만 길드 제한 예외, JSONL 로그·일별 로테이션, YouTube·AI 일시 오류 백오프
- `/yawn`: 맥락 표시 푸터, 스레드·포럼 부모 포함 옵션, 짧은 TTL 세션 메모리, 길드별 프리셋, PII 마스킹
- 음악: NP 채널 분리, pause/resume·볼륨, 대기열 영속화, 라이브 스트림 NP 표시
- 가드: 카테고리·역할 단위 허용, 가드 실패 debug 로그
- 기타: 시즌 이벤트, 주식 요약 웹훅, 컴포넌트·컨텍스트 메뉴 진입, `kakao-export` 재시도·큐, DX(타입·목 음성 mock)

---

## 참고 링크

- [Discord — Voice Close Event Codes](https://discord.com/developers/docs/topics/opcodes-and-status-codes#voice-voice-close-event-codes)
- [discord.js — 이슈 #11419 (DAVE 관련 논의 예시)](https://github.com/discordjs/discord.js/issues/11419)
- 워크스페이스: [`apps/discord-bots/README.md`](https://github.com/mascari4615/mascari4615.github.io/blob/master/apps/discord-bots/README.md)
- 봇 README: [`apps/discord-bots/apps/yawnbot/README.md`](https://github.com/mascari4615/mascari4615.github.io/blob/master/apps/discord-bots/apps/yawnbot/README.md)
