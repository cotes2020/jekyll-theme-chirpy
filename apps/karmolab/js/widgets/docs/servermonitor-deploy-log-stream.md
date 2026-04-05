# 서버 모니터 · Deploy 로그 스트림 (설계 초안)

**상태:** 미구현 — 구현 전 검토용 설계만 정리합니다.

**범위:** KarmoLab 데스크톱(Tauri) **서버 모니터** 위젯에서, 프로필 카드의 **deploy**(및 선택적으로 **npm i**) 실행 시 **표준 출력·에러를 실시간으로** 같은 화면 아래에 보여 주는 기능.

---

## 1. 목표

- **deploy** 버튼을 눌렀을 때, 터미널에서 보이던 것과 유사하게 **줄 단위로** 로그가 쌓이는 영역을 **해당 카드(또는 카드 바로 아래)** 에 둔다.
- 긴 작업(Discord API 등) 중에도 **응답 없음처럼 보이지 않게** 한다.
- 실패 시에도 **마지막까지의 로그**가 남아 원인 파악에 쓰인다.

## 2. 비목표 (초기 버전에서 제외)

- 일반 **시작** 프로세스의 장기 로그(백그라운드 detach)까지 통합하지 않는다. 우선 **한 번 끝나는 동기 작업**(deploy, npm install)만 대상으로 한다.
- 터미널 **입력**(stdin)·**PTY**(색상·진행 막대 완벽 재현)는 하지 않는다. **텍스트 스트림**만 다룬다.
- 원격 서버나 브라우저만 켠 상태(GitHub Pages)에서의 스트리밍은 하지 않는다. **Tauri WebView 안**에서만 동작한다.

## 3. 현재 구현과 차이

- `localdev_deploy` / `localdev_npm_install` 은 자식 프로세스의 **stdout·stderr를 끝까지 모은 뒤**(`output()` 등) 성공/실패와 함께 한 번에 반환한다.
- 설계에서는 **파이프를 열고** 읽기 루프를 돌며 **중간 결과를 프론트로 보내는** 경로가 필요하다.

## 4. UX

| 항목 | 제안 |
|------|------|
| 위치 | **프로필 카드 하단**에 접을 수 있는 **로그 패널**(또는 카드 확장 영역). 여러 카드가 세로로 있으면 각 카드에 독립 패널. |
| 표시 | 단색 **모노스페이스** 영역, 자동 스크롤(사용자가 위로 스크롤하면 일시 정지 옵션은 후속). |
| 동시 실행 | 동일 프로필에 대해 deploy **중복 클릭 방지**(버튼 disabled). 서로 다른 프로필은 동시에 돌 수 있음 → 패널은 프로필별로 분리. |
| 종료 후 | 성공/실패 **토스트**는 유지. 패널은 **지우기** 버튼 또는 **다음 실행 전 자동 클리어** 정책 중 하나를 택한다(초기는 “다음 실행 시 위에 이어 붙이지 않고 클리어”가 단순). |
| 길이 | 메모리·렌더 부하를 위해 **최대 줄 수 또는 최대 문자 수** 상한(예: 500줄 또는 256KB) 후 오래된 줄부터 드롭. |

## 5. 아키텍처 (Tauri 중심)

### 5.1 흐름

1. WebView에서 `invoke('localdev_deploy_stream_start', { profileId })` 같은 **시작 커맨드**를 호출하거나, 기존 이름을 유지하고 **옵션 플래그**로 스트림 모드를 켠다.
2. Rust 쪽에서 기존과 동일하게 **repo root**, **프로필 조회**, **`deploy_args` / npm install 검증**, **`cwd` 해석**을 수행한다.
3. `Stdio::piped()` 로 자식 프로세스를 띄우고, **stdout / stderr** 각각 또는 **합쳐진 스트림**을 비동기로 읽는다.
4. 읽은 **줄(또는 UTF-8 덩어리)** 마다 **Tauri 이벤트**로 WebView에 전달한다. 예: 채널명 `localdev-log`, 페이로드 `{ runId, profileId, stream: 'out'|'err', line }`.
5. 프로세스 **종료** 시 `status`와 함께 `localdev-log-done` 이벤트(또는 동일 채널의 `done` 타입)를 한 번 보낸다.
6. WebView는 **해당 profileId 카드**의 패널에만 append.

### 5.2 Windows `cmd /C npm …`

- 현재와 같이 `npm`을 `cmd /C`로 감싸는 경우, **파이프는 자식 cmd 프로세스에 연결**되면 된다. stderr는 npm이 에러를 stderr로 보내는 경우가 많으므로 **stdout·stderr 둘 다 읽지 않으면 데드락**에 걸릴 수 있음 → **비동기로 두 파이프를 동시에 읽거나**, `stderr`를 `stdout`에 합치는 옵션이 있으면 단일 스트림으로 단순화 가능.
- 구현 시 **데드락 방지**를 명시적으로 검증할 것.

### 5.3 `runId`

- 한 세션에서 여러 번 deploy할 때 로그를 구분하기 위해 **UUID 또는 타임스탬프 기반 id**를 시작 시 생성해 모든 이벤트에 실는다.
- UI는 **가장 최근 runId**만 표시하거나, 이전 실행 로그를 접어 두는 것은 후속.

## 6. API 스케치 (이름은 구현 시 조정)

| 방향 | 제안 |
|------|------|
| Rust → Web | `app.emit_all` 또는 `window.emit` 으로 `localdev-log` / `localdev-log-done` |
| Web → Rust | `localdev_deploy_with_log(profile_id)` 가 **즉시** `Ok({ run_id })` 를 반환하고, 실제 종료는 이벤트로만 알림 **또는** 기존처럼 **한 번의 invoke가 Promise로 끝까지 대기**하되 중간에 `listen`으로 청크 수신(Fire-and-forget + listen 패턴은 Tauri v2에서 권장 패턴 확인 필요). |

**권장 패턴(초안):** 프론트에서 `listen` 을 먼저 등록한 뒤 `invoke`로 스트리밍 작업 시작; `invoke`는 프로세스 spawn 직후 `run_id` 반환 또는 **프로세스가 끝날 때까지 await** 하면서도 중간 이벤트는 별도 스레드에서 emit — 후자는 단일 async 커맨드로 구현 가능.

## 7. 보안·운영

- 기존 **`args_are_safe`**, **허용 program**, **cwd가 repo 내부** 등 검증은 **그대로 유지**한다.
- **로그에 민감 정보**가 섞일 수 있음(API 키, 토큰, 쿠키 경로 등). npm/Discord 스크립트 출력 특성상 **경고 문구**를 패널 상단에 두고, 화면 공유 시 주의를 문서에 명시한다.
- 로그는 **디스크에 영구 저장하지 않는 것**을 기본으로 한다(필요 시 후속 “세션 로그 파일” 옵션).

## 8. 구현 단계 (제안)

1. **Phase 1:** `localdev_deploy` 만 파이프 + 이벤트 스트림; UI는 서버 모니터 카드 하단 텍스트 영역.
2. **Phase 2:** `localdev_npm_install` 동일 적용, 상한·스크롤 정책 다듬기.
3. **Phase 3:** (선택) 로그 복사 버튼, 다크/라이트 대비, 실패 시 마지막 N줄만 토스트에 요약.

## 9. 대안 (스트리밍을 하지 않을 때)

- **deploy 시 OS 기본 터미널에서 명령 한 줄 실행**(`open terminal` 스타일) — 플랫폼별 구현 부담.
- 사용자에게 **README의 `npm run deploy:yawnbot` 를 터미널에서 실행**하도록 안내만 강화.

## 10. 관련 코드·설정 (레포)

- Tauri: `apps/karmolab-tauri/src-tauri/src/local_dev.rs` (`run_npm_deploy_blocking` 등).
- UI: `apps/karmolab/src/widgets/servermonitor.ts` (deploy 버튼).
- 설정: `apps/karmolab/data/servermonitor-config.json` 의 `deployArgs`.

---

## 문서 보완 (구현 후)

- [ ] 실제 이벤트 이름·페이로드 스키마로 본문 갱신
- [ ] `local-dev-runner.md` 의 사용자 안내(버튼 동작) 한 줄 반영
- [ ] 스크린샷 또는 짧은 GIF(선택)
