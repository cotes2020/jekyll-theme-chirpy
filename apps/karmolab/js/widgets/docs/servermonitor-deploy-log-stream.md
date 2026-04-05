# 서버 모니터 · Deploy / npm 로그 스트림

**상태:** 구현됨 (데스크톱 Tauri + 서버 모니터 위젯).

**범위:** KarmoLab 데스크톱(Tauri) **서버 모니터** 위젯에서, 프로필 카드의 **deploy**·**npm i** 실행 시 **표준 출력·에러를 실시간으로** 카드 아래 패널에 보여 줍니다.

---

## 구현 요약

| 항목 | 내용 |
|------|------|
| Rust 커맨드 | `localdev_deploy_stream`, `localdev_npm_install_stream` (`spawn_blocking` + 파이프) |
| 블로킹 커맨드(유지) | `localdev_deploy`, `localdev_npm_install` — devtools·호환용 |
| 이벤트 | `localdev-log` (줄), `localdev-log-done` (종료) |
| WebView | `window.__TAURI__.event.listen` 등록 후 `invoke` (순서 고정) |
| 권한 | `capabilities/default.json`에 `core:event:allow-listen` |

### 이벤트 페이로드 (camelCase)

**`localdev-log`**

- `runId` (string, UUID)
- `profileId` (string)
- `stream`: `"out"` | `"err"`
- `line` (string, 한 줄; 끝의 `\r`/`\n` 제거)

**`localdev-log-done`**

- `runId`, `profileId`, `kind` (`"deploy"` | `"npm_install"`), `success` (boolean), `code` (number \| 생략, exit code)

---

## 1. 목표

- **deploy** / **npm i** 시 터미널과 비슷하게 **줄 단위** 로그가 카드 하단에 쌓인다.
- 긴 작업 중에도 **응답 없음처럼 보이지 않게** 한다.
- 실패 시 **마지막까지의 로그**로 원인 파악에 도움을 준다.

## 2. 비목표 (당분간)

- **시작**으로 띄운 백그라운드 프로세스의 장기 로그는 포함하지 않는다.
- stdin·PTY(색상·진행 막대 완벽 재현)는 하지 않는다.
- GitHub Pages만 연 브라우저에서는 동작하지 않는다(**Tauri WebView 전용**).

## 3. 구현과 블로킹 API

- 스트리밍: 파이프 + **stdout/stderr 각각 별도 스레드**에서 `read_line` 후 `app.emit`.
- 기존 `localdev_deploy` / `localdev_npm_install`은 `output()` 기반으로 그대로 두었다.

## 4. UX (적용됨)

| 항목 | 동작 |
|------|------|
| 위치 | 프로필 카드 하단 **로그 패널** (`npm i` 또는 `deploy`가 있는 카드만) |
| 표시 | 모노스페이스 스타일, 자동 아래로 스크롤 |
| 동시 실행 | 같은 프로필에서 deploy/npm 동시에 두 번 못 함(UI 버튼 비활성 + Rust `stream_busy`) |
| 새 실행 | 시작 시 패널 **클리어** 후 다시 쌓음 |
| 상한 | 약 **500줄** 또는 **256KB** 넘으면 앞줄부터 제거 |

패널 상단에 **민감 정보·화면 공유 주의** 문구를 둔다.

## 5. Windows `cmd /C npm …`

- 파이프는 `cmd` 자식에 연결한다.
- **stdout·stderr를 동시에 읽는다**(데드락 방지).

## 6. 보안·운영

- `args_are_safe`, 허용 `program`, `cwd`가 repo 내부인지 등 기존 검증 유지.
- 로그에 토큰·경로가 섞일 수 있음 → UI·이 문서에 경고.
- 로그는 기본적으로 **디스크에 저장하지 않음**.

## 7. 후속 (선택)

- 로그 복사 버튼, 스크롤 일시 정지, 실패 시 토스트에 마지막 N줄 요약 등.

## 8. 관련 코드·설정

- Rust: `apps/karmolab-tauri/src-tauri/src/local_dev.rs`
- UI: `apps/karmolab/src/widgets/servermonitor.ts`
- 설정: `apps/karmolab/data/servermonitor-config.json`의 `deployArgs`, `npmInstall`

---

## 문서 보완

- [x] 이벤트 이름·페이로드 스키마 반영
- [x] `local-dev-runner.md` 사용자 안내 반영
- [ ] 스크린샷 또는 짧은 GIF(선택)
