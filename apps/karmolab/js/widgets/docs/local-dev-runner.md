# 로컬 데브 러너 (Tauri 전용)

KarmoLab **데스크톱 앱**(Tauri) 안에서만 쓰는 기능이에요. 일반 브라우저로 GitHub Pages만 열면 **서버 모니터** 위젯 자체가 메뉴에 없습니다.

> **원본 파일:** `apps/karmolab/js/widgets/docs/local-dev-runner.md` (GitHub에서 직접 열어도 동일)

---

## 앱 실행하기

터미널에서:

```bash
cd apps/karmolab-tauri
npm install
npm run dev
```

- `npm run dev`: 저장소 루트에서 **`python -m http.server 8899`** 로 정적 서버를 띄운 뒤 **`http://127.0.0.1:8899/apps/karmolab/`** 가 응답할 때까지 기다렸다가 Tauri가 그 URL을 엽니다(KarmoLab만; Jekyll 없음). **데스크톱 창만 닫아도 정적 서버는 그대로** 둡니다(다시 `npm run dev:app`만 실행하면 됨). 둘 다 끄려면 터미널에서 **Ctrl+C** 하세요.
- 이미 **8899**에 같은 방식으로 서버가 떠 있으면: `npm run dev:app` (Tauri만).
- **블로그·Jekyll까지** 로컬로 쓰려면: `npm run dev:with-jekyll` (4000번 Jekyll + 별도 `devUrl` 설정).
- Windows에서 Jekyll을 쓸 때 Listen이 **같은 폴더를 두 경로로 감시**한다고 에러를 내면, `_config.yml`의 `exclude`와 `dev:jekyll`의 **`--force_polling`** 을 참고하세요.

빌드·원격 URL 등은 **문서 → 프로젝트 명령** 탭과 `apps/karmolab-tauri/README.md` 를 참고하세요.

---

## 위젯에서 쓰는 법

1. 앱에서 상단 **데스크톱 앱** 메뉴 → **서버 모니터**를 엽니다.
2. **로컬** 블록에서 **프로젝트(저장소) 루트 경로**에 이 레포의 최상위 폴더를 넣습니다. (예: Windows `C:\Users\…\Mascari4615.github.io`)
3. **루트 저장**을 누릅니다. 값은 WebView `localStorage`와 Rust 쪽 상태에 같이 반영됩니다.
4. **새로고침**으로 **로컬** 카드(같은 **`id`** 의 `localMonitors` URL 응답 + `devProfiles` 프로세스가 **한 장**에 묶임)를 갱신합니다. **목록 새로고침**은 설정·추적 상태만 다시 읽고 URL ping은 하지 않습니다.
5. **환경 변수(.env):** **로컬** 아래 **환경 변수**에서 `servermonitor-config.json`의 **`envFiles`**에 적은 파일을 탐색기로 열거나, 앱 안에서 편집·저장할 수 있습니다. **저장소 루트**는 같은 **로컬** 블록 상단에서 먼저 저장해야 합니다.

---

## envFiles (선택)

- **`apps/karmolab/data/servermonitor-config.json`** 의 **`envFiles`** 배열에 `{ "label", "path", "hint?" }` 를 둡니다. `path`는 레포 루트 기준 **상대 경로**만 됩니다(상위 폴더 `..` 불가).
- 데스크톱 앱에서 **탐색기에서 표시**, 기본 앱으로 열기, **편집·저장**(임시 파일 후 이름 바꿈, 최대 512KB)을 제공합니다. 파일이 없으면 빈 편집기로 시작해 저장 시 새로 만들 수 있습니다.

---

## 프로필 설정 (`devProfiles`)

- 명령 목록은 **`apps/karmolab/data/servermonitor-config.json`** 의 **`devProfiles`** 배열만 읽습니다.
- UI는 **프로필 `id`** 만 Rust에 넘기고, `program`·`args`·`cwd`는 Rust가 그 JSON에서 다시 읽어 검증합니다.
- **허용 `program`:** `npm`, `npx`, `bundle`, `ruby`, `node` (및 확장자 변형).
- **`cwd`:** 레포 루트 기준 상대 경로(예: `.`, `apps/discord-bots`). 반드시 루트 **아래** 실제 폴더여야 합니다.
- **`npmInstall: true`:** 그 프로필에 **npm install** 버튼이 보이고, 해당 `cwd`에서 동기 실행됩니다.
- **`localMonitors`:** 항목마다 `title`·`subtitle`(선택)·`url`(선택)·`noHealthUrl`(의도적으로 ping 안 함) 등을 둘 수 있습니다. 예전처럼 `label`만 있어도 됩니다. **ATKUp** 봇은 기본 **`http://127.0.0.1:8081/health`** (`ATKUP_HEALTH_PORT`, 끄려면 `0`). 데스크톱에서는 **`devProfiles` 항목과 `id`가 같으면 카드 한 장**에 URL 상태와 시작·종료가 같이 나옵니다.
- **`healthUrl`:** (선택) `devProfiles` 전용. `localMonitors`와 주소를 맞춰 두면 카드 ping과 의미가 같아집니다.

---

## 동작·제한

- **시작:** 백그라운드 프로세스로 실행합니다(표준 출력은 숨깁니다).
- **종료 (Windows):** `taskkill /T /F` 로 프로세스 트리를 끊습니다.
- **종료 (Linux/macOS):** 부모 PID에 `kill`을 보내는 수준이라, 자식이 남을 수 있어요. 필요하면 `pkill` 등으로 정리하세요.
- **앱 완전 종료:** Rust 쪽 PID 추적이 초기화됩니다. 남은 프로세스는 작업 관리자 등으로 확인하세요.
- **Windows:** `npm` / `npx` / `bundle` / `ruby` 는 내부적으로 `cmd /C` 로 호출합니다(`.cmd`/배치 런처 호환). **Node·Ruby(`bundle`)** 가 사용자 PATH에 있어야 합니다.

---

## 관련 문서

- 터미널 명령 모음: **문서 → 프로젝트 명령**
- Tauri 앱 빌드·업데이트·트레이: `apps/karmolab-tauri/README.md`
- Tauri 업데이트·릴리스 체크리스트: **문서 → 로드맵** (`roadmap.md` 의 「Tauri 데스크톱 · 자동 업데이트」)
