# 로컬 데브 러너 (Tauri 전용)

KarmoLab **데스크톱 앱**(Tauri) 안에서만 쓰는 기능이에요. 브라우저로 GitHub Pages만 열면 `서버 모니터`에 **로컬 실행** 탭이 나오지 않아요.

> **원본 파일:** `apps/karmolab/js/widgets/docs/local-dev-runner.md` (GitHub에서 직접 열어도 동일)

---

## 앱 실행하기

터미널에서:

```bash
cd apps/karmolab-tauri
npm install
npm run dev
```

- `npm run dev`: 저장소 루트에서 Jekyll(포트 4000)을 띄운 뒤, **`/karmolab/` 가 HTTP로 응답할 때까지** 기다렸다가 Tauri가 같은 URL을 엽니다. 사이트가 크면 **첫 `jekyll serve` 생성에 수 분** 걸릴 수 있어요(스크립트는 최대 약 10분까지 대기).
- 이미 Jekyll만 다른 터미널에서 돌리고 있으면: `npm run dev:app` (Tauri만).

빌드·원격 URL 등은 **문서 → 프로젝트 명령** 탭과 `apps/karmolab-tauri/README.md` 를 참고하세요.

---

## 위젯에서 쓰는 법

1. 앱에서 **서버 모니터** 위젯을 엽니다.
2. **로컬 실행** 탭으로 갑니다.
3. **프로젝트(저장소) 루트 경로**에 이 레포의 최상위 폴더를 넣습니다. (예: Windows `C:\Users\…\Mascari4615.github.io`)
4. **루트 저장**을 누릅니다. 값은 WebView `localStorage`와 Rust 쪽 상태에 같이 반영됩니다.
5. 표에서 **시작** / **종료** / (해당 시) **npm install** 을 사용합니다.

---

## 프로필 설정 (`devProfiles`)

- 명령 목록은 **`apps/karmolab/data/servermonitor-config.json`** 의 **`devProfiles`** 배열만 읽습니다.
- UI는 **프로필 `id`** 만 Rust에 넘기고, `program`·`args`·`cwd`는 Rust가 그 JSON에서 다시 읽어 검증합니다.
- **허용 `program`:** `npm`, `npx`, `bundle`, `ruby`, `node` (및 확장자 변형).
- **`cwd`:** 레포 루트 기준 상대 경로(예: `.`, `apps/discord-bots`). 반드시 루트 **아래** 실제 폴더여야 합니다.
- **`npmInstall: true`:** 그 프로필에 **npm install** 버튼이 보이고, 해당 `cwd`에서 동기 실행됩니다.
- **`healthUrl`:** (선택) 로컬 실행 탭에서 응답 여부를 짧게 ping 해 보는 데 씁니다. **상태** 탭의 `localMonitors` 와 맞춰 두면 한눈에 보기 좋아요.

---

## 동작·제한

- **시작:** 백그라운드 프로세스로 실행합니다(표준 출력은 숨깁니다).
- **종료 (Windows):** `taskkill /T /F` 로 프로세스 트리를 끊습니다.
- **종료 (Linux/macOS):** 부모 PID에 `kill`을 보내는 수준이라, 자식이 남을 수 있어요. 필요하면 `pkill` 등으로 정리하세요.
- **앱 완전 종료:** Rust 쪽 PID 추적이 초기화됩니다. 남은 프로세스는 작업 관리자 등으로 확인하세요.
- **Windows:** `npm` / `npx` 는 내부적으로 `cmd /C` 로 호출합니다. **Node·Ruby(`bundle`)** 가 사용자 PATH에 있어야 합니다.

---

## 관련 문서

- 터미널 명령 모음: **문서 → 프로젝트 명령**
- Tauri 앱 빌드·업데이트·트레이: `apps/karmolab-tauri/README.md`
- Tauri 업데이트·릴리스 체크리스트: **문서 → 로드맵** (`roadmap.md` 의 「Tauri 데스크톱 · 자동 업데이트」)
