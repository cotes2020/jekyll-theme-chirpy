# KarmoLab (Tauri)

데스크톱에서 시스템 WebView로 [KarmoLab](https://mascari4615.github.io/karmolab/)을 여는 얇은 셸입니다. UI는 GitHub Pages에 배포된 웹과 동일합니다.

## 준비물

- [Rust](https://www.rust-lang.org/learn/get-started) + [Tauri 사전 요구사항](https://v2.tauri.app/start/prerequisites/) (Windows: Visual Studio Build Tools, WebView2)

## 명령

```bash
cd apps/karmolab-tauri
npm install
npm run dev   # 개발 실행 (내부적으로 tauri dev)
npm run build # 설치 패키지 빌드
```

로컬 Jekyll로 테스트하려면 `src-tauri/tauri.conf.json`의 `build.devUrl`을 예: `http://127.0.0.1:4000/karmolab/` 로 바꾼 뒤 `npm run dev`를 실행하세요.

## 데스크톱 전용

- Tauri가 주입: `window.__KARMOLAB_DESKTOP__ === true` (웹 브라우저에는 없음)
- KarmoLab 스크립트: `Toolbox.isDesktopApp()` 로 동일 분기
- **단일 인스턴스**: 앱을 다시 실행하면 새 창 대신 기존 창이 앞으로 옵니다(최소화 해제 포함).
- **창 닫기(X)**: 앱이 종료되지 않고 트레이로 숨겨집니다. 완전 종료는 트레이 메뉴의 **종료**만 사용하세요.
- 트레이: **왼쪽 클릭**으로 메뉴(창 보이기 / 브라우저에서 열기 / 종료). Windows에서는 트레이 아이콘 **왼쪽 더블클릭**으로 메뉴 없이 창만 앞으로.
- 헤더에 **브라우저** 링크(앱 전용): 기본 브라우저로 같은 사이트를 엽니다.
- **OS 알림**: `showToast`가 에러·성공을 띄울 때(일부 짧은 성공 메시지 제외) 데스크톱 알림으로도 보냅니다. Windows에서는 알림 권한·집중 모드 설정에 따라 표시가 달라질 수 있습니다.

## 사이드바에 「디버그」가 없을 때

**브라우저에서는 보이는데 앱에서만 안 보일 때**는 WebView2가 `toolbox.js` 등을 **예전 버전으로 캐시**한 경우가 많습니다. 앱은 `devtools.js`를 **Rust에 포함**해 `Toolbox.init` 직전에 한 번 더 주입하므로, 앱을 **다시 빌드**(`npm run dev` / `npm run build`)하면 최소한 디버그 위젯은 캐시와 무관하게 붙습니다.

그 외에는: 앱이 **GitHub Pages**를 읽는다면 로컬만 고친 내용은 **푸시·배포 후**에 반영됩니다. 배포 후에도 옛 JS면 WebView2 캐시를 지우거나 `devUrl` URL에 `?v=1` 등으로 쿼리를 바꿔 보세요. 로컬 Jekyll은 `devUrl`을 `http://127.0.0.1:4000/karmolab/`로 두고 `npm run dev`로 확인하면 됩니다.
