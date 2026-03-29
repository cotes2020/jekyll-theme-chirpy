# KarmoLab (Tauri)

데스크톱에서 시스템 WebView로 [KarmoLab](https://mascari4615.github.io/karmolab/)을 여는 얇은 셸입니다. UI는 GitHub Pages에 배포된 웹과 동일합니다.

## 준비물

- [Rust](https://www.rust-lang.org/learn/get-started) + [Tauri 사전 요구사항](https://v2.tauri.app/start/prerequisites/) (Windows: Visual Studio Build Tools, WebView2)

## 명령

```bash
cd apps/karmolab-tauri
npm install
npm run dev    # tauri dev
npm run build  # 설치 패키지 빌드
```

로컬 Jekyll로 테스트하려면 `src-tauri/tauri.conf.json`의 `build.devUrl`을 예: `http://127.0.0.1:4000/karmolab/` 로 바꾼 뒤 `npm run dev`를 실행하세요.

## 데스크톱 전용

- Tauri가 주입: `window.__KARMOLAB_DESKTOP__ === true` (웹 브라우저에는 없음)
- KarmoLab 스크립트: `Toolbox.isDesktopApp()` 로 동일 분기
- 트레이: 왼쪽 클릭으로 창 표시, 우클릭으로 메뉴(창 보이기 / 종료)
