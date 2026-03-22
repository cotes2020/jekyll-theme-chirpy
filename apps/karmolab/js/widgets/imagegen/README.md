# imagegen 위젯

AI 이미지 생성 위젯. 모듈별로 분리되어 유지보수하기 쉽게 구성됨.

## 파일 구조

| 파일 | 역할 |
|------|------|
| `presets.js` | CONTEXT_PRESETS, CHARACTER_PRESETS, VIBE_OPTIONS 등 프리셋 데이터 |
| `config.js` | GALLERY_SESSION_KEY, PROMPT_HISTORY_KEY 등 설정 상수 |
| `styles.js` | Mdd.injectCSS로 주입하는 CSS |
| `core.js` | 큐 시스템, 유틸(showLightbox, downloadImage), 히스토리 |
| `imagegen.js` | 메인 엔트리 - buildMain, UI, window._ig, Toolbox.register |

## 로드 순서

manifest에서 다음 순서로 로드됨:

1. presets → 2. config → 3. styles → 4. core → 5. imagegen

## 네임스페이스

모든 모듈이 `window.ImageGen` (IG)에 값을 추가함.

- presets: CONTEXT_PRESETS, CHARACTER_PRESETS, getSlotsFromPrompt, getCharacterOptions 등
- config: GALLERY_SESSION_KEY, PROMPT_HISTORY_MAX 등
- core: queue, enqueue, processQueue, showLightbox, initCore 등
- imagegen: state, buildMain, window._ig

## core 초기화

`imagegen.js`의 buildMain에서 `IG.initCore({ state, renderQueue, ... })` 호출.
core의 processQueue 등이 이 deps를 통해 UI 업데이트를 수행함.
