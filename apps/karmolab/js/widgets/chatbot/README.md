# chatbot 위젯

AI 채팅 UI는 **`chatbot.js` 단일 파일**로 동작합니다. CSS는 `Mdd.injectCSS('chatbot', ...)`로 같은 파일에서 주입하고, 세션·마크다운·전송 로직도 모두 여기에 있습니다.

## 로드

`apps/karmolab/js/widgets-lazy-meta.js` 에서:

```text
lazyScriptPaths: ['chatbot/chatbot']
```

다른 `widgets/chatbot/*.js` 파일은 로드되지 않습니다.

## 공개 API

- `Toolbox.register` 로 탭 `chatbot-main` 등록
- 스트리밍/캐릭터/프리셋 등은 `window._cb` (파일 하단 참고)

## 기본 캐릭터 시드

첫 실행 시 `카레 (비서)` 한 명만 넣고, 이후 로드마다 id 기준으로 없으면 병합: `c_mascot_yon`(욘), `c_mascot_alisa`(알리사), `c_mascot_ling`(링). 컨셉은 이미지 생성 [`imagegen/presets.js`](../imagegen/presets.js) 의 `CHARACTER_PRESETS`(마녀 욘·메이드 알리사·강시 링)와 맞춤.

## 과거 분리 모듈

예전에 `session.js`, `ui.js`, `actions.js`, `styles.js`, `markdown.js`, `presets.json` 등으로 나뉘어 있었으나, 실제 번들에 포함되지 않아 유지보수 혼선만 주므로 제거되었습니다. 프리셋 문구는 `chatbot.js` 내 `SYSTEM_PROMPT_PRESETS` 를 수정합니다.
