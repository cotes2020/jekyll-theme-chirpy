# chatbot 위젯

AI 채팅 UI는 여러 스크립트로 나뉘어 지연 로드되며, **`chatbot.js`** 가 세션·전송·`window._cb` 를 담당합니다.

## 로드

`apps/karmolab/js/widgets-lazy-meta.js` 에서:

```text
lazyScriptPaths: ['world/world', 'world/parse-md', 'world/load-characters-from-wiki', 'chatbot/styles', 'chatbot/markdown', 'chatbot/characters', 'chatbot/karmo-image', 'chatbot/prompt', 'chatbot/chatbot']
```

- `styles.js` — `Mdd.injectCSS('chatbot', …)`
- `markdown.js` — `window.ChatbotMarkdown.renderMarkdown`
- `characters.js` — `window.ChatbotCharacters` (캐릭터 저장·PNG/JSON 가져오기·모달 UI)
- `karmo-image.js` — `window.ChatbotKarmoImage` (KARMO_IMAGE 태그·캐릭터 이미지 생성)
- `prompt.js` — `window.ChatbotPrompt` (`SYSTEM_PROMPT_PRESETS`, `assembleSystemPrompt`)
- `chatbot.js` — 메인 로직

## 공개 API

- `Toolbox.register` 로 탭 `chatbot-main` 등록
- 스트리밍/캐릭터/프리셋 등은 `window._cb` (파일 하단 참고)

## 기본 캐릭터 시드

첫 실행 시 `카레 (비서)` 한 명만 넣고, 이후 로드마다 id 기준으로 없으면 병합: `c_mascot_yon`(욘), `c_mascot_alisa`(알리사), `c_mascot_ling`(링). 문구·프롬프트 단일 출처는 [`world/wiki/entities/characters/`](../../../world/wiki/entities/characters/) Markdown frontmatter이며, 로드 시 `load-characters-from-wiki.js`가 `KarmoWorld.bindings.chatbot`에 넣습니다. 실패 시 `characters.js`의 내장 기본값을 씁니다.

## 프리셋 문구

역할·톤 프리셋은 `prompt.js` 의 `SYSTEM_PROMPT_PRESETS` 를 수정합니다.
