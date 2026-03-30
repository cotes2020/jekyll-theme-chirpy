## KarmoWorld (세계관 SSoT)

캐릭터 등 개체는 **`wiki/` 아래 Markdown 한 파일**에 사람이 읽는 본문과, 도구가 쓰는 필드를 **같은 파일의 YAML frontmatter**에 둡니다. 이미지 생성·챗봇은 앱이 로드할 때 `load-characters-from-wiki.js`가 MD를 가져와 파싱해 `KarmoWorld.bindings`에 채웁니다.

### 구조

- `wiki/entities/characters/*.md` — 캐릭터 문서 + 메타(frontmatter)
- `parse-md.js` — frontmatter 분리, 제한된 YAML 파싱
- `load-characters-from-wiki.js` — 위 MD fetch → `entities.characters`, `bindings.imagegen`, `bindings.chatbot` 구성
- `artifacts/` — 향후 아이템·문헌 등 (위키 경로 `wiki/artifacts/...` 예정)

### Frontmatter 키 (캐릭터)

- 공통: `entityId`, `slug`, `nameKo`, `nameEn`, `oneLine`, `aliases`, `tags`, `title`, `kind`, `type`
- 이미지: `imagegen_presetId`, `imagegen_icon`, `imagegen_label`, `imagegen_shortLabel`, `imagegen_prompt` (`|` 블록)
- 챗봇: `chatbot_id`, `chatbot_name`, `chatbot_userName`, `chatbot_userNote`, `chatbot_visualDescription`, `chatbot_description`, `chatbot_personality`, `chatbot_scenario`, `chatbot_firstMes`

### 원칙

- 문서와 데이터를 **한 파일**에서 맞추고, 도구 전용 JS에 같은 문구를 복제하지 않습니다.
- 로컬에서 `file://` 로 열면 fetch가 막힐 수 있으므로, 개발 시에는 정적 서버나 GitHub Pages 경로로 확인합니다.
