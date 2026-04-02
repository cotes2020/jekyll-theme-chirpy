## KarmoWorld (세계관 SSoT)

캐릭터 등 개체는 **메타는 `*.yaml`**, 사람이 읽는 본문은 **같은 슬러그의 `*.md`(front matter 없음)**에 둡니다. 이미지 생성·챗봇은 앱이 로드할 때 `load-characters-from-wiki.js`가 YAML·MD를 가져와 파싱해 `KarmoWorld.bindings`에 채웁니다.

### 구조

- `wiki/entities/characters/{slug}.yaml` — 도구용 필드(이미지·챗봇·엔티티 메타)
- `wiki/entities/characters/{slug}.md` — 위키 본문만 (YAML `---` 블록 없음)
- `parse-md.js` — 제한된 YAML 파싱, 단일 MD front matter 또는 yaml+md 분리
- `load-characters-from-wiki.js` — `{slug}.yaml` + `{slug}.md` fetch → `entities.characters`, `bindings.imagegen`, `bindings.chatbot` 구성
- `artifacts/` — 향후 아이템·문헌 등 (위키 경로 `wiki/artifacts/...` 예정)

### YAML 키 (캐릭터)

- 공통: `entityId`, `slug`, `nameKo`, `nameEn`, `oneLine`, `aliases`, `tags`, `title`, `kind`, `type`
- 이미지: `imagegen_presetId`, `imagegen_icon`, `imagegen_label`, `imagegen_shortLabel`, `imagegen_prompt` (`|` 블록)
- 챗봇: `chatbot_id`, `chatbot_name`, `chatbot_userName`, `chatbot_userNote`, `chatbot_visualDescription`, `chatbot_description`, `chatbot_personality`, `chatbot_scenario`, `chatbot_firstMes`

### 원칙

- 메타와 본문을 **같은 슬러그**로 맞추고, 도구 전용 JS에 같은 문구를 복제하지 않습니다.
- `.md`에 front matter를 넣지 않으면 Jekyll이 해당 파일을 **페이지로 변환하지 않고** `_site`에 원본 경로로 복사합니다. 별도 `exclude`나 빌드 후 복사 단계가 필요 없습니다.
- 로컬에서 `file://` 로 열면 fetch가 막힐 수 있으므로, 개발 시에는 정적 서버나 GitHub Pages 경로로 확인합니다.

### Jekyll 배포

`bundle exec jekyll build` 한 번이면 `apps/karmolab/world/wiki/` 아래 `.yaml`·본문만 `.md`가 `_site`에 포함됩니다.
