# 로드맵 & 기획

제품 방향, 실행 계획, 데스크톱 배포 메모를 한곳에 모았어요.

---

## 제품 로드맵

- 연구소 안정도 표시
- 스토리 로그 (마스코트 우클릭)
- SEO/메타: MVP 툴 설명·검색 키워드 정리
- 티메토 대사: `mdd.js`의 `LINE_PRESETS` · `Mdd.linePreset()` 로 관리 (위젯에서 `msg`/`mood` 덮어쓰기)

---

## Tauri 데스크톱 · 자동 업데이트

트레이 **「업데이트 확인…」**은 GitHub Releases의 `latest.json` manifest를 보고, 새 버전이 있으면 내려받아 설치합니다. **당장 릴리스를 안 돌릴 때**는 아래 순서만 보면 됩니다.

1. 버전 맞추기: `apps/karmolab-tauri/src-tauri/tauri.conf.json`의 `version`과 `src-tauri/Cargo.toml`의 `version`.
2. 서명 후 빌드: `TAURI_SIGNING_PRIVATE_KEY` (필요 시 `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`). PowerShell 예: `$env:TAURI_SIGNING_PRIVATE_KEY = "...\karmolab-updater.key"` → 앱 폴더에서 `npm run build`.
3. GitHub Release: 설치 파일 + 짝 `.sig` + 에셋 이름 **`latest.json`**. 내용은 `apps/karmolab-tauri/updater/latest.json.example` 참고. URL은 `tauri.conf.json`의 `plugins.updater.endpoints`와 일치.
4. (선택) CI: `.github/workflows/karmolab-tauri.yml`에 서명 시크릿 + `tauri build` + Release 업로드.

상세는 `apps/karmolab-tauri/README.md` 「앱 내 업데이트」·[Tauri Updater](https://v2.tauri.app/plugin/updater/) 문서.

---

## 기획 (Director Notes)

> 유틸 + 미니게임/킬링타임 + 서브컬처(MDD)

### 확정 사항

- **브랜드**: **KarmoLab**
- **마스코트**: **티메토(Timeto)** — 연보라 머리, 소녀 연구소장
- **톤**: 밝은 연구소 + HUD(계측) + 청량한 학원 연구부 느낌

### 비전

**서브컬처 감성의 마스코트가 안내하는, 123apps급 유틸 + 미니게임/킬링타임 허브.**

### MDD 운영 원칙

- 기능은 마스코트 “실험/장비”로 의인화
- UI 피드백은 표정·대사·효과로 (말투: 1인칭 “저”, 호칭 “조수님”, 존댓말)

### MVP 장비 목록 (요약)

**유틸**

1. **텍스트 계측기** — 문자·바이트·줄·공백 제외 등
2. **비밀번호 생성기** (기존) — 옵션·복사 UX
3. **카운트다운** (기존/확장) — 타임스탬프 ↔ 날짜
4. **GIF 실험기** — 프레임 추출·최적화
5. **짤 생성기** — 템플릿·자막 프리셋

**프로필 / 세계관**

6. **연구 노트** (기존 “내 정보” 확장) — 조수 닉네임, 안정도, 업적, 설정

*이미 있는 툴(암호화·이미지 변환 등)은 MVP 표에 안 적어도 사이트에 포함. 필요 시 라벨·설명만 티메토 톤으로 맞춤.*

### 티메토 캐릭터 시트

- **외형**: 연보라 머리, 흰 연구복, HUD 고글
- **말투**: 실험·측정 용어 자연스럽게 사용
- **감정·대사 프리셋** (기본 문구는 `LINE_PRESETS`, 코드에서 `Mdd.linePreset('success' | 'error' | 'tool_run' | …)`)

| 상황 | 포즈 | 기본 대사(요지) |
|------|------|-----------------|
| 첫 방문 | pointing | 환영, 조수님 |
| 데일리/허브 | happy | 오늘 실험·즐겨찾기 안내 |
| 도구 실행 | think | 측정 개시 |
| 성공 | cheer | 샘플 확보·기록 |
| 실패 | sad | 장비 삐끗 |
| 데이터 경고 | angry | 중요 데이터 확인 |
| 방치 | sleep | zzZ… 조수님…? |
| 반응 | shock | 돌아오셨군요 |
| 업적 | love | 연구소 안정 |
| 짤·드립 | smug | 명작 될지도 |

### 신규·확장 위젯 스펙 (상세)

#### 텍스트 계측기

- **입력**: textarea 1
- **출력(실시간)**: 문자수, 공백 제외, UTF-8 바이트, 줄수, 단어수
- **부가**: 복사, 티메토 반응 (`measure_done` 등)
- **레이아웃**: form

#### GIF 실험기

- **입력**: GIF 업로드(드래그 앤 드롭)
- **기능**: 프레임 목록, PNG보내기, 간격 조절, 용량 최적화(“다이어트”)
- **출력**: 최적화 GIF / 프레임 ZIP
- **레이아웃**: wide

#### 짤 생성기

- **입력**: 이미지 업로드 또는 템플릿 + 캡션
- **캡션**: 프리셋 여러 종(테두리·자막·말풍선 등)
- **출력**: PNG 다운로드, 클립보드 (워터마크 옵션)
- **레이아웃**: wide

#### 연구 노트 (내 정보 확장)

- **추가**: 조수 닉네임, 연구소 안정도(수치·바), 업적, 해금 대사/스킨 미리보기
- **유지**: 테마, API 키, 사용량 등 기존 설정

### 기존 위젯 확장 포인트

- **암호화/복호화**: 카드 설명·라벨 티메토 톤
- **카운트다운**: D-day 대사 `linePreset` 연동
