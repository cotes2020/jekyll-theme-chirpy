# 프로젝트 통합 명령·경로 가이드

이 레포(`Mascari4615.github.io`) 안에서 자주 쓰는 **터미널 명령**과 **로컬 URL**을 한곳에 모았습니다. 블로그(Jekyll), KarmoLab, 부가 앱(`apps/*`)을 오갈 때 **복사해서 실행**하면 됩니다.

> **원본 파일:** `apps/karmolab/js/widgets/docs/project-commands-guide.md` (GitHub에서 직접 열어도 동일)

---

## 레포 구조 한눈에

| 경로 | 역할 |
|------|------|
| 루트 `/` | **Jekyll + Chirpy** 블로그, 테마용 `npm`(Rollup·PurgeCSS 등) |
| `apps/karmolab/` | **KarmoLab** 웹앱(TypeScript → `js/` 빌드 산출) |
| `apps/karmolab-tauri/` | KarmoLab **데스크톱**(Tauri + 로컬 Jekyll 연동) |
| `apps/discord-bots/` | Discord 봇 워크스페이스(여러 패키지) |
| `apps/chat-overlay/` | 방송용 오버레이(Tauri + Vite) |
| `apps/karmo-web-extension/` | Chrome 확장(MV3) |
| `tools/run.sh` | Jekyll `serve` 래퍼(증분·호스트 옵션) |

세부 README: 각 `apps/*/README.md` 및 루트 `README.md`(업스트림 Chirpy 안내).

---

## 블로그: Jekyll

Ruby 의존성은 **`Gemfile`** 기준으로 루트에서 한 번 설치합니다.

```bash
cd /path/to/mascari4615.github.io
bundle install
```

### 로컬 미리보기 (가장 흔함)

증분 빌드 + 라이브 리로드:

```bash
bundle exec jekyll serve --host 127.0.0.1 --port 4000 --livereload --incremental
```

- 사이트 루트: **http://127.0.0.1:4000/**
- KarmoLab: **http://127.0.0.1:4000/karmolab/**

**래퍼 스크립트** (Git Bash / WSL / macOS / Linux):

```bash
bash tools/run.sh
bash tools/run.sh --production   # JEKYLL_ENV=production (예: SW·프로덕션 전용 동작 확인)
```

### 프로덕션 모드로 serve (PowerShell 예)

```powershell
$env:JEKYLL_ENV = "production"
bundle exec jekyll serve --host 127.0.0.1 --port 4000 --livereload --incremental
```

### 정적 사이트만 생성

```bash
bundle exec jekyll build
```

출력은 기본 `_site/` (Chirpy 설정 따름).

---

## 블로그: 루트 npm (테마 JS/CSS·그래프)

Chirpy 테마의 **프런트 빌드**와 포스트 그래프 생성입니다. CI와 맞추려면 Jekyll 전에 자주 돌립니다.

```bash
cd /path/to/mascari4615.github.io
npm ci
npm run site:prep
```

- `npm run site:prep` = `npm run build` + `npm run build:graph`
- `npm run build` = CSS/JS 동시 빌드(`concurrently`)
- `npm run test` = ESLint + Stylelint

개발 시 JS만 감시:

```bash
npm run watch:js
```

---

## KarmoLab: TypeScript → `js/` 빌드

위젯·툴박스 소스는 `src/`, 배포물은 `js/`입니다. **KarmoLab 코드를 수정했다면** Jekyll 전에 빌드하세요.

```bash
cd apps/karmolab
npm install
npm run build
```

타입만 검사:

```bash
npm run typecheck
```

---

## KarmoLab: Jekyll 없이 정적 파일만 띄우기

`apps/karmolab/index.html` 주석과 같이, **Jekyll 빌드 없이** 파일만 서빙해 볼 때 씁니다. 상단 프론트매터가 HTML에 그대로 보일 수 있으나 **스크립트는 대체로 동작**합니다.

**Python 3** (레포 루트에서):

```bash
cd /path/to/mascari4615.github.io
python -m http.server 8899
```

브라우저: **http://localhost:8899/apps/karmolab/index.html**

**Node** (정적 서버 예):

```bash
npx --yes serve -l 8899
```

그다음 같은 경로로 `index.html`을 열면 됩니다.

---

## KarmoLab Tauri (데스크톱)

Rust·Tauri·WebView2 전제. 자세한 설명은 **`apps/karmolab-tauri/README.md`**.

```bash
cd apps/karmolab-tauri
npm install
npm run dev
```

- Jekyll을 **4000**에서 띄운 뒤 Tauri가 **http://127.0.0.1:4000/karmolab/** 를 연다.
- 이미 Jekyll이 돌아가 중이면: `npm run dev:app`
- WebView만 **배포 URL**로 테스트: `npm run dev:remote`
- 설치 패키지 빌드: `npm run build`
- 데스크톱 앱에서 **KarmoLab → 서버 모니터** 화면 아래 **로컬 프로세스** 영역에서 Jekyll·Discord 봇 등 프로필 시작·종료·`npm install`: **문서 → 데스크톱·로컬** 탭 (`apps/karmolab/js/widgets/docs/local-dev-runner.md`).

---

## Discord 봇 워크스페이스

슬래시 명령 목록이 아니라 **워크스페이스 진입·빌드**만 여기 둡니다. 세부 스크립트·환경 변수는 **`apps/discord-bots/README.md`** 와 각 앱 README를 보세요.

```bash
cd apps/discord-bots
npm install
npm run build
npm run start:yawnbot
npm run deploy:yawnbot
```

---

## 기타 `apps/`

| 앱 | 대표 명령 |
|----|-----------|
| `chat-overlay` | `cd apps/chat-overlay` → `npm install` → `npm run tauri:dev` |
| `karmo-web-extension` | Chrome에서 **압축해제된 확장 로드** → 폴더 선택 (`README.md` 참고) |

---

## 유용한 링크

| 설명 | URL |
|------|-----|
| 이 레포 | https://github.com/Mascari4615/Mascari4615.github.io |
| Jekyll 문서 | https://jekyllrb.com/docs/ |
| Chirpy 테마 위키 | https://github.com/cotes2020/jekyll-theme-chirpy/wiki |
| Tauri v2 | https://v2.tauri.app/ |

---

## KarmoLab에서 이 문서 열기

**[KarmoLab](https://mascari4615.github.io/karmolab/)** → **문서** → **프로젝트 명령** 탭.
