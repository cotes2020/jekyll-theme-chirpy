# CLAUDE.md

This file provides guidance for AI assistants working with this repository.

## Project Overview

This is **mascari4615.github.io**, a Jekyll-based blog and personal portfolio site using the **Chirpy theme (v7.5.0)**. It is deployed to GitHub Pages at `https://mascari4615.github.io`. The repository also functions as a monorepo containing several companion applications.

- **Site title**: KarmoDDrine
- **Language**: Korean (`ko-KR`), Timezone: `Asia/Seoul`
- **Owner**: mascari4615 (mascari4615@gmail.com)

---

## Repository Structure

```
mascari4615.github.io/
├── _config.yml              # Main Jekyll configuration
├── _posts/                  # Blog content (see Content Structure below)
├── _tabs/                   # Navigation tab pages (about, archives, categories, tags, graph, works)
├── _layouts/                # Chirpy theme HTML layouts
├── _includes/               # Chirpy theme HTML partials
├── _sass/                   # SCSS source files
├── _data/                   # Locale and data files
├── _plugins/                # Jekyll Ruby plugins
├── _javascript/             # TypeScript/JS source (compiled to assets/js/dist/)
├── assets/                  # Static assets (images, CSS, JS)
├── apps/                    # Sub-applications (monorepo)
│   ├── karmolab/            # TypeScript/Node.js dashboard app
│   ├── karmolab-react-src/  # React frontend (Vite)
│   ├── karmolab-tauri/      # Tauri desktop app
│   ├── chat-overlay/        # Tauri overlay app
│   ├── discord-bots/        # Discord bot implementations
│   └── karmo-web-extension/ # Browser extension
├── packages/
│   └── karmolab-ai/         # Shared Google Cloud / Vertex AI utilities
├── scripts/                 # Build scripts (e.g., build-post-graph.cjs)
├── tools/                   # Release scripts
├── .github/workflows/       # CI/CD workflows
├── rollup.config.js         # JS bundler config
├── purgecss.js              # CSS purging config
├── tsconfig.json            # TypeScript config
├── eslint.config.js         # ESLint config
├── Gemfile                  # Ruby dependencies
└── package.json             # Node.js dependencies and scripts
```

---

## Content Structure (`_posts/`)

Posts are organized hierarchically by topic:

```
_posts/
├── computer/                # Technical / CS content
│   ├── algorithm/           # AI, backtracking, DP, etc.
│   ├── graphics/            # Animation, color, modeling, rendering, shader
│   ├── programming/         # Design patterns, paradigms, workflow
│   ├── software/            # Unity, third-party tools
│   ├── system/              # Assembly, memory, OS, processor, Windows
│   └── data-structure/
├── stone/                   # Personal / creative content
│   ├── diary/
│   ├── library/             # Anime/manga, blog, game, music, misc
│   └── think/               # Personal, strategy, theory
└── works/                   # Portfolio / project documentation
    ├── art/
    ├── game-dev/
    │   ├── witch-mendokusai/  # Main game project (dev-log, game-design, world)
    │   └── maplestory-clone/
    ├── karmo-lab/
    └── virtual/             # VRChat / metaverse projects
```

---

## Post Conventions

### File Naming

- Published posts: `YYYY-MM-DD-slug-name.md` (kebab-case)
- Drafts: `slug-name-DRAFT.md` or `YYYY-MM-DD-slug-name-draft.md`
- Templates: `slug-name-TEMPLATE.md`

### Front Matter

```yaml
---
title: "Post Title"
description: "Short description of the post."
categories: [Category1, Category2]
tags: [tag1, tag2]
image: "/assets/img/background/image.jpg"
hidden: true   # omit or set false to show in listings

date: YYYY-MM-DD. HH:MM
# last_modified_at: YYYY-MM-DD. HH:MM   # Uncomment/update on content changes
---
```

- **Date format**: `YYYY-MM-DD. HH:MM` (note the period and space before time)
- **`last_modified_at`**: Update only when content changes meaningfully. Typo fixes, ordering changes, and formatting changes do NOT count.
- **Categories/Tags**: Use Korean or English consistently per the post's context. Category and Tag labels are always English; heading section labels are Korean.
- **`hidden: true`**: Hides the post from list pages (used for internal/reference posts).

### `last_modified_at` Tracking Pattern

Previous modification dates are kept as commented-out lines with brief descriptions:

```yaml
# last_modified_at: 2024-10-22. 11:27 # Init
# last_modified_at: 2025-03-15. 10:34 # 말머리 -> 머리말
last_modified_at: 2025-05-02. 01:35 # 절제, 통일
```

The `_plugins/posts-lastmod-hook.rb` auto-detects `last_modified_at` from git history for posts with more than one commit.

### Content Style

- **머리말 (Introduction)**: Opens each post — states the topic, purpose, and direction.
- **꼬리말 (Conclusion)**: Optional closing section.
- **메모 (Memo)**: Raw notes; includes 참고 (references), 키워드 (keywords), 도토리 (undigested raw ideas), 기록 (historical records).
- **Links**: Internal links as `[title](link)`, external links as `['source': 'title'](link)`.
- **Horizontal rules**: Always follow with a blank line.
- **Time stamps in content**: Use format `2025-05-01. 13:45`. Image filenames use `250501-134500`.
- **Writing style**: Concise and restrained (절제). Remove redundant or obvious information. Avoid unnecessary decorative emoji in titles/categories.

---

## Development Workflow

### Prerequisites

- Ruby ~3.1 (3.4 recommended)
- Node.js 20+
- Bundler

### Setup

```bash
bundle install
npm install
```

### Building Assets

```bash
npm run build          # PurgeCSS + Rollup (CSS + JS) in parallel
npm run build:css      # CSS only (PurgeCSS)
npm run build:js       # JS only (Rollup, production)
npm run watch:js       # JS watch mode for development
npm run build:graph    # Generate post graph data
npm run site:prep      # Full prep: build + graph
```

### Running Locally

```bash
bundle exec jekyll serve         # Serve with live reload
bundle exec jekyll serve --draft # Include draft posts
bundle exec jekyll b             # Build only
```

### Linting

```bash
npm run test           # JS + SCSS lint
npm run lint:js        # ESLint only
npm run lint:scss      # Stylelint only
npm run lint:fix:scss  # Auto-fix SCSS issues
```

### Sub-Application Builds

Each app has its own `npm install` + `npm run build` workflow:

```bash
# KarmoLab React app
cd apps/karmolab-react-src && npm ci && npm run build
# Requires VITE_GOOGLE_CLIENT_ID env var

# KarmoLab TypeScript app
cd apps/karmolab && npm ci && npm run build

# Tauri apps: managed via GitHub Actions (karmolab-tauri.yml)
```

---

## CI/CD Workflows (`.github/workflows/`)

| Workflow | Trigger | Description |
|---|---|---|
| `verify.yml` | Push to `main`/`master`, PR | **Master invariant 단일 게이트** — `npm run verify` (apps/karmolab build + packages/karmolab-ai build + apps/karmolab-tauri cargo check) + typos. apps/blog lint 은 자동화 빚 (config 누락) — follow-up. branch protection 의 required status check 로 `verify (master invariant)` 등록 (사용자 액션). 폐기 흡수: `ai-quality.yml`, `code-quality.yml`, `karmolab-ts.yml`, `karmolab-tauri.yml`. |
| `pages-deploy.yml` | Push to `main`/`master`, manual | Full site build and deploy to GitHub Pages |
| `karmolab-tauri-release.yml` | Tag/manual | Tauri auto-update release pipeline |
| `auto-merge.yml` | PR | Auto-merge after checks pass |
| `claude.yml` | issue_comment | Claude Code Action |

### Deployment Pipeline Steps

1. Checkout (full history for lastmod detection)
2. Setup Ruby 3.4 + Node.js 20
3. `npm ci` (root dependencies)
4. `npm run build` (CSS purge + JS bundle)
5. `npm run build:graph` (post graph data)
6. Build KarmoLab React app
7. Build KarmoLab TypeScript app
8. `bundle exec jekyll b` (production)
9. Upload and deploy to GitHub Pages

---

## Git Workflow

본 § 가 본 레포 git workflow 의 정본. CodeRabbit 이 자동 픽업해 같은 룰로 PR 리뷰.

### Branches

- `master` / `main`: Active development source. **직접 push 금지** (예외 단락 참조).
- `production`: Release branch (triggers semantic-release). 직접 push 절대 금지.
- Feature/fix branches: `feature/<주제>`, `fix/<주제>`, `chore/<주제>`, `refactor/<주제>`

### 브랜치 + PR 강제 (AI Native 게이트)

작업마다 브랜치 분기 후 PR. **첫 커밋 시 바로 Draft PR 생성** +
`.github/pull_request_template.md` 의도 채움 → push → CodeRabbit 코멘트 대응 →
완료 시 PR 리뷰 후 `master` 머지.

### Master 직접 push — 차단됨

Branch Protection (아래 § 참조) 으로 PR 강제 + Include administrators + `verify (master invariant)` required. **모든 master 변경은 PR 통해야 함** — 1~3줄 chore / 응급 fix 도 예외 X. 응급 시 PR 만들고 review 0 + verify 통과 즉시 머지 (review 강제 0). 본 단락의 옛 「예외」 룰은 자동화 강제로 폐기됨.

### Commit Messages

Conventional Commits are enforced via commitlint (`.husky/commit-msg`):

```
feat: add new feature
fix: resolve a bug
perf: performance improvement
refactor: code restructuring (no behavior change)
docs: documentation update
chore: maintenance task
style: formatting/style change
test: add or update tests
```

- **Merge commits** are exempt from linting.
- Semantic versioning and changelog generation run automatically on the `production` branch via `semantic-release`.

### Branch Protection (사용자 GitHub 측 설정)

룰을 *기계적으로 강제* 하려면 GitHub repo → Settings → Branches 에서 `master` (와 `production`) 에 protection rule:
- Require a pull request before merging
- Require status checks to pass (Code Quality CI 통과 필수)
- Restrict who can push to matching branches (직접 push 차단)
- Include administrators (예외 없음)

이 설정 안 되어있으면 본 § 룰은 *수동 약속* 만 됨.

---

## master invariant — Machine-checkable Contract

master 브랜치는 항상 다음을 만족:

- `apps/karmolab` 의 build (typecheck 포함) 통과 (필수)
- `packages/karmolab-ai` 의 build 통과 (필수)
- `apps/karmolab-tauri/src-tauri` 의 `cargo check --all-targets` 통과 (필수)
- `apps/blog` 의 lint:js + lint:scss — *자동화 빚*: chirpy monorepo 분리 시 `apps/blog/eslint.config.js` + `.stylelintrc.json` 누락. follow-up TASK 로 chirpy upstream config 흡수 후 verify 에 재추가.
- typos check (`crate-ci/typos`) — `continue-on-error: true` (기존 code-quality.yml 행동 보존). *자동화 빚*: master 데이터 약어 false-positive 다수 (anime/tierlist json). follow-up TASK: `_typos.toml` 등록 + 진짜 typo fix → strict 게이트.

검증의 단일 진실: **`npm run verify`** (`scripts/verify.mjs`). 모든 게이트가 이 한 명령만 호출.

3중 게이트:

1. **로컬 pre-push** (`.husky/pre-push`) — 자동 호출. 우회: `git push --no-verify` (응급용).
2. **CI** (`.github/workflows/verify.yml`) — `npm run verify` 호출. 아래 § Branch Protection 의 required status check 로 `verify (master invariant)` 등록 필요 (사용자 액션).
3. **Branch Protection** (사용자 GitHub UI) — § Branch Protection 참조.

추가 hook:
- **`.husky/commit-msg`** — Conventional Commits (commitlint via `apps/blog`) 강제. `apps/blog/node_modules/@commitlint` 미설치 시 silent skip — 사용자가 `cd apps/blog && npm ci` 해야 활성.

위반 발견 시 SLO: 1시간 내 revert. 책임자 = 마지막 머지자.

신규 자동화 룰 추가 시 분류 정합: `memo/CLAUDE-karmoddrine.md` § "자동화 가능 룰은 코드로" 참고. 텍스트 룰만 두면 클로드는 잊는다.

---

## Code Quality Configuration

| Tool | Config File | Scope |
|---|---|---|
| ESLint | `apps/blog/eslint.config.js` | `apps/blog/_javascript/`, `*.config.js` files |
| Stylelint | `apps/blog/.stylelintrc.json` | `apps/blog/_sass/**/*.scss` |
| Markdownlint | `.markdownlint.json` | Markdown posts |
| EditorConfig | `.editorconfig` | All files |
| TypeScript | `apps/karmolab/tsconfig.json`, `apps/blog/tsconfig.json` | sub-apps 각자 |
| Commitlint | `apps/blog/package.json` (`commitlint` key) — `.husky/commit-msg` 가 호출 | Git commit messages |

### Editor Defaults (`.editorconfig`)

- Encoding: UTF-8
- Indentation: 2 spaces
- Quotes: single for JS/CSS/SCSS, double for YAML
- Line endings: LF
- No trailing whitespace (except Markdown)

---

## Key Plugins and Features

- **`_plugins/posts-lastmod-hook.rb`**: Auto-sets `last_modified_at` from git log for posts with >1 commit.
- **`scripts/build-post-graph.cjs`**: Generates post dependency graph data → `assets/js/graph-view/`.
- **PWA**: Offline caching enabled via Chirpy theme.
- **Comments**: Giscus (GitHub Discussions), repo `mascari4615/blog-comments`.
- **Analytics**: Google Analytics (`G-QRNK1L0YH7`) + GoatCounter (`mascari4615`).
- **CDN**: `https://cdn.jsdelivr.net/gh/mascari4615/mascari4615.github.io@master`

---

## Important Notes for AI Assistants

1. **Do not modify `assets/js/dist/`** — this is compiled output from `_javascript/`. Edit source in `_javascript/` instead.
2. **Do not add posts without proper front matter** including `title`, `description`, `categories`, `tags`, `date`.
3. **Respect the `_drafts/` directories** — files there are works in progress and should not be published by moving them out without intent.
4. **The `hidden: true` front matter key** suppresses posts from list pages but keeps them accessible by URL. Use it for internal/reference content.
5. **SCSS lives in `_sass/`** — run `npm run build:css` or `npm run build` after changes to regenerate purged CSS.
6. **Do not edit `_config.yml` lightly** — changes affect the entire site behavior and build pipeline.
7. **Apps in `apps/` are independent projects** with their own `package.json` and build processes. They are excluded from the main Jekyll build.
8. **Commit messages must follow Conventional Commits** — the pre-commit hook (`husky`) will reject non-conforming messages.
9. **The `production` branch** is for releases only; normal development goes to `master`/`main` — but `master`/`main` 도 PR 거쳐 머지 (`§ Git Workflow` 참조). 직접 push 는 1~3줄 chore 등 예외만.
10. **공통 작업 원칙 (레거시 금지 / 마이그레이션 자기소멸 / 커밋 전 확인 / 커밋 전 테스트 / 한 commit 한 주제 / 푸시는 지시 시에만)** — 단일 출처: `karmoddrine/memo/UMBRELLA.md` § 공통 작업 원칙 — 모든 레포. 본 레포에도 동일 적용. 충돌 시 K 가 우선.
11. **Vertex AI is preferred over AI Studio** — the user has Vertex AI credits. When both surfaces support the same capability (text generation, embeddings), default to Vertex. Use `KARMOLAB_AI_SURFACE=vertex` as the standard. AI Studio is a fallback only. Do not propose AI Studio as the primary option. When adding new AI features, implement both surfaces via `karmolab-ai` and respect the surface env var.
12. **새 로컬 서버·dev 프로세스는 KarmoLab Server Monitor (`devProfiles`) 등록 우선** — 사용자는 `apps/karmolab-tauri` 데스크톱 앱을 상시 띄워두고 그 안의 **서버 모니터** 위젯에서 시작/종료/로그 스트림/deploy 를 한다. 새 봇·로컬 서버·dev runner 를 추가할 때는 **반드시 `apps/karmolab/data/servermonitor-config.json` 의 `devProfiles` (그리고 같은 `id` 로 `localMonitors`) 에 등록을 함께 제안**한다. 사용자가 외울 터미널 명령이 늘어나면 안 됨.
    - 허용 `program`: `npm`, `npx`, `bundle`, `ruby`, `node`. 그 외 바이너리 (`cloudflared`, `python`, `cargo` 등) 는 `package.json` script 로 한 번 감싼 뒤 등록.
    - `cwd` 는 레포 루트 기준 상대 경로만. `..` 금지. Rust 측에서 canonicalize 후 루트 prefix 검증함.
    - `npmInstall: true` / `deployArgs: ["run", "..."]` 를 두면 카드에 버튼이 자동 생성되고 stdout/stderr 가 카드 패널에 라이브 스트림.
    - `.env` 파일은 `envFiles[]` 에 `relPath` 로 등록 — 사용자에게 "탐색기에서 .env 만들어주세요" 떠넘기지 말 것. 앱 내 편집기 (512KB 한도) 로 처리하게 한다.
    - 봉제 명령 (`localdev_send_stdin`), 외부 실행 PID 발견·종료 (`localdev_list_external_pids` / `localdev_stop_external`), 재기동 후 PID reattach 등도 자동.
    - 트레이 「개발 모드 (로컬 8899)」 토글은 KarmoLab 자체 (WebView 가 보는 페이지) 를 로컬 정적 서버로 띄울 때만. 일반 봇/서버는 `devProfiles` 가 정답.
    - 코드: `apps/karmolab-tauri/src-tauri/src/local_dev.rs`. 사용자 문서: `apps/karmolab/js/widgets/docs/local-dev-runner.md`.
