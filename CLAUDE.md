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
| `pages-deploy.yml` | Push to `main`/`master`, manual | Full site build and deploy to GitHub Pages |
| `lint-js.yml` | Changes to `_javascript/`, `*.config.js`, `tsconfig.json` | ESLint |
| `lint-scss.yml` | Changes to `_sass/` | Stylelint |
| `karmolab-ts.yml` | Changes to `apps/karmolab/` | TypeScript typecheck |
| `karmolab-tauri.yml` | Changes to `apps/karmolab-tauri/` | Tauri build/test |
| `ai-quality.yml` | Changes to KarmoLab AI surfaces (`gemini.ts`, chatbot, `packages/karmolab-ai/`) | AI-related typecheck/build quality gate |

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

## Git Conventions

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

### Branches

- `master` / `main`: Active development / deployment source
- `production`: Release branch (triggers semantic-release)
- Feature/fix branches: Work in progress

---

## Code Quality Configuration

| Tool | Config File | Scope |
|---|---|---|
| ESLint | `eslint.config.js` | `_javascript/`, `*.config.js` files |
| Stylelint | `.stylelintrc.json` | `_sass/**/*.scss` |
| Markdownlint | `.markdownlint.json` | Markdown posts |
| EditorConfig | `.editorconfig` | All files |
| TypeScript | `tsconfig.json` | `_javascript/`, apps |
| Commitlint | `package.json` (`commitlint` key) | Git commit messages |

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
9. **The `production` branch** is for releases only; normal development goes to `master`/`main`.
10. **No legacy or compatibility code** — when upgrading a feature, delete the old implementation entirely. No fallbacks, no deprecated wrappers, no runtime version branches. Always leave only the latest approach.
11. **Migrations must be immediate and self-destructing** — when a data migration is needed, write a one-time script, have it run immediately against all existing data, then delete the script from the codebase. Never leave migration code or legacy data formats in the repo. Do not add compatibility shims to handle old data shapes at runtime — migrate everything upfront.
12. **Always ask before committing** — after finishing work, explain the changes and ask the user whether to commit. Never commit automatically.
13. **Test before committing** — never commit untested code. Run a type check, unit test, or manual smoke test appropriate to the change before asking to commit.
14. **Vertex AI is preferred over AI Studio** — the user has Vertex AI credits. When both surfaces support the same capability (text generation, embeddings), default to Vertex. Use `KARMOLAB_AI_SURFACE=vertex` as the standard. AI Studio is a fallback only. Do not propose AI Studio as the primary option. When adding new AI features, implement both surfaces via `karmolab-ai` and respect the surface env var.
15. **One commit, one topic** — each commit must cover exactly one logical change. Do not bundle unrelated fixes, features, or refactors into a single commit. If work spans multiple topics, stage and commit them separately.
