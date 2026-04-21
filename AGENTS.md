# AGENTS.md

This file provides context and operating rules for AI coding agents working in this repository.

## Project Snapshot

- Repository: mascari4615.github.io (Jekyll + Chirpy monorepo)
- Main site: Korean blog/portfolio deployed to GitHub Pages
- Companion apps: KarmoLab, React app, Tauri app, Discord bots, browser extension

## Important Working Boundaries

- Do not edit compiled output under `assets/js/dist/`. Edit source files in `_javascript/` instead.
- Treat each app under `apps/` as an independent project with its own install/build flow.
- Avoid changing `_config.yml` unless the task explicitly requires global site behavior changes.
- Keep changes focused. Do not mix unrelated refactors into feature/fix work.

## High-Signal Paths

- Site source: `_posts/`, `_tabs/`, `_layouts/`, `_includes/`, `_sass/`, `_javascript/`
- KarmoLab source: `apps/karmolab/src/`
- Shared AI utilities: `packages/karmolab-ai/`
- CI workflows: `.github/workflows/`

## Common Commands

### Root (site)

```bash
npm run build
npm run build:css
npm run build:js
npm run test
bundle exec jekyll serve
bundle exec jekyll b
```

### KarmoLab

```bash
cd apps/karmolab
npm ci
npm run typecheck
npm run build
```

### Shared AI Package

```bash
cd packages/karmolab-ai
npm ci
npm run build
```

## AI-Related Change Checklist

When editing AI-related paths (for example `apps/karmolab/src/gemini.ts`, chatbot widgets, or `packages/karmolab-ai/`):

1. Run type checks for affected app/package.
2. Run build for affected app/package.
3. Keep provider-specific behavior explicit (AI Studio vs Vertex).
4. Preserve existing env-based configuration contracts unless migration is requested.
5. Update docs/comments only when behavior or public config contracts changed.

## Commit and PR Conventions

- Use Conventional Commits (`feat:`, `fix:`, `chore:`, etc.).
- Keep PR scope single-purpose.
- In PRs, mark AI usage in `.github/PULL_REQUEST_TEMPLATE.md` when applicable.
