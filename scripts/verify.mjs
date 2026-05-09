#!/usr/bin/env node
// master invariant 게이트 — 단일 진실.
// 호출처: `npm run verify` / `.husky/pre-push` / `.github/workflows/verify.yml`.
// 정본: memo/UMBRELLA.md § 자동화 가능 룰은 코드로 — 텍스트 룰은 잊힌다.

import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

function run(label, cwd, command) {
  console.log(`\n[verify] ${label}: ${command} (cwd: ${cwd})`);
  const r = spawnSync(command, { cwd, stdio: 'inherit', shell: true });
  if (r.status !== 0) {
    console.error(`[verify] X ${label} 실패 (exit ${r.status ?? '?'})`);
    process.exit(r.status ?? 1);
  }
}

function requireDeps(sub) {
  if (!existsSync(`${sub}/node_modules`)) {
    console.error(`[verify] X ${sub}/node_modules 없음 — 'cd ${sub} && npm ci' 필요`);
    process.exit(1);
  }
}

console.log('[verify] master invariant 게이트 시작');

// 1. apps/karmolab — build (typecheck 포함). 이전 karmolab-ts.yml + ai-quality.yml karmolab-ai-surface 흡수.
requireDeps('apps/karmolab');
run('apps/karmolab build', 'apps/karmolab', 'npm run build');

// 2. packages/karmolab-ai — build. 이전 ai-quality.yml shared-ai-package-build 흡수.
if (existsSync('packages/karmolab-ai/node_modules')) {
  run('packages/karmolab-ai build', 'packages/karmolab-ai', 'npm run build');
} else {
  console.log('[verify] ! packages/karmolab-ai/node_modules 없음 — build skip (정합: cd packages/karmolab-ai && npm ci)');
}

// 3. apps/karmolab-tauri — cargo check. 이전 karmolab-tauri.yml 흡수.
//    PR #15 의 DOMAIN_DIRS private E0603 같은 사고 방지.
if (existsSync('apps/karmolab-tauri/src-tauri/Cargo.toml')) {
  run('apps/karmolab-tauri cargo check', 'apps/karmolab-tauri/src-tauri', 'cargo check --all-targets');
}

// 4. apps/blog lint — chirpy v7.5.0 의 root config (eslint.config.js + .stylelintrc.json)
//    흡수 후 복원 (TASK-KL-031). node_modules 없으면 skip — pre-push 가 매번 npm ci 강요하면
//    개발 흐름 깨짐. CI 는 verify.yml 의 'Install blog deps' step 이 보장.
if (existsSync('apps/blog/node_modules')) {
  run('apps/blog lint:js', 'apps/blog', 'npm run lint:js');
  run('apps/blog lint:scss', 'apps/blog', 'npm run lint:scss');
} else {
  console.log('[verify] ! apps/blog/node_modules 없음 — lint skip (정합: cd apps/blog && npm ci)');
}

// 5. typos — CI 의 verify.yml 별 step (crate-ci/typos action) 이 책임. local 은 binary 미설치 가정 → skip.

console.log('\n[verify] OK — master invariant 통과');
