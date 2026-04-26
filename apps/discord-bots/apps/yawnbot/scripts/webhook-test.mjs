/**
 * 로컬에서 실행 중인 yawnbot 의 GitHub webhook 엔드포인트로
 * 합성(synthetic) payload 를 POST 하는 수동 테스트 도구.
 *
 * 실행: npm run test:webhook -- <event>
 *   event: ping | push | issues | pr-opened | pr-merged | pr-closed
 *          | pr-reopened | pr-ready | pr-ignored | release | release-pre | release-ignored
 *
 * 환경변수:
 *   WEBHOOK_PORT (기본 4615)
 *   WEBHOOK_URL  (지정 시 PORT 무시. 예: http://localhost:4615/webhook/github)
 *   WEBHOOK_TEST_REPO (기본 mascari4615/test-repo) — payload.repository.full_name
 *     data/webhook-routes.json 의 routes 매칭을 검증할 때 본인 repo 로 지정.
 */
import process from 'node:process';

const PORT = process.env.WEBHOOK_PORT || 4615;
const URL = process.env.WEBHOOK_URL || `http://localhost:${PORT}/webhook/github`;

const sender = {
  login: 'mascari4615',
  avatar_url: 'https://avatars.githubusercontent.com/u/61608119?v=4',
};
const repository = { full_name: process.env.WEBHOOK_TEST_REPO || 'mascari4615/test-repo' };

const pr = (overrides = {}) => ({
  number: 7,
  title: 'feat: PR 알림 테스트',
  html_url: 'https://github.com/mascari4615/test-repo/pull/7',
  merged: false,
  head: { ref: 'feat/webhook-pr' },
  base: { ref: 'main' },
  ...overrides,
});

const release = (overrides = {}) => ({
  tag_name: 'v1.2.3',
  name: 'v1.2.3 - 봇 알림 개선',
  html_url: 'https://github.com/mascari4615/test-repo/releases/tag/v1.2.3',
  prerelease: false,
  ...overrides,
});

const fixtures = {
  ping: {
    event: 'ping',
    payload: { sender, repository, zen: 'Non-blocking is better than blocking.' },
  },
  push: {
    event: 'push',
    payload: {
      sender,
      repository,
      ref: 'refs/heads/main',
      commits: [
        {
          id: 'abcdef1234567890',
          url: 'https://github.com/mascari4615/test-repo/commit/abcdef1',
          message: 'feat: add github webhook test fixture',
        },
        {
          id: '1234567890abcdef',
          url: 'https://github.com/mascari4615/test-repo/commit/1234567',
          message: 'chore: bump deps',
        },
      ],
    },
  },
  issues: {
    event: 'issues',
    payload: {
      sender,
      repository,
      action: 'opened',
      issue: {
        number: 42,
        title: '테스트 이슈입니다',
        html_url: 'https://github.com/mascari4615/test-repo/issues/42',
      },
    },
  },
  'pr-opened': {
    event: 'pull_request',
    payload: { sender, repository, action: 'opened', pull_request: pr() },
  },
  'pr-merged': {
    event: 'pull_request',
    payload: { sender, repository, action: 'closed', pull_request: pr({ merged: true }) },
  },
  'pr-closed': {
    event: 'pull_request',
    payload: {
      sender,
      repository,
      action: 'closed',
      pull_request: pr({
        number: 8,
        title: 'wip: drop this',
        html_url: 'https://github.com/mascari4615/test-repo/pull/8',
        merged: false,
        head: { ref: 'wip/throwaway' },
      }),
    },
  },
  'pr-reopened': {
    event: 'pull_request',
    payload: { sender, repository, action: 'reopened', pull_request: pr() },
  },
  'pr-ready': {
    event: 'pull_request',
    payload: { sender, repository, action: 'ready_for_review', pull_request: pr() },
  },
  'pr-ignored': {
    // synchronize 는 임베드 X, 200 만 응답해야 정상
    event: 'pull_request',
    payload: { sender, repository, action: 'synchronize', pull_request: pr() },
  },
  release: {
    event: 'release',
    payload: { sender, repository, action: 'published', release: release() },
  },
  'release-pre': {
    event: 'release',
    payload: {
      sender,
      repository,
      action: 'published',
      release: release({ tag_name: 'v1.2.3-rc.1', name: 'v1.2.3-rc.1', prerelease: true }),
    },
  },
  'release-ignored': {
    // created/edited 는 임베드 X
    event: 'release',
    payload: { sender, repository, action: 'created', release: release() },
  },
};

const eventKey = process.argv[2];
if (!eventKey || !fixtures[eventKey]) {
  console.error('Usage: npm run test:webhook -- <event>');
  console.error('  event:', Object.keys(fixtures).join(' | '));
  process.exit(1);
}

const { event, payload } = fixtures[eventKey];
const deliveryId = `test-${eventKey}-${Date.now()}`;

console.log(`POST ${URL}`);
console.log(`  x-github-event:    ${event}`);
console.log(`  x-github-delivery: ${deliveryId}`);

const res = await fetch(URL, {
  method: 'POST',
  headers: {
    'content-type': 'application/json',
    'x-github-event': event,
    'x-github-delivery': deliveryId,
  },
  body: JSON.stringify(payload),
});

console.log(`→ ${res.status} ${res.statusText}`);
process.exit(res.ok ? 0 : 1);
