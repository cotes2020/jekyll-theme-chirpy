/**
 * Cloudflare Quick Tunnel + GitHub webhook 자동 등록/갱신.
 *
 * 동작:
 *   1. cloudflared tunnel --url http://localhost:<port> 자식 프로세스로 실행
 *   2. 출력에서 https://*.trycloudflare.com URL 추출
 *   3. data/webhook-routes.json 의 githubRepos 각 repo 에 대해
 *      yawnbot webhook (config.url 끝이 /webhook/github) 을 찾아
 *      있으면 PATCH(URL 갱신), 없으면 POST(신규 생성).
 *
 * 사전 요구:
 *   - cloudflared, gh CLI (둘 다 npm run setup:env 로 설치/로그인)
 *
 * 환경변수:
 *   WEBHOOK_PORT (기본 4615)
 *
 * 실행:
 *   npm run tunnel        — 터널만 (봇은 별도 터미널)
 *   npm run dev           — --with-bot 옵션으로 npm run dev:bot 까지 spawn (통합)
 */
import { spawn, execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import process from 'node:process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROUTES_PATH = path.resolve(__dirname, '..', 'data', 'webhook-routes.json');
const PORT = process.env.WEBHOOK_PORT || '4615';
const HOOK_PATH = '/webhook/github';
const EVENTS = ['push', 'pull_request', 'release', 'issues'];
const URL_RE = /https:\/\/[a-z0-9-]+\.trycloudflare\.com/i;

function loadRepos() {
  const parsed = JSON.parse(readFileSync(ROUTES_PATH, 'utf-8'));
  const repos = Array.isArray(parsed.githubRepos) ? parsed.githubRepos : [];
  if (!repos.length) {
    console.error('[Tunnel] data/webhook-routes.json 의 githubRepos 가 비어 있습니다.');
    process.exit(1);
  }
  return repos;
}

function gh(args) {
  return execFileSync('gh', args, { encoding: 'utf-8' });
}

function ensureGh() {
  try {
    gh(['auth', 'status']);
  } catch {
    console.error('[Tunnel] gh CLI 미설치 또는 미로그인 — 먼저 npm run setup:env 를 실행해주세요.');
    process.exit(1);
  }
}

function listHooks(repo) {
  return JSON.parse(gh(['api', `repos/${repo}/hooks`, '--paginate']));
}

function upsertHook(repo, webhookUrl) {
  let hooks;
  try {
    hooks = listHooks(repo);
  } catch (e) {
    console.error(`[Tunnel] ${repo}: hook 목록 조회 실패 — ${e.message}`);
    return;
  }
  const existing = hooks.find(
    (h) => typeof h?.config?.url === 'string' && h.config.url.endsWith(HOOK_PATH),
  );

  if (existing) {
    if (existing.config.url === webhookUrl) {
      console.log(`[Tunnel] ${repo}: hook URL 동일 — 갱신 생략`);
      return;
    }
    try {
      gh([
        'api',
        `repos/${repo}/hooks/${existing.id}`,
        '-X',
        'PATCH',
        '-f',
        `config[url]=${webhookUrl}`,
        '-f',
        'config[content_type]=json',
      ]);
      console.log(`[Tunnel] ${repo}: hook ${existing.id} 갱신 → ${webhookUrl}`);
    } catch (e) {
      console.error(`[Tunnel] ${repo}: hook 갱신 실패 — ${e.message}`);
    }
    return;
  }

  const args = [
    'api',
    `repos/${repo}/hooks`,
    '-X',
    'POST',
    '-f',
    'name=web',
    '-F',
    'active=true',
    '-f',
    `config[url]=${webhookUrl}`,
    '-f',
    'config[content_type]=json',
  ];
  for (const ev of EVENTS) {
    args.push('-f', `events[]=${ev}`);
  }
  try {
    gh(args);
    console.log(`[Tunnel] ${repo}: hook 신규 생성 → ${webhookUrl}`);
  } catch (e) {
    console.error(`[Tunnel] ${repo}: hook 생성 실패 — ${e.message}`);
  }
}

const withBot = process.argv.slice(2).includes('--with-bot');

ensureGh();
const repos = loadRepos();

const children = [];

if (withBot) {
  console.log('[Tunnel] --with-bot — npm run dev:bot 동시 시작');
  const bot = spawn('npm', ['run', 'dev:bot'], { stdio: 'inherit', shell: true });
  children.push(bot);
  bot.on('exit', (code) => {
    console.log(`[Tunnel] 봇 프로세스 종료 (code=${code}) — cloudflared 정리`);
    for (const c of children) if (c !== bot && !c.killed) c.kill('SIGINT');
    process.exit(code ?? 0);
  });
}

console.log(
  `[Tunnel] cloudflared quick tunnel 시작 (localhost:${PORT}) — 등록 대상 repo ${repos.length}개`,
);

const cf = spawn('cloudflared', ['tunnel', '--url', `http://localhost:${PORT}`], {
  stdio: ['ignore', 'pipe', 'pipe'],
});
children.push(cf);

let updated = false;
function handleChunk(chunk) {
  const text = chunk.toString();
  process.stderr.write(text);
  if (updated) return;
  const m = text.match(URL_RE);
  if (!m) return;

  updated = true;
  const tunnelBase = m[0];
  const webhookUrl = `${tunnelBase}${HOOK_PATH}`;
  console.log(`\n[Tunnel] 발급된 URL: ${tunnelBase}`);
  console.log(`[Tunnel] webhook 등록/갱신 시작 → ${webhookUrl}\n`);
  for (const repo of repos) upsertHook(repo, webhookUrl);
  const tail = withBot
    ? '봇은 같은 창에서 떠 있습니다.'
    : '이 창은 켜둔 채로 봇을 별도 터미널에서 실행하세요.';
  console.log(`\n[Tunnel] webhook 동기화 완료. ${tail}\n`);
}

cf.stdout.on('data', handleChunk);
cf.stderr.on('data', handleChunk);

cf.on('exit', (code) => {
  console.log(`[Tunnel] cloudflared 종료 (code=${code})`);
  for (const c of children) if (c !== cf && !c.killed) c.kill('SIGINT');
  process.exit(code ?? 0);
});

const forwardSignal = (sig) => () => {
  for (const c of children) if (!c.killed) c.kill(sig);
};
process.on('SIGINT', forwardSignal('SIGINT'));
process.on('SIGTERM', forwardSignal('SIGTERM'));
