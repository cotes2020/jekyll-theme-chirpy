/**
 * `npm run dev:static`: 레포 루트를 127.0.0.1:<PORT> 에 서빙.
 * Python이 PATH에 있으면 `http.server`를 쓰고, 없으면 Node 내장 http로 폴백.
 *
 * 포트 결정: `argv[2]` > `KARMOLAB_DEV_STATIC_PORT` env > 8899 (default).
 * `dev:dual` 은 prod KarmoLab 의 트레이 「개발 모드」 (8899) 와 충돌하지 않게 8898 사용.
 */
import { spawn, spawnSync } from 'node:child_process';
import http from 'node:http';
import fs from 'node:fs';
import fsPromises from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline } from 'node:stream/promises';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '../../..');
const HOST = '127.0.0.1';
const PORT = Number(process.argv[2]) || Number(process.env.KARMOLAB_DEV_STATIC_PORT) || 8899;
// `--node-only` 플래그 또는 env: Python 스킵. dev:dual 처럼 *webview cache 회피 (no-store header)*
// 가 필수일 때 사용. Python http.server 는 cache header 통제 불가라서.
const NODE_ONLY = process.argv.includes('--node-only') || process.env.KARMOLAB_DEV_NODE_ONLY === '1';

const PYTHON_LAUNCHERS = [
  { cmd: 'python', args: ['-m', 'http.server', String(PORT), '--bind', HOST] },
  { cmd: 'python3', args: ['-m', 'http.server', String(PORT), '--bind', HOST] },
  { cmd: 'py', args: ['-3', '-m', 'http.server', String(PORT), '--bind', HOST], probe: ['-3', '-c', 'print(1)'] },
  { cmd: 'py', args: ['-m', 'http.server', String(PORT), '--bind', HOST], probe: ['-c', 'print(1)'] },
];

function tryStartPython() {
  for (const { cmd, args, probe } of PYTHON_LAUNCHERS) {
    const probeArgs = probe ?? ['-c', 'print(1)'];
    const r = spawnSync(cmd, probeArgs, {
      stdio: 'ignore',
      cwd: REPO_ROOT,
      windowsHide: true,
    });
    if (r.error || r.status !== 0) continue;

    console.error(`[dev-static] ${cmd} ${args.join(' ')} (cwd: ${REPO_ROOT})`);
    const child = spawn(cmd, args, {
      stdio: 'inherit',
      cwd: REPO_ROOT,
      windowsHide: true,
    });
    child.on('error', (err) => {
      console.error('[dev-static] spawn failed:', err.message);
      process.exit(1);
    });
    child.on('exit', (code, signal) => {
      if (signal) process.exit(1);
      process.exit(code ?? 0);
    });
    return true;
  }
  return false;
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.htm': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.map': 'application/json',
  '.webmanifest': 'application/manifest+json',
  '.xml': 'application/xml; charset=utf-8',
  '.wasm': 'application/wasm',
  '.webp': 'image/webp',
};

function guessMime(filePath) {
  return MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
}

function isPathUnderRepo(fsPath) {
  const rootResolved = path.resolve(REPO_ROOT);
  const resolved = path.resolve(fsPath);
  const rel = path.relative(rootResolved, resolved);
  return !rel.startsWith('..') && !path.isAbsolute(rel);
}

async function statSafe(p) {
  try {
    return await fsPromises.stat(p);
  } catch {
    return null;
  }
}

async function sendFile(req, res, filePath, st) {
  res.writeHead(200, {
    'Content-Type': guessMime(filePath),
    'Content-Length': st.size,
    'Cache-Control': 'no-store, no-cache, must-revalidate',
  });
  if (req.method === 'HEAD') {
    res.end();
    return;
  }
  await pipeline(fs.createReadStream(filePath), res);
}

async function handleRequest(req, res) {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.writeHead(405, { Allow: 'GET, HEAD' });
    res.end('Method Not Allowed');
    return;
  }

  let urlPath;
  try {
    urlPath = new URL(req.url || '/', `http://${HOST}`).pathname;
  } catch {
    res.writeHead(400);
    res.end('Bad Request');
    return;
  }

  try {
    let decoded;
    try {
      decoded = decodeURIComponent(urlPath);
    } catch {
      res.writeHead(400);
      res.end('Bad Request');
      return;
    }

    const rel = decoded === '/' || decoded === '' ? '.' : decoded.replace(/^\/+/, '');
    const fsPath = path.resolve(REPO_ROOT, rel);

    if (!isPathUnderRepo(fsPath)) {
      res.writeHead(403);
      res.end('Forbidden');
      return;
    }

    let st = await statSafe(fsPath);
    let filePath = fsPath;

    if (st?.isDirectory()) {
      const indexPath = path.join(fsPath, 'index.html');
      const indexSt = await statSafe(indexPath);
      if (indexSt?.isFile()) {
        filePath = indexPath;
        st = indexSt;
      } else {
        res.writeHead(404);
        res.end('Not Found');
        return;
      }
    }

    if (!st?.isFile()) {
      res.writeHead(404);
      res.end('Not Found');
      return;
    }

    await sendFile(req, res, filePath, st);
  } catch (err) {
    console.error('[dev-static]', err);
    res.writeHead(500);
    res.end('Internal Server Error');
  }
}

function startNodeServer() {
  const server = http.createServer((req, res) => {
    handleRequest(req, res).catch((err) => {
      console.error('[dev-static]', err);
      if (!res.writableEnded) {
        res.writeHead(500);
        res.end('Internal Server Error');
      }
    });
  });

  server.listen(PORT, HOST, () => {
    console.error(`[dev-static] Node http server at http://${HOST}:${PORT}/ (root: ${REPO_ROOT})`);
  });

  server.on('error', (err) => {
    console.error('[dev-static]', err.message);
    process.exit(1);
  });
}

if (!NODE_ONLY && tryStartPython()) {
  // 자식 프로세스가 유지됨
} else {
  if (NODE_ONLY) console.error('[dev-static] --node-only mode (cache-control 강제 위해 Python skip).');
  else console.error('[dev-static] Python not found; using Node static server.');
  startNodeServer();
}
