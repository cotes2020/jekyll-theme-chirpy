/**
 * kakao-export.mjs --watch 를 백그라운드에서 실행합니다.
 * PID: %USERPROFILE%\.karmolab\kakao-export-watch.pid
 * 종료: npm run kakao-export-watch-stop
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const script = path.join(__dirname, 'kakao-export.mjs');
const pidFile = path.join(os.homedir(), '.karmolab', 'kakao-export-watch.pid');

const child = spawn(process.execPath, [script, '--watch'], {
  detached: true,
  stdio: 'ignore',
  cwd: path.join(__dirname, '..'),
  env: process.env,
});

child.unref();

fs.mkdirSync(path.dirname(pidFile), { recursive: true });
fs.writeFileSync(pidFile, String(child.pid), 'utf8');

console.error(`kakao-export --watch 백그라운드 (pid ${child.pid})`);
console.error('종료: npm run kakao-export-watch-stop');
console.error('저장+요약 한 번에: npm run kakao-export');
