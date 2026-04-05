/**
 * kakao-export-watch-background.mjs 가 기록한 PID 로 감시 프로세스를 종료합니다.
 */

import fs from 'fs';
import path from 'path';
import os from 'os';
import process from 'node:process';

const pidFile = path.join(os.homedir(), '.karmolab', 'kakao-export-watch.pid');

let pid;
try {
  pid = parseInt(fs.readFileSync(pidFile, 'utf8').trim(), 10);
} catch {
  console.error('pid 파일이 없습니다. 감시가 실행 중이 아닐 수 있습니다.');
  process.exit(1);
}

if (!Number.isFinite(pid) || pid <= 0) {
  console.error('잘못된 pid:', pid);
  process.exit(1);
}

try {
  process.kill(pid, 'SIGTERM');
  console.error(`종료 신호 보냄 (pid ${pid})`);
} catch (e) {
  console.error(e.message || e);
  process.exit(1);
}

try {
  fs.unlinkSync(pidFile);
} catch {}
