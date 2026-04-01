/**
 * 런타임 작업 디렉터리(패키지 루트). `npm start` / `npm run dev` 는 이 디렉터리에서 실행한다고 가정.
 *
 * NOTE: 기존 yawnbot-server는 process.cwd() 기반이었는데, 워크스페이스 분리 후에도
 * 실행 위치에 흔들리지 않도록 기본값을 "현재 패키지 루트"로 고정합니다.
 */
import fs from 'fs';
import path from 'path';

export const PKG_ROOT = path.resolve(__dirname, '..');

export function enhancementImgDir(): string {
  return path.join(PKG_ROOT, 'resources', 'img', 'enhancement');
}

export function memeImgDir(): string {
  return path.join(PKG_ROOT, 'resources', 'img', 'meme');
}

/** `npm run build` 후 생성되는 컴파일된 러너 */
export function cursorRunnerScript(): string {
  return path.join(PKG_ROOT, 'dist', 'cli', 'cursor-local-runner.js');
}

export function cursorRunnerExists(): boolean {
  return fs.existsSync(cursorRunnerScript());
}

