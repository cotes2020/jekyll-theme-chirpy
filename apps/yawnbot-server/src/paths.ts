/**
 * 런타임 작업 디렉터리(패키지 루트). `npm start` / `npm run dev` 는 이 디렉터리에서 실행한다고 가정.
 */
import fs from 'fs';
import path from 'path';

export const PKG_ROOT = process.cwd();

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
