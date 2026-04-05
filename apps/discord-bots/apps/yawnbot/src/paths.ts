/**
 * 런타임 작업 디렉터리(패키지 루트). `npm start` / `npm run dev` 는 이 디렉터리에서 실행한다고 가정.
 *
 * NOTE: 기존 yawnbot-server는 process.cwd() 기반이었는데, 워크스페이스 분리 후에도
 * 실행 위치에 흔들리지 않도록 기본값을 "현재 패키지 루트"로 고정합니다.
 */
import fs from 'fs';
import os from 'os';
import path from 'path';

/**
 * `.env`의 CURSOR_LOCAL_REPO_DIR — `~`·`%NAME%`(Windows)만 풀어서 절대 경로로 만듭니다.
 */
export function resolveCursorRepoDir(raw: string | undefined | null): string {
  if (raw == null) return '';
  let s = String(raw).trim();
  if (!s) return '';
  if (s[0] === '~' && (s.length === 1 || s[1] === '/' || s[1] === '\\')) {
    s = path.join(os.homedir(), s.slice(1));
  }
  if (process.platform === 'win32') {
    s = s.replace(/%([^%]+)%/g, (_, name: string) => process.env[name] ?? '');
  }
  return path.resolve(s);
}

// Compiled to dist/src/*.js → two levels up to package root (apps/discord-bots/apps/yawnbot)
export const PKG_ROOT = path.resolve(__dirname, '..', '..');

/**
 * 이 패키지가 `…/apps/discord-bots/apps/yawnbot`에 있다고 가정할 때의 git 워크스페이스 루트(레포 최상위).
 * 클론을 어디에 두었는지(`source\repos` 등)와 무관합니다.
 */
export function defaultCursorRepoRoot(): string {
  return path.resolve(PKG_ROOT, '..', '..', '..', '..');
}

/**
 * `/cursor-edit`: `CURSOR_LOCAL_REPO_DIR`이 비어 있으면 위 기본 루트를 씁니다.
 * env를 썼는데 경로가 틀리면 빈 문자열(호출부에서 오류 처리).
 */
export function resolveCursorRepoDirForSlash(): string {
  const raw = process.env.CURSOR_LOCAL_REPO_DIR;
  if (raw != null && String(raw).trim()) {
    const resolved = resolveCursorRepoDir(raw);
    if (resolved && fs.existsSync(resolved) && fs.statSync(resolved).isDirectory()) {
      return resolved;
    }
    return '';
  }
  const fallback = defaultCursorRepoRoot();
  if (fallback && fs.existsSync(fallback) && fs.statSync(fallback).isDirectory()) {
    return fallback;
  }
  return '';
}

export function enhancementImgDir(): string {
  return path.join(PKG_ROOT, 'resources', 'img', 'enhancement');
}

export function memeImgDir(): string {
  return path.join(PKG_ROOT, 'resources', 'img', 'meme');
}

/** `/music sound clip:` — 이 폴더 안의 파일명만 허용 (경로 조작 방지) */
export function packagedAudioDir(): string {
  return path.join(PKG_ROOT, 'resources', 'audio');
}

/** `npm run build` 후 생성되는 컴파일된 러너 */
export function cursorRunnerScript(): string {
  return path.join(PKG_ROOT, 'dist', 'cli', 'cursor-local-runner.js');
}

export function cursorRunnerExists(): boolean {
  return fs.existsSync(cursorRunnerScript());
}

