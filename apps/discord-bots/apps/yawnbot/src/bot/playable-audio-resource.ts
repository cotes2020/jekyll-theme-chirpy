import ffmpegPath from 'ffmpeg-static';
import { createAudioResource, demuxProbe, type AudioResource } from '@discordjs/voice';
import { createReadStream } from 'node:fs';
import { Readable } from 'node:stream';

if (typeof ffmpegPath === 'string' && ffmpegPath.length > 0 && !process.env.FFMPEG_PATH) {
  process.env.FFMPEG_PATH = ffmpegPath;
}

function assertHttpUrlSafe(u: URL): void {
  const h = u.hostname.toLowerCase();
  if (h === 'localhost' || h === '127.0.0.1' || h === '0.0.0.0' || h === '[::1]' || h === '::1') {
    throw new Error('localhost/루프백 URL은 사용할 수 없습니다.');
  }
}

/**
 * HTTP(S) 직접 링크 → FFmpeg demux 후 재생 (mp3, ogg, wav 등).
 */
export async function createAudioResourceFromHttpUrl(urlString: string): Promise<AudioResource> {
  const u = new URL(urlString.trim());
  if (u.protocol !== 'http:' && u.protocol !== 'https:') {
    throw new Error('http(s) URL만 지원합니다.');
  }
  assertHttpUrlSafe(u);

  const res = await fetch(u.href, {
    redirect: 'follow',
    headers: { 'User-Agent': 'Unyibinda/1.0 (Discord voice bot)' },
  });
  if (!res.ok) {
    throw new Error(`다운로드 실패 HTTP ${res.status}`);
  }
  if (!res.body) {
    throw new Error('응답 스트림이 없습니다.');
  }

  const nodeStream = Readable.fromWeb(res.body as import('stream/web').ReadableStream);
  const probe = await demuxProbe(nodeStream as Readable);
  return createAudioResource(probe.stream, { inputType: probe.type, silencePaddingFrames: 5 });
}

/**
 * 로컬 파일(절대 경로) — 패키지 `resources/audio/` 등에서만 쓰는 것을 권장.
 */
export async function createAudioResourceFromLocalFile(filePath: string): Promise<AudioResource> {
  const rs = createReadStream(filePath);
  const probe = await demuxProbe(rs as Readable);
  return createAudioResource(probe.stream, { inputType: probe.type, silencePaddingFrames: 5 });
}
