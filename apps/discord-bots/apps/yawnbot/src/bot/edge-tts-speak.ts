import ffmpegPath from 'ffmpeg-static';
import { createAudioResource, demuxProbe, type AudioResource } from '@discordjs/voice';
import { createReadStream } from 'node:fs';
import type { Readable } from 'node:stream';
import { mkdtemp, unlink, rmdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { EdgeTTS } from 'node-edge-tts';

if (typeof ffmpegPath === 'string' && ffmpegPath.length > 0 && !process.env.FFMPEG_PATH) {
  process.env.FFMPEG_PATH = ffmpegPath;
}

/**
 * Microsoft Edge 온라인 TTS(별도 API 키 없음) → 임시 MP3 → Discord 재생용 AudioResource.
 * 재생 종료 후 임시 파일을 지웁니다.
 */
export async function createEdgeTtsAudioResource(text: string): Promise<AudioResource> {
  const dir = await mkdtemp(join(tmpdir(), 'yawnbot-tts-'));
  const filePath = join(dir, 'speech.mp3');
  const proxy = process.env.SPEAK_TTS_PROXY?.trim() || process.env.HTTPS_PROXY?.trim();
  const voice = process.env.SPEAK_VOICE?.trim() || 'ko-KR-SunHiNeural';
  const lang = process.env.SPEAK_LANG?.trim() || 'ko-KR';
  const timeout = Math.min(
    Math.max(parseInt(process.env.SPEAK_TTS_TIMEOUT_MS || '30000', 10) || 30000, 5000),
    120000,
  );
  const tts = new EdgeTTS({
    voice,
    lang,
    outputFormat: 'audio-24khz-48kbitrate-mono-mp3',
    timeout,
    ...(proxy ? { proxy } : {}),
  });
  await tts.ttsPromise(text, filePath);

  let cleaned = false;
  const cleanup = async (): Promise<void> => {
    if (cleaned) return;
    cleaned = true;
    try {
      await unlink(filePath);
    } catch {
      /* ignore */
    }
    try {
      await rmdir(dir);
    } catch {
      /* ignore */
    }
  };

  const rs = createReadStream(filePath);
  rs.once('error', () => void cleanup());

  const probe = await demuxProbe(rs as Readable);
  probe.stream.once('close', () => void cleanup());
  probe.stream.once('end', () => void cleanup());

  return createAudioResource(probe.stream, { inputType: probe.type, silencePaddingFrames: 5 });
}
