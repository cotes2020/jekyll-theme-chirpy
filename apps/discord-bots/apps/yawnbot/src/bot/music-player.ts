/**
 * 네이티브 Opus — 있으면 우선 사용.
 * Node Current(예: 25)처럼 @discordjs/opus prebuild가 아직 없는 ABI면 require 실패 → opusscript만 씀.
 */
import 'opusscript';
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  require('@discordjs/opus');
} catch {
  console.warn(
    '[music] @discordjs/opus 네이티브 바이너리를 불러오지 못했습니다. opusscript로 진행합니다. ' +
      '네이티브 Opus를 쓰려면 Node.js LTS(22·24)로 맞추거나, Visual Studio Build Tools 설치 후 `npm rebuild @discordjs/opus`를 실행하세요.',
  );
}
import ffmpegPath from 'ffmpeg-static';
import {
  createAudioPlayer,
  createAudioResource,
  demuxProbe,
  NoSubscriberBehavior,
  AudioPlayerStatus,
  StreamType,
  type AudioPlayer,
  type AudioResource,
  type VoiceConnection,
} from '@discordjs/voice';
import ytdl from '@distube/ytdl-core';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { Readable } from 'node:stream';
import play from 'play-dl';
import {
  ensureGuildVoiceContext,
  joinVoiceChannelForMusic,
  leaveVoiceChannel,
  VOICE_READY_TIMEOUT_MS,
  waitForVoiceReady,
} from './voice-connection';
import type { VoiceBasedChannel } from 'discord.js';
import { createEdgeTtsAudioResource } from './edge-tts-speak';

/** play-dl / YouTube 조회·스트림이 끝없이 걸리면 `/play`가 생각 중에서 안 풀림 */
export const YOUTUBE_RESOLVE_TIMEOUT_MS = 45_000;
export const YOUTUBE_STREAM_TIMEOUT_MS = 90_000;

/**
 * 2025–2026 권장: 순수 JS 추출기보다 yt-dlp 바이너리가 YouTube 변경에 가장 빨리 따라감.
 * - YT_DLP_PATH / YAWNBOT_YT_DLP_PATH: 실행 파일 직접 지정
 * - YT_DLP_COOKIES_PATH / YAWNBOT_YOUTUBE_COOKIES_PATH: Netscape cookies.txt (연령·로그인 제한 완화)
 * - youtube-dl-exec postinstall 로 내려받은 바이너리(있으면) 자동 사용
 */
function resolveYtDlpBinary(): string {
  const fromEnv = process.env.YT_DLP_PATH || process.env.YAWNBOT_YT_DLP_PATH;
  if (fromEnv && existsSync(fromEnv)) return fromEnv;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { YOUTUBE_DL_PATH } = require('youtube-dl-exec/src/constants.js') as { YOUTUBE_DL_PATH: string };
    if (YOUTUBE_DL_PATH && existsSync(YOUTUBE_DL_PATH)) return YOUTUBE_DL_PATH;
  } catch {
    /* ignore */
  }
  return 'yt-dlp';
}

function resolveYtDlpCookiesPath(): string | undefined {
  const p = process.env.YT_DLP_COOKIES_PATH || process.env.YAWNBOT_YOUTUBE_COOKIES_PATH;
  if (p && existsSync(p)) return p;
  return undefined;
}

/** yt-dlp stdout → demuxProbe (FFmpeg는 ffmpeg-static) */
async function resourceFromYtDlpExec(url: string): Promise<AudioResource> {
  const bin = resolveYtDlpBinary();
  const args: string[] = [
    '-f',
    'bestaudio[ext=webm]/bestaudio[ext=m4a]/ba/b',
    '-o',
    '-',
    '--no-playlist',
    '--quiet',
    '--no-warnings',
  ];
  const cookies = resolveYtDlpCookiesPath();
  if (cookies) {
    args.push('--cookies', cookies);
  }
  args.push(url);

  const child = spawn(bin, args, {
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });

  let stderrBuf = '';
  child.stderr?.on('data', (ch: Buffer) => {
    stderrBuf = (stderrBuf + ch.toString()).slice(-4000);
  });

  await new Promise<void>((resolve, reject) => {
    child.once('error', reject);
    child.once('spawn', () => resolve());
  });

  const { stdout } = child;
  if (!stdout) {
    child.kill();
    throw new Error('yt-dlp: stdout 없음');
  }

  try {
    const probe = await demuxProbe(stdout as any);
    return createAudioResource(probe.stream, { inputType: probe.type, silencePaddingFrames: 5 });
  } catch (e: any) {
    child.kill('SIGKILL');
    const hint = stderrBuf.trim() ? ` (${stderrBuf.trim().slice(0, 500)})` : '';
    throw new Error(`${e?.message ?? e}${hint}`);
  }
}

/**
 * play-dl / ytdl 이 기대하는 표준 watch URL로 맞춤 (검색 결과·단축 URL 등).
 * 내부 스트림 URL이 비어 Invalid URL 이 나는 경우를 줄입니다.
 */
export function canonicalYoutubeWatchUrl(input: string): string {
  const s = input.trim();
  if (!s) return s;
  if (play.yt_validate(s) !== 'video') return s;
  try {
    const id = play.extractID(s);
    if (id) return `https://www.youtube.com/watch?v=${id}`;
  } catch {
    /* ignore */
  }
  return s;
}

export function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => {
      reject(new Error(`${label} 시간 초과 (${Math.round(ms / 1000)}초)`));
    }, ms);
    promise.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}

/** prism-media / @discordjs/voice 가 FFmpeg를 찾도록 (PATH에 없어도 재생 가능) */
if (typeof ffmpegPath === 'string' && ffmpegPath.length > 0) {
  process.env.FFMPEG_PATH = ffmpegPath;
}

type QueueItem =
  | { kind: 'youtube'; title: string; url: string }
  | { kind: 'custom'; title: string; load: () => Promise<AudioResource> };

type GuildMusicState = {
  player: AudioPlayer;
  queue: QueueItem[];
  subscribed: boolean;
};

const states = new Map<string, GuildMusicState>();

const musicConnectionErrorHooked = new WeakSet<VoiceConnection>();

function wireMusicConnectionErrorOnce(connection: VoiceConnection): void {
  if (musicConnectionErrorHooked.has(connection)) return;
  musicConnectionErrorHooked.add(connection);
  connection.on('error', (e) => console.error('[music connection]', e));
}

const VOICE_READY_RETRY_DELAY_MS = 400;

/** entersState 실패 시 1회만 destroy 후 재연결 (일시적 네트워크/어댑터 이슈 완화) */
async function joinAndWaitUntilVoiceReady(channel: VoiceBasedChannel): Promise<VoiceConnection> {
  await ensureGuildVoiceContext(channel);
  let connection = joinVoiceChannelForMusic(channel);
  wireMusicConnectionErrorOnce(connection);
  try {
    await waitForVoiceReady(connection, VOICE_READY_TIMEOUT_MS);
  } catch (e) {
    console.warn('[music] 음성 Ready 실패, 1회 재연결 시도:', e instanceof Error ? e.message : e);
    try {
      connection.destroy();
    } catch {
      /* ignore */
    }
    leaveVoiceChannel(channel.guild.id);
    await new Promise((r) => setTimeout(r, VOICE_READY_RETRY_DELAY_MS));
    await ensureGuildVoiceContext(channel);
    connection = joinVoiceChannelForMusic(channel);
    wireMusicConnectionErrorOnce(connection);
    await waitForVoiceReady(connection, VOICE_READY_TIMEOUT_MS);
  }
  return connection;
}

function getOrCreatePlayer(guildId: string): GuildMusicState {
  let s = states.get(guildId);
  if (s) return s;
  const player = createAudioPlayer({
    behaviors: { noSubscriber: NoSubscriberBehavior.Play },
  });
  player.on('error', (e) => console.error('[music] AudioPlayer error:', e));
  player.on('stateChange', (oldState, newState) => {
    if (oldState.status !== newState.status) {
      console.log('[music] player state:', oldState.status, '->', newState.status);
    }
  });
  const state: GuildMusicState = { player, queue: [], subscribed: false };
  player.on(AudioPlayerStatus.Idle, () => {
    void playNext(guildId);
  });
  states.set(guildId, state);
  return state;
}

function mapPlayDlTypeToStreamType(t: unknown): StreamType | undefined {
  if (t === StreamType.WebmOpus || t === StreamType.OggOpus || t === StreamType.Opus) return t;
  if (typeof t === 'string') {
    const s = t.toLowerCase();
    if (s.includes('webm')) return StreamType.WebmOpus;
    if (s.includes('ogg')) return StreamType.OggOpus;
    if (s === 'opus') return StreamType.Opus;
  }
  return undefined;
}

/** ytdl-core 기본 playerClients 에 WEB 이 없으면 playable formats 가 0개로 떨어질 수 있음 */
const YTDL_STREAM_OPTIONS: ytdl.downloadOptions = {
  filter: 'audioonly',
  /** highestaudio 는 포맷 메타가 비정상일 때 chooseFormat 에서 실패하는 경우가 있음 */
  quality: 'best',
  highWaterMark: 1 << 25,
  playerClients: ['WEB', 'WEB_EMBEDDED', 'IOS', 'ANDROID', 'TV'],
  requestOptions: {
    headers: {
      'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    },
  },
};

/**
 * TS `module: CommonJS` 이면 `import('youtubei.js')` 가 `require()` 로 바뀌어 ESM 전용 패키지가 깨짐.
 * 문자열로만 `import()` 를 호출해 Node 네이티브 동적 import 를 쓴다.
 */
function importYoutubeiModule(): Promise<typeof import('youtubei.js')> {
  return new Function('return import("youtubei.js")')() as Promise<typeof import('youtubei.js')>;
}

/**
 * play-dl이 실패할 때 @distube/ytdl-core (InnerTube 다중 클라이언트).
 */
async function resourceFromYtdlCore(url: string): Promise<AudioResource> {
  const stream = ytdl(url, YTDL_STREAM_OPTIONS);
  try {
    const probe = await demuxProbe(stream as any);
    return createAudioResource(probe.stream, { inputType: probe.type, silencePaddingFrames: 5 });
  } catch (e) {
    stream.destroy();
    throw e;
  }
}

let innertubePromise: Promise<any> | null = null;

function getInnertubeSingleton() {
  if (!innertubePromise) {
    innertubePromise = (async () => {
      const { default: Innertube } = await importYoutubeiModule();
      return Innertube.create();
    })();
  }
  return innertubePromise;
}

/**
 * play-dl `search()` 가 YouTube 응답 변경으로 실패할 때(예: `browseId` 접근 오류) InnerTube 검색으로 첫 동영상만 고름.
 */
export async function searchYoutubeFirstVideoViaYoutubei(
  query: string,
): Promise<{ title: string; url: string } | null> {
  const q = (query ?? '').trim();
  if (!q) return null;
  try {
    const innertube = await getInnertubeSingleton();
    const search = await innertube.search(q, { type: 'video' });
    const vids = search.videos;
    if (!vids?.length) return null;
    const v = vids[0] as { video_id?: string; id?: string; title?: { toString(): string; text?: string } };
    const vid =
      typeof v.video_id === 'string' && v.video_id.length > 0
        ? v.video_id
        : typeof v.id === 'string' && v.id.length > 0
          ? v.id
          : null;
    if (!vid) return null;
    let titleStr = q;
    if (v.title) {
      const t = v.title as { toString?: () => string; text?: string };
      titleStr = typeof t.toString === 'function' ? String(t.toString()) : String(t.text ?? q);
    }
    return {
      title: titleStr.trim() || 'YouTube',
      url: `https://www.youtube.com/watch?v=${vid}`,
    };
  } catch (e) {
    console.warn('[music] youtubei.js 검색 폴백 실패:', e instanceof Error ? e.message : e);
    return null;
  }
}

/**
 * ytdl-core까지 실패 시 youtubei.js (InnerTube 전용 클라이언트) — 포맷/서명 변경에 더 잘 버팀.
 */
async function resourceFromYoutubei(videoId: string): Promise<AudioResource> {
  const innertube = await getInnertubeSingleton();
  const webStream = await innertube.download(videoId, { type: 'audio', quality: 'best' });
  const nodeStream = Readable.fromWeb(webStream as any);
  try {
    const probe = await demuxProbe(nodeStream as any);
    return createAudioResource(probe.stream, { inputType: probe.type, silencePaddingFrames: 5 });
  } catch (e) {
    nodeStream.destroy();
    throw e;
  }
}

/** play-dl 스트림 → demuxProbe로 타입 확정 후 재생. 실패 시 play-dl 타입 매핑. */
async function resourceFromPlayDlStream(yt: { stream: NodeJS.ReadableStream; type: unknown }): Promise<AudioResource> {
  try {
    const probe = await demuxProbe(yt.stream as any);
    return createAudioResource(probe.stream, { inputType: probe.type, silencePaddingFrames: 5 });
  } catch (e: any) {
    console.warn('[music] demuxProbe 실패:', e?.message ?? e);
    const m = mapPlayDlTypeToStreamType(yt.type);
    if (m) {
      return createAudioResource(yt.stream as any, { inputType: m, silencePaddingFrames: 5 });
    }
    return createAudioResource(yt.stream as any, { inputType: yt.type as any, silencePaddingFrames: 5 });
  }
}

async function createYoutubeAudioResourceInner(url: string): Promise<AudioResource> {
  const canonical = canonicalYoutubeWatchUrl(url);
  const lastErrors: string[] = [];

  try {
    return await resourceFromYtDlpExec(canonical);
  } catch (e: any) {
    lastErrors.push(`yt-dlp: ${e?.message ?? e}`);
    console.warn('[music] yt-dlp 실패, play-dl·JS 추출기 폴백:', e instanceof Error ? e.message : e);
  }

  try {
    const info = await play.video_info(canonical);
    const yt = await play.stream_from_info(info, { discordPlayerCompatibility: true });
    return resourceFromPlayDlStream(yt);
  } catch (e: any) {
    lastErrors.push(`stream_from_info: ${e?.message ?? e}`);
  }
  try {
    const yt = await play.stream(canonical, { discordPlayerCompatibility: true });
    return resourceFromPlayDlStream(yt);
  } catch (e: any) {
    lastErrors.push(`stream(compat): ${e?.message ?? e}`);
  }
  try {
    const yt = await play.stream(canonical);
    return resourceFromPlayDlStream(yt);
  } catch (e: any) {
    lastErrors.push(`stream: ${e?.message ?? e}`);
  }
  try {
    return await resourceFromYtdlCore(canonical);
  } catch (e: any) {
    lastErrors.push(`ytdl-core: ${e?.message ?? e}`);
  }
  try {
    const id = play.extractID(canonical);
    return await resourceFromYoutubei(id);
  } catch (e: any) {
    lastErrors.push(`youtubei.js: ${e?.message ?? e}`);
    throw new Error(lastErrors.join(' | '));
  }
}

function createYoutubeAudioResource(url: string): Promise<AudioResource> {
  return withTimeout(
    createYoutubeAudioResourceInner(url),
    YOUTUBE_STREAM_TIMEOUT_MS,
    'YouTube 스트림(FFmpeg/demux)',
  );
}

async function playNext(guildId: string): Promise<void> {
  const s = states.get(guildId);
  if (!s) return;
  const item = s.queue.shift();
  if (!item) return;
  console.log(
    '[music] playNext:',
    item.title.slice(0, 60),
    item.kind === 'youtube' ? item.url.slice(0, 48) : '(custom)',
  );
  try {
    const resource =
      item.kind === 'youtube' ? await createYoutubeAudioResource(item.url) : await item.load();
    s.player.play(resource);
  } catch (e: any) {
    console.error('[music] 재생 실패:', item.kind === 'youtube' ? item.url : item.title, e?.message ?? e);
    await playNext(guildId);
  }
}

async function appendToMusicQueue(
  channel: VoiceBasedChannel,
  item: QueueItem,
): Promise<{ ok: true; position: number; started: boolean } | { ok: false; error: string }> {
  try {
    console.log('[music] 음성 연결 대기(Ready)...', channel.guild.id);
    const connection = await joinAndWaitUntilVoiceReady(channel);
    console.log('[music] 음성 Ready');

    const guildId = channel.guild.id;
    const s = getOrCreatePlayer(guildId);
    const wasIdle = s.player.state.status === AudioPlayerStatus.Idle;
    s.queue.push(item);
    const position = s.queue.length;

    if (!s.subscribed) {
      connection.subscribe(s.player);
      s.subscribed = true;
    }

    if (wasIdle) {
      await playNext(guildId);
    }

    return { ok: true, position, started: wasIdle };
  } catch (e: any) {
    try {
      leaveVoiceChannel(channel.guild.id);
    } catch {
      /* ignore */
    }
    return { ok: false, error: e?.message || String(e) };
  }
}

/** 음성 채널에 연결하고 큐에 추가한 뒤 재생 시작(또는 대기열에만 추가). */
export async function enqueueYouTube(
  channel: VoiceBasedChannel,
  title: string,
  url: string,
): Promise<
  { ok: true; position: number; started: boolean } | { ok: false; error: string }
> {
  return appendToMusicQueue(channel, { kind: 'youtube', title, url });
}

/** TTS·URL·파일 등 임의 오디오 소스를 `/play`와 같은 대기열에 넣습니다. */
export async function enqueueCustomTrack(
  channel: VoiceBasedChannel,
  title: string,
  load: () => Promise<AudioResource>,
): Promise<{ ok: true; position: number; started: boolean } | { ok: false; error: string }> {
  return appendToMusicQueue(channel, { kind: 'custom', title, load });
}

export function skipTrack(guildId: string): boolean {
  const s = states.get(guildId);
  if (!s) return false;
  if (s.player.state.status === AudioPlayerStatus.Playing || s.player.state.status === AudioPlayerStatus.Buffering) {
    s.player.stop(true);
    return true;
  }
  return false;
}

export function stopMusic(guildId: string): boolean {
  const s = states.get(guildId);
  if (!s) return false;
  s.queue.length = 0;
  s.player.stop(true);
  return true;
}

export function destroyMusicForGuild(guildId: string): void {
  const s = states.get(guildId);
  if (!s) return;
  try {
    s.queue.length = 0;
    s.player.stop(true);
    s.player.removeAllListeners();
  } catch {
    /* ignore */
  }
  states.delete(guildId);
}

export function destroyAllMusicPlayers(): void {
  for (const id of states.keys()) {
    destroyMusicForGuild(id);
  }
}

export function getQueueSummary(guildId: string): string[] {
  const s = states.get(guildId);
  if (!s || s.queue.length === 0) return [];
  return s.queue.map((q) => q.title);
}

/** Edge TTS 문장을 `/play`와 동일한 대기열에 넣어 재생합니다. */
export async function enqueueSpokenText(
  channel: VoiceBasedChannel,
  title: string,
  text: string,
): Promise<
  { ok: true; position: number; started: boolean } | { ok: false; error: string }
> {
  return enqueueCustomTrack(channel, title, () => createEdgeTtsAudioResource(text));
}

