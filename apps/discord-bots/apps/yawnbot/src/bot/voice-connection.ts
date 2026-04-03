import {
  joinVoiceChannel,
  getVoiceConnection,
  getVoiceConnections,
  VoiceConnectionStatus,
  entersState,
  type VoiceConnection,
  type VoiceConnectionState,
} from '@discordjs/voice';
import type { VoiceBasedChannel } from 'discord.js';

/** `entersState` 대기 — 일부 환경에서 내부 타임아웃만으로는 끝나지 않을 수 있어 외부 watchdog과 함께 씀 */
export const VOICE_READY_TIMEOUT_MS = 60_000;

/**
 * `VoiceConnectionStatus.Ready` 만 성공으로 본다. (UDP/미디어 열림)
 * 길드 API상 봇이 채널에 있다고 해서 조기 종료하면 **무음·가짜 연결**만 남는다.
 */
export async function waitForVoiceReady(connection: VoiceConnection, timeoutMs: number): Promise<void> {
  const onState = (oldState: VoiceConnectionState, newState: VoiceConnectionState) => {
    console.log('[voice] state:', oldState.status, '->', newState.status);
    if (newState.status === VoiceConnectionStatus.Disconnected && 'reason' in newState) {
      console.warn('[voice] disconnected, reason:', (newState as { reason?: unknown }).reason);
    }
  };
  connection.on('stateChange', onState);

  const readyWait = entersState(connection, VoiceConnectionStatus.Ready, timeoutMs);
  void readyWait.catch(() => {});

  let watchdogTimer: ReturnType<typeof setTimeout> | undefined;
  const watchdog = new Promise<never>((_, reject) => {
    watchdogTimer = setTimeout(() => {
      try {
        connection.destroy();
      } catch {
        /* ignore */
      }
      reject(
        new Error(
          `음성 Ready 대기 초과(${Math.round(timeoutMs / 1000)}초). 네트워크·방화벽·VPN을 확인하거나 잠시 후 다시 시도하세요.`,
        ),
      );
    }, timeoutMs);
  });

  try {
    await Promise.race([readyWait, watchdog]);
  } finally {
    if (watchdogTimer !== undefined) clearTimeout(watchdogTimer);
    connection.off('stateChange', onState);
  }
}

/** `.env` 는 load-env 이후에만 보이므로, 모듈 상수가 아니라 사용 시점에 읽음 */
function isVoiceDebug(): boolean {
  return process.env.VOICE_DEBUG === '1';
}

/**
 * Discord DAVE(E2E) 음성 암호화 — 일부 환경에서 signalling↔connecting 루프·무음 유발 보고됨.
 * `DISCORD_VOICE_DAVE=1` 이면 DAVE 사용(기본은 끔).
 * @see https://github.com/discordjs/discord.js/issues/11419
 */
function useDaveEncryptionFromEnv(): boolean {
  return process.env.DISCORD_VOICE_DAVE === '1';
}

const voiceJoinErrorHooked = new WeakSet<VoiceConnection>();
const voiceDebugHooked = new WeakSet<VoiceConnection>();

function wireVoiceJoinErrorOnce(connection: VoiceConnection): void {
  if (voiceJoinErrorHooked.has(connection)) return;
  voiceJoinErrorHooked.add(connection);
  connection.on('error', (e) => console.error('[voice connection]', e));
}

function wireVoiceDebugOnce(connection: VoiceConnection): void {
  if (!isVoiceDebug() || voiceDebugHooked.has(connection)) return;
  voiceDebugHooked.add(connection);
  connection.on('debug', (msg) => console.log('[voice]', msg));
}

function joinVoiceChannelOpts(
  base: Parameters<typeof joinVoiceChannel>[0],
): Parameters<typeof joinVoiceChannel>[0] {
  const dave = useDaveEncryptionFromEnv() ? { daveEncryption: true as const } : { daveEncryption: false as const };
  const merged: Parameters<typeof joinVoiceChannel>[0] = { ...base, ...dave };
  return isVoiceDebug() ? { ...merged, debug: true } : merged;
}

/** 게이트웨이 캐시를 한 번 갱신해 VOICE_* 이벤트가 안정적으로 오게 함 (일부 환경에서만 효과) */
export async function ensureGuildVoiceContext(channel: VoiceBasedChannel): Promise<void> {
  try {
    await channel.guild.fetch();
  } catch {
    /* ignore */
  }
  try {
    await channel.fetch();
  } catch {
    /* ignore */
  }
  const uid = channel.guild.client.user?.id;
  if (uid) {
    await channel.guild.members.fetch(uid).catch(() => {});
  }
}

/** 음악 재생용: 같은 채널이면 Ready일 때만 재사용. Ready 아닌 연결을 재사용하면 signalling↔connecting 루프가 고정될 수 있음. */
export function joinVoiceChannelForMusic(channel: VoiceBasedChannel): VoiceConnection {
  const existing = getVoiceConnection(channel.guild.id);
  if (existing && existing.joinConfig.channelId === channel.id) {
    if (existing.state.status === VoiceConnectionStatus.Ready) {
      return existing;
    }
    try {
      existing.destroy();
    } catch {
      /* ignore */
    }
  } else {
    existing?.destroy();
  }
  const connection = joinVoiceChannel(
    joinVoiceChannelOpts({
      channelId: channel.id,
      guildId: channel.guild.id,
      adapterCreator: channel.guild.voiceAdapterCreator,
      selfDeaf: false,
      selfMute: false,
    }),
  );
  wireVoiceDebugOnce(connection);
  return connection;
}

export async function joinVoiceChannelSafe(
  channel: VoiceBasedChannel,
): Promise<{ ok: true } | { ok: false; error: string }> {
  await ensureGuildVoiceContext(channel);
  const guild = channel.guild;
  const existing = getVoiceConnection(guild.id);

  if (existing && existing.joinConfig.channelId === channel.id) {
    try {
      if (existing.state.status !== VoiceConnectionStatus.Ready) {
        await waitForVoiceReady(existing, VOICE_READY_TIMEOUT_MS);
      }
      return { ok: true };
    } catch (e: any) {
      try {
        existing.destroy();
      } catch {
        /* ignore */
      }
      return { ok: false, error: formatVoiceJoinError(e) };
    }
  }

  existing?.destroy();
  try {
    const connection = joinVoiceChannel(
      joinVoiceChannelOpts({
        channelId: channel.id,
        guildId: guild.id,
        adapterCreator: guild.voiceAdapterCreator,
        selfDeaf: true,
        selfMute: false,
      }),
    );
    wireVoiceJoinErrorOnce(connection);
    wireVoiceDebugOnce(connection);
    await waitForVoiceReady(connection, VOICE_READY_TIMEOUT_MS);
    return { ok: true };
  } catch (e: any) {
    getVoiceConnection(guild.id)?.destroy();
    return { ok: false, error: formatVoiceJoinError(e) };
  }
}

function formatVoiceJoinError(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e);
  if (msg === 'The operation was aborted' || (e instanceof Error && e.name === 'AbortError')) {
    return (
      `음성 준비 시간 초과 또는 연결 중단(${Math.round(VOICE_READY_TIMEOUT_MS / 1000)}초). ` +
      `Windows 방화벽에서 Node.js(또는 실행 중인 프로세스)의 UDP 발신을 허용했는지, VPN·회사망을 끄고 시도했는지 확인하세요. ` +
      `같은 PC에서 여전히 signalling↔connecting만 반복되면 봇을 VPS나 다른 네트워크에서 돌려 보는 편이 빠릅니다.`
    );
  }
  return msg;
}

export function leaveVoiceChannel(guildId: string): boolean {
  const c = getVoiceConnection(guildId);
  if (!c) return false;
  c.destroy();
  return true;
}

export function destroyAllVoiceConnections(): void {
  for (const c of getVoiceConnections().values()) {
    try {
      c.destroy();
    } catch {
      /* ignore */
    }
  }
}
