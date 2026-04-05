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
import { inspect } from 'node:util';

/** `entersState` 대기 — 일부 환경에서 내부 타임아웃만으로는 끝나지 않을 수 있어 외부 watchdog과 함께 씀 */
export const VOICE_READY_TIMEOUT_MS = 60_000;

/**
 * `VoiceConnectionStatus.Ready` 만 성공으로 본다. (UDP/미디어 열림)
 * 길드 API상 봇이 채널에 있다고 해서 조기 종료하면 **무음·가짜 연결**만 남는다.
 */
export async function waitForVoiceReady(connection: VoiceConnection, timeoutMs: number): Promise<void> {
  const onState = (oldState: VoiceConnectionState, newState: VoiceConnectionState) => {
    console.log('[voice] state:', oldState.status, '->', newState.status);
    if (
      oldState.status === VoiceConnectionStatus.Connecting &&
      newState.status === VoiceConnectionStatus.Signalling
    ) {
      console.warn(
        '[voice] connecting → signalling: 음성 WS가 Ready(UDP 정보) 전에 끊긴 뒤 @discordjs/voice 가 자동 재접속 중입니다. ' +
          '직전 줄의 `[voice] [NW] Discord 음성 WebSocket close code:` 숫자를 확인하세요 (4014만 Disconnected·full state 로 남음).',
      );
    }
    if (newState.status === VoiceConnectionStatus.Disconnected) {
      console.warn('[voice] disconnected, full state:', inspect(newState, { depth: 8, colors: false }));
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
 * Discord DAVE(E2E) 음성 암호화 — E2EE 필수 채널(음성 close 4017)에 대응하려면 켜 두는 편이 안전함.
 * 기본값 **켬**. 끄려면 `DISCORD_VOICE_DAVE=0` / `false` / `off` / `no` (대소문자 무시).
 * @see https://github.com/discordjs/discord.js/issues/11419
 */
function useDaveEncryptionFromEnv(): boolean {
  const v = process.env.DISCORD_VOICE_DAVE?.trim().toLowerCase();
  if (v === '0' || v === 'false' || v === 'off' || v === 'no') return false;
  return true;
}

const voiceJoinErrorHooked = new WeakSet<VoiceConnection>();
const voiceDebugHooked = new WeakSet<VoiceConnection>();

function wireVoiceJoinErrorOnce(connection: VoiceConnection): void {
  if (voiceJoinErrorHooked.has(connection)) return;
  voiceJoinErrorHooked.add(connection);
  connection.on('error', (e) => {
    console.error('[voice connection]', e);
    if (e && typeof e === 'object' && 'code' in e) {
      console.error('[voice connection] .code:', (e as { code: unknown }).code);
    }
  });
}

/** Voice Close Event Codes — @discordjs/voice 는 4014만 Disconnected 로 두고, 그 외 코드는 바로 signalling 으로 돌려 closeCode 가 상태에 안 남음 */
function voiceCloseCodeHint(code: number): string {
  const table: Record<number, string> = {
    4017: 'DAVE(E2EE) 필수 채널 — 기본은 DAVE 켬; 여전히 4017이면 @discordjs/voice·davey 버전 확인',
    4016: '암호화 모드 불일치 — @discordjs/voice·의존성 버전 확인',
    4014: '개별 연결 종료(킥 등) — 재연결하지 말 것(문서)',
    4012: 'Select Protocol 에서 보낸 protocol 미인식',
    4006: '세션 무효 — 잠시 후 재시도·길드 음성 상태 동기화',
    4004: 'Identify 토큰/인증 실패',
  };
  return table[code] ? ` — ${table[code]}` : '';
}

function wireVoiceDebugOnce(connection: VoiceConnection): void {
  if (!isVoiceDebug() || voiceDebugHooked.has(connection)) return;
  voiceDebugHooked.add(connection);
  connection.on('debug', (msg) => console.log('[voice]', msg));

  const hookNetworkingClose = (_old: VoiceConnectionState, neu: VoiceConnectionState) => {
    if (neu.status !== VoiceConnectionStatus.Connecting) return;
    const nw = Reflect.get(neu, 'networking') as
      | { once(event: 'close', listener: (code: number) => void): void }
      | undefined;
    nw?.once('close', (code: number) => {
      console.warn(`[voice] [NW] Discord 음성 WebSocket close code: ${code}${voiceCloseCodeHint(code)}`);
    });
  };
  connection.on('stateChange', hookNetworkingClose);
  hookNetworkingClose(connection.state, connection.state);
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
  wireVoiceJoinErrorOnce(connection);
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
