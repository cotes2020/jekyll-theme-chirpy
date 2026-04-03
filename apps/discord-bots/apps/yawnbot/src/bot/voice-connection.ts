import {
  joinVoiceChannel,
  getVoiceConnection,
  getVoiceConnections,
  VoiceConnectionStatus,
  entersState,
} from '@discordjs/voice';
import type { VoiceBasedChannel } from 'discord.js';

export async function joinVoiceChannelSafe(
  channel: VoiceBasedChannel,
): Promise<{ ok: true } | { ok: false; error: string }> {
  const guild = channel.guild;
  getVoiceConnection(guild.id)?.destroy();
  try {
    const connection = joinVoiceChannel({
      channelId: channel.id,
      guildId: guild.id,
      adapterCreator: guild.voiceAdapterCreator,
      selfDeaf: true,
      selfMute: false,
    });
    connection.on('error', (e) => console.error('[voice connection]', e));
    await entersState(connection, VoiceConnectionStatus.Ready, 25_000);
    return { ok: true };
  } catch (e: any) {
    getVoiceConnection(guild.id)?.destroy();
    return { ok: false, error: e?.message || String(e) };
  }
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
