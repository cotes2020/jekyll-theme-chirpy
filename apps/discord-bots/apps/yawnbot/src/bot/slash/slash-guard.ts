// @ts-nocheck
import { MessageFlags } from 'discord.js';
import { parseCommaSeparatedEnv } from '@discord-bots/common';

function allowedGuildIdSet(): Set<string> | null {
  const raw = process.env.YAWNBOT_ALLOWED_GUILD_IDS ?? process.env.YAWNBOT_SLASH_GUILD_IDS;
  const ids = parseCommaSeparatedEnv(raw);
  if (ids.length === 0) return null;
  return new Set(ids);
}

function allowedSlashChannelIdSet(): Set<string> | null {
  const ids = parseCommaSeparatedEnv(process.env.YAWNBOT_ALLOWED_SLASH_CHANNEL_IDS);
  if (ids.length === 0) return null;
  return new Set(ids);
}

/**
 * 허용 목록이 비어 있으면 통과. 설정된 경우에만 길드·채널을 검사합니다.
 * @returns true면 계속 처리, false면 이미 reply 했음.
 */
export async function guardSlashInteraction(interaction): Promise<boolean> {
  if (!interaction.isChatInputCommand()) return true;

  const guildAllow = allowedGuildIdSet();
  if (guildAllow) {
    if (!interaction.guildId || !guildAllow.has(interaction.guildId)) {
      await interaction
        .reply({
          content: '이 봇은 등록된 서버에서만 슬래시 명령을 쓸 수 있습니다.',
          flags: MessageFlags.Ephemeral,
        })
        .catch(() => {});
      return false;
    }
  }

  const chAllow = allowedSlashChannelIdSet();
  if (chAllow) {
    const ch = interaction.channelId;
    if (!ch || !chAllow.has(ch)) {
      await interaction
        .reply({
          content: '이 채널에서는 슬래시 명령을 쓸 수 없습니다. 안내된 채널에서 시도하세요.',
          flags: MessageFlags.Ephemeral,
        })
        .catch(() => {});
      return false;
    }
  }

  return true;
}
