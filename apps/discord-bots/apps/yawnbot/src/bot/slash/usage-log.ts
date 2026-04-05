// @ts-nocheck

function usageLogEnabled(): boolean {
  const s = String(process.env.YAWNBOT_SLASH_USAGE_LOG ?? '').trim().toLowerCase();
  return s === '1' || s === 'true' || s === 'yes' || s === 'on';
}

/**
 * 콘솔 한 줄: 명령·서브커맨드·길드·채널·유저
 */
export function logSlashUsage(interaction): void {
  if (!usageLogEnabled() || !interaction.isChatInputCommand()) return;
  let sub = '';
  try {
    sub = interaction.options.getSubcommand(false) ?? '';
  } catch {
    /* no subcommand */
  }
  const parts = [
    '[slash]',
    interaction.commandName,
    sub || '-',
    `g=${interaction.guildId || 'dm'}`,
    `c=${interaction.channelId || '-'}`,
    `u=${interaction.user?.id || '-'}`,
  ];
  console.log(parts.join(' '));
}
