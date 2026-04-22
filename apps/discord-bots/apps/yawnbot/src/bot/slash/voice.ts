import { MessageFlags, PermissionFlagsBits } from 'discord.js';
import type { ChatInputCommandInteraction, GuildMember, VoiceBasedChannel, GuildChannel } from 'discord.js';
import { joinVoiceChannelSafe, leaveVoiceChannel } from '../voice-connection';
import { destroyMusicForGuild } from '../music-player';
import type { BotContext } from './bot-context';

export async function handleVoiceJoin(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  if (!interaction.guild || !interaction.member) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const opt = interaction.options.getChannel('채널') ?? interaction.options.getChannel('channel');
  let target = opt;
  if (!target) {
    const member = interaction.member as GuildMember;
    const vc = member.voice?.channel;
    if (!vc || !vc.isVoiceBased()) {
      await interaction.reply({
        content: '음성 채널을 지정하거나, 명령을 실행할 때 음성 채널에 참가해 있어야 합니다.',
        flags: MessageFlags.Ephemeral,
      });
      return;
    }
    target = vc;
  }
  const guildChannel = target as GuildChannel;
  if (!('isVoiceBased' in guildChannel) || !guildChannel.isVoiceBased()) {
    await interaction.reply({ content: '음성 채널이 아닙니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const voiceChannel = guildChannel as unknown as VoiceBasedChannel;
  const botMember = interaction.guild.members.me;
  if (!botMember) {
    await interaction.reply({ content: '봇 멤버 정보를 불러올 수 없습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const perms = voiceChannel.permissionsFor(botMember);
  if (!perms?.has([PermissionFlagsBits.Connect, PermissionFlagsBits.ViewChannel])) {
    await interaction.reply({
      content: '봇에게 해당 채널의 **보기**·**연결** 권한이 필요합니다.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }
  await interaction.deferReply({ flags: MessageFlags.Ephemeral });
  await interaction.editReply({
    content: '음성 서버에 연결하는 중입니다. 잠시만 기다려 주세요…',
  });
  /** Ready 대기가 길면 슬래시 응답이 멈춘 것처럼 보이므로 주기적으로 갱신 */
  let tick = 0;
  const progress = setInterval(() => {
    tick += 15;
    void interaction
      .editReply({ content: `음성 서버에 연결하는 중입니다… (${tick}초 경과)` })
      .catch(() => {});
  }, 15_000);
  let result;
  try {
    result = await joinVoiceChannelSafe(voiceChannel);
  } finally {
    clearInterval(progress);
  }
  if (!result.ok) {
    await interaction.editReply({
      content: `음성 채널 입장 실패: ${result.error}`,
    });
    return;
  }
  await interaction.editReply({
    content: `**${target.name}** 음성 채널에 연결했습니다.`,
  });
}

export async function handleVoiceLeave(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  if (!interaction.guildId) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  destroyMusicForGuild(interaction.guildId);
  const ok = leaveVoiceChannel(interaction.guildId);
  if (!ok) {
    await interaction.reply({ content: '이 서버에서 음성 채널에 연결되어 있지 않습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  await interaction.reply({ content: '음성 채널에서 나갔습니다.', flags: MessageFlags.Ephemeral });
}
