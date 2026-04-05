// @ts-nocheck
import { MessageFlags, PermissionFlagsBits } from 'discord.js';
import { enqueueSpokenText, withTimeout } from '../music-player';

const SPEAK_MAX_CHARS = 500;
const DEFAULT_LINE = 'Hello, world. 운이빈다 Edge TTS 테스트입니다.';
const SYNTH_TIMEOUT_MS = 45_000;

export async function handleSpeak(ctx, interaction) {
  if (!interaction.inGuild()) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }

  try {
    await interaction.deferReply();
  } catch (e) {
    const code = e && typeof e === 'object' && 'code' in e ? e.code : undefined;
    if (code === 10062) {
      console.warn('[speak] deferReply 10062 (Unknown interaction)');
      return;
    }
    throw e;
  }

  if (!interaction.member) {
    await interaction.editReply({ content: '멤버 정보를 불러올 수 없습니다. 잠시 후 다시 시도하세요.' });
    return;
  }

  const vc = interaction.member.voice?.channel;
  if (!vc || !vc.isVoiceBased()) {
    await interaction.editReply({ content: '음성 채널에 들어간 뒤 `/music speak`를 사용하세요.' });
    return;
  }

  const botMember = interaction.guild.members.me;
  if (!botMember) {
    await interaction.editReply({ content: '봇 멤버 정보를 불러올 수 없습니다.' });
    return;
  }

  const perms = vc.permissionsFor(botMember);
  if (!perms?.has([PermissionFlagsBits.Connect, PermissionFlagsBits.Speak, PermissionFlagsBits.ViewChannel])) {
    await interaction.editReply({
      content: '봇에게 해당 음성 채널 **보기·연결·말하기(Speak)** 권한이 필요합니다.',
    });
    return;
  }

  const raw = interaction.options.getString('text');
  const text = (raw?.trim() || DEFAULT_LINE).slice(0, SPEAK_MAX_CHARS);
  const queueTitle = `TTS: ${text.slice(0, 80)}${text.length > 80 ? '…' : ''}`;

  await interaction.editReply({ content: '음성 합성 중… (Microsoft Edge TTS)' });

  let result;
  try {
    result = await withTimeout(
      enqueueSpokenText(vc, queueTitle, text),
      SYNTH_TIMEOUT_MS,
      'TTS 합성·재생 준비',
    );
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    await interaction.editReply({ content: `실패: ${msg}` });
    return;
  }

  if (!result.ok) {
    await interaction.editReply({ content: `재생 준비 실패: ${result.error}` });
    return;
  }

  await interaction.editReply({
    content: result.started
      ? `**${queueTitle}** 재생을 시작합니다.`
      : `대기열 **${result.position}번**에 추가했습니다: **${queueTitle}**`,
  });
}
