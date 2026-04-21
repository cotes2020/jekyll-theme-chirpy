// @ts-nocheck
import { EmbedBuilder, MessageFlags } from 'discord.js';
import { CharacterService } from '../../services/character-service';

function resolveScheduleForInteraction(ctx, interaction) {
  const cs = ctx.characterService;
  const getSchedule = ctx.getSchedule;
  if (!cs || !getSchedule) {
    return null;
  }
  const isDM = !interaction.guildId;
  const channelKey = CharacterService.channelKey({
    isDM,
    userId: interaction.user.id,
    channelId: interaction.channelId ?? '',
  });
  const card = cs.resolveCard(channelKey);
  if (!card) return null;
  return { card, schedule: getSchedule(card.slug) };
}

/** "YYYY-MM-DD HH:MM" (KST) → UTC ISO string */
function parseKSTDatetime(input: string): string | null {
  const m = input.trim().match(/^(\d{4})-(\d{2})-(\d{2})[T\s](\d{2}):(\d{2})$/);
  if (!m) return null;
  const [, y, mo, d, h, mi] = m.map(Number);
  const utcMs = Date.UTC(y, mo - 1, d, h - 9, mi);
  return isNaN(utcMs) ? null : new Date(utcMs).toISOString();
}

function formatKST(iso: string): string {
  return new Date(iso).toLocaleString('ko-KR', {
    timeZone: 'Asia/Seoul',
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  });
}

export async function handleScheduleAdd(ctx, interaction): Promise<void> {
  const resolved = resolveScheduleForInteraction(ctx, interaction);
  if (!resolved) {
    await interaction.reply({ content: '활성 캐릭터가 없어요. `/character list` 로 확인해봐요.', flags: MessageFlags.Ephemeral });
    return;
  }

  const title = interaction.options.getString('제목', true);
  const datetimeRaw = interaction.options.getString('날짜시간', true);
  const notifyMinutes = interaction.options.getInteger('알림') ?? 10;

  const iso = parseKSTDatetime(datetimeRaw);
  if (!iso) {
    await interaction.reply({
      content: '날짜 형식이 올바르지 않아요. `YYYY-MM-DD HH:MM` 형식으로 입력해주세요.\n예: `2026-04-25 14:30`',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const entry = resolved.schedule.add(title, iso, notifyMinutes);
  const unixSec = Math.floor(new Date(iso).getTime() / 1000);
  const embed = new EmbedBuilder()
    .setTitle('✅ 일정 추가됨')
    .addFields(
      { name: '제목', value: entry.title },
      { name: '일시 (KST)', value: `<t:${unixSec}:f>`, inline: true },
      { name: '알림', value: `${entry.notifyMinutes}분 전`, inline: true },
      { name: 'ID', value: `\`${entry.id}\``, inline: true },
    )
    .setColor(0x4caf50);
  await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
}

export async function handleScheduleList(ctx, interaction): Promise<void> {
  const resolved = resolveScheduleForInteraction(ctx, interaction);
  if (!resolved) {
    await interaction.reply({ content: '활성 캐릭터가 없어요. `/character list` 로 확인해봐요.', flags: MessageFlags.Ephemeral });
    return;
  }

  const entries = resolved.schedule.list().filter((e) => !e.notified);
  if (entries.length === 0) {
    await interaction.reply({ content: '등록된 일정이 없어요.', flags: MessageFlags.Ephemeral });
    return;
  }

  const lines = entries.map((e) => {
    const unixSec = Math.floor(new Date(e.datetime).getTime() / 1000);
    return `\`${e.id}\` **${e.title}** — <t:${unixSec}:f> (${e.notifyMinutes}분 전 알림)`;
  });

  const embed = new EmbedBuilder()
    .setTitle('📅 예정된 일정')
    .setDescription(lines.join('\n'))
    .setColor(0x7c4dff);
  await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
}

export async function handleScheduleDelete(ctx, interaction): Promise<void> {
  const resolved = resolveScheduleForInteraction(ctx, interaction);
  if (!resolved) {
    await interaction.reply({ content: '활성 캐릭터가 없어요. `/character list` 로 확인해봐요.', flags: MessageFlags.Ephemeral });
    return;
  }

  const id = interaction.options.getString('id', true);
  const deleted = resolved.schedule.remove(id);
  await interaction.reply({
    content: deleted ? `일정 \`${id}\` 삭제됨.` : `일정 \`${id}\`를 찾을 수 없어요.`,
    flags: MessageFlags.Ephemeral,
  });
}
