/**
 * /프로필 — 현재 활성 캐릭터 기준 유저 대시보드
 *
 * 친밀도 · 기분 · 오늘 일정 · 다가오는 기념일 · 뉴스 키워드를 한 화면에.
 */
import { EmbedBuilder, MessageFlags } from 'discord.js';
import type { ChatInputCommandInteraction } from 'discord.js';
import type { BotContext } from './bot-context';
import { RELATIONSHIP_LEVELS } from '../../services/relationship-service';
import { CharacterService } from '../../services/character-service';

export async function handleProfile(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const cs = ctx.characterService;
  if (!cs || !ctx.memoRepoPath) {
    await interaction.reply({ content: 'MEMO_REPO_PATH가 설정되지 않아 프로필 기능이 비활성화되어 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }

  const isDM = !interaction.guildId;
  const channelKey = CharacterService.channelKey({
    isDM,
    userId: interaction.user.id,
    channelId: interaction.channelId ?? '',
  });
  const card = cs.resolveCard(channelKey);
  if (!card) {
    await interaction.reply({ content: '활성 캐릭터 카드가 없어요. `/character list` 로 확인해봐요.', flags: MessageFlags.Ephemeral });
    return;
  }

  // ── 친밀도
  const relationship = ctx.getRelationship ? ctx.getRelationship(card.slug) : null;
  let relationshipField = '데이터 없음';
  if (relationship) {
    const info = relationship.getLevelInfo();
    const maxLevel = RELATIONSHIP_LEVELS[RELATIONSHIP_LEVELS.length - 1].level;
    const filled = '●'.repeat(info.level);
    const empty = '○'.repeat(maxLevel - info.level);
    const nextLevel = RELATIONSHIP_LEVELS.find((l) => l.level === info.level + 1);
    const progressLine = nextLevel
      ? `${relationship.conversationCount}회 / 다음 레벨 ${nextLevel.threshold}회`
      : `${relationship.conversationCount}회 (최고 레벨)`;
    const moodLine = relationship.moodScore !== 0
      ? ` · 호감도 ${relationship.moodScore >= 0 ? '+' : ''}${relationship.moodScore}`
      : '';
    relationshipField = `Lv.${info.level} **${info.label}** ${filled}${empty}\n${progressLine}${moodLine}`;
  }

  // ── 기분
  const mood = ctx.getMood ? ctx.getMood(card.slug) : null;
  const currentMood = mood?.get()?.mood || '알 수 없음';
  const carryOver = mood?.getCarryOver();
  const moodField = carryOver
    ? `${currentMood}\n*어제 기분(${carryOver.displayMood})이 남아 있어요*`
    : currentMood;

  // ── 오늘 일정
  const schedule = ctx.getSchedule ? ctx.getSchedule(card.slug) : null;
  let scheduleField = '일정 없음';
  if (schedule) {
    const kstNow = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
    const todayStart = new Date(kstNow.getFullYear(), kstNow.getMonth(), kstNow.getDate()).getTime();
    const todayEnd = todayStart + 24 * 60 * 60 * 1000;
    const todayEntries = schedule.list().filter((e) => {
      const t = new Date(e.datetime).getTime();
      return t >= todayStart && t < todayEnd;
    });
    if (todayEntries.length > 0) {
      scheduleField = todayEntries.slice(0, 5).map((e) => {
        const kstTime = new Date(e.datetime).toLocaleString('ko-KR', {
          timeZone: 'Asia/Seoul', hour: '2-digit', minute: '2-digit', hour12: false,
        });
        return `${kstTime} ${e.title}`;
      }).join('\n');
      if (todayEntries.length > 5) scheduleField += `\n외 ${todayEntries.length - 5}개`;
    }
  }

  // ── 다가오는 기념일 (14일 이내)
  const anniversary = ctx.getAnniversary ? ctx.getAnniversary(card.slug) : null;
  let anniversaryField = '없음';
  if (anniversary) {
    const upcoming = anniversary.getUpcoming(14);
    if (upcoming.length > 0) {
      anniversaryField = upcoming.slice(0, 3).map((a) => {
        const dLabel = a.dDay === 0 ? 'D-Day!' : `D-${a.dDay}`;
        const yearsLabel = a.years != null ? ` (${a.years}주년)` : '';
        return `${dLabel} **${a.label}**${yearsLabel}`;
      }).join('\n');
    }
  }

  // ── 뉴스 키워드
  const news = ctx.getNews ? ctx.getNews(card.slug) : null;
  const keywords = news?.getKeywords() ?? [];
  const newsField = keywords.length > 0 ? keywords.map((k) => k.keyword).join(', ') : '없음';

  const embed = new EmbedBuilder()
    .setTitle(`${card.name} 프로필`)
    .setColor(0x7c4dff)
    .addFields(
      { name: '🤝 친밀도', value: relationshipField },
      { name: '😊 기분', value: moodField, inline: true },
      { name: '📅 오늘 일정', value: scheduleField, inline: true },
      { name: '🎂 기념일 (14일 이내)', value: anniversaryField, inline: true },
      { name: '📰 뉴스 키워드', value: newsField, inline: true },
    )
    .setFooter({ text: `캐릭터: ${card.slug}` })
    .setTimestamp();

  await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
}
