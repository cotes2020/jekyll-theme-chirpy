import { EmbedBuilder, MessageFlags } from 'discord.js';
import type { ChatInputCommandInteraction } from 'discord.js';
import type { BotContext } from './bot-context';

export async function handleNewsKeywordList(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  if (!ctx.getNews) {
    await interaction.reply({ content: 'MEMO_REPO_PATH가 설정되지 않아 뉴스 기능이 비활성화돼 있어요.', flags: MessageFlags.Ephemeral });
    return;
  }
  const slug = ctx.characterService?.getDefaultSlug() ?? 'default';
  const keywords = ctx.getNews(slug).getKeywords();
  const embed = new EmbedBuilder()
    .setTitle('📰 관심사 키워드 목록')
    .setColor(0x2196f3);
  if (keywords.length === 0) {
    embed.setDescription('등록된 키워드가 없어요. `/일정 키워드 추가`로 키워드를 등록해봐요.');
  } else {
    embed.setDescription(
      keywords.map((k, i) => `\`${k.id}\` **${i + 1}.** ${k.keyword}`).join('\n'),
    );
    embed.setFooter({ text: `총 ${keywords.length}개` });
  }
  await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
}

export async function handleNewsKeywordAdd(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  if (!ctx.getNews) {
    await interaction.reply({ content: 'MEMO_REPO_PATH가 설정되지 않아 뉴스 기능이 비활성화돼 있어요.', flags: MessageFlags.Ephemeral });
    return;
  }
  const keyword = (interaction.options.getString('키워드', true) || '').trim();
  const slug = ctx.characterService?.getDefaultSlug() ?? 'default';
  const entry = ctx.getNews(slug).addKeyword(keyword);
  if (!entry) {
    await interaction.reply({ content: `이미 등록된 키워드이거나 유효하지 않아요: **${keyword}**`, flags: MessageFlags.Ephemeral });
    return;
  }
  await interaction.reply({ content: `📰 키워드 추가 완료: **${keyword}**`, flags: MessageFlags.Ephemeral });
}

export async function handleNewsKeywordDelete(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  if (!ctx.getNews) {
    await interaction.reply({ content: 'MEMO_REPO_PATH가 설정되지 않아 뉴스 기능이 비활성화돼 있어요.', flags: MessageFlags.Ephemeral });
    return;
  }
  const id = (interaction.options.getString('id', true) || '').trim();
  const slug = ctx.characterService?.getDefaultSlug() ?? 'default';
  const ok = ctx.getNews(slug).removeKeyword(id);
  if (!ok) {
    await interaction.reply({ content: `해당 ID의 키워드를 찾을 수 없어요: \`${id}\``, flags: MessageFlags.Ephemeral });
    return;
  }
  await interaction.reply({ content: `🗑️ 키워드 삭제 완료 (\`${id}\`)`, flags: MessageFlags.Ephemeral });
}
