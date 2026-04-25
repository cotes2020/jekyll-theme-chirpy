/**
 * gallery.ts — /갤러리 커맨드
 *
 * 캐릭터별 image-cache/index.json을 읽어 최근 이미지를 Discord embed로 표시.
 * hitCount(많이 재사용된) 또는 createdAt(최신) 기준 정렬 선택 가능.
 * 한 번에 1장씩 표시, ◀ ▶ 버튼으로 페이지 이동.
 */
import fs from 'fs';
import path from 'path';
import {
  EmbedBuilder,
  MessageFlags,
  AttachmentBuilder,
  ActionRowBuilder,
  ButtonBuilder,
  ButtonStyle,
} from 'discord.js';
import type { ChatInputCommandInteraction, ButtonInteraction } from 'discord.js';
import type { BotContext } from './bot-context';
import { CharacterService } from '../../services/character-service';
import type { CacheEntry } from '../../services/image-cache-service';

interface GalleryIndex { scenes: CacheEntry[] }

function loadIndex(memoRepoPath: string, slug: string): CacheEntry[] {
  const indexPath = path.join(memoRepoPath, 'characters', slug, 'image-cache', 'index.json');
  if (!fs.existsSync(indexPath)) return [];
  try {
    return (JSON.parse(fs.readFileSync(indexPath, 'utf-8')) as GalleryIndex).scenes ?? [];
  } catch {
    return [];
  }
}

function buildGalleryEmbed(entry: CacheEntry, index: number, total: number, slug: string): EmbedBuilder {
  const createdAt = new Date(entry.createdAt).toLocaleString('ko-KR', { timeZone: 'Asia/Seoul' });
  return new EmbedBuilder()
    .setTitle(`🖼️ ${slug} 갤러리 (${index + 1} / ${total})`)
    .setDescription(`**Scene**: ${entry.sceneDesc?.slice(0, 300) || '(없음)'}`)
    .addFields(
      { name: '생성일', value: createdAt, inline: true },
      { name: '재사용', value: `${entry.hitCount}회`, inline: true },
    )
    .setImage(`attachment://scene.png`)
    .setColor(0x7c4dff);
}

function buildNavRow(index: number, total: number, slug: string, sort: string): ActionRowBuilder<ButtonBuilder> {
  return new ActionRowBuilder<ButtonBuilder>().addComponents(
    new ButtonBuilder()
      .setCustomId(`gallery:prev:${slug}:${index}:${sort}`)
      .setLabel('◀')
      .setStyle(ButtonStyle.Secondary)
      .setDisabled(index === 0),
    new ButtonBuilder()
      .setCustomId(`gallery:next:${slug}:${index}:${sort}`)
      .setLabel('▶')
      .setStyle(ButtonStyle.Secondary)
      .setDisabled(index === total - 1),
  );
}

function sortedEntries(entries: CacheEntry[], sort: string): CacheEntry[] {
  if (sort === 'popular') {
    return [...entries].sort((a, b) => b.hitCount - a.hitCount);
  }
  // 기본: 최신순
  return [...entries].sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
}

export async function handleGallery(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  if (!ctx.characterService || !ctx.memoRepoPath) {
    await interaction.reply({ content: 'MEMO_REPO_PATH 또는 캐릭터 서비스 미설정.', flags: MessageFlags.Ephemeral });
    return;
  }

  const isDM = !interaction.guildId;
  const channelKey = CharacterService.channelKey({ isDM, userId: interaction.user.id, channelId: interaction.channelId });
  const card = ctx.characterService.resolveCard(channelKey);
  if (!card) {
    await interaction.reply({ content: '활성 캐릭터가 없어요.', flags: MessageFlags.Ephemeral });
    return;
  }

  const sort = interaction.options.getString('정렬') ?? 'recent';
  const entries = sortedEntries(loadIndex(ctx.memoRepoPath, card.slug), sort);

  if (!entries.length) {
    await interaction.reply({ content: `${card.displayName}의 이미지 캐시가 비어 있어요.`, flags: MessageFlags.Ephemeral });
    return;
  }

  const entry = entries[0];
  if (!fs.existsSync(entry.filePath)) {
    await interaction.reply({ content: '이미지 파일을 찾을 수 없어요.', flags: MessageFlags.Ephemeral });
    return;
  }

  const attachment = new AttachmentBuilder(entry.filePath, { name: 'scene.png' });
  const embed = buildGalleryEmbed(entry, 0, entries.length, card.slug);
  const row = buildNavRow(0, entries.length, card.slug, sort);

  await interaction.reply({ embeds: [embed], files: [attachment], components: [row], flags: MessageFlags.Ephemeral });
}

/** buttons.ts에서 호출 */
export async function handleGalleryButton(ctx: BotContext, interaction: ButtonInteraction): Promise<void> {
  const [, dir, slug, indexStr, sort] = interaction.customId.split(':');
  const currentIndex = parseInt(indexStr, 10);
  const nextIndex = dir === 'next' ? currentIndex + 1 : currentIndex - 1;

  if (!ctx.memoRepoPath) { await interaction.deferUpdate(); return; }

  const entries = sortedEntries(loadIndex(ctx.memoRepoPath, slug), sort);
  if (nextIndex < 0 || nextIndex >= entries.length) { await interaction.deferUpdate(); return; }

  const entry = entries[nextIndex];
  if (!fs.existsSync(entry.filePath)) { await interaction.deferUpdate(); return; }

  const attachment = new AttachmentBuilder(entry.filePath, { name: 'scene.png' });
  const embed = buildGalleryEmbed(entry, nextIndex, entries.length, slug);
  const row = buildNavRow(nextIndex, entries.length, slug, sort);

  await interaction.update({ embeds: [embed], files: [attachment], components: [row] });
}
