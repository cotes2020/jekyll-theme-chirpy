/**
 * /character 슬래시 — DM/채널별 활성 캐릭터 관리
 */
import path from 'path';
import { EmbedBuilder, MessageFlags, AttachmentBuilder } from 'discord.js';
import type { ChatInputCommandInteraction } from 'discord.js';
import type { CharacterService, CharacterCard } from '../../services/character-service';
import { CharacterService as CSHelper } from '../../services/character-service';
import { buildCharacterImagePrompt, runImageGeneration } from './image';
import { ImageCacheService } from '../../services/image-cache-service';
import { RELATIONSHIP_LEVELS } from '../../services/relationship-service';
import type { BotContext } from './bot-context';

function getChannelKey(interaction: ChatInputCommandInteraction): string {
  const isDM = !interaction.guildId;
  return CSHelper.channelKey({
    isDM,
    userId: interaction.user.id,
    channelId: interaction.channelId ?? '',
  });
}

function summarizeCard(card: CharacterCard): string {
  const parts: string[] = [];
  if (card.tone) parts.push(`톤: ${card.tone}`);
  if (card.speechStyle) parts.push(`말투: ${card.speechStyle}`);
  if (card.relationship) parts.push(`관계: ${card.relationship}`);
  return parts.join(' · ') || '(frontmatter 없음)';
}

export async function handleCharacterList(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const cs: CharacterService | null = ctx.characterService;
  if (!cs) {
    await interaction.reply({
      content: 'MEMO_REPO_PATH가 설정되지 않아 캐릭터 시스템이 비활성화돼 있어요.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const channelKey = getChannelKey(interaction);
  const activeSlug = cs.resolveSlug(channelKey);
  const defaultSlug = cs.getDefaultSlug();
  const slugs = cs.listCharacters();

  if (slugs.length === 0) {
    await interaction.reply({
      content: '등록된 캐릭터가 없어요. `memo/characters/<slug>/card.md` 를 만들어주세요.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const embed = new EmbedBuilder()
    .setTitle('🎭 캐릭터 목록')
    .setDescription(
      `이 곳의 활성 캐릭터: **${activeSlug}**` +
        (activeSlug !== defaultSlug ? ` (default: ${defaultSlug})` : ' (default)'),
    )
    .setColor(0x7c4dff);

  for (const slug of slugs) {
    const card = cs.loadCard(slug);
    if (!card) {
      embed.addFields({ name: `${slug} ⚠️`, value: 'card.md 로드 실패' });
      continue;
    }
    const marker = slug === activeSlug ? ' ✅' : '';
    embed.addFields({
      name: `${card.displayName} (${slug})${marker}`,
      value: summarizeCard(card),
    });
  }

  await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
}

export async function handleCharacterSwitch(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const cs: CharacterService | null = ctx.characterService;
  if (!cs) {
    await interaction.reply({
      content: 'MEMO_REPO_PATH가 설정되지 않아 캐릭터 시스템이 비활성화돼 있어요.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const slug = interaction.options.getString('slug', true).trim();
  const channelKey = getChannelKey(interaction);

  try {
    cs.setChannelSlug(channelKey, slug);
  } catch (e) {
    const available = cs.listCharacters().join(', ') || '(없음)';
    await interaction.reply({
      content: `캐릭터 전환 실패: ${e instanceof Error ? e.message : String(e)}\n사용 가능: ${available}`,
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const card = cs.loadCard(slug);
  const embed = new EmbedBuilder()
    .setTitle('🎭 캐릭터 전환 완료')
    .setDescription(
      `이 ${interaction.guildId ? '채널' : 'DM'}에서 이제 **${card?.displayName ?? slug}** (${slug}) 가 응답합니다.`,
    )
    .addFields({ name: '설정', value: summarizeCard(card ?? ({} as CharacterCard)) })
    .setColor(0x4caf50);

  await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
}

export async function handleCharacterInfo(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const cs: CharacterService | null = ctx.characterService;
  if (!cs) {
    await interaction.reply({
      content: 'MEMO_REPO_PATH가 설정되지 않아 캐릭터 시스템이 비활성화돼 있어요.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const requested = interaction.options.getString('slug')?.trim();
  const slug = requested || cs.resolveSlug(getChannelKey(interaction));
  const card = cs.loadCard(slug);
  if (!card) {
    await interaction.reply({
      content: `캐릭터를 찾을 수 없어요: ${slug}`,
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const bodyPreview = card.body.length > 1000 ? card.body.slice(0, 1000) + '\n…' : card.body;

  const embed = new EmbedBuilder()
    .setTitle(`🎭 ${card.displayName} (${card.slug})`)
    .setDescription(summarizeCard(card))
    .addFields({ name: '시스템 프롬프트 (card.md 본문)', value: '```md\n' + bodyPreview + '\n```' })
    .setColor(0x7c4dff);

  await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
}

export async function handleCharacterReset(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const cs: CharacterService | null = ctx.characterService;
  if (!cs) {
    await interaction.reply({
      content: 'MEMO_REPO_PATH가 설정되지 않아 캐릭터 시스템이 비활성화돼 있어요.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const channelKey = getChannelKey(interaction);
  const wasSet = cs.resetChannel(channelKey);
  const defaultSlug = cs.getDefaultSlug();

  if (!wasSet) {
    await interaction.reply({
      content: `이 ${interaction.guildId ? '채널' : 'DM'}은 이미 default(**${defaultSlug}**)를 쓰고 있어요.`,
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  await interaction.reply({
    content: `매핑을 제거했어요. 이제 default(**${defaultSlug}**) 가 응답합니다.`,
    flags: MessageFlags.Ephemeral,
  });
}

export async function handleCharacterReload(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const cs: CharacterService | null = ctx.characterService;
  if (!cs) {
    await interaction.reply({ content: 'MEMO_REPO_PATH가 설정되지 않아 캐릭터 시스템이 비활성화돼 있어요.', flags: MessageFlags.Ephemeral });
    return;
  }

  const slugOpt = (interaction.options.getString('slug') || '').trim();
  const slug = slugOpt || cs.resolveSlug(getChannelKey(interaction));

  const card = cs.reloadCard(slug);
  if (!card) {
    await interaction.reply({ content: `캐릭터를 찾을 수 없어요: \`${slug}\``, flags: MessageFlags.Ephemeral });
    return;
  }

  await interaction.reply({
    content: `🔄 **${card.displayName}** (\`${slug}\`) 카드 캐시 재로드 완료.`,
    flags: MessageFlags.Ephemeral,
  });
}

/**
 * /character image — 현재 채널 활성 캐릭터의 외형(appearance.md)으로 이미지 생성.
 * 상황 비우면 외형만 써서 기본 포즈·프로필 이미지.
 */
export async function handleCharacterImage(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const cs: CharacterService | null = ctx.characterService;
  if (!cs) {
    await interaction.reply({
      content: 'MEMO_REPO_PATH가 설정되지 않아 캐릭터 시스템이 비활성화돼 있어요.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const situation = (interaction.options.getString('상황') || '').trim();
  const aspectRatio = interaction.options.getString('비율') || undefined;
  const count = interaction.options.getInteger('개수') ?? 1;

  const channelKey = getChannelKey(interaction);
  const card = cs.resolveCard(channelKey);
  if (!card) {
    await interaction.reply({
      content: '활성 캐릭터 카드가 없어요. `/character list` 로 확인해봐요.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  try {
    await interaction.deferReply();
  } catch (e) {
    if (e && typeof e === 'object' && 'code' in e && (e as { code: unknown }).code === 10062) {
      console.warn('[character.image] deferReply 10062');
      return;
    }
    throw e;
  }

  const finalPrompt = buildCharacterImagePrompt(card, situation || undefined);

  let saveDir: string | undefined;
  if (ctx.memoRepoPath) {
    const dateStr = new Date(Date.now() + 9 * 60 * 60 * 1000).toISOString().slice(0, 10);
    saveDir = path.join(ctx.memoRepoPath, 'image-log', card.slug, dateStr);
  }

  await runImageGeneration(interaction, finalPrompt, {
    aspectRatio,
    sampleCount: count,
    displayPrompt: situation || `${card.displayName} 기본 외형`,
    characterLabel: card.slug,
    saveDir,
  });
}

/**
 * /character image-history — 자동 생성된 씬 이미지 캐시 목록 조회.
 */
export async function handleCharacterImageHistory(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const cs: CharacterService | null = ctx.characterService;
  if (!cs) {
    await interaction.reply({
      content: 'MEMO_REPO_PATH가 설정되지 않아 캐릭터 시스템이 비활성화돼 있어요.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const channelKey = getChannelKey(interaction);
  const card = cs.resolveCard(channelKey);
  if (!card) {
    await interaction.reply({
      content: '활성 캐릭터 카드가 없어요. `/character list` 로 확인해봐요.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const memoRepoPath = process.env.MEMO_REPO_PATH?.trim() || '';
  const cacheService = new ImageCacheService(card.dir, memoRepoPath, card.slug);
  const scenes = cacheService.listScenes();

  if (scenes.length === 0) {
    await interaction.reply({
      content: `${card.displayName}의 이미지 캐시가 없어요. 대화하다 보면 자동으로 생성됩니다.`,
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  // 최신순 정렬, 최대 10개
  const sorted = [...scenes].sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
  ).slice(0, 10);

  const embed = new EmbedBuilder()
    .setTitle(`🖼️ ${card.displayName} 씬 이미지 캐시 (${scenes.length}개)`)
    .setDescription('최근 10개. 재사용 횟수 높을수록 자주 매칭된 장면.')
    .setColor(0x7c4dff);

  for (const entry of sorted) {
    const date = new Date(entry.createdAt).toLocaleDateString('ko-KR');
    const sceneLabel = entry.sceneDesc?.slice(0, 100) || '(no scene)';
    embed.addFields({
      name: sceneLabel,
      value: `히트: ${entry.hitCount}회 · ${date} · \`${entry.id.slice(0, 8)}\``,
      inline: false,
    });
  }

  // 가장 최근 이미지 첨부
  const latest = sorted[0];
  let files: AttachmentBuilder[] = [];
  try {
    const fs = await import('fs');
    if (fs.existsSync(latest.filePath)) {
      const buffer = fs.readFileSync(latest.filePath);
      const ext = (latest.mimeType.split('/')[1] || 'png').replace(/[^a-z0-9]/gi, '');
      files = [new AttachmentBuilder(buffer, { name: `latest.${ext}` })];
      embed.setImage(`attachment://latest.${ext}`);
    }
  } catch {
    /* ignore */
  }

  await interaction.reply({ embeds: [embed], files, flags: MessageFlags.Ephemeral });
}

export async function handleCharacterRelationship(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const cs = ctx.characterService;
  if (!cs) {
    await interaction.reply({ content: 'MEMO_REPO_PATH가 설정되지 않아 캐릭터 시스템이 비활성화돼 있어요.', flags: MessageFlags.Ephemeral });
    return;
  }
  if (!ctx.getRelationship) {
    await interaction.reply({ content: 'MEMO_REPO_PATH가 설정되지 않아 친밀도 시스템을 사용할 수 없어요.', flags: MessageFlags.Ephemeral });
    return;
  }

  const channelKey = getChannelKey(interaction);
  const card = cs.resolveCard(channelKey);
  if (!card) {
    await interaction.reply({ content: '활성 캐릭터 카드가 없어요. `/character switch`로 캐릭터를 선택해주세요.', flags: MessageFlags.Ephemeral });
    return;
  }

  const rel = ctx.getRelationship(card.slug);
  const info = rel.getLevelInfo();
  const nextLevel = RELATIONSHIP_LEVELS.find((l) => l.level === info.level + 1);
  const nextThreshold = nextLevel ? `${nextLevel.threshold}회` : '최대 레벨';
  const progressToNext = nextLevel
    ? `${rel.conversationCount} / ${nextLevel.threshold} (${Math.round(rel.conversationCount / nextLevel.threshold * 100)}%)`
    : '최대 달성!';

  const LEVEL_BARS = ['○○○○', '●○○○', '●●○○', '●●●○', '●●●●'];
  const bar = LEVEL_BARS[info.level] ?? '●●●●';

  const embed = new EmbedBuilder()
    .setTitle(`💞 ${card.displayName}와의 친밀도`)
    .setColor(0xe91e8c)
    .addFields(
      { name: '레벨', value: `Lv.${info.level} **${info.label}** ${bar}`, inline: true },
      { name: '총 대화', value: `${rel.conversationCount}회`, inline: true },
      { name: '호감도', value: `${rel.moodScore >= 0 ? '+' : ''}${rel.moodScore}`, inline: true },
      { name: '다음 단계까지', value: nextLevel ? `${info.label} → **${nextLevel.label}** (${progressToNext})` : '이미 최고 레벨이에요!', inline: false },
    )
    .setFooter({ text: info.hint });

  await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
}
