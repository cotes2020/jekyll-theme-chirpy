// @ts-nocheck
/**
 * /character 슬래시 — DM/채널별 활성 캐릭터 관리
 */
import { EmbedBuilder, MessageFlags } from 'discord.js';
import type { CharacterService, CharacterCard } from '../../services/character-service';
import { CharacterService as CSHelper } from '../../services/character-service';
import { buildCharacterImagePrompt, runImageGeneration } from './image';

function getChannelKey(interaction): string {
  // 길드 채널이 아니면 DM 으로 간주 (interaction 의 channel 이 DM 이거나 guildId 없음)
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

export async function handleCharacterList(ctx, interaction) {
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
      embed.addFields({
        name: `${slug} ⚠️`,
        value: 'card.md 로드 실패',
      });
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

export async function handleCharacterSwitch(ctx, interaction) {
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

export async function handleCharacterInfo(ctx, interaction) {
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

export async function handleCharacterReset(ctx, interaction) {
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

/**
 * /character image — 현재 채널 활성 캐릭터의 외형(appearance.md)으로 이미지 생성.
 * 상황 비우면 외형만 써서 기본 포즈/프로필 이미지.
 */
export async function handleCharacterImage(ctx, interaction) {
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
    if (e && typeof e === 'object' && 'code' in e && e.code === 10062) {
      console.warn('[character.image] deferReply 10062');
      return;
    }
    throw e;
  }

  const finalPrompt = buildCharacterImagePrompt(card, situation || undefined);
  await runImageGeneration(interaction, finalPrompt, {
    aspectRatio,
    sampleCount: count,
    displayPrompt: situation || `${card.displayName} 기본 외형`,
    characterLabel: card.slug,
  });
}
