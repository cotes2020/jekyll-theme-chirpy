// @ts-nocheck
import fs from 'fs';
import path from 'path';
import { MessageFlags, AttachmentBuilder, EmbedBuilder } from 'discord.js';
import { generateImageFromEnvWithOptions } from 'karmolab-ai/node';
import { CharacterService } from '../../services/character-service';
import type { CharacterCard } from '../../services/character-service';

const MAX_PROMPT_CHARS = 1500;

/** card.dir/appearance.md 본문 (frontmatter 제거). 없으면 card.imageStyle 폴백. */
export function loadAppearance(card: CharacterCard): string {
  const appearancePath = path.join(card.dir, 'appearance.md');
  if (fs.existsSync(appearancePath)) {
    try {
      const raw = fs.readFileSync(appearancePath, 'utf-8');
      const fm = raw.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n?([\s\S]*)$/);
      const body = (fm ? fm[1] : raw).trim();
      if (body) return body;
    } catch {
      /* ignore */
    }
  }
  return card.imageStyle?.trim() || '';
}

/** appearance.md(또는 image_style) 본문 + 상황을 Scene 포맷으로 결합 */
export function buildCharacterImagePrompt(card: CharacterCard, situation: string): string {
  const appearance = loadAppearance(card);
  const s = situation.trim();
  if (!appearance) return s;
  return `${appearance}\n\nScene: ${s}`;
}

/**
 * 공용 이미지 생성 + embed 응답.
 * 호출 전에 interaction.deferReply() 가 완료돼있어야 함 (editReply 로 응답하므로).
 */
export async function runImageGeneration(
  interaction,
  finalPrompt: string,
  opts: {
    modelId?: string | null;
    aspectRatio?: string;
    sampleCount?: number;
    negativePrompt?: string;
    /** embed에 표시할 원본 프롬프트(사용자가 입력한 것) */
    displayPrompt?: string;
    /** embed에 표시할 캐릭터 라벨 (slug 또는 "none") */
    characterLabel?: string;
  } = {},
): Promise<void> {
  const startedAt = Date.now();
  try {
    const { images, modelId: effectiveModelId } = await generateImageFromEnvWithOptions(
      process.env,
      finalPrompt,
      {
        modelId: opts.modelId,
        sampleCount: Math.max(1, Math.min(4, opts.sampleCount ?? 1)),
        aspectRatio: opts.aspectRatio,
        negativePrompt: opts.negativePrompt,
      },
    );

    const files = images.map((img, idx) => {
      const ext = (img.mimeType.split('/')[1] || 'png').replace(/[^a-z0-9]/gi, '');
      return new AttachmentBuilder(img.buffer, { name: `image-${idx + 1}.${ext}` });
    });

    const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
    const embed = new EmbedBuilder()
      .setTitle('🎨 이미지 생성')
      .setDescription((opts.displayPrompt || finalPrompt).slice(0, 400))
      .addFields(
        { name: '캐릭터', value: opts.characterLabel || 'none', inline: true },
        { name: '모델', value: effectiveModelId, inline: true },
        { name: '비율', value: opts.aspectRatio || '1:1', inline: true },
        { name: '개수', value: String(images.length), inline: true },
        { name: '소요', value: `${elapsed}초`, inline: true },
      )
      .setColor(0x7c4dff);

    if (opts.negativePrompt) {
      embed.addFields({ name: '네거티브', value: opts.negativePrompt.slice(0, 256) });
    }

    await interaction.editReply({ embeds: [embed], files });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    await interaction.editReply({ content: `이미지 생성 실패: ${msg.slice(0, 800)}` });
  }
}

export async function handleImage(ctx, interaction) {
  const prompt = (interaction.options.getString('프롬프트', true) || '').trim();
  const modelId = interaction.options.getString('모델') || null;
  const aspectRatio = interaction.options.getString('비율') || undefined;
  const count = interaction.options.getInteger('개수') ?? 1;
  const negativePrompt = interaction.options.getString('네거티브') || undefined;
  const charOpt = (interaction.options.getString('캐릭터') || '').trim().toLowerCase();

  if (!prompt) {
    await interaction.reply({
      content: '프롬프트를 입력해주세요.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }
  if (prompt.length > MAX_PROMPT_CHARS) {
    await interaction.reply({
      content: `프롬프트가 너무 깁니다 (${MAX_PROMPT_CHARS}자 제한, 현재 ${prompt.length}자).`,
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  // 캐릭터 해석: ''=활성, 'none'=캐릭터 없이, slug=지정 캐릭터
  const cs = ctx.characterService;
  let card: CharacterCard | null = null;
  let characterLabel = 'none';
  if (charOpt !== 'none' && cs) {
    if (charOpt) {
      const available = cs.listCharacters();
      if (!available.includes(charOpt)) {
        await interaction.reply({
          content: `캐릭터 슬러그 없음: ${charOpt}. 사용 가능: ${available.join(', ') || '없음'}`,
          flags: MessageFlags.Ephemeral,
        });
        return;
      }
      card = cs.loadCard(charOpt);
    } else {
      const isDM = !interaction.guildId;
      const channelKey = CharacterService.channelKey({
        isDM,
        userId: interaction.user.id,
        channelId: interaction.channelId ?? '',
      });
      card = cs.resolveCard(channelKey);
    }
    if (card) characterLabel = card.slug;
  }

  try {
    await interaction.deferReply();
  } catch (e) {
    if (e && typeof e === 'object' && 'code' in e && e.code === 10062) {
      console.warn('[image] deferReply 10062 (Unknown interaction)');
      return;
    }
    throw e;
  }

  const finalPrompt = card ? buildCharacterImagePrompt(card, prompt) : prompt;

  await runImageGeneration(interaction, finalPrompt, {
    modelId,
    aspectRatio,
    sampleCount: count,
    negativePrompt,
    displayPrompt: prompt,
    characterLabel,
  });
}
