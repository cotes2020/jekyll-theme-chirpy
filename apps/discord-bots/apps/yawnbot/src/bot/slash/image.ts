import fs from 'fs';
import path from 'path';
import { MessageFlags, AttachmentBuilder, EmbedBuilder } from 'discord.js';
import type { ChatInputCommandInteraction } from 'discord.js';
import { generateImageFromEnvWithOptions } from 'karmolab-ai/node';
import type { CharacterCard } from '../../services/character-service';
import type { BotContext } from './bot-context';

const MAX_PROMPT_CHARS = 1500;

/**
 * Vertex Imagen 이미지당 단가 (USD, 2026-04 기준 공개 가격).
 * 알 수 없는 모델은 null — embed에서 "비용" 필드 생략.
 * 가격 변경 시 업데이트.
 */
export const IMAGEN_PRICE_PER_IMAGE: Record<string, number> = {
  // Imagen 4
  'imagen-4.0-fast-generate-001': 0.02,
  'imagen-4.0-generate-001': 0.04,
  'imagen-4.0-ultra-generate-001': 0.06,
  // Imagen 3 (참고용)
  'imagen-3.0-fast-generate-001': 0.02,
  'imagen-3.0-generate-001': 0.04,
  'imagen-3.0-generate-002': 0.04,
};

function estimateCost(modelId: string, count: number): string | null {
  const price = IMAGEN_PRICE_PER_IMAGE[modelId];
  if (price == null) return null;
  const total = price * count;
  return `$${total.toFixed(3)} (${count}장 × $${price.toFixed(2)})`;
}

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

/**
 * appearance(+image_style) + 상황을 결합.
 * - 상황 비어 있으면 외형만 (기본 포즈·프로필 이미지)
 * - 외형만 있으면 외형
 * - 상황만 있으면 상황
 * - 둘 다 비면 "{name} portrait" 폴백
 */
export function buildCharacterImagePrompt(card: CharacterCard, situation?: string): string {
  const appearance = loadAppearance(card);
  const s = (situation || '').trim();
  if (appearance && s) return `${appearance}\n\nScene: ${s}`;
  if (appearance) return appearance;
  if (s) return s;
  return `${card.displayName} portrait`;
}

/**
 * 공용 이미지 생성 + embed 응답.
 * 호출 전에 interaction.deferReply() 가 완료돼있어야 함 (editReply 로 응답하므로).
 */
export async function runImageGeneration(
  interaction: Pick<ChatInputCommandInteraction, 'editReply'>,
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
        aspectRatio: opts.aspectRatio as Parameters<typeof generateImageFromEnvWithOptions>[2]['aspectRatio'],
        negativePrompt: opts.negativePrompt,
      },
    );

    const files = images.map((img, idx) => {
      const ext = (img.mimeType.split('/')[1] || 'png').replace(/[^a-z0-9]/gi, '');
      return new AttachmentBuilder(img.buffer, { name: `image-${idx + 1}.${ext}` });
    });

    const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
    const cost = estimateCost(effectiveModelId, images.length);

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

    if (cost) {
      embed.addFields({ name: '예상 비용', value: cost, inline: true });
    }
    if (opts.negativePrompt) {
      embed.addFields({ name: '네거티브', value: opts.negativePrompt.slice(0, 256) });
    }

    await interaction.editReply({ embeds: [embed], files });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    await interaction.editReply({ content: `이미지 생성 실패: ${msg.slice(0, 800)}` });
  }
}

export async function handleImage(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
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

  // 캐릭터 해석: 비었거나 'none' → 캐릭터 없이 raw prompt. slug 입력 시에만 캐릭터 적용.
  const cs = ctx.characterService;
  let card: CharacterCard | null = null;
  let characterLabel = 'none';
  if (charOpt && charOpt !== 'none') {
    if (!cs) {
      await interaction.reply({
        content: 'MEMO_REPO_PATH가 설정되지 않아 캐릭터 기능이 비활성화돼 있어요.',
        flags: MessageFlags.Ephemeral,
      });
      return;
    }
    const available = cs.listCharacters();
    if (!available.includes(charOpt)) {
      await interaction.reply({
        content: `캐릭터 슬러그 없음: ${charOpt}. 사용 가능: ${available.join(', ') || '없음'}`,
        flags: MessageFlags.Ephemeral,
      });
      return;
    }
    card = cs.loadCard(charOpt);
    if (card) characterLabel = card.slug;
  }

  try {
    await interaction.deferReply();
  } catch (e) {
    if (e && typeof e === 'object' && 'code' in e && (e as { code: unknown }).code === 10062) {
      console.warn('[image] deferReply 10062 (Unknown interaction)');
      return;
    }
    throw e;
  }

  const finalPrompt = card ? buildCharacterImagePrompt(card, prompt) : prompt;

  // 카드에 negative_prompt가 있으면 사용자 입력 앞에 붙임
  const mergedNegativePrompt = [card?.negativePrompt, negativePrompt]
    .filter(Boolean)
    .join(', ') || undefined;

  await runImageGeneration(interaction, finalPrompt, {
    modelId,
    aspectRatio,
    sampleCount: count,
    negativePrompt: mergedNegativePrompt,
    displayPrompt: prompt,
    characterLabel,
  });
}
