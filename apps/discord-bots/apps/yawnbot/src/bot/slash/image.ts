// @ts-nocheck
import { MessageFlags, AttachmentBuilder, EmbedBuilder } from 'discord.js';
import { generateImageFromEnvWithOptions } from 'karmolab-ai/node';

const MAX_PROMPT_CHARS = 1500;

export async function handleImage(ctx, interaction) {
  const prompt = (interaction.options.getString('프롬프트', true) || '').trim();
  const modelId = interaction.options.getString('모델') || null;
  const aspectRatio = interaction.options.getString('비율') || undefined;
  const count = interaction.options.getInteger('개수') ?? 1;
  const negativePrompt = interaction.options.getString('네거티브') || undefined;

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

  try {
    await interaction.deferReply();
  } catch (e) {
    if (e && typeof e === 'object' && 'code' in e && e.code === 10062) {
      console.warn('[image] deferReply 10062 (Unknown interaction)');
      return;
    }
    throw e;
  }

  const startedAt = Date.now();
  try {
    const { images, modelId: effectiveModelId } = await generateImageFromEnvWithOptions(
      process.env,
      prompt,
      {
        modelId,
        sampleCount: Math.max(1, Math.min(4, count)),
        aspectRatio,
        negativePrompt,
      },
    );

    const files = images.map((img, idx) => {
      const ext = (img.mimeType.split('/')[1] || 'png').replace(/[^a-z0-9]/gi, '');
      return new AttachmentBuilder(img.buffer, { name: `image-${idx + 1}.${ext}` });
    });

    const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
    const embed = new EmbedBuilder()
      .setTitle('🎨 이미지 생성')
      .setDescription(prompt.slice(0, 400))
      .addFields(
        { name: '모델', value: effectiveModelId, inline: true },
        { name: '비율', value: aspectRatio || '1:1', inline: true },
        { name: '개수', value: String(images.length), inline: true },
        { name: '소요', value: `${elapsed}초`, inline: true },
      )
      .setColor(0x7c4dff);

    if (negativePrompt) {
      embed.addFields({ name: '네거티브', value: negativePrompt.slice(0, 256) });
    }

    await interaction.editReply({ embeds: [embed], files });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    await interaction.editReply({ content: `이미지 생성 실패: ${msg.slice(0, 800)}` });
  }
}
