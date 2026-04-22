import { EmbedBuilder, MessageFlags, ButtonInteraction, AttachmentBuilder } from 'discord.js';
import { getRandomImage } from '../services/gamedata';
import { showHelpPage, handleEnhance, handleSell } from './game-ui';
import { tryHandleMusicQueueButton } from './slash/music';
import type { BotContext } from './slash/bot-context';
import { MOOD_REACTION_MAP, type MoodReactionEmoji } from './assistant-handler';
import { handleGalleryButton } from './slash/gallery';
import { generateImageFromEnvWithOptions } from 'karmolab-ai/node';

export async function handleButtonInteraction(ctx: BotContext, interaction: ButtonInteraction): Promise<void> {
  if (!interaction.isButton()) return;
  if (await tryHandleMusicQueueButton(interaction)) return;
  const customId = interaction.customId;
  const userId = interaction.user.id;
  const userName = interaction.user.displayName || interaction.user.username;
  const { gameData, getImageAttachment } = ctx;

  try {
    if (customId.startsWith('help_page:')) {
      const pageIndex = parseInt(customId.split(':')[1], 10);
      await showHelpPage(ctx, interaction, pageIndex, true);
      return;
    }

    if (customId === 'consolation') {
      const imageName = getRandomImage('위로(놀림)_');
      const wImg = getImageAttachment(imageName);
      const embed = new EmbedBuilder()
        .setTitle(gameData.getMessage('Consolation_Title'))
        .setDescription(gameData.getMessage('Consolation_Desc', `<@${userId}>`))
        .setColor(0xff00ff);
      const payload: Parameters<typeof interaction.channel.send>[0] = { embeds: [embed] };
      if (wImg) Object.assign(payload, { files: [wImg.file], embeds: [embed.setImage(`attachment://${wImg.name}`)] });

      await interaction.channel!.send(payload);
      await interaction.deferUpdate();
      return;
    }

    if (customId === 'enhance_retry') {
      await handleEnhance(ctx, interaction, userId, userName, true);
      return;
    }
    if (customId === 'sell_sword') {
      await handleSell(ctx, interaction, userId, true);
      return;
    }

    if (customId.startsWith('image_vary:')) {
      const action = customId.split(':')[1];
      const embed = interaction.message.embeds[0];
      const basePrompt = embed?.description?.trim();
      if (!basePrompt) { await interaction.reply({ content: '프롬프트를 찾을 수 없어요.', flags: MessageFlags.Ephemeral }); return; }

      const aspectRatio = action === 'wide' ? '16:9' : action === 'portrait' ? '9:16' : undefined;
      const finalPrompt = action === 'pose'
        ? `${basePrompt}, different pose, different angle, dynamic composition`
        : basePrompt;

      await interaction.deferReply();
      try {
        const { images, modelId } = await generateImageFromEnvWithOptions(process.env, finalPrompt, {
          sampleCount: 1,
          aspectRatio: aspectRatio as Parameters<typeof generateImageFromEnvWithOptions>[2]['aspectRatio'],
        });
        if (!images.length) { await interaction.editReply('이미지 생성 결과 없음'); return; }
        const img = images[0];
        const ext = (img.mimeType.split('/')[1] || 'png').replace(/[^a-z0-9]/gi, '');
        const attachment = new AttachmentBuilder(img.buffer, { name: `vary.${ext}` });
        await interaction.editReply({ content: `🎨 변형 완료 (${modelId})`, files: [attachment] });
      } catch (e) {
        await interaction.editReply(`실패: ${e instanceof Error ? e.message : String(e)}`);
      }
      return;
    }

    if (customId.startsWith('gallery:')) {
      await handleGalleryButton(ctx, interaction);
      return;
    }

    if (customId.startsWith('mood_reaction:')) {
      const [, emoji, slug] = customId.split(':');
      const moodDesc = MOOD_REACTION_MAP[emoji as MoodReactionEmoji];
      if (moodDesc && slug) {
        if (ctx.getMood) {
          ctx.getMood(slug).set(moodDesc);
          console.log(`[Button] 기분 반응 수신: ${emoji} → ${moodDesc} (slug=${slug})`);
        }
        if (ctx.getRelationship) {
          ctx.getRelationship(slug).addMoodReaction(emoji);
        }
      }
      await interaction.reply({ content: emoji, flags: MessageFlags.Ephemeral });
      return;
    }
  } catch (err) {
    console.error('[Button Error]', err);
    await interaction.reply({ content: '오류가 발생했습니다.', flags: MessageFlags.Ephemeral }).catch(() => {});
  }
}
