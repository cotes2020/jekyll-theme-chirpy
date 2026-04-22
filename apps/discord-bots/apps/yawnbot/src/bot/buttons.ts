import { EmbedBuilder, MessageFlags, ButtonInteraction } from 'discord.js';
import { getRandomImage } from '../services/gamedata';
import { showHelpPage, handleEnhance, handleSell } from './game-ui';
import { tryHandleMusicQueueButton } from './slash/music';
import type { BotContext } from './slash/bot-context';
import { MOOD_REACTION_MAP, type MoodReactionEmoji } from './assistant-handler';

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

    if (customId.startsWith('mood_reaction:')) {
      const [, emoji, slug] = customId.split(':');
      const moodDesc = MOOD_REACTION_MAP[emoji as MoodReactionEmoji];
      if (moodDesc && slug && ctx.getMood) {
        ctx.getMood(slug).set(moodDesc);
        console.log(`[Button] 기분 반응 수신: ${emoji} → ${moodDesc} (slug=${slug})`);
      }
      await interaction.reply({ content: emoji, flags: MessageFlags.Ephemeral });
      return;
    }
  } catch (err) {
    console.error('[Button Error]', err);
    await interaction.reply({ content: '오류가 발생했습니다.', flags: MessageFlags.Ephemeral }).catch(() => {});
  }
}
