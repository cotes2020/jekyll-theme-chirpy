import { EmbedBuilder, MessageFlags } from 'discord.js';
import type { ChatInputCommandInteraction } from 'discord.js';
import type { BotContext } from './bot-context';

export async function handleAdminReload(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string): Promise<void> {
  const { gameData, isAdmin } = ctx;
  if (!isAdmin(userId)) {
    await interaction.reply({ content: gameData.getMessage('Admin_AccessDenied_Desc'), flags: MessageFlags.Ephemeral });
    return;
  }
  await gameData.initialize();
  const embed = new EmbedBuilder()
    .setTitle(gameData.getMessage('Admin_Reload_Title'))
    .setDescription(gameData.getMessage('Admin_Reload_Desc'))
    .setColor(0x4caf50);
  await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
}

export async function handleAdminSave(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string): Promise<void> {
  const { gameData, isAdmin } = ctx;
  if (!isAdmin(userId)) {
    await interaction.reply({ content: gameData.getMessage('Admin_AccessDenied_Desc'), flags: MessageFlags.Ephemeral });
    return;
  }
  gameData.saveGameData();
  const embed = new EmbedBuilder()
    .setTitle(gameData.getMessage('Admin_Save_Title'))
    .setDescription(gameData.getMessage('Admin_Save_Desc'))
    .setColor(0x4caf50);
  await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
}

