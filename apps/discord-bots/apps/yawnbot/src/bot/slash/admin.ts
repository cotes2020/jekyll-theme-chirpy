// @ts-nocheck
import { EmbedBuilder, MessageFlags } from 'discord.js';

export async function handleAdminReload(ctx, interaction, userId) {
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

export async function handleAdminSave(ctx, interaction, userId) {
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

