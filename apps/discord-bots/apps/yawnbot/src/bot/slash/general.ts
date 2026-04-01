// @ts-nocheck
import { EmbedBuilder } from 'discord.js';
import { showHelpPage } from '../game-ui';

export async function handlePing(ctx, interaction) {
  const { gameData, client } = ctx;
  const embed = new EmbedBuilder()
    .setTitle(gameData.getMessage('General_Ping_Title'))
    .setDescription(gameData.getMessage('General_Ping_Desc', client.ws.ping))
    .setColor(0x00bcd4);
  await interaction.reply({ embeds: [embed] });
}

export async function handleHelp(ctx, interaction) {
  await showHelpPage(ctx, interaction, 0);
}

