import { EmbedBuilder, MessageFlags } from 'discord.js';
import type { ChatInputCommandInteraction } from 'discord.js';
import { formatMoney, getLevelColor } from '../../services/gamedata';
import { handleEnhance, handleSell } from '../game-ui';
import type { BotContext } from './bot-context';

export async function handleEnhanceSlash(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string, userName: string): Promise<void> {
  await handleEnhance(ctx, interaction, userId, userName);
}

export async function handleSellSlash(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string): Promise<void> {
  await handleSell(ctx, interaction, userId);
}

export async function handleInfo(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string, userName: string): Promise<void> {
  const { gameData, enhancement, getImageAttachment } = ctx;
  const user = gameData.getUser(userId);
  if (!user.sword.weaponType || !user.sword.imageName) enhancement.ensureSword(user);

  const swordName = user.sword.name || gameData.getMessage('Info_NoName');
  const embed = new EmbedBuilder()
    .setTitle(gameData.getMessage('Info_Title', userName))
    .addFields(
      { name: gameData.getMessage('Info_SwordName'), value: `**${swordName}** (+${user.sword.level}강)`, inline: true },
      { name: gameData.getMessage('Info_MaxLevel'), value: `+${user.maxLevel}강`, inline: true },
      { name: gameData.getMessage('Info_Balance'), value: `${formatMoney(user.money)}원`, inline: true },
    )
    .setColor(getLevelColor(user.sword.level));

  const attachment = getImageAttachment(user.sword.imageName);
  const payload: any = { embeds: [embed] };
  if (attachment) {
    embed.setThumbnail(`attachment://${attachment.name}`);
    payload.files = [attachment.file];
  }
  await interaction.reply(payload);
}

export async function handleMoney(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string, userName: string): Promise<void> {
  const { gameData } = ctx;
  const user = gameData.getUser(userId);
  const embed = new EmbedBuilder()
    .setTitle(gameData.getMessage('Money_Title'))
    .setDescription(gameData.getMessage('Money_Desc', userName, formatMoney(user.money)))
    .setColor(0xffd700);
  await interaction.reply({ embeds: [embed] });
}

export async function handleRank(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const { gameData, client } = ctx;
  const all = Object.entries(gameData.users)
    .map(([id, u]) => ({ id, money: (u as any).money, level: (u as any).sword?.level || 0 }))
    .sort((a, b) => b.money - a.money)
    .slice(0, 10);
  let desc = '';
  for (let i = 0; i < all.length; i++) {
    const u = all[i];
    const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `${i + 1}.`;
    let name;
    try {
      const member = await client.users.fetch(u.id);
      name = member.displayName || member.username;
    } catch {
      name = gameData.getMessage('Rank_UnknownUser');
    }
    desc += `${medal} **${name}** — ${formatMoney(u.money)}원 (+${u.level}강)\n`;
  }
  const embed = new EmbedBuilder()
    .setTitle(gameData.getMessage('Rank_Title'))
    .setDescription(desc || gameData.getMessage('Rank_NoData'))
    .setColor(0xffd700);
  await interaction.reply({ embeds: [embed] });
}

export async function handleAttendance(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string): Promise<void> {
  const { gameData, enhancement } = ctx;
  const r = enhancement.checkAttendance(userId);
  const embed = new EmbedBuilder();
  if (r.type === 'ok') {
    embed
      .setTitle(gameData.getMessage('Attend_Complete_Title'))
      .setDescription(gameData.getMessage('Attend_Complete_Desc', formatMoney(r.reward)))
      .addFields({ name: gameData.getMessage('Attend_CurrentBalance'), value: `${formatMoney(r.balance)}원` })
      .setColor(0x4caf50);
  } else {
    embed
      .setTitle(gameData.getMessage('Attend_Wait_Title'))
      .setDescription(gameData.getMessage('Attend_Wait_Desc', r.min, r.sec))
      .setColor(0xff9800);
  }
  await interaction.reply({ embeds: [embed] });
}

export async function handleGiveMeMoney(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string): Promise<void> {
  const { gameData, enhancement } = ctx;
  const r = enhancement.giveMeMoney(userId);
  const embed = new EmbedBuilder()
    .setTitle(gameData.getMessage('GMM_Title'))
    .setDescription(gameData.getMessage('GMM_Desc'))
    .addFields(
      { name: gameData.getMessage('GMM_Amount'), value: `${formatMoney(r.amount)}원`, inline: true },
      { name: '잔액', value: `${formatMoney(r.balance)}원`, inline: true },
    )
    .setColor(0xffd700);
  await interaction.reply({ embeds: [embed] });
}

export async function handleBattle(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string, userName: string): Promise<void> {
  const { gameData, enhancement, getImageAttachment } = ctx;
  const target = interaction.options.getUser('상대');
  if (!target) {
    await interaction.reply({ content: gameData.getMessage('Battle_NoTarget_Desc'), flags: MessageFlags.Ephemeral });
    return;
  }
  if (target.id === userId) {
    await interaction.reply({ content: gameData.getMessage('Battle_Self_Desc'), flags: MessageFlags.Ephemeral });
    return;
  }
  if (target.bot) {
    await interaction.reply({ content: gameData.getMessage('Battle_Bot_Desc'), flags: MessageFlags.Ephemeral });
    return;
  }

  const targetUser = gameData.getUser(target.id);
  if (!targetUser.sword.weaponType) enhancement.ensureSword(targetUser);

  const r = enhancement.battle(userId, target.id);
  const embed = new EmbedBuilder();
  let attachment = null;

  if (r.type === 'limit') {
    embed.setTitle(gameData.getMessage('Battle_Limit_Title')).setDescription(gameData.getMessage('Battle_Limit_Desc')).setColor(0xf44336);
  } else {
    embed
      .setTitle(r.type === 'win' ? gameData.getMessage('Battle_Win_Title') : gameData.getMessage('Battle_Lose_Title'))
      .setDescription(
        r.type === 'win'
          ? gameData.getMessage('Battle_Win_Desc', userName, target.displayName || target.username)
          : gameData.getMessage('Battle_Lose_Desc', userName, target.displayName || target.username),
      )
      .addFields(
        { name: gameData.getMessage('Battle_MySword'), value: `+${r.myLevel}강 ${r.mySword}`, inline: true },
        { name: gameData.getMessage('Battle_TargetSword'), value: `+${r.targetLevel}강 ${r.targetSword}`, inline: true },
        { name: gameData.getMessage('Battle_Remaining'), value: `${r.remaining}회`, inline: true },
      )
      .setColor(r.type === 'win' ? 0x4caf50 : 0xf44336);
    if (r.reward) embed.addFields({ name: gameData.getMessage('Battle_Reward'), value: `${formatMoney(r.reward)}원` });
    attachment = getImageAttachment(r.battleImage);
  }

  const payload: any = { embeds: [embed] };
  if (attachment) {
    embed.setThumbnail(`attachment://${attachment.name}`);
    payload.files = [attachment.file];
  }
  await interaction.reply(payload);
}

export async function handleSlot(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string): Promise<void> {
  const { gameData, enhancement } = ctx;
  const bet = interaction.options.getInteger('금액', true);
  const r = enhancement.slot(userId, bet);
  const embed = new EmbedBuilder();
  if (r.type === 'error') {
    await interaction.reply({ content: r.msg, flags: MessageFlags.Ephemeral });
    return;
  }
  if (r.type === 'no_money') {
    await interaction.reply({ content: gameData.getMessage('Game_NoMoney', formatMoney(r.balance)), flags: MessageFlags.Ephemeral });
    return;
  }

  embed
    .setTitle(gameData.getMessage('Slot_Result_Title'))
    .setDescription(`**[ ${r.symbols.join(' | ')} ]**\n\n${r.payout > 0 ? `🎉 ${r.msg}` : '😢 ' + gameData.getMessage('Slot_Lose')}`)
    .addFields(
      { name: gameData.getMessage('Slot_Bet'), value: `${formatMoney(r.bet)}원`, inline: true },
      { name: gameData.getMessage('Slot_Earn'), value: `${formatMoney(r.payout)}원`, inline: true },
      { name: gameData.getMessage('Slot_Balance'), value: `${formatMoney(r.balance)}원`, inline: true },
    )
    .setColor(r.payout > 0 ? 0xffd700 : 0x9e9e9e);
  await interaction.reply({ embeds: [embed] });
}

export async function handleOddEven(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string): Promise<void> {
  const { gameData, enhancement } = ctx;
  const choice = interaction.options.getString('선택', true);
  const bet = interaction.options.getInteger('금액', true);
  const r = enhancement.oddEven(userId, choice, bet);
  if (r.type === 'error') {
    await interaction.reply({ content: r.msg, flags: MessageFlags.Ephemeral });
    return;
  }
  if (r.type === 'no_money') {
    await interaction.reply({ content: gameData.getMessage('Game_NoMoney', formatMoney(r.balance)), flags: MessageFlags.Ephemeral });
    return;
  }

  const DICE = ['⚀', '⚁', '⚂', '⚃', '⚄', '⚅'];
  const embed = new EmbedBuilder()
    .setTitle(gameData.getMessage('OddEven_Title'))
    .setDescription(
      `${DICE[r.dice - 1]} ${gameData.getMessage('OddEven_Result', r.result)}\n\n${
        r.type === 'win' ? gameData.getMessage('OddEven_Win', formatMoney(r.payout)) : gameData.getMessage('OddEven_Lose')
      }`,
    )
    .addFields(
      { name: gameData.getMessage('OddEven_Choice'), value: String(choice), inline: true },
      { name: gameData.getMessage('OddEven_Bet'), value: `${formatMoney(r.bet)}원`, inline: true },
      { name: '잔액', value: `${formatMoney(r.balance)}원`, inline: true },
    )
    .setColor(r.type === 'win' ? 0x4caf50 : 0xf44336);
  await interaction.reply({ embeds: [embed] });
}

export async function handleRps(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string): Promise<void> {
  const { gameData, enhancement } = ctx;
  const choice = interaction.options.getString('선택', true);
  const bet = interaction.options.getInteger('금액', true);
  const r = enhancement.rps(userId, choice, bet);
  if (r.type === 'error') {
    await interaction.reply({ content: r.msg, flags: MessageFlags.Ephemeral });
    return;
  }
  if (r.type === 'no_money') {
    await interaction.reply({ content: gameData.getMessage('Game_NoMoney', formatMoney(r.balance)), flags: MessageFlags.Ephemeral });
    return;
  }

  const RPS_EMOJI: Record<string, string> = { 가위: '✌️', 바위: '✊', 보: '🖐️' };
  const result =
    r.type === 'win'
      ? gameData.getMessage('RPS_Win', formatMoney(r.payout))
      : r.type === 'draw'
        ? gameData.getMessage('RPS_Draw')
        : gameData.getMessage('RPS_Lose');
  const embed = new EmbedBuilder()
    .setTitle(gameData.getMessage('RPS_Title'))
    .addFields(
      { name: gameData.getMessage('RPS_User'), value: `${RPS_EMOJI[r.userChoice]} ${r.userChoice}`, inline: true },
      { name: gameData.getMessage('RPS_Bot'), value: `${RPS_EMOJI[r.botChoice]} ${r.botChoice}`, inline: true },
      { name: gameData.getMessage('RPS_Result'), value: result },
      { name: '잔액', value: `${formatMoney(r.balance)}원`, inline: true },
    )
    .setColor(r.type === 'win' ? 0x4caf50 : r.type === 'draw' ? 0xff9800 : 0xf44336);
  await interaction.reply({ embeds: [embed] });
}

