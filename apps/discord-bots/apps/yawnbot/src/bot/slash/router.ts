// @ts-nocheck
import { handlePing, handleHelp } from './general';
import {
  handleEnhanceSlash,
  handleSellSlash,
  handleInfo,
  handleMoney,
  handleRank,
  handleAttendance,
  handleGiveMeMoney,
  handleBattle,
  handleSlot,
  handleOddEven,
  handleRps,
} from './core-game';
import { handleStockList, handleStockChart, handleBuy, handleSellStock, handleMyStock } from './stock';
import { handleRaidInfo, handleRaidSpawn, handleRaidAttack } from './raid';
import { handleCursorEdit, handleYawn } from './ai';
import { handleAdminReload, handleAdminSave } from './admin';

export async function dispatchSlashCommand(ctx, interaction) {
  if (!interaction.isChatInputCommand()) return;

  const userId = interaction.user.id;
  const userName = interaction.user.displayName || interaction.user.username;

  try {
    switch (interaction.commandName) {
      case 'ping':
        await handlePing(ctx, interaction);
        break;
      case '도움말':
        await handleHelp(ctx, interaction);
        break;
      case '강화':
        await handleEnhanceSlash(ctx, interaction, userId, userName);
        break;
      case '판매':
        await handleSellSlash(ctx, interaction, userId);
        break;
      case '정보':
        await handleInfo(ctx, interaction, userId, userName);
        break;
      case '돈':
        await handleMoney(ctx, interaction, userId, userName);
        break;
      case '랭킹':
        await handleRank(ctx, interaction);
        break;
      case '출첵':
        await handleAttendance(ctx, interaction, userId);
        break;
      case '돈내놔':
        await handleGiveMeMoney(ctx, interaction, userId);
        break;
      case '배틀':
        await handleBattle(ctx, interaction, userId, userName);
        break;
      case '슬롯':
        await handleSlot(ctx, interaction, userId);
        break;
      case '홀짝':
        await handleOddEven(ctx, interaction, userId);
        break;
      case '가위바위보':
        await handleRps(ctx, interaction, userId);
        break;
      case '주식목록':
        await handleStockList(ctx, interaction);
        break;
      case '주식차트':
        await handleStockChart(ctx, interaction);
        break;
      case '매수':
        await handleBuy(ctx, interaction, userId);
        break;
      case '매도':
        await handleSellStock(ctx, interaction, userId);
        break;
      case '내주식':
        await handleMyStock(ctx, interaction, userId, userName);
        break;
      case '레이드정보':
        await handleRaidInfo(ctx, interaction);
        break;
      case '레이드소환':
        await handleRaidSpawn(ctx, interaction);
        break;
      case '공격':
        await handleRaidAttack(ctx, interaction, userId);
        break;
      case 'cursor-edit':
        await handleCursorEdit(ctx, interaction, userId);
        break;
      case 'yawn':
        await handleYawn(ctx, interaction);
        break;
      case 'admin-reload':
        await handleAdminReload(ctx, interaction, userId);
        break;
      case 'admin-save':
        await handleAdminSave(ctx, interaction, userId);
        break;
      default:
        await interaction.reply({ content: '알 수 없는 명령어입니다.', ephemeral: true });
    }
  } catch (err) {
    console.error(`[Error] ${interaction.commandName}:`, err);
    const reply = interaction.replied || interaction.deferred ? interaction.editReply : interaction.reply;
    await reply.call(interaction, { content: `오류가 발생했습니다: ${err.message}`, ephemeral: true }).catch(() => {});
  }
}

