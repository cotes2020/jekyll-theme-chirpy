// @ts-nocheck
import { MessageFlags, EmbedBuilder } from 'discord.js';
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
import { handleVoiceJoin, handleVoiceLeave } from './voice';
import {
  handlePlay,
  handleSkip,
  handleStopMusic,
  handleQueue,
  handleShuffle,
  handleRemove,
  handleLoop,
} from './music';
import { handleSpeak } from './speak';
import { handleSound } from './sound';
import { handleAdminReload, handleAdminSave } from './admin';
import { guardSlashInteraction } from './slash-guard';
import { logSlashUsage } from './usage-log';

export async function dispatchSlashCommand(ctx, interaction) {
  if (!interaction.isChatInputCommand()) return;
  if (!(await guardSlashInteraction(interaction))) return;
  logSlashUsage(interaction);

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
      case '음성입장':
      case 'voice-join':
        await handleVoiceJoin(ctx, interaction);
        break;
      case '음성퇴장':
      case 'voice-leave':
        await handleVoiceLeave(ctx, interaction);
        break;
      case 'music': {
        const sub = interaction.options.getSubcommand();
        switch (sub) {
          case 'play':
            await handlePlay(ctx, interaction);
            break;
          case 'speak':
            await handleSpeak(ctx, interaction);
            break;
          case 'sound':
            await handleSound(ctx, interaction);
            break;
          case 'skip':
            await handleSkip(ctx, interaction);
            break;
          case 'stop':
            await handleStopMusic(ctx, interaction);
            break;
          case 'shuffle':
            await handleShuffle(ctx, interaction);
            break;
          case 'remove':
            await handleRemove(ctx, interaction);
            break;
          case 'loop':
            await handleLoop(ctx, interaction);
            break;
          case 'queue':
            await handleQueue(ctx, interaction);
            break;
          default:
            await interaction.reply({ content: '알 수 없는 music 하위 명령입니다.', flags: MessageFlags.Ephemeral });
        }
        break;
      }
      case 'admin-reload':
        await handleAdminReload(ctx, interaction, userId);
        break;
      case 'admin-save':
        await handleAdminSave(ctx, interaction, userId);
        break;
      case '기억': {
        const memory = ctx.memory;
        if (!memory) {
          await interaction.reply({ content: 'MEMO_REPO_PATH가 설정되지 않아 기억 기능이 비활성화되어 있습니다.', flags: MessageFlags.Ephemeral });
          break;
        }

        const sub = interaction.options.getSubcommand();
        switch (sub) {
          case '확인': {
            try {
              const userMd = memory.getUserMd();
              const selfMd = memory.getSelfMd();
              const embed = new EmbedBuilder()
                .setTitle('🧠 YawnBot 메모리')
                .addFields(
                  { name: '나에 대한 정보', value: userMd.slice(0, 1000) || '(없음)' },
                  { name: '봇 자신에 대한 정보', value: selfMd.slice(0, 500) || '(없음)' },
                )
                .setColor(0x7c4dff);
              await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
            } catch (e) {
              await interaction.reply({ content: `기억 조회 실패: ${e instanceof Error ? e.message : String(e)}`, flags: MessageFlags.Ephemeral });
            }
            break;
          }

          case '저장': {
            await interaction.deferReply({ flags: MessageFlags.Ephemeral });
            try {
              memory.commitIfDirty();
              await interaction.editReply('대화 기록을 memo 레포에 저장했습니다.');
            } catch (e) {
              await interaction.editReply(`저장 실패: ${e instanceof Error ? e.message : String(e)}`);
            }
            break;
          }

          case '수정': {
            const content = interaction.options.getString('내용');
            if (!content) {
              await interaction.reply({ content: '내용을 입력해주세요.', flags: MessageFlags.Ephemeral });
              break;
            }
            await interaction.deferReply({ flags: MessageFlags.Ephemeral });
            try {
              const { generateAssistantText } = await import('karmolab-ai/node');
              const currentUserMd = memory.getUserMd();
              const { text: updatedUserMd } = await generateAssistantText(
                process.env,
                `너는 mascari4615의 개인 AI 비서야.\n다음은 현재 user.md의 내용이야:\n${currentUserMd}\n\n사용자가 요청한 수정 사항:\n${content}\n\n이를 반영해서 업데이트된 user.md 내용을 마크다운 형식으로 작성해줘. 기존 정보는 유지하면서 새로운 정보를 추가/수정해.`,
              );
              memory.appendHotMemory(`[기억수정] ${content.slice(0, 50)}`);
              // user.md 파일 직접 업데이트
              const path = await import('path');
              const fs = await import('fs');
              const userMdPath = path.default.join((memory as any).memoryDir, 'user.md');
              fs.default.writeFileSync(userMdPath, `# 나에 대한 정보\n\n${updatedUserMd.trim()}\n`, 'utf-8');
              await interaction.editReply(`✅ user.md를 업데이트했습니다.\n\n수정 내용: ${content}`);
            } catch (e) {
              await interaction.editReply(`수정 실패: ${e instanceof Error ? e.message : String(e)}`);
            }
            break;
          }

          default:
            await interaction.reply({ content: '알 수 없는 기억 하위 명령입니다.', flags: MessageFlags.Ephemeral });
        }
        break;
      }
      default:
        await interaction.reply({ content: '알 수 없는 명령어입니다.', flags: MessageFlags.Ephemeral });
    }
  } catch (err) {
    console.error(`[Error] ${interaction.commandName}:`, err);
    const msg = err instanceof Error ? err.message : String(err);
    const reply = interaction.replied || interaction.deferred ? interaction.editReply : interaction.reply;
    await reply
      .call(interaction, { content: `오류가 발생했습니다: ${msg}`, flags: MessageFlags.Ephemeral })
      .catch(() => {});
  }
}

