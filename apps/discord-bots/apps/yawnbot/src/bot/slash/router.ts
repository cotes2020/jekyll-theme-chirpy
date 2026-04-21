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
import {
  handleCharacterList,
  handleCharacterSwitch,
  handleCharacterInfo,
  handleCharacterReset,
} from './character';
import { CharacterService } from '../../services/character-service';
import { guardSlashInteraction } from './slash-guard';
import { logSlashUsage } from './usage-log';

/** 현재 /기억 호출 컨텍스트의 활성 슬러그 memory 를 돌려준다. 없으면 null + 안내. */
async function resolveMemoryForInteraction(ctx, interaction) {
  const cs = ctx.characterService;
  const getMem = ctx.getMemory;
  if (!cs || !getMem) {
    await interaction.reply({
      content: 'MEMO_REPO_PATH가 설정되지 않아 기억 기능이 비활성화되어 있습니다.',
      flags: MessageFlags.Ephemeral,
    });
    return null;
  }
  const isDM = !interaction.guildId;
  const channelKey = CharacterService.channelKey({
    isDM,
    userId: interaction.user.id,
    channelId: interaction.channelId ?? '',
  });
  const card = cs.resolveCard(channelKey);
  if (!card) {
    await interaction.reply({
      content: '활성 캐릭터 카드가 없어요. `/character list` 로 확인해봐요.',
      flags: MessageFlags.Ephemeral,
    });
    return null;
  }
  return { card, memory: getMem(card.slug) };
}

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
      case '게임': {
        const sub = interaction.options.getSubcommand();
        switch (sub) {
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
          default:
            await interaction.reply({ content: '알 수 없는 게임 하위 명령입니다.', flags: MessageFlags.Ephemeral });
        }
        break;
      }
      case '미니게임': {
        const sub = interaction.options.getSubcommand();
        switch (sub) {
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
          default:
            await interaction.reply({ content: '알 수 없는 미니게임 하위 명령입니다.', flags: MessageFlags.Ephemeral });
        }
        break;
      }
      case '주식': {
        const sub = interaction.options.getSubcommand();
        switch (sub) {
          case '목록':
            await handleStockList(ctx, interaction);
            break;
          case '차트':
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
          default:
            await interaction.reply({ content: '알 수 없는 주식 하위 명령입니다.', flags: MessageFlags.Ephemeral });
        }
        break;
      }
      case '레이드': {
        const sub = interaction.options.getSubcommand();
        switch (sub) {
          case '정보':
            await handleRaidInfo(ctx, interaction);
            break;
          case '소환':
            await handleRaidSpawn(ctx, interaction);
            break;
          case '공격':
            await handleRaidAttack(ctx, interaction, userId);
            break;
          default:
            await interaction.reply({ content: '알 수 없는 레이드 하위 명령입니다.', flags: MessageFlags.Ephemeral });
        }
        break;
      }
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
      case 'character': {
        const sub = interaction.options.getSubcommand();
        switch (sub) {
          case 'list':
            await handleCharacterList(ctx, interaction);
            break;
          case 'switch':
            await handleCharacterSwitch(ctx, interaction);
            break;
          case 'info':
            await handleCharacterInfo(ctx, interaction);
            break;
          case 'reset':
            await handleCharacterReset(ctx, interaction);
            break;
          default:
            await interaction.reply({ content: '알 수 없는 character 하위 명령입니다.', flags: MessageFlags.Ephemeral });
        }
        break;
      }
      case '기억': {
        const resolved = await resolveMemoryForInteraction(ctx, interaction);
        if (!resolved) break;
        const { card, memory } = resolved;

        const sub = interaction.options.getSubcommand();
        switch (sub) {
          case '확인': {
            try {
              const userMd = memory.getUserMd();
              const selfMd = memory.getSelfMd();

              const userContent = userMd.slice(0, 1024) || '(없음)';
              const selfContent = selfMd.slice(0, 1024) || '(없음)';

              const userSize = Buffer.byteLength(userMd, 'utf-8');
              const selfSize = Buffer.byteLength(selfMd, 'utf-8');

              const embed = new EmbedBuilder()
                .setTitle(`🧠 ${card.displayName} 메모리 (${card.slug})`)
                .addFields(
                  {
                    name: `나에 대한 정보 (${(userSize / 1024).toFixed(1)}KB)`,
                    value: userContent,
                  },
                  {
                    name: `봇 자신에 대한 정보 (${(selfSize / 1024).toFixed(1)}KB)`,
                    value: selfContent,
                  },
                )
                .setFooter({ text: userSize > 1024 || selfSize > 1024 ? '⚠️ 전체 내용을 보려면 /기억 핫로그를 사용하세요' : undefined })
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
              await interaction.editReply(`${card.displayName}(${card.slug}) 대화 기록을 memo 레포에 저장했습니다.`);
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
              const fs = await import('fs');
              fs.default.writeFileSync(
                memory.getUserMdPath(),
                `# 나에 대한 정보\n\n${updatedUserMd.trim()}\n`,
                'utf-8',
              );

              const oldLines = currentUserMd.split('\n').length;
              const newLines = updatedUserMd.split('\n').length;
              const diff = `${oldLines}줄 → ${newLines}줄`;

              const embed = new EmbedBuilder()
                .setTitle(`✅ ${card.displayName} user.md 업데이트 완료`)
                .addFields(
                  { name: '요청', value: content.slice(0, 256) },
                  { name: '변화', value: diff, inline: true },
                  { name: '새 용량', value: `${(Buffer.byteLength(updatedUserMd, 'utf-8') / 1024).toFixed(1)}KB`, inline: true },
                )
                .setColor(0x4caf50);
              await interaction.editReply({ embeds: [embed] });
            } catch (e) {
              await interaction.editReply(`수정 실패: ${e instanceof Error ? e.message : String(e)}`);
            }
            break;
          }

          case '핫로그': {
            try {
              const hotLog = memory.getHotMemoryLog(20);
              const embed = new EmbedBuilder()
                .setTitle(`🔥 ${card.displayName} 핫 메모리 로그`)
                .setDescription('최근 중요 기억들 (최대 20개)')
                .addFields({
                  name: '기록',
                  value: `\`\`\`\n${hotLog}\n\`\`\``,
                })
                .setColor(0xff9800);
              await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
            } catch (e) {
              await interaction.reply({ content: `핫로그 조회 실패: ${e instanceof Error ? e.message : String(e)}`, flags: MessageFlags.Ephemeral });
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
