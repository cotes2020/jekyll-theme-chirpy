/**
 * 슬래시 커맨드 등록 (Discord API에 등록하는 스크립트)
 * — 한국어 기본 + en-US 설명·이름 로컬라이즈 (클라이언트 언어에 맞게 표시)
 *
 * 큰 빌더들은 deploy-builders/ 하위 파일로 분리. 여기는 독립 커맨드 + 조립만 담당.
 */
import './load-env';
import './install-console-timestamps';
import { SlashCommandBuilder, ChannelType, Locale } from 'discord.js';
import { deployApplicationCommands } from '@discord-bots/common';

import { voiceJoin, voiceLeave, musicCommandGroup } from './deploy-builders/voice-music';
import {
  gameCommandGroup,
  minigameCommandGroup,
  stockCommandGroup,
  raidCommandGroup,
} from './deploy-builders/game-stock';
import { characterCommand } from './deploy-builders/character';

const EN = Locale.EnglishUS;
const enUS = (s: string): Record<string, string> => ({ [EN]: s });

const commands = [
  new SlashCommandBuilder()
    .setName('음성입장')
    .setDescription('봇을 음성 또는 스테이지 채널에 연결합니다.')
    .setDescriptionLocalizations(enUS('Connect the bot to a voice or stage channel'))
    .addChannelOption((opt) =>
      opt
        .setName('채널')
        .setNameLocalizations(enUS('channel'))
        .setDescription('입장할 채널 (비우면 본인이 있는 음성 채널)')
        .setDescriptionLocalizations(
          enUS('Channel to join (empty = your current voice channel)'),
        )
        .addChannelTypes(ChannelType.GuildVoice, ChannelType.GuildStageVoice)
        .setRequired(false),
    ),
  new SlashCommandBuilder()
    .setName('음성퇴장')
    .setDescription('봇을 음성 채널 연결에서 끊습니다.')
    .setDescriptionLocalizations(enUS('Disconnect the bot from voice')),

  voiceJoin(),
  voiceLeave(),
  musicCommandGroup(),
  gameCommandGroup(),
  minigameCommandGroup(),
  stockCommandGroup(),
  raidCommandGroup(),

  new SlashCommandBuilder()
    .setName('ping')
    .setDescription('봇의 응답 속도를 확인합니다.')
    .setDescriptionLocalizations(enUS('Check bot latency')),
  new SlashCommandBuilder()
    .setName('도움말')
    .setNameLocalizations(enUS('help'))
    .setDescription('카테고리별 도움말 (주제를 비우면 개요)')
    .setDescriptionLocalizations(enUS('Help by category (empty topic = overview)'))
    .addStringOption((opt) =>
      opt
        .setName('주제')
        .setNameLocalizations(enUS('topic'))
        .setDescription('게임 · /music · AI·ping 등')
        .setDescriptionLocalizations(enUS('game, music, utility, or overview'))
        .setRequired(false)
        .addChoices(
          { name: '개요', name_localizations: enUS('Overview'), value: 'overview' },
          { name: '음성 · /music', name_localizations: enUS('Voice · /music'), value: 'music' },
          { name: '검 · 미니게임 · 주식 · 레이드', name_localizations: enUS('Sword · minigames · stocks · raid'), value: 'game' },
          { name: 'AI · ping · 음성 입장', name_localizations: enUS('AI · ping · voice join'), value: 'utility' },
        ),
    ),

  new SlashCommandBuilder()
    .setName('yawn')
    .setDescription('Gemini AI에게 무엇이든 물어보세요!')
    .setDescriptionLocalizations(enUS('Ask the Gemini AI anything'))
    .addStringOption((opt) =>
      opt
        .setName('질문')
        .setNameLocalizations(enUS('question'))
        .setDescription('AI에게 전달할 메시지')
        .setDescriptionLocalizations(enUS('Message for the AI'))
        .setRequired(true),
    )
    .addStringOption((opt) =>
      opt
        .setName('api')
        .setDescription('호출 API (비우면 .env의 KARMOLAB_AI_SURFACE 등 기본)')
        .setDescriptionLocalizations(
          enUS('API surface (default: .env KARMOLAB_AI_SURFACE / GEMINI_SURFACE)'),
        )
        .addChoices(
          { name: '기본 (.env)', name_localizations: enUS('Default (.env)'), value: 'inherit' },
          { name: 'Google AI Studio', name_localizations: enUS('Google AI Studio'), value: 'ai_studio' },
          { name: 'Vertex AI', name_localizations: enUS('Vertex AI'), value: 'vertex' },
        ),
    )
    .addStringOption((opt) =>
      opt
        .setName('model')
        .setDescription('모델 ID (예: gemini-2.5-flash). 비우면 GEMINI_MODEL·패키지 기본')
        .setDescriptionLocalizations(enUS('Model id; empty = GEMINI_MODEL / package default'))
        .setMaxLength(64),
    ),

  new SlashCommandBuilder()
    .setName('이미지')
    .setNameLocalizations(enUS('image'))
    .setDescription('Vertex Imagen으로 이미지를 생성합니다.')
    .setDescriptionLocalizations(enUS('Generate images via Vertex Imagen'))
    .addStringOption((opt) =>
      opt
        .setName('프롬프트')
        .setNameLocalizations(enUS('prompt'))
        .setDescription('이미지 프롬프트 (영어 권장)')
        .setDescriptionLocalizations(enUS('Image prompt (English recommended)'))
        .setRequired(true)
        .setMaxLength(1500),
    )
    .addStringOption((opt) =>
      opt
        .setName('모델')
        .setNameLocalizations(enUS('model'))
        .setDescription('모델 ID (비우면 IMAGE_MODEL_ID 기본)')
        .setDescriptionLocalizations(enUS('Model id; empty = IMAGE_MODEL_ID default'))
        .addChoices(
          { name: 'Imagen 4 Generate', value: 'imagen-4.0-generate-001' },
          { name: 'Imagen 4 Ultra', value: 'imagen-4.0-ultra-generate-001' },
          { name: 'Imagen 4 Fast', value: 'imagen-4.0-fast-generate-001' },
        ),
    )
    .addStringOption((opt) =>
      opt
        .setName('비율')
        .setNameLocalizations(enUS('aspect'))
        .setDescription('가로세로 비율 (기본 1:1)')
        .setDescriptionLocalizations(enUS('Aspect ratio (default 1:1)'))
        .addChoices(
          { name: '1:1 (정사각)', value: '1:1' },
          { name: '16:9 (가로)', value: '16:9' },
          { name: '9:16 (세로)', value: '9:16' },
          { name: '4:3', value: '4:3' },
          { name: '3:4', value: '3:4' },
        ),
    )
    .addIntegerOption((opt) =>
      opt
        .setName('개수')
        .setNameLocalizations(enUS('count'))
        .setDescription('생성할 이미지 개수 (1~4)')
        .setDescriptionLocalizations(enUS('Number of images (1-4)'))
        .setMinValue(1)
        .setMaxValue(4),
    )
    .addStringOption((opt) =>
      opt
        .setName('네거티브')
        .setNameLocalizations(enUS('negative'))
        .setDescription('피하고 싶은 요소 (negative prompt)')
        .setDescriptionLocalizations(enUS('Elements to avoid (negative prompt)'))
        .setMaxLength(500),
    ),

  new SlashCommandBuilder()
    .setName('cursor-edit')
    .setDescription('[관리자] 로컬 저장소에서 Cursor agent(acp)로 프롬프트 실행')
    .setDescriptionLocalizations(enUS('[Admin] Run a Cursor agent prompt on the local repo'))
    .addStringOption((opt) =>
      opt
        .setName('prompt')
        .setDescription('에이전트에 전달할 지시')
        .setDescriptionLocalizations(enUS('Instructions for the agent'))
        .setRequired(true),
    )
    .addStringOption((opt) =>
      opt
        .setName('mode')
        .setDescription('세션 모드')
        .setDescriptionLocalizations(enUS('Session mode'))
        .addChoices(
          { name: 'agent', value: 'agent' },
          { name: 'ask', value: 'ask' },
          { name: 'plan', value: 'plan' },
        ),
    ),

  new SlashCommandBuilder()
    .setName('admin-reload')
    .setDescription('[관리자] 데이터를 다시 불러옵니다.')
    .setDescriptionLocalizations(enUS('[Admin] Reload persisted data')),
  new SlashCommandBuilder()
    .setName('admin-save')
    .setDescription('[관리자] 데이터를 저장합니다.')
    .setDescriptionLocalizations(enUS('[Admin] Save data to disk')),

  characterCommand(),

  new SlashCommandBuilder()
    .setName('기억')
    .setDescription('YawnBot 메모리 관리')
    .setDescriptionLocalizations(enUS('Manage YawnBot memory'))
    .addSubcommand((sub) =>
      sub
        .setName('확인')
        .setNameLocalizations({ 'en-US': 'view' })
        .setDescription('저장된 나에 대한 정보를 출력합니다.')
        .setDescriptionLocalizations({ 'en-US': 'View saved information about you' }),
    )
    .addSubcommand((sub) =>
      sub
        .setName('저장')
        .setNameLocalizations({ 'en-US': 'save' })
        .setDescription('지금까지의 대화를 memo 레포에 즉시 커밋합니다.')
        .setDescriptionLocalizations({ 'en-US': 'Immediately commit current conversation to memo repo' }),
    )
    .addSubcommand((sub) =>
      sub
        .setName('수정')
        .setNameLocalizations({ 'en-US': 'edit' })
        .setDescription('user.md를 AI 도움으로 수정합니다.')
        .setDescriptionLocalizations({ 'en-US': 'Edit user.md with AI assistance' })
        .addStringOption((opt) =>
          opt
            .setName('내용')
            .setNameLocalizations({ 'en-US': 'content' })
            .setDescription('추가하거나 수정할 사항')
            .setDescriptionLocalizations({ 'en-US': 'What to add or modify' })
            .setRequired(true),
        ),
    )
    .addSubcommand((sub) =>
      sub
        .setName('핫로그')
        .setNameLocalizations({ 'en-US': 'hotlog' })
        .setDescription('최근 중요 기억들을 확인합니다.')
        .setDescriptionLocalizations({ 'en-US': 'View recent important memories' }),
    ),
].map((cmd) => cmd.toJSON());

async function main(): Promise<void> {
  const token = process.env.DISCORD_TOKEN;
  const clientId = process.env.CLIENT_ID;
  if (!token || !clientId) {
    console.error('[Deploy] DISCORD_TOKEN 또는 CLIENT_ID가 없습니다.');
    process.exitCode = 1;
    return;
  }
  const guildId = process.env.DISCORD_GUILD_ID?.trim();
  await deployApplicationCommands({ token, clientId, commands, logPrefix: '[Deploy]', guildId });
}

void main();
