/**
 * 슬래시 커맨드 등록 (Discord API에 등록하는 스크립트)
 * — 한국어 기본 + en-US 설명·이름 로컬라이즈 (클라이언트 언어에 맞게 표시)
 *
 * 큰 빌더들은 deploy-builders/ 하위 파일로 분리. 여기는 독립 커맨드 + 조립만 담당.
 */
import './load-env';
import './install-console-timestamps';
import { SlashCommandBuilder, Locale } from 'discord.js';
import { deployApplicationCommands } from '@discord-bots/common';

import { musicCommandGroup } from './deploy-builders/voice-music';
import { gameCommandGroup } from './deploy-builders/game-stock';
import { characterCommand } from './deploy-builders/character';
import { scheduleCommand } from './deploy-builders/schedule';
import { adminCommand } from './deploy-builders/admin';
import { loadOpsReportContext, reportDeploy } from './services/ops-self-report';

const EN = Locale.EnglishUS;
const enUS = (s: string): Record<string, string> => ({ [EN]: s });

const commands = [
  musicCommandGroup(),
  gameCommandGroup(),

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
        .setName('캐릭터')
        .setNameLocalizations(enUS('character'))
        .setDescription('캐릭터 슬러그 (비우면 활성 캐릭터, "none"=캐릭터 없이)')
        .setDescriptionLocalizations(
          enUS('Character slug (empty=active, "none"=no character)'),
        )
        .setMaxLength(64)
        .setAutocomplete(true),
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

  adminCommand(),
  characterCommand(),
  scheduleCommand(),

  new SlashCommandBuilder()
    .setName('프로필')
    .setNameLocalizations(enUS('profile'))
    .setDescription('친밀도·기분·일정·기념일·뉴스 키워드 대시보드')
    .setDescriptionLocalizations(enUS('Your relationship & schedule dashboard')),

  new SlashCommandBuilder()
    .setName('갤러리')
    .setNameLocalizations(enUS('gallery'))
    .setDescription('캐릭터 이미지 캐시 갤러리 (◀▶ 페이지 이동)')
    .setDescriptionLocalizations(enUS('Browse character image cache gallery'))
    .addStringOption((opt) =>
      opt.setName('정렬').setNameLocalizations(enUS('sort'))
        .setDescription('정렬 기준 (기본: 최신순)')
        .setDescriptionLocalizations(enUS('Sort order (default: recent)'))
        .addChoices(
          { name: '최신순', value: 'recent' },
          { name: '인기순', value: 'popular' },
        ),
    ),

  new SlashCommandBuilder()
    .setName('사용량')
    .setNameLocalizations(enUS('usage'))
    .setDescription('이미지 생성 비용 대시보드 (모델별/일별 집계)')
    .setDescriptionLocalizations(enUS('Image generation cost dashboard')),

].map((cmd) => cmd.toJSON());

async function main(): Promise<void> {
  const token = process.env.DISCORD_TOKEN;
  const clientId = process.env.CLIENT_ID;
  const guildId = process.env.DISCORD_GUILD_ID?.trim();
  if (!token || !clientId) {
    console.error('[Deploy] DISCORD_TOKEN 또는 CLIENT_ID가 없습니다.');
    process.exitCode = 1;
    return;
  }
  if (!guildId) {
    console.error('[Deploy] DISCORD_GUILD_ID가 없습니다. 글로벌 배포를 방지하기 위해 길드 ID가 필요합니다.');
    console.error('[Deploy] 글로벌 커맨드를 초기화하려면 npm run deploy:clear-global 을 사용하세요.');
    process.exitCode = 1;
    return;
  }
  await deployApplicationCommands({ token, clientId, commands, logPrefix: '[Deploy]', guildId, guildOnly: true });

  const opsCtx = loadOpsReportContext();
  if (opsCtx) {
    await reportDeploy(opsCtx, { count: commands.length, target: `guild:${guildId}` });
  }
}

void main();
