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

import { voiceJoin, voiceLeave, musicCommandGroup } from './deploy-builders/voice-music';
import { gameCommandGroup } from './deploy-builders/game-stock';
import { characterCommand } from './deploy-builders/character';
import { scheduleCommand } from './deploy-builders/schedule';
import { adminCommand } from './deploy-builders/admin';

const EN = Locale.EnglishUS;
const enUS = (s: string): Record<string, string> => ({ [EN]: s });

const commands = [
  voiceJoin(),
  voiceLeave(),
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

  new SlashCommandBuilder()
    .setName('기념일')
    .setNameLocalizations(enUS('anniversary'))
    .setDescription('기념일 관리')
    .setDescriptionLocalizations(enUS('Manage anniversaries'))
    .addSubcommand((sub) =>
      sub.setName('목록').setNameLocalizations(enUS('list'))
        .setDescription('기념일 목록 조회').setDescriptionLocalizations(enUS('List anniversaries')),
    )
    .addSubcommand((sub) =>
      sub.setName('추가').setNameLocalizations(enUS('add'))
        .setDescription('기념일 추가').setDescriptionLocalizations(enUS('Add anniversary'))
        .addStringOption((o) => o.setName('이름').setNameLocalizations(enUS('label')).setDescription('기념일 이름').setDescriptionLocalizations(enUS('Label')).setRequired(true))
        .addIntegerOption((o) => o.setName('월').setNameLocalizations(enUS('month')).setDescription('월 (1-12)').setDescriptionLocalizations(enUS('Month')).setRequired(true).setMinValue(1).setMaxValue(12))
        .addIntegerOption((o) => o.setName('일').setNameLocalizations(enUS('day')).setDescription('일 (1-31)').setDescriptionLocalizations(enUS('Day')).setRequired(true).setMinValue(1).setMaxValue(31))
        .addIntegerOption((o) => o.setName('연도').setNameLocalizations(enUS('year')).setDescription('시작 연도 (N주년 계산용)').setDescriptionLocalizations(enUS('Start year for anniversary count'))),
    )
    .addSubcommand((sub) =>
      sub.setName('삭제').setNameLocalizations(enUS('delete'))
        .setDescription('기념일 삭제').setDescriptionLocalizations(enUS('Delete anniversary'))
        .addStringOption((o) => o.setName('id').setDescription('목록에서 확인한 ID').setDescriptionLocalizations(enUS('ID from list')).setRequired(true)),
    ),

  new SlashCommandBuilder()
    .setName('뉴스키워드')
    .setNameLocalizations(enUS('news-keywords'))
    .setDescription('뉴스 관심사 키워드 관리 (자발적 메시지에서 관련 뉴스 언급)')
    .setDescriptionLocalizations(enUS('Manage news interest keywords'))
    .addSubcommand((sub) =>
      sub.setName('목록').setNameLocalizations(enUS('list'))
        .setDescription('키워드 목록').setDescriptionLocalizations(enUS('List keywords')),
    )
    .addSubcommand((sub) =>
      sub.setName('추가').setNameLocalizations(enUS('add'))
        .setDescription('키워드 추가').setDescriptionLocalizations(enUS('Add keyword'))
        .addStringOption((o) => o.setName('키워드').setNameLocalizations(enUS('keyword')).setDescription('관심사 키워드').setDescriptionLocalizations(enUS('Interest keyword')).setRequired(true)),
    )
    .addSubcommand((sub) =>
      sub.setName('삭제').setNameLocalizations(enUS('delete'))
        .setDescription('키워드 삭제').setDescriptionLocalizations(enUS('Delete keyword'))
        .addStringOption((o) => o.setName('id').setDescription('목록에서 확인한 ID').setDescriptionLocalizations(enUS('ID from list')).setRequired(true)),
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
