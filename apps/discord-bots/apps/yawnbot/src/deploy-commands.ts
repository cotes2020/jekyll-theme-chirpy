/**
 * 슬래시 커맨드 등록 (Discord API에 등록하는 스크립트)
 */
import './load-env';
import { SlashCommandBuilder, ChannelType } from 'discord.js';
import { deployApplicationCommands } from '@discord-bots/common';

const voiceJoin = () =>
  new SlashCommandBuilder()
    .setName('voice-join')
    .setDescription('봇을 음성·스테이지 채널에 연결 (/음성입장 과 동일)')
    .addChannelOption((opt) =>
      opt
        .setName('channel')
        .setDescription('입장할 채널 (비우면 본인이 있는 음성 채널)')
        .addChannelTypes(ChannelType.GuildVoice, ChannelType.GuildStageVoice)
        .setRequired(false),
    );

const voiceLeave = () =>
  new SlashCommandBuilder().setName('voice-leave').setDescription('봇 음성 연결 해제 (/음성퇴장 과 동일)');

const commands = [
  new SlashCommandBuilder()
    .setName('음성입장')
    .setDescription('봇을 음성 또는 스테이지 채널에 연결합니다.')
    .addChannelOption((opt) =>
      opt
        .setName('채널')
        .setDescription('입장할 채널 (비우면 본인이 있는 음성 채널)')
        .addChannelTypes(ChannelType.GuildVoice, ChannelType.GuildStageVoice)
        .setRequired(false),
    ),
  new SlashCommandBuilder().setName('음성퇴장').setDescription('봇을 음성 채널 연결에서 끊습니다.'),
  voiceJoin(),
  voiceLeave(),

  new SlashCommandBuilder()
    .setName('play')
    .setDescription('YouTube URL 또는 검색어로 재생 (음성 채널에 있어야 함)')
    .addStringOption((opt) =>
      opt.setName('query').setDescription('YouTube 동영상 URL 또는 검색어').setRequired(true),
    ),
  new SlashCommandBuilder().setName('skip').setDescription('지금 재생 중인 곡 건너뛰기'),
  new SlashCommandBuilder().setName('stop').setDescription('재생 중지 및 대기열 비우기'),
  new SlashCommandBuilder().setName('queue').setDescription('대기열 확인'),

  new SlashCommandBuilder().setName('강화').setDescription('검을 강화합니다. (확률 존재)'),
  new SlashCommandBuilder().setName('판매').setDescription('검을 판매하여 돈을 얻습니다.'),
  new SlashCommandBuilder().setName('정보').setDescription('내 검과 재산 정보를 확인합니다.'),
  new SlashCommandBuilder().setName('돈').setDescription('현재 보유한 돈을 확인합니다.'),
  new SlashCommandBuilder().setName('랭킹').setDescription('전체 유저 랭킹을 확인합니다.'),
  new SlashCommandBuilder().setName('출첵').setDescription('매일 출석체크 보상을 받습니다.'),
  new SlashCommandBuilder().setName('돈내놔').setDescription('일정 시간마다 랜덤 용돈을 받습니다.'),

  new SlashCommandBuilder()
    .setName('배틀')
    .setDescription('다른 유저와 대결합니다.')
    .addUserOption((opt) => opt.setName('상대').setDescription('대결할 상대를 선택하세요').setRequired(true)),
  new SlashCommandBuilder()
    .setName('슬롯')
    .setDescription('슬롯 머신을 돌립니다.')
    .addIntegerOption((opt) => opt.setName('금액').setDescription('배팅할 금액').setRequired(true).setMinValue(1)),
  new SlashCommandBuilder()
    .setName('홀짝')
    .setDescription('홀짝 게임을 합니다.')
    .addStringOption((opt) =>
      opt
        .setName('선택')
        .setDescription('홀 또는 짝')
        .setRequired(true)
        .addChoices({ name: '홀', value: '홀' }, { name: '짝', value: '짝' }),
    )
    .addIntegerOption((opt) => opt.setName('금액').setDescription('배팅할 금액').setRequired(true).setMinValue(1)),
  new SlashCommandBuilder()
    .setName('가위바위보')
    .setDescription('가위바위보를 합니다.')
    .addStringOption((opt) =>
      opt
        .setName('선택')
        .setDescription('가위, 바위, 보')
        .setRequired(true)
        .addChoices({ name: '가위', value: '가위' }, { name: '바위', value: '바위' }, { name: '보', value: '보' }),
    )
    .addIntegerOption((opt) => opt.setName('금액').setDescription('배팅할 금액').setRequired(true).setMinValue(1)),

  new SlashCommandBuilder().setName('주식목록').setDescription('현재 상장된 주식 시세를 확인합니다.'),
  new SlashCommandBuilder()
    .setName('주식차트')
    .setDescription('특정 주식의 차트를 확인합니다.')
    .addStringOption((opt) =>
      opt
        .setName('종목')
        .setDescription('종목 심볼')
        .setRequired(true)
        .addChoices(
          { name: '떡락전자 (SAMSUNG)', value: 'SAMSUNG' },
          { name: '화성갈끄니까 (DOGE)', value: 'DOGE' },
          { name: '테슬라 (TESLA)', value: 'TESLA' },
          { name: '사과 (APPLE)', value: 'APPLE' },
          { name: '비트코인 (BITCOIN)', value: 'BITCOIN' },
        ),
    ),
  new SlashCommandBuilder()
    .setName('매수')
    .setDescription('주식을 매수합니다.')
    .addStringOption((opt) =>
      opt
        .setName('종목')
        .setDescription('종목 심볼')
        .setRequired(true)
        .addChoices(
          { name: '떡락전자 (SAMSUNG)', value: 'SAMSUNG' },
          { name: '화성갈끄니까 (DOGE)', value: 'DOGE' },
          { name: '테슬라 (TESLA)', value: 'TESLA' },
          { name: '사과 (APPLE)', value: 'APPLE' },
          { name: '비트코인 (BITCOIN)', value: 'BITCOIN' },
        ),
    )
    .addIntegerOption((opt) => opt.setName('수량').setDescription('매수할 수량').setRequired(true).setMinValue(1)),
  new SlashCommandBuilder()
    .setName('매도')
    .setDescription('주식을 매도합니다.')
    .addStringOption((opt) =>
      opt
        .setName('종목')
        .setDescription('종목 심볼')
        .setRequired(true)
        .addChoices(
          { name: '떡락전자 (SAMSUNG)', value: 'SAMSUNG' },
          { name: '화성갈끄니까 (DOGE)', value: 'DOGE' },
          { name: '테슬라 (TESLA)', value: 'TESLA' },
          { name: '사과 (APPLE)', value: 'APPLE' },
          { name: '비트코인 (BITCOIN)', value: 'BITCOIN' },
        ),
    )
    .addIntegerOption((opt) => opt.setName('수량').setDescription('매도할 수량').setRequired(true).setMinValue(1)),
  new SlashCommandBuilder().setName('내주식').setDescription('내 주식 잔고를 확인합니다.'),

  new SlashCommandBuilder().setName('레이드정보').setDescription('현재 진행 중인 레이드 정보를 확인합니다.'),
  new SlashCommandBuilder().setName('공격').setDescription('레이드 보스를 공격합니다.'),
  new SlashCommandBuilder().setName('레이드소환').setDescription('새로운 레이드 보스를 소환합니다.'),

  new SlashCommandBuilder().setName('ping').setDescription('봇의 응답 속도를 확인합니다.'),
  new SlashCommandBuilder().setName('도움말').setDescription('도움말을 확인합니다.'),

  new SlashCommandBuilder()
    .setName('yawn')
    .setDescription('Gemini AI에게 무엇이든 물어보세요!')
    .addStringOption((opt) => opt.setName('질문').setDescription('AI에게 전달할 메시지').setRequired(true)),

  new SlashCommandBuilder()
    .setName('cursor-edit')
    .setDescription('[관리자] 로컬 저장소에서 Cursor agent(acp)로 프롬프트 실행')
    .addStringOption((opt) => opt.setName('prompt').setDescription('에이전트에 전달할 지시').setRequired(true))
    .addStringOption((opt) =>
      opt
        .setName('mode')
        .setDescription('세션 모드')
        .addChoices({ name: 'agent', value: 'agent' }, { name: 'ask', value: 'ask' }, { name: 'plan', value: 'plan' }),
    ),

  new SlashCommandBuilder().setName('admin-reload').setDescription('[관리자] 데이터를 다시 불러옵니다.'),
  new SlashCommandBuilder().setName('admin-save').setDescription('[관리자] 데이터를 저장합니다.'),
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

