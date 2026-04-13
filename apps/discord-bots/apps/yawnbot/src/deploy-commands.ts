/**
 * 슬래시 커맨드 등록 (Discord API에 등록하는 스크립트)
 * — 한국어 기본 + en-US 설명·이름 로컬라이즈 (클라이언트 언어에 맞게 표시)
 */
import './load-env';
import './install-console-timestamps';
import { SlashCommandBuilder, ChannelType, Locale } from 'discord.js';
import { deployApplicationCommands } from '@discord-bots/common';

const EN = Locale.EnglishUS;
const enUS = (s: string): Record<string, string> => ({ [EN]: s });

const voiceJoin = () =>
  new SlashCommandBuilder()
    .setName('voice-join')
    .setDescription('봇을 음성·스테이지 채널에 연결 (/음성입장 과 동일)')
    .setDescriptionLocalizations(enUS('Connect the bot to a voice/stage channel (same as /음성입장)'))
    .addChannelOption((opt) =>
      opt
        .setName('channel')
        .setDescription('입장할 채널 (비우면 본인이 있는 음성 채널)')
        .setDescriptionLocalizations(enUS('Channel to join (empty = your current voice channel)'))
        .addChannelTypes(ChannelType.GuildVoice, ChannelType.GuildStageVoice)
        .setRequired(false),
    );

const voiceLeave = () =>
  new SlashCommandBuilder()
    .setName('voice-leave')
    .setDescription('봇 음성 연결 해제 (/음성퇴장 과 동일)')
    .setDescriptionLocalizations(enUS('Disconnect the bot from voice (same as /음성퇴장)'));

/** 음성 재생·대기열 — YouTube·TTS·클립·skip/stop/shuffle/remove/loop/queue 서브커맨드 */
const musicCommandGroup = () =>
  new SlashCommandBuilder()
    .setName('music')
    .setDescription('음성 채널 YouTube·TTS·클립 재생 및 대기열')
    .setDescriptionLocalizations(enUS('YouTube, TTS, and clip playback with a shared queue'))
    .addSubcommand((sc) =>
      sc
        .setName('play')
        .setDescription('YouTube 동영상·플레이리스트 URL 또는 검색어로 재생 (음성 채널 필수)')
        .setDescriptionLocalizations(enUS('Play from YouTube URL, playlist, or search (voice channel required)'))
        .addStringOption((opt) =>
          opt
            .setName('query')
            .setDescription('동영상/playlist?list= URL, watch?…&list=, 또는 검색어')
            .setDescriptionLocalizations(enUS('Video or playlist URL, or search text'))
            .setRequired(true),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('speak')
        .setDescription('Edge TTS로 문장을 읽어 재생 (YouTube·클립과 동일 대기열, 디스코드 내장 TTS 아님)')
        .setDescriptionLocalizations(enUS('Speak text via Edge TTS (same queue as music; not built-in Discord TTS)'))
        .addStringOption((opt) =>
          opt
            .setName('text')
            .setDescription('읽을 문장 (비우면 데모 문장)')
            .setDescriptionLocalizations(enUS('Text to read (empty = demo phrase)'))
            .setRequired(false),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('sound')
        .setDescription('첨부·URL·로컬 클립 오디오 재생 (YouTube·TTS와 동일 대기열, file/url/clip 중 하나)')
        .setDescriptionLocalizations(enUS('Play attachment, URL, or packaged clip (exactly one of file/url/clip)'))
        .addAttachmentOption((opt) =>
          opt
            .setName('file')
            .setDescription('오디오 첨부 (mp3, wav, ogg 등)')
            .setDescriptionLocalizations(enUS('Audio attachment (mp3, wav, ogg, …)'))
            .setRequired(false),
        )
        .addStringOption((opt) =>
          opt
            .setName('url')
            .setDescription('직접 링크 (http(s) 오디오 파일)')
            .setDescriptionLocalizations(enUS('Direct http(s) link to an audio file'))
            .setRequired(false),
        )
        .addStringOption((opt) =>
          opt
            .setName('clip')
            .setDescription('봇 패키지 resources/audio/ 안의 파일명만 (예: hello.mp3)')
            .setDescriptionLocalizations(enUS('Filename under resources/audio/ (e.g. hello.mp3)'))
            .setRequired(false),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('skip')
        .setDescription('지금 재생 중인 곡 건너뛰기')
        .setDescriptionLocalizations(enUS('Skip the current track')),
    )
    .addSubcommand((sc) =>
      sc
        .setName('stop')
        .setDescription('재생 중지 및 대기열 비우기')
        .setDescriptionLocalizations(enUS('Stop playback and clear the queue')),
    )
    .addSubcommand((sc) =>
      sc
        .setName('shuffle')
        .setDescription('대기 중인 곡 순서만 무작위로 섞기 (지금 재생 곡은 유지)')
        .setDescriptionLocalizations(enUS('Shuffle only the waiting queue (current track unchanged)')),
    )
    .addSubcommand((sc) =>
      sc
        .setName('remove')
        .setDescription('대기열에서 번호로 곡 제거 (/music queue 목록의 1·2·3… 과 동일)')
        .setDescriptionLocalizations(enUS('Remove a track by queue index (same numbers as /music queue)'))
        .addIntegerOption((opt) =>
          opt
            .setName('index')
            .setDescription('제거할 대기 곡 번호 (1부터, queue 목록과 같음)')
            .setDescriptionLocalizations(enUS('1-based index in the queue list'))
            .setRequired(true)
            .setMinValue(1)
            .setMaxValue(9999),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('loop')
        .setDescription('반복: 끔 / 지금 곡 한 곡 / 대기열 순환')
        .setDescriptionLocalizations(enUS('Loop: off / one track / whole queue'))
        .addStringOption((opt) =>
          opt
            .setName('mode')
            .setDescription('off=끔, track=한 곡, queue=대기열 끝나면 처음부터')
            .setDescriptionLocalizations(enUS('off | track | queue'))
            .setRequired(true)
            .addChoices(
              { name: '끔', name_localizations: enUS('Off'), value: 'off' },
              { name: '한 곡 (지금 재생)', name_localizations: enUS('One track'), value: 'track' },
              { name: '대기열 순환', name_localizations: enUS('Queue'), value: 'queue' },
            ),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('queue')
        .setDescription('대기열 확인 (페이지·이전/다음 버튼)')
        .setDescriptionLocalizations(enUS('View queue with paging buttons'))
        .addIntegerOption((opt) =>
          opt
            .setName('page')
            .setDescription('페이지 번호 (기본 1)')
            .setDescriptionLocalizations(enUS('Page number (default 1)'))
            .setRequired(false)
            .setMinValue(1),
        ),
    );

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
        .setDescriptionLocalizations(enUS('Channel to join (empty = your current voice channel)'))
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

  new SlashCommandBuilder()
    .setName('강화')
    .setDescription('검을 강화합니다. (확률 존재)')
    .setDescriptionLocalizations(enUS('Enhance your sword (RNG)')),
  new SlashCommandBuilder()
    .setName('판매')
    .setDescription('검을 판매하여 돈을 얻습니다.')
    .setDescriptionLocalizations(enUS('Sell your sword for money')),
  new SlashCommandBuilder()
    .setName('정보')
    .setDescription('내 검과 재산 정보를 확인합니다.')
    .setDescriptionLocalizations(enUS('View sword and balance')),
  new SlashCommandBuilder()
    .setName('돈')
    .setDescription('현재 보유한 돈을 확인합니다.')
    .setDescriptionLocalizations(enUS('Check your balance')),
  new SlashCommandBuilder()
    .setName('랭킹')
    .setDescription('전체 유저 랭킹을 확인합니다.')
    .setDescriptionLocalizations(enUS('Leaderboard')),
  new SlashCommandBuilder()
    .setName('출첵')
    .setDescription('매일 출석체크 보상을 받습니다.')
    .setDescriptionLocalizations(enUS('Daily attendance reward')),
  new SlashCommandBuilder()
    .setName('돈내놔')
    .setDescription('일정 시간마다 랜덤 용돈을 받습니다.')
    .setDescriptionLocalizations(enUS('Random pocket money on cooldown')),

  new SlashCommandBuilder()
    .setName('배틀')
    .setDescription('다른 유저와 대결합니다.')
    .setDescriptionLocalizations(enUS('Battle another user'))
    .addUserOption((opt) =>
      opt
        .setName('상대')
        .setNameLocalizations(enUS('opponent'))
        .setDescription('대결할 상대를 선택하세요')
        .setDescriptionLocalizations(enUS('Opponent'))
        .setRequired(true),
    ),
  new SlashCommandBuilder()
    .setName('슬롯')
    .setDescription('슬롯 머신을 돌립니다.')
    .setDescriptionLocalizations(enUS('Spin the slots'))
    .addIntegerOption((opt) =>
      opt
        .setName('금액')
        .setNameLocalizations(enUS('amount'))
        .setDescription('배팅할 금액')
        .setDescriptionLocalizations(enUS('Bet amount'))
        .setRequired(true)
        .setMinValue(1),
    ),
  new SlashCommandBuilder()
    .setName('홀짝')
    .setDescription('홀짝 게임을 합니다.')
    .setDescriptionLocalizations(enUS('Odd or even game'))
    .addStringOption((opt) =>
      opt
        .setName('선택')
        .setNameLocalizations(enUS('pick'))
        .setDescription('홀 또는 짝')
        .setDescriptionLocalizations(enUS('Odd or even'))
        .setRequired(true)
        .addChoices(
          { name: '홀', name_localizations: enUS('Odd'), value: '홀' },
          { name: '짝', name_localizations: enUS('Even'), value: '짝' },
        ),
    )
    .addIntegerOption((opt) =>
      opt
        .setName('금액')
        .setNameLocalizations(enUS('amount'))
        .setDescription('배팅할 금액')
        .setDescriptionLocalizations(enUS('Bet amount'))
        .setRequired(true)
        .setMinValue(1),
    ),
  new SlashCommandBuilder()
    .setName('가위바위보')
    .setDescription('가위바위보를 합니다.')
    .setDescriptionLocalizations(enUS('Rock paper scissors'))
    .addStringOption((opt) =>
      opt
        .setName('선택')
        .setNameLocalizations(enUS('pick'))
        .setDescription('가위, 바위, 보')
        .setDescriptionLocalizations(enUS('Rock, paper, or scissors'))
        .setRequired(true)
        .addChoices(
          { name: '가위', name_localizations: enUS('Scissors'), value: '가위' },
          { name: '바위', name_localizations: enUS('Rock'), value: '바위' },
          { name: '보', name_localizations: enUS('Paper'), value: '보' },
        ),
    )
    .addIntegerOption((opt) =>
      opt
        .setName('금액')
        .setNameLocalizations(enUS('amount'))
        .setDescription('배팅할 금액')
        .setDescriptionLocalizations(enUS('Bet amount'))
        .setRequired(true)
        .setMinValue(1),
    ),

  new SlashCommandBuilder()
    .setName('주식목록')
    .setDescription('현재 상장된 주식 시세를 확인합니다.')
    .setDescriptionLocalizations(enUS('Listed stock prices')),
  new SlashCommandBuilder()
    .setName('주식차트')
    .setDescription('특정 주식의 차트를 확인합니다.')
    .setDescriptionLocalizations(enUS('Stock chart'))
    .addStringOption((opt) =>
      opt
        .setName('종목')
        .setNameLocalizations(enUS('symbol'))
        .setDescription('종목 심볼')
        .setDescriptionLocalizations(enUS('Ticker symbol'))
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
    .setDescriptionLocalizations(enUS('Buy stock'))
    .addStringOption((opt) =>
      opt
        .setName('종목')
        .setNameLocalizations(enUS('symbol'))
        .setDescription('종목 심볼')
        .setDescriptionLocalizations(enUS('Ticker symbol'))
        .setRequired(true)
        .addChoices(
          { name: '떡락전자 (SAMSUNG)', value: 'SAMSUNG' },
          { name: '화성갈끄니까 (DOGE)', value: 'DOGE' },
          { name: '테슬라 (TESLA)', value: 'TESLA' },
          { name: '사과 (APPLE)', value: 'APPLE' },
          { name: '비트코인 (BITCOIN)', value: 'BITCOIN' },
        ),
    )
    .addIntegerOption((opt) =>
      opt
        .setName('수량')
        .setNameLocalizations(enUS('qty'))
        .setDescription('매수할 수량')
        .setDescriptionLocalizations(enUS('Shares to buy'))
        .setRequired(true)
        .setMinValue(1),
    ),
  new SlashCommandBuilder()
    .setName('매도')
    .setDescription('주식을 매도합니다.')
    .setDescriptionLocalizations(enUS('Sell stock'))
    .addStringOption((opt) =>
      opt
        .setName('종목')
        .setNameLocalizations(enUS('symbol'))
        .setDescription('종목 심볼')
        .setDescriptionLocalizations(enUS('Ticker symbol'))
        .setRequired(true)
        .addChoices(
          { name: '떡락전자 (SAMSUNG)', value: 'SAMSUNG' },
          { name: '화성갈끄니까 (DOGE)', value: 'DOGE' },
          { name: '테슬라 (TESLA)', value: 'TESLA' },
          { name: '사과 (APPLE)', value: 'APPLE' },
          { name: '비트코인 (BITCOIN)', value: 'BITCOIN' },
        ),
    )
    .addIntegerOption((opt) =>
      opt
        .setName('수량')
        .setNameLocalizations(enUS('qty'))
        .setDescription('매도할 수량')
        .setDescriptionLocalizations(enUS('Shares to sell'))
        .setRequired(true)
        .setMinValue(1),
    ),
  new SlashCommandBuilder()
    .setName('내주식')
    .setDescription('내 주식 잔고를 확인합니다.')
    .setDescriptionLocalizations(enUS('Your stock holdings')),

  new SlashCommandBuilder()
    .setName('레이드정보')
    .setDescription('현재 진행 중인 레이드 정보를 확인합니다.')
    .setDescriptionLocalizations(enUS('Current raid status')),
  new SlashCommandBuilder()
    .setName('공격')
    .setDescription('레이드 보스를 공격합니다.')
    .setDescriptionLocalizations(enUS('Attack the raid boss')),
  new SlashCommandBuilder()
    .setName('레이드소환')
    .setDescription('새로운 레이드 보스를 소환합니다.')
    .setDescriptionLocalizations(enUS('Summon a new raid boss')),

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
          {
            name: '개요',
            name_localizations: enUS('Overview'),
            value: 'overview',
          },
          {
            name: '음성 · /music',
            name_localizations: enUS('Voice · /music'),
            value: 'music',
          },
          {
            name: '검 · 미니게임 · 주식 · 레이드',
            name_localizations: enUS('Sword · minigames · stocks · raid'),
            value: 'game',
          },
          {
            name: 'AI · ping · 음성 입장',
            name_localizations: enUS('AI · ping · voice join'),
            value: 'utility',
          },
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
        .setDescriptionLocalizations(enUS('API surface (default: .env KARMOLAB_AI_SURFACE / GEMINI_SURFACE)'))
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
