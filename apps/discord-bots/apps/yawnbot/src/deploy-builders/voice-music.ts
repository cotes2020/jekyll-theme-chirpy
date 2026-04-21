/**
 * 음성·음악 관련 슬래시 빌더 — deploy-commands.ts 에서 분리.
 */
import { SlashCommandBuilder, ChannelType, Locale } from 'discord.js';

const EN = Locale.EnglishUS;
const enUS = (s: string): Record<string, string> => ({ [EN]: s });

export const voiceJoin = () =>
  new SlashCommandBuilder()
    .setName('voice-join')
    .setDescription('봇을 음성·스테이지 채널에 연결 (/음성입장 과 동일)')
    .setDescriptionLocalizations(
      enUS('Connect the bot to a voice/stage channel (same as /음성입장)'),
    )
    .addChannelOption((opt) =>
      opt
        .setName('channel')
        .setDescription('입장할 채널 (비우면 본인이 있는 음성 채널)')
        .setDescriptionLocalizations(
          enUS('Channel to join (empty = your current voice channel)'),
        )
        .addChannelTypes(ChannelType.GuildVoice, ChannelType.GuildStageVoice)
        .setRequired(false),
    );

export const voiceLeave = () =>
  new SlashCommandBuilder()
    .setName('voice-leave')
    .setDescription('봇 음성 연결 해제 (/음성퇴장 과 동일)')
    .setDescriptionLocalizations(enUS('Disconnect the bot from voice (same as /음성퇴장)'));

/** 음성 재생·대기열 — YouTube·TTS·클립·skip/stop/shuffle/remove/loop/queue 서브커맨드 */
export const musicCommandGroup = () =>
  new SlashCommandBuilder()
    .setName('music')
    .setDescription('음성 채널 YouTube·TTS·클립 재생 및 대기열')
    .setDescriptionLocalizations(
      enUS('YouTube, TTS, and clip playback with a shared queue'),
    )
    .addSubcommand((sc) =>
      sc
        .setName('play')
        .setDescription('YouTube 동영상·플레이리스트 URL 또는 검색어로 재생 (음성 채널 필수)')
        .setDescriptionLocalizations(
          enUS('Play from YouTube URL, playlist, or search (voice channel required)'),
        )
        .addStringOption((opt) =>
          opt
            .setName('query')
            .setDescription('동영상/playlist?list= URL, watch?…&list=, 또는 검색어')
            .setDescriptionLocalizations(
              enUS('Video or playlist URL, or search text'),
            )
            .setRequired(true),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('speak')
        .setDescription('Edge TTS로 문장을 읽어 재생 (YouTube·클립과 동일 대기열, 디스코드 내장 TTS 아님)')
        .setDescriptionLocalizations(
          enUS('Speak text via Edge TTS (same queue as music; not built-in Discord TTS)'),
        )
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
        .setDescriptionLocalizations(
          enUS('Play attachment, URL, or packaged clip (exactly one of file/url/clip)'),
        )
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
        .setDescriptionLocalizations(
          enUS('Shuffle only the waiting queue (current track unchanged)'),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('remove')
        .setDescription('대기열에서 번호로 곡 제거 (/music queue 목록의 1·2·3… 과 동일)')
        .setDescriptionLocalizations(
          enUS('Remove a track by queue index (same numbers as /music queue)'),
        )
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
