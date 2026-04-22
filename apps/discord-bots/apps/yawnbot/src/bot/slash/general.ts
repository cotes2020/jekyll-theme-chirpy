import { EmbedBuilder, MessageFlags } from 'discord.js';
import type { ChatInputCommandInteraction } from 'discord.js';
import { showHelpPage } from '../game-ui';
import type { BotContext } from './bot-context';

export async function handlePing(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const { gameData, client } = ctx;
  const embed = new EmbedBuilder()
    .setTitle(gameData.getMessage('General_Ping_Title'))
    .setDescription(gameData.getMessage('General_Ping_Desc', client.ws.ping))
    .setColor(0x00bcd4);
  await interaction.reply({ embeds: [embed] });
}

/** `/도움말` 주제: 개요 | music | game(페이지 임베드) | utility — 채널 스팸을 줄이려고 기본은 ephemeral */
export async function handleHelp(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const topic = interaction.options.getString('주제');

  if (!topic || topic === 'overview') {
    const embed = new EmbedBuilder()
      .setTitle('🦦 욘봇 도움말')
      .setDescription(
        '`/도움말 주제`로 자세한 내용을 확인할 수 있어요.\n' +
          '모든 응답은 **나만 보기**로 표시되니 안심하세요!',
      )
      .addFields(
        {
          name: '🎮 게임',
          value:
            '주제: **game**\n' +
            '검, 미니게임(슬롯·홀짝·가위바위보), 주식, 레이드, 출석 등\n' +
            '각 게임의 규칙, 아이템, 경제 시스템 설명',
        },
        {
          name: '🎵 음악',
          value:
            '주제: **music**\n' +
            '`/music play` / `speak` / `sound` / `queue` / `skip` / `stop` / `shuffle` / `remove` / `loop`\n' +
            '음성 채널이 필요합니다.',
        },
        {
          name: '🧠 기억 & AI',
          value:
            '주제: **memory**\n' +
            '`/기억 확인` / `저장` / `수정` / `핫로그` — 대화 기록 및 프로필 관리\n' +
            '주제: **utility** — `/yawn` (AI 질문), `/ping` (봇 상태)',
        },
        {
          name: '🎧 음성 & 관리',
          value:
            '주제: **utility**\n' +
            '`/음성입장` / `음성퇴장` / `/cursor-edit` / `/admin-reload` / `/admin-save`',
        },
      )
      .setFooter({ text: '예: /도움말 game  |  /도움말 music  |  /도움말 memory' })
      .setColor(0x7c4dff);
    await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
    return;
  }

  if (topic === 'music') {
    const embed = new EmbedBuilder()
      .setTitle('/music — 음악 플레이어')
      .setDescription('한 **대기열**을 공유합니다. 명령을 친 사람이 **음성(또는 스테이지) 채널**에 있어야 해요.')
      .addFields(
        {
          name: '▶️ play',
          value: 'YouTube URL 또는 검색어로 노래 재생\n`query` 필수 — URL, 플레이리스트, 검색어 모두 지원',
        },
        {
          name: '🎤 speak',
          value: 'Edge TTS로 텍스트 음성 변환 재생\n`text` 선택 — 입력 안 하면 질문을 음성으로 읽음',
        },
        {
          name: '🔊 sound',
          value: '효과음 또는 사운드 재생\n`file` (로컬) / `url` (웹) / `clip` (미리 저장된) 중 하나만 사용',
        },
        {
          name: '📋 queue',
          value: '대기열 확인 (페이지로 넘김)',
          inline: true,
        },
        {
          name: '⏭️ skip',
          value: '다음 곡으로 건너뛰기',
          inline: true,
        },
        {
          name: '⏹️ stop',
          value: '재생 중지 & 대기열 초기화',
          inline: true,
        },
        {
          name: '🔀 shuffle',
          value: '대기열 섞기',
          inline: true,
        },
        {
          name: '🗑️ remove',
          value: '특정 곡 제거 (index 번호)',
          inline: true,
        },
        {
          name: '🔁 loop',
          value: '반복 설정: off (반복 안 함) / track (한곡 반복) / queue (전체 반복)',
          inline: true,
        },
      )
      .setColor(0x7c4dff);
    await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
    return;
  }

  if (topic === 'utility') {
    const embed = new EmbedBuilder()
      .setTitle('🤖 AI · 유틸 · 음성')
      .addFields(
        {
          name: '/yawn — AI에 질문',
          value:
            '`질문` (필수) + 선택 `api` (기본 / AI Studio / Vertex) + 선택 `model`\n' +
            '→ Gemini 또는 Claude CLI로 응답 (env ASSISTANT_AI_PROVIDER 설정)',
        },
        {
          name: '/ping — 봇 상태',
          value: '디스코드 웹소켓 지연시간(ms) 표시 (낮을수록 빠름)',
        },
        {
          name: '🎧 /음성입장, /voice-join',
          value: '봇을 음성 채널로 초대\n옵션: 채널 지정 또는 본인이 있는 채널로 자동 입장',
          inline: true,
        },
        {
          name: '🎧 /음성퇴장, /voice-leave',
          value: '봇이 음성 채널에서 나감',
          inline: true,
        },
        {
          name: '👨‍💻 /cursor-edit (관리자)',
          value: '로컬 Cursor agent 실행 (코드 수정 AI)',
          inline: true,
        },
        {
          name: '♻️ /admin-reload (관리자)',
          value: '봇 설정 재로드',
          inline: true,
        },
        {
          name: '💾 /admin-save (관리자)',
          value: '게임 데이터 즉시 저장',
          inline: true,
        },
      )
      .setColor(0x7c4dff);
    await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
    return;
  }

  if (topic === 'memory') {
    const embed = new EmbedBuilder()
      .setTitle('🧠 기억 커맨드 — AI 비서 메모리 관리')
      .setDescription(
        '대화 기록 및 프로필을 관리합니다.\n' +
          '모든 정보는 MEMO_REPO_PATH에 자동으로 git commit됩니다.',
      )
      .addFields(
        {
          name: '📖 /기억 확인',
          value:
            '저장된 정보 확인\n' +
            '→ user.md (나에 대한 정보) + self.md (봇 자신에 대한 정보)\n' +
            '→ 파일 용량 표시 + 크면 핫로그 사용 안내',
        },
        {
          name: '💾 /기억 저장',
          value: '지금까지의 대화를 memo 레포에 즉시 커밋 (1시간마다 자동 저장되지만 수동 저장도 가능)',
        },
        {
          name: '✏️ /기억 수정 [내용]',
          value:
            'AI 도움으로 user.md 수정\n' +
            '예: `/기억 수정 새로운 취미는 게임`\n' +
            '→ 수정 전후 줄 수 비교 + 새 파일 크기 표시',
        },
        {
          name: '🔥 /기억 핫로그',
          value:
            '최근 중요 기억들 조회 (최대 20개)\n' +
            '→ 날짜별로 태깅된 즉시 저장된 정보들\n' +
            '→ 파일이 클 때 빠르게 확인용',
        },
        {
          name: '📚 메모리 계층 구조',
          value:
            '`logs/` — 일일 대화 원본 (즉시 기록)\n' +
            '`daily/` — 어제 대화 요약 (매일 생성)\n' +
            '`weekly/` — 주간 요약 (매주 생성)\n' +
            '`user.md` — 나에 대한 누적 정보\n' +
            '`self.md` — 봇 자신에 대한 누적 정보',
        },
      )
      .setColor(0x7c4dff);
    await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
    return;
  }

  if (topic === 'game') {
    await showHelpPage(ctx, interaction, 0, false, { ephemeral: true });
    return;
  }

  await interaction.reply({
    content: '알 수 없는 주제입니다. `/도움말`에서 주제를 다시 선택해 주세요.',
    flags: MessageFlags.Ephemeral,
  });
}
