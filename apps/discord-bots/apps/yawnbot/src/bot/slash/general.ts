// @ts-nocheck
import { EmbedBuilder, MessageFlags } from 'discord.js';
import { showHelpPage } from '../game-ui';

export async function handlePing(ctx, interaction) {
  const { gameData, client } = ctx;
  const embed = new EmbedBuilder()
    .setTitle(gameData.getMessage('General_Ping_Title'))
    .setDescription(gameData.getMessage('General_Ping_Desc', client.ws.ping))
    .setColor(0x00bcd4);
  await interaction.reply({ embeds: [embed] });
}

/** `/도움말` 주제: 개요 | music | game(페이지 임베드) | utility — 채널 스팸을 줄이려고 기본은 ephemeral */
export async function handleHelp(ctx, interaction) {
  const topic = interaction.options.getString('주제');

  if (!topic || topic === 'overview') {
    const embed = new EmbedBuilder()
      .setTitle('🦦 욘봇 도움말')
      .setDescription(
        '`/도움말`에서 **주제**를 고르면 카테고리별 안내를 **나만 보기**로 받을 수 있어요. (게임 쪽은 페이지 버튼으로 넘겨요.)',
      )
      .addFields(
        {
          name: '🎮 검 · 미니게임 · 주식 · 레이드',
          value:
            '주제 **검 · 미니게임 · 주식 · 레이드** — `/강화`, `/출첵`, `/슬롯`, `/주식목록`, `/레이드정보` 등 (상세 문구는 서버 메시지 설정)',
        },
        {
          name: '🎵 음성 · /music',
          value:
            '`/music play` · `speak` · `sound` · `queue` · `skip` · `stop` · `shuffle` · `remove` · `loop` — **음성 채널** 필요',
        },
        {
          name: '🧠 기억',
          value: '주제 **기억** — `/기억 확인`, `/기억 저장`, `/기억 수정` — 대화 기록 및 프로필 관리',
        },
        {
          name: '✨ 기타',
          value: '`/yawn` (Gemini/Vertex) · `/ping` · `/음성입장`·`/voice-join` · `/음성퇴장`·`/voice-leave`',
        },
      )
      .setFooter({ text: '관리자: /cursor-edit, /admin-reload, /admin-save' })
      .setColor(0x7c4dff);
    await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
    return;
  }

  if (topic === 'music') {
    const embed = new EmbedBuilder()
      .setTitle('/music')
      .setDescription('한 **대기열**을 공유해요. 명령을 친 사람이 **음성(또는 스테이지) 채널**에 있어야 해요.')
      .addFields(
        { name: 'play', value: 'YouTube URL·플레이리스트·검색어 (`query` 필수)' },
        { name: 'speak', value: 'Edge TTS (`text` 선택)' },
        { name: 'sound', value: '`file` / `url` / `clip` 중 **하나만**' },
        {
          name: '대기열 · 제어',
          value: '`queue` (페이지) · `skip` · `stop` · `shuffle` · `remove index` · `loop` (off / track / queue)',
        },
      )
      .setColor(0x7c4dff);
    await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
    return;
  }

  if (topic === 'utility') {
    const embed = new EmbedBuilder()
      .setTitle('AI · 유틸 · 음성')
      .addFields(
        {
          name: '/yawn',
          value: '`질문` + 선택 `api`(기본·AI Studio·Vertex), `model` — KarmoLabAI(`karmolab-ai/node`).',
        },
        { name: '/ping', value: '웹소켓 지연(ms).' },
        {
          name: '음성',
          value: '`/음성입장`·`/voice-join` (채널 옵션 또는 본인 음성) · `/음성퇴장`·`/voice-leave`',
        },
        {
          name: '관리자 전용',
          value: '`/cursor-edit` — 로컬 Cursor agent. `/admin-reload` · `/admin-save`',
        },
      )
      .setColor(0x7c4dff);
    await interaction.reply({ embeds: [embed], flags: MessageFlags.Ephemeral });
    return;
  }

  if (topic === 'memory') {
    const embed = new EmbedBuilder()
      .setTitle('🧠 기억 커맨드')
      .setDescription('대화 기록 및 프로필을 관리해요. 모든 정보는 MEMO_REPO_PATH에 자동으로 git commit됩니다.')
      .addFields(
        { name: '/기억 확인', value: '저장된 나에 대한 정보와 봇 자신에 대한 정보를 출력해요.' },
        { name: '/기억 저장', value: '대화 기록을 memo 레포에 즉시 커밋해요.' },
        { name: '/기억 수정 [내용]', value: 'AI 도움으로 user.md를 수정해요. 예: `/기억 수정 새로운 취미는 게임` ' },
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
