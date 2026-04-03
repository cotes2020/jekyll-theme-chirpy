// @ts-nocheck
import { MessageFlags, PermissionFlagsBits } from 'discord.js';
import play from 'play-dl';
import {
  enqueueYouTube,
  skipTrack,
  stopMusic,
  getQueueSummary,
  withTimeout,
  YOUTUBE_RESOLVE_TIMEOUT_MS,
  canonicalYoutubeWatchUrl,
} from '../music-player';

async function resolveYouTube(query: string) {
  const q = (query ?? '').trim();
  if (!q) return { error: 'empty' };
  const v = play.yt_validate(q);
  if (v === 'video') {
    const info = await play.video_info(q);
    const title = info.video_details?.title || 'YouTube';
    const url = canonicalYoutubeWatchUrl(info.video_details?.url || q);
    return { title, url };
  }
  if (v === 'playlist') {
    return { error: 'playlist' };
  }
  const results = await play.search(q, { limit: 1, source: { youtube: 'video' } });
  if (!results.length) return { error: 'notfound' };
  const first = results[0];
  if (first.type !== 'video') return { error: 'notfound' };
  return { title: first.title || q, url: canonicalYoutubeWatchUrl(first.url) };
}

export async function handlePlay(ctx, interaction) {
  if (!interaction.inGuild()) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  /** 3초 내 첫 ACK 필수. 동기 검사만 하고 바로 defer (member 접근은 그 다음). */
  try {
    await interaction.deferReply();
  } catch (e) {
    const code = e && typeof e === 'object' && 'code' in e ? (e as { code: unknown }).code : undefined;
    if (code === 10062) {
      console.warn(
        '[play] deferReply 10062 (Unknown interaction): 인터랙션 만료 — 게이트웨이/회선 지연이 크거나 봇이 바쁩니다. 같은 명령을 다시 시도하거나 네트워크·호스트를 바꿔 보세요.',
      );
      return;
    }
    throw e;
  }

  if (!interaction.member) {
    await interaction.editReply({ content: '멤버 정보를 불러올 수 없습니다. 잠시 후 다시 시도하세요.' });
    return;
  }

  const vc = interaction.member.voice?.channel;
  if (!vc || !vc.isVoiceBased()) {
    await interaction.editReply({ content: '음성 채널에 들어간 뒤 `/play`를 사용하세요.' });
    return;
  }
  const botMember = interaction.guild.members.me;
  if (!botMember) {
    await interaction.editReply({ content: '봇 멤버 정보를 불러올 수 없습니다.' });
    return;
  }
  const perms = vc.permissionsFor(botMember);
  if (!perms?.has([PermissionFlagsBits.Connect, PermissionFlagsBits.Speak, PermissionFlagsBits.ViewChannel])) {
    await interaction.editReply({
      content: '봇에게 해당 음성 채널 **보기·연결·말하기(Speak)** 권한이 필요합니다.',
    });
    return;
  }

  const query = interaction.options.getString('query') ?? '';
  await interaction.editReply({ content: 'YouTube에서 곡 정보를 확인하는 중…' });
  console.log('[play] deferred, query=', query.slice(0, 120));

  let resolved;
  try {
    console.log('[play] YouTube 검색/조회 시작...');
    resolved = await withTimeout(resolveYouTube(query), YOUTUBE_RESOLVE_TIMEOUT_MS, 'YouTube 검색/조회');
    console.log('[play] YouTube 조회 완료');
  } catch (e: any) {
    console.error('[play] YouTube 조회 실패:', e?.message ?? e);
    await interaction.editReply({ content: `검색/조회 실패: ${e?.message || String(e)}` });
    return;
  }
  if (resolved.error === 'playlist') {
    await interaction.editReply({ content: '플레이리스트는 지원하지 않습니다. 동영상 하나의 URL을 넣어주세요.' });
    return;
  }
  if (resolved.error === 'notfound' || resolved.error === 'empty') {
    await interaction.editReply({ content: 'YouTube에서 결과를 찾지 못했습니다.' });
    return;
  }

  await interaction.editReply({
    content: `**${resolved.title.slice(0, 80)}** — 음성 채널에 연결하는 중…`,
  });
  console.log('[play] enqueue 시작...');
  let tick = 0;
  const progress = setInterval(() => {
    tick += 15;
    void interaction
      .editReply({
        content: `**${resolved.title.slice(0, 80)}** — 음성 연결 중… (${tick}초 경과)`,
      })
      .catch(() => {});
  }, 15_000);
  let result;
  try {
    result = await enqueueYouTube(vc, resolved.title, resolved.url);
  } finally {
    clearInterval(progress);
  }
  console.log('[play] enqueue 결과:', result.ok ? { position: result.position, started: result.started } : result);
  if (!result.ok) {
    await interaction.editReply({ content: `재생 준비 실패: ${result.error}` });
    return;
  }

  await interaction.editReply({
    content: result.started
      ? `**${resolved.title}** 재생을 시작합니다.`
      : `대기열 **${result.position}번**에 추가: **${resolved.title}**`,
  });
}

export async function handleSkip(ctx, interaction) {
  if (!interaction.guildId) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const ok = skipTrack(interaction.guildId);
  await interaction.reply({
    content: ok ? '다음 곡으로 넘깁니다.' : '건너뛸 재생이 없습니다.',
    flags: MessageFlags.Ephemeral,
  });
}

export async function handleStopMusic(ctx, interaction) {
  if (!interaction.guildId) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const ok = stopMusic(interaction.guildId);
  await interaction.reply({
    content: ok ? '재생을 멈추고 대기열을 비웠습니다.' : '멈출 재생이 없습니다.',
    flags: MessageFlags.Ephemeral,
  });
}

export async function handleQueue(ctx, interaction) {
  if (!interaction.guildId) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const list = getQueueSummary(interaction.guildId);
  await interaction.reply({
    content: list.length ? `대기열:\n${list.map((t, i) => `${i + 1}. ${t}`).join('\n')}`.slice(0, 1900) : '대기열이 비어 있습니다.',
    flags: MessageFlags.Ephemeral,
  });
}
