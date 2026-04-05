// @ts-nocheck
/**
 * 음성 재생 명령 응답 정책(요약)
 * - 성공 알림: 채널에 보이게(public) — `/music queue`·`/music skip`·`/music stop`·`/music play`·`/music shuffle`·`/music remove`·`/music loop` 완료
 * - 조용한 거절: ephemeral — 길드 밖, skip/stop 무동작
 * - 재생 실패: 로그 + 슬래시 친 텍스트 채널에 짧은 한 줄(약 15초 쿨다운)
 */
import {
  ActionRowBuilder,
  ButtonBuilder,
  ButtonStyle,
  MessageFlags,
  PermissionFlagsBits,
} from 'discord.js';
import play from 'play-dl';
import {
  enqueueYouTube,
  enqueueYouTubeTracks,
  fetchYoutubePlaylistEntries,
  skipTrack,
  stopMusic,
  shuffleWaitingQueue,
  removeWaitingTrackAt,
  setMusicLoopMode,
  getMusicQueuePage,
  withTimeout,
  YOUTUBE_RESOLVE_TIMEOUT_MS,
  getYoutubePlaylistMaxTracks,
  canonicalYoutubeWatchUrl,
  searchYoutubeFirstVideoViaYoutubei,
} from '../music-player';

/** 플레이리스트 메타(항목 수)만 가져올 때는 검색 단일 곡보다 여유 있게 */
const YOUTUBE_PLAYLIST_RESOLVE_MS = 90_000;

const QUEUE_BTN_PREFIX = 'music_queue:';

function buildQueuePayload(guildId: string, page: number) {
  const q = getMusicQueuePage(guildId, page);
  const parts = [];
  if (q.nowPlaying) {
    parts.push(`**재생 중:** ${q.nowPlaying}`);
  }
  parts.push(`**반복:** ${q.loopLabel}`);
  if (q.totalWaiting === 0) {
    parts.push(q.nowPlaying ? '대기열이 비어 있습니다.' : '재생 중인 곡과 대기열이 없습니다.');
  } else {
    parts.push(`**대기열** ${q.page}/${q.totalPages}페이지 · 총 ${q.totalWaiting}곡`);
    parts.push(q.lines.join('\n') || '…');
  }
  const content = parts.join('\n\n').slice(0, 1900);
  const components = [];
  if (q.totalPages > 1) {
    const row = new ActionRowBuilder().addComponents(
      new ButtonBuilder()
        .setCustomId(`${QUEUE_BTN_PREFIX}${q.page - 1}`)
        .setLabel('이전')
        .setStyle(ButtonStyle.Secondary)
        .setDisabled(q.page <= 1),
      new ButtonBuilder()
        .setCustomId(`${QUEUE_BTN_PREFIX}${q.page + 1}`)
        .setLabel('다음')
        .setStyle(ButtonStyle.Secondary)
        .setDisabled(q.page >= q.totalPages),
    );
    components.push(row);
  }
  return { content, components };
}

/**
 * 대기열 페이지 버튼. 처리했으면 true.
 */
export async function tryHandleMusicQueueButton(interaction) {
  if (!interaction.isButton() || !interaction.customId?.startsWith(QUEUE_BTN_PREFIX)) {
    return false;
  }
  if (!interaction.guildId) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return true;
  }
  const raw = interaction.customId.slice(QUEUE_BTN_PREFIX.length);
  const page = parseInt(raw, 10);
  if (Number.isNaN(page) || page < 1) {
    await interaction.reply({ content: '잘못된 페이지입니다.', flags: MessageFlags.Ephemeral });
    return true;
  }
  try {
    const payload = buildQueuePayload(interaction.guildId, page);
    await interaction.update(payload);
  } catch (e) {
    console.error('[queue button]', e);
    await interaction.reply({ content: '대기열을 갱신하지 못했습니다.', flags: MessageFlags.Ephemeral }).catch(() => {});
  }
  return true;
}

/**
 * `playlist?list=` / `watch?…&list=` / `youtu.be/…?list=` → 정규화된 playlist URL.
 * 그 외 `yt_validate === 'playlist'` 인 입력은 그대로 반환.
 */
function tryYoutubePlaylistPageUrl(query: string): string | null {
  const q = query.trim();
  if (!q) return null;
  if (play.yt_validate(q) === 'playlist') return q;
  if (!q.includes('list=')) return null;
  try {
    const u = q.includes('://') ? new URL(q) : new URL(`https://www.youtube.com/watch?v=0&${q.replace(/^\?/, '')}`);
    const list = u.searchParams.get('list');
    if (!list) return null;
    return `https://www.youtube.com/playlist?list=${encodeURIComponent(list)}`;
  } catch {
    return null;
  }
}

async function resolveYouTube(query: string) {
  const q = (query ?? '').trim();
  if (!q) return { error: 'empty' };

  const plUrl = tryYoutubePlaylistPageUrl(q);
  if (plUrl) {
    try {
      const { title, entries } = await withTimeout(
        fetchYoutubePlaylistEntries(plUrl),
        YOUTUBE_PLAYLIST_RESOLVE_MS,
        '플레이리스트 로드',
      );
      return { kind: 'playlist', playlistTitle: title, tracks: entries };
    } catch (e) {
      console.warn('[play] 플레이리스트 로드 실패:', e instanceof Error ? e.message : e);
      if (play.yt_validate(q) !== 'video') {
        return { error: 'playlist_unavailable' };
      }
      /* watch URL에 list=만 붙은 경우 등: 단일 동영상으로 폴백 */
    }
  }

  const v = play.yt_validate(q);
  if (v === 'video') {
    const info = await play.video_info(q);
    const title = info.video_details?.title || 'YouTube';
    const url = canonicalYoutubeWatchUrl(info.video_details?.url || q);
    return { kind: 'single', title, url };
  }
  if (v === 'playlist') {
    try {
      const { title, entries } = await withTimeout(
        fetchYoutubePlaylistEntries(q),
        YOUTUBE_PLAYLIST_RESOLVE_MS,
        '플레이리스트 로드',
      );
      return { kind: 'playlist', playlistTitle: title, tracks: entries };
    } catch (e) {
      console.warn('[play] 플레이리스트(직접 URL) 로드 실패:', e instanceof Error ? e.message : e);
      return { error: 'playlist_unavailable' };
    }
  }
  let results;
  try {
    results = await play.search(q, { limit: 1, source: { youtube: 'video' } });
  } catch (e) {
    console.warn('[play] play-dl 검색 실패, youtubei.js 폴백:', e instanceof Error ? e.message : e);
    const fb = await searchYoutubeFirstVideoViaYoutubei(q);
    if (fb) return { kind: 'single', title: fb.title, url: canonicalYoutubeWatchUrl(fb.url) };
    throw e;
  }
  if (!results.length) {
    const fb = await searchYoutubeFirstVideoViaYoutubei(q);
    if (fb) return { kind: 'single', title: fb.title, url: canonicalYoutubeWatchUrl(fb.url) };
    return { error: 'notfound' };
  }
  const first = results[0];
  if (first.type !== 'video') {
    const fb = await searchYoutubeFirstVideoViaYoutubei(q);
    if (fb) return { kind: 'single', title: fb.title, url: canonicalYoutubeWatchUrl(fb.url) };
    return { error: 'notfound' };
  }
  return { kind: 'single', title: first.title || q, url: canonicalYoutubeWatchUrl(first.url) };
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
    await interaction.editReply({ content: '음성 채널에 들어간 뒤 `/music play`를 사용하세요.' });
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
    const resolveMs =
      tryYoutubePlaylistPageUrl(query) || play.yt_validate(query) === 'playlist'
        ? YOUTUBE_PLAYLIST_RESOLVE_MS
        : YOUTUBE_RESOLVE_TIMEOUT_MS;
    resolved = await withTimeout(resolveYouTube(query), resolveMs, 'YouTube 검색/조회');
    console.log('[play] YouTube 조회 완료');
  } catch (e: any) {
    console.error('[play] YouTube 조회 실패:', e?.message ?? e);
    await interaction.editReply({ content: `검색/조회 실패: ${e?.message || String(e)}` });
    return;
  }
  if (resolved.error === 'playlist_unavailable') {
    await interaction.editReply({
      content:
        '플레이리스트를 불러오지 못했습니다. 공개 목록인지, `youtube.com/playlist?list=` 링크인지 확인해 주세요. (비공개·삭제된 목록은 불가)',
    });
    return;
  }
  if (resolved.error === 'notfound' || resolved.error === 'empty') {
    await interaction.editReply({ content: 'YouTube에서 결과를 찾지 못했습니다.' });
    return;
  }

  if (resolved.kind === 'playlist') {
    const pt = resolved.playlistTitle.slice(0, 80);
    const n = resolved.tracks.length;
    const cap = getYoutubePlaylistMaxTracks();
    const policy = Number.isFinite(cap) ? `상한 ${cap}곡` : '무제한(목록 끝까지)';
    await interaction.editReply({
      content: `**${pt}** — ${n}곡을 대기열에 넣는 중… (${policy})`,
    });
    console.log('[play] enqueue playlist 시작...', n);
    let tick = 0;
    const progress = setInterval(() => {
      tick += 15;
      void interaction
        .editReply({
          content: `**${pt}** — 음성 연결·대기열 추가 중… (${tick}초 경과)`,
        })
        .catch(() => {});
    }, 15_000);
    let result;
    const notify = interaction.channelId ? { notifyTextChannelId: interaction.channelId } : undefined;
    try {
      result = await enqueueYouTubeTracks(vc, resolved.tracks, notify);
    } finally {
      clearInterval(progress);
    }
    console.log('[play] enqueue playlist 결과:', result);
    if (!result.ok) {
      await interaction.editReply({ content: `재생 준비 실패: ${result.error}` });
      return;
    }
    const tl = result.totalListed;
    const summary = `목록 **${tl}곡** 중 대기열 추가 **${result.added}곡** · URL·형식 제외 **${result.skipped}곡**`;
    const action = result.started ? '첫 곡 재생을 시작했습니다.' : '대기열에 반영했습니다.';
    await interaction.editReply({
      content: `**${resolved.playlistTitle.slice(0, 80)}**\n${summary}\n${action}`.slice(0, 2000),
    });
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
  const notifySingle = interaction.channelId ? { notifyTextChannelId: interaction.channelId } : undefined;
  try {
    result = await enqueueYouTube(vc, resolved.title, resolved.url, notifySingle);
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
  if (ok) {
    await interaction.reply({ content: '다음 곡으로 넘깁니다.' });
  } else {
    await interaction.reply({ content: '건너뛸 재생이 없습니다.', flags: MessageFlags.Ephemeral });
  }
}

export async function handleStopMusic(ctx, interaction) {
  if (!interaction.guildId) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const ok = stopMusic(interaction.guildId);
  if (ok) {
    await interaction.reply({ content: '재생을 멈추고 대기열을 비웠습니다.' });
  } else {
    await interaction.reply({ content: '멈출 재생이 없습니다.', flags: MessageFlags.Ephemeral });
  }
}

export async function handleShuffle(ctx, interaction) {
  if (!interaction.guildId) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const r = shuffleWaitingQueue(interaction.guildId);
  if (!r.ok) {
    const msg =
      r.reason === 'single'
        ? '대기열이 한 곡뿐이라 섞을 수 없습니다.'
        : '대기 중인 곡이 없습니다. (재생 중인 곡만 있으면 `/music skip` 후 다시 시도하세요.)';
    await interaction.reply({ content: msg, flags: MessageFlags.Ephemeral });
    return;
  }
  await interaction.reply({ content: `대기 중 **${r.shuffled}곡** 순서를 섞었습니다.` });
}

export async function handleRemove(ctx, interaction) {
  if (!interaction.guildId) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const raw = interaction.options.getInteger('index');
  if (raw == null) {
    await interaction.reply({ content: '제거할 번호(`index`)가 필요합니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const r = removeWaitingTrackAt(interaction.guildId, raw);
  if (!r.ok) {
    if (r.reason === 'empty') {
      await interaction.reply({
        content: '대기 중인 곡이 없습니다. (`/music queue`에 번호가 보일 때만 제거할 수 있습니다. 지금 재생 중인 곡은 `/music skip`으로 건너뜁니다.)',
        flags: MessageFlags.Ephemeral,
      });
      return;
    }
    await interaction.reply({
      content: `번호는 **1~${r.max}** 사이로 입력하세요. (/music queue 목록과 같은 번호)`,
      flags: MessageFlags.Ephemeral,
    });
    return;
  }
  const label = r.title.slice(0, 120);
  await interaction.reply({ content: `대기열에서 제거했습니다: **${label}**` });
}

export async function handleLoop(ctx, interaction) {
  if (!interaction.guildId) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const raw = interaction.options.getString('mode');
  const mode = raw === 'off' || raw === 'track' || raw === 'queue' ? raw : null;
  if (!mode) {
    await interaction.reply({ content: 'mode는 off / track / queue 중 하나여야 합니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const r = setMusicLoopMode(interaction.guildId, mode);
  if (!r.ok) {
    await interaction.reply({ content: r.error, flags: MessageFlags.Ephemeral });
    return;
  }
  const label = r.mode === 'track' ? '**한 곡** (지금 재생)' : r.mode === 'queue' ? '**대기열 순환**' : '**끔**';
  await interaction.reply({ content: `반복: ${label}` });
}

export async function handleQueue(ctx, interaction) {
  if (!interaction.guildId) {
    await interaction.reply({ content: '서버에서만 사용할 수 있습니다.', flags: MessageFlags.Ephemeral });
    return;
  }
  const raw = interaction.options.getInteger('page');
  const page = raw != null && raw >= 1 ? raw : 1;
  const payload = buildQueuePayload(interaction.guildId, page);
  await interaction.reply(payload);
}
