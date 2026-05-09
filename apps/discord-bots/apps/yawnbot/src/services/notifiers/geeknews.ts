/**
 * Hacker News 상위 스토리 (긱뉴스) 알림 (atkup-bot 흡수, TASK-YB-003).
 *
 * - 공개 Firebase API → topstories.json
 * - 슬래시 `/atkup news [count]` 호출 시 즉시 전송 (cron 없음)
 * - 정본: https://github.com/HackerNews/API
 */
import { EmbedBuilder, type Client, type SendableChannels } from 'discord.js';

const HN_COLOR = 0xff6600;

export interface HnStoryLine {
  title: string;
  href: string;
  score: number;
  by: string;
  host: string;
}

interface HnItem {
  id: number;
  title?: string;
  url?: string;
  score?: number;
  by?: string;
}

function storyHref(it: HnItem): string {
  if (it.url && /^https?:\/\//i.test(it.url)) return it.url;
  return `https://news.ycombinator.com/item?id=${it.id}`;
}

function hostLabel(it: HnItem): string {
  if (!it.url || !/^https?:\/\//i.test(it.url)) return 'news.ycombinator.com';
  try {
    return new URL(it.url).hostname.replace(/^www\./, '');
  } catch {
    return 'link';
  }
}

function truncateTitle(s: string, max = 120): string {
  const t = s.trim();
  if (t.length <= max) return t;
  return `${t.slice(0, max - 1)}…`;
}

/** 마크다운 링크용: 제목에 [ ] 가 있으면 깨지므로 제거 */
function sanitizeTitleForMdLink(title: string): string {
  return title.replace(/[\[\]]/g, '').trim() || '제목';
}

/**
 * @param limit 5~15 권장 (Discord 임베드 설명 길이)
 */
export async function fetchHnTopStories(limit: number): Promise<HnStoryLine[]> {
  const cap = Math.min(15, Math.max(5, Math.floor(limit)));
  const res = await fetch('https://hacker-news.firebaseio.com/v0/topstories.json');
  if (!res.ok) {
    throw new Error(`HN topstories 요청 실패: ${res.status}`);
  }
  const ids = (await res.json()) as number[];
  const slice = ids.slice(0, cap);
  const raw = await Promise.all(
    slice.map((id) =>
      fetch(`https://hacker-news.firebaseio.com/v0/item/${id}.json`).then(
        (r) => r.json() as Promise<HnItem | null>,
      ),
    ),
  );

  const lines: HnStoryLine[] = [];
  for (const it of raw) {
    if (!it || typeof it.title !== 'string' || !it.title.trim()) continue;
    lines.push({
      title: truncateTitle(it.title),
      href: storyHref(it),
      score: typeof it.score === 'number' ? it.score : 0,
      by: it.by || '—',
      host: hostLabel(it),
    });
  }
  return lines;
}

function buildGeekNewsEmbed(lines: HnStoryLine[]): EmbedBuilder {
  const desc = lines
    .map((s, i) => {
      const t = sanitizeTitleForMdLink(s.title);
      return `${i + 1}. [${t}](${s.href}) · ${s.score}pt · ${s.host} · ${s.by}`;
    })
    .join('\n');

  return new EmbedBuilder()
    .setTitle('📰 Hacker News · 긱 뉴스')
    .setDescription(desc.slice(0, 4090))
    .setColor(HN_COLOR)
    .setFooter({ text: 'YawnBot · GeekNews' });
}

export async function sendGeekNewsToChannel(channel: SendableChannels, count: number): Promise<number> {
  const stories = await fetchHnTopStories(count);
  if (stories.length === 0) {
    await channel.send({ content: 'Hacker News에서 가져온 글이 없습니다.' });
    return 0;
  }
  const embed = buildGeekNewsEmbed(stories);
  await channel.send({ embeds: [embed] });
  return stories.length;
}

/**
 * 슬래시 `/atkup news [count]` 진입점.
 * 환경변수 YAWNBOT_GEEKNEWS_CHANNEL_ID 미설정 시 'no_channel' 반환.
 */
export async function triggerGeekNewsOnce(
  client: Client,
  count: number,
): Promise<{ status: 'sent' | 'no_channel' | 'channel_unreachable'; sent: number }> {
  const channelId = process.env.YAWNBOT_GEEKNEWS_CHANNEL_ID?.trim();
  if (!channelId) {
    return { status: 'no_channel', sent: 0 };
  }
  const channel = await client.channels.fetch(channelId).catch(() => null);
  if (!channel?.isSendable()) {
    return { status: 'channel_unreachable', sent: 0 };
  }
  const sent = await sendGeekNewsToChannel(channel, count);
  return { status: 'sent', sent };
}
