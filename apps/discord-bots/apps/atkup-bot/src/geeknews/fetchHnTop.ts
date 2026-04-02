/**
 * Hacker News 공개 Firebase API — 상위 스토리 목록
 * @see https://github.com/HackerNews/API
 */

export type HnStoryLine = {
  title: string;
  href: string;
  score: number;
  by: string;
  host: string;
};

type HnItem = {
  id: number;
  title?: string;
  url?: string;
  score?: number;
  by?: string;
};

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
      fetch(`https://hacker-news.firebaseio.com/v0/item/${id}.json`).then((r) => r.json() as Promise<HnItem | null>)
    )
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
