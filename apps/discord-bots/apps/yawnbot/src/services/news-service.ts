/**
 * NewsService — 관심사 키워드 기반 Google News RSS 뉴스 조회
 *
 * - 키워드 저장: {characterDir}/news-keywords.json
 * - Google News RSS (https://news.google.com/rss/search?q=...) 무료, 키 불필요
 * - 최근 N시간 내 기사만 필터링, 기사 dedup 위해 본 타이틀 메모리 캐싱
 */
import fs from 'fs';
import path from 'path';
import https from 'https';

export interface NewsKeyword {
  id: string;
  keyword: string;
  addedAt: string;
}

export interface NewsArticle {
  title: string;
  link: string;
  pubDate: string;
  keyword: string;
}

function fetchRaw(url: string): Promise<string> {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (chunk: Buffer) => chunks.push(chunk));
      res.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
      res.on('error', reject);
    }).on('error', reject);
  });
}

function parseRssItems(xml: string): { title: string; link: string; pubDate: string }[] {
  const items: { title: string; link: string; pubDate: string }[] = [];
  const itemMatches = xml.match(/<item>[\s\S]*?<\/item>/g) || [];
  for (const item of itemMatches) {
    const title =
      item.match(/<title><!\[CDATA\[([\s\S]*?)\]\]><\/title>/)?.[1]?.trim() ||
      item.match(/<title>([\s\S]*?)<\/title>/)?.[1]?.trim() || '';
    const link =
      item.match(/<link>([\s\S]*?)<\/link>/)?.[1]?.trim() ||
      item.match(/<guid[^>]*>([\s\S]*?)<\/guid>/)?.[1]?.trim() || '';
    const pubDate = item.match(/<pubDate>([\s\S]*?)<\/pubDate>/)?.[1]?.trim() || '';
    if (title) items.push({ title, link, pubDate });
  }
  return items;
}

export class NewsService {
  private filePath: string;
  private keywords: NewsKeyword[];
  private seenTitles = new Set<string>();

  constructor(characterDir: string) {
    this.filePath = path.join(characterDir, 'news-keywords.json');
    this.keywords = this._load();
  }

  private _load(): NewsKeyword[] {
    try {
      if (fs.existsSync(this.filePath)) {
        return JSON.parse(fs.readFileSync(this.filePath, 'utf-8')) as NewsKeyword[];
      }
    } catch { /* ignore */ }
    return [];
  }

  private _save(): void {
    try {
      fs.writeFileSync(this.filePath, JSON.stringify(this.keywords, null, 2), 'utf-8');
    } catch (e) {
      console.warn('[NewsService] 저장 실패:', e instanceof Error ? e.message : String(e));
    }
  }

  getKeywords(): NewsKeyword[] { return this.keywords; }

  addKeyword(keyword: string): NewsKeyword | null {
    const kw = keyword.trim();
    if (!kw || this.keywords.some((k) => k.keyword === kw)) return null;
    const entry: NewsKeyword = { id: Date.now().toString(), keyword: kw, addedAt: new Date().toISOString() };
    this.keywords.push(entry);
    this._save();
    return entry;
  }

  removeKeyword(id: string): boolean {
    const before = this.keywords.length;
    this.keywords = this.keywords.filter((k) => k.id !== id);
    if (this.keywords.length !== before) { this._save(); return true; }
    return false;
  }

  /**
   * 등록된 키워드 중 하나를 골라 최근 기사 조회.
   * maxAgeHours 이내 기사만 반환. 이미 본 기사(seenTitles)는 제외.
   */
  async fetchFreshArticle(maxAgeHours = 6): Promise<NewsArticle | null> {
    if (this.keywords.length === 0) return null;

    // 키워드 순서를 섞어서 골고루 사용
    const shuffled = [...this.keywords].sort(() => Math.random() - 0.5);
    const cutoff = Date.now() - maxAgeHours * 3600_000;

    for (const { keyword } of shuffled) {
      try {
        const encoded = encodeURIComponent(keyword);
        const url = `https://news.google.com/rss/search?q=${encoded}&hl=ko&gl=KR&ceid=KR:ko`;
        const xml = await fetchRaw(url);
        const items = parseRssItems(xml);

        for (const item of items) {
          if (this.seenTitles.has(item.title)) continue;
          const pub = item.pubDate ? new Date(item.pubDate).getTime() : 0;
          if (pub && pub < cutoff) continue;

          this.seenTitles.add(item.title);
          return { ...item, keyword };
        }
      } catch (e) {
        console.warn(`[NewsService] RSS 조회 실패 (${keyword}):`, e instanceof Error ? e.message : String(e));
      }
    }
    return null;
  }
}
