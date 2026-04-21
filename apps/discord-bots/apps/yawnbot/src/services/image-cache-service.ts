/**
 * ImageCacheService — 자동 이미지 생성 캐시
 *
 * 구조:
 *   {characterDir}/image-cache/
 *     index.json     — 씬 메타데이터 목록
 *     {id}.png       — 실제 이미지 파일
 *
 * 유사도: 태그 Jaccard similarity (|A∩B| / |A∪B|)
 * 캐시 최대: MAX_CACHE_ENTRIES개, 초과 시 hitCount 낮은 것 삭제
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { execSync } from 'child_process';

const MAX_CACHE_ENTRIES = 50;
const DEFAULT_THRESHOLD = 0.4;

export interface CacheEntry {
  id: string;
  tags: string[];
  prompt: string;
  filePath: string;
  mimeType: string;
  createdAt: string;
  hitCount: number;
}

interface ImageCacheIndex {
  scenes: CacheEntry[];
}

function jaccardSimilarity(a: string[], b: string[]): number {
  if (a.length === 0 && b.length === 0) return 1;
  const setA = new Set(a.map((t) => t.toLowerCase().trim()));
  const setB = new Set(b.map((t) => t.toLowerCase().trim()));
  let intersection = 0;
  for (const tag of setA) {
    if (setB.has(tag)) intersection++;
  }
  const union = setA.size + setB.size - intersection;
  return union === 0 ? 1 : intersection / union;
}

export class ImageCacheService {
  private cacheDir: string;
  private indexPath: string;
  private memoRepoPath: string;
  private slug: string;
  private index: ImageCacheIndex | null = null;

  constructor(characterDir: string, memoRepoPath: string, slug: string) {
    this.cacheDir = path.join(characterDir, 'image-cache');
    this.indexPath = path.join(this.cacheDir, 'index.json');
    this.memoRepoPath = memoRepoPath;
    this.slug = slug;
  }

  private ensureDir(): void {
    if (!fs.existsSync(this.cacheDir)) {
      fs.mkdirSync(this.cacheDir, { recursive: true });
    }
  }

  private readIndex(): ImageCacheIndex {
    if (this.index) return this.index;
    try {
      if (fs.existsSync(this.indexPath)) {
        const raw = fs.readFileSync(this.indexPath, 'utf-8');
        const parsed = JSON.parse(raw) as ImageCacheIndex;
        this.index = {
          scenes: Array.isArray(parsed.scenes) ? parsed.scenes : [],
        };
        return this.index;
      }
    } catch {
      /* ignore parse errors */
    }
    this.index = { scenes: [] };
    return this.index;
  }

  private writeIndex(): void {
    this.ensureDir();
    fs.writeFileSync(
      this.indexPath,
      JSON.stringify(this.index, null, 2) + '\n',
      'utf-8',
    );
  }

  /** tags와 유사도 ≥ threshold인 캐시 엔트리 반환 (가장 높은 유사도 우선). 없으면 null. */
  findSimilar(tags: string[], threshold = DEFAULT_THRESHOLD): CacheEntry | null {
    const index = this.readIndex();
    let best: CacheEntry | null = null;
    let bestScore = 0;
    for (const entry of index.scenes) {
      if (!fs.existsSync(entry.filePath)) continue;
      const score = jaccardSimilarity(tags, entry.tags);
      if (score >= threshold && score > bestScore) {
        bestScore = score;
        best = entry;
      }
    }
    if (best) {
      console.log(
        `[ImageCache] 히트: id=${best.id}, 유사도=${bestScore.toFixed(2)}, 태그=${best.tags.join(',')}`,
      );
    }
    return best;
  }

  /** 캐시 히트 횟수 +1 후 저장 */
  incrementHit(entry: CacheEntry): void {
    entry.hitCount++;
    this.writeIndex();
  }

  /** 이미지 버퍼를 파일로 저장하고 인덱스에 추가 */
  add(
    tags: string[],
    prompt: string,
    buffer: Buffer,
    mimeType: string,
  ): CacheEntry {
    this.ensureDir();
    const id = crypto.randomBytes(8).toString('hex');
    const ext = (mimeType.split('/')[1] || 'png').replace(/[^a-z0-9]/gi, '');
    const filePath = path.join(this.cacheDir, `${id}.${ext}`);
    fs.writeFileSync(filePath, buffer);

    const entry: CacheEntry = {
      id,
      tags,
      prompt,
      filePath,
      mimeType,
      createdAt: new Date().toISOString(),
      hitCount: 0,
    };

    const index = this.readIndex();
    index.scenes.push(entry);
    this.prune(index);
    this.writeIndex();
    console.log(`[ImageCache] 저장: id=${id}, 태그=${tags.join(',')}`);
    this.commitToGit(entry);
    return entry;
  }

  private commitToGit(entry: CacheEntry): void {
    try {
      const relCacheDir = path.relative(this.memoRepoPath, this.cacheDir).replace(/\\/g, '/');
      execSync(
        `git -C "${this.memoRepoPath}" add "${relCacheDir}/index.json" "${relCacheDir}/${entry.id}.${entry.mimeType.split('/')[1] || 'png'}"`,
        { stdio: 'pipe' },
      );
      execSync(
        `git -C "${this.memoRepoPath}" commit -m "feat(${this.slug}): 씬 이미지 캐시 추가 [${entry.tags.join(', ')}]"`,
        { stdio: 'pipe' },
      );
      console.log(`[ImageCache:${this.slug}] git commit 완료 (${entry.id})`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      if (!msg.includes('nothing to commit')) {
        console.warn(`[ImageCache:${this.slug}] git commit 실패:`, msg.slice(0, 200));
      }
    }
  }

  private prune(index: ImageCacheIndex): void {
    if (index.scenes.length <= MAX_CACHE_ENTRIES) return;
    index.scenes.sort((a, b) => a.hitCount - b.hitCount);
    const toRemove = index.scenes.splice(0, index.scenes.length - MAX_CACHE_ENTRIES);
    for (const entry of toRemove) {
      try {
        if (fs.existsSync(entry.filePath)) fs.unlinkSync(entry.filePath);
      } catch {
        /* ignore */
      }
    }
    console.log(`[ImageCache] 정리: ${toRemove.length}개 삭제`);
  }
}
