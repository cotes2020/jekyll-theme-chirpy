/**
 * ImageCacheService — 자동 이미지 생성 캐시
 *
 * 구조:
 *   {characterDir}/image-cache/
 *     index.json        — 씬 메타데이터 목록 (embedding 포함)
 *     image-log.jsonl   — 생성/히트 로그 (JSON Lines)
 *     {id}.png          — 실제 이미지 파일
 *
 * 유사도: Gemini text-embedding-004 코사인 유사도
 * 캐시 최대: MAX_CACHE_ENTRIES개, 초과 시 hitCount 낮은 것 삭제
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { execSync } from 'child_process';

const MAX_CACHE_ENTRIES = 50;
const DEFAULT_COSINE_THRESHOLD = 0.75;
const EMBEDDING_MODEL = 'text-embedding-004';
const EMBED_API_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

export interface CacheEntry {
  id: string;
  tags: string[];
  prompt: string;
  filePath: string;
  mimeType: string;
  createdAt: string;
  hitCount: number;
  /** Gemini text-embedding-004 벡터 (정규화됨). 없으면 Jaccard 폴백. */
  embedding?: number[];
}

interface ImageCacheIndex {
  scenes: CacheEntry[];
}

interface ImageLogEntry {
  timestamp: string;
  type: 'generated' | 'cache_hit';
  id: string;
  tags: string[];
  model?: string;
  costUsd?: number;
}

// ─── 유사도 헬퍼 ──────────────────────────────────────────────────────────

/** 정규화된 벡터의 dot product = cosine similarity */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// ─── Gemini Embedding REST 호출 ──────────────────────────────────────────

async function embedText(text: string, apiKey: string): Promise<number[] | null> {
  try {
    const url = `${EMBED_API_BASE}/${EMBEDDING_MODEL}:embedContent?key=${encodeURIComponent(apiKey)}`;
    const body = JSON.stringify({
      model: `models/${EMBEDDING_MODEL}`,
      content: { parts: [{ text }] },
    });
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
    });
    if (!res.ok) {
      console.warn(`[ImageCache] embedding API ${res.status}: ${await res.text().catch(() => '')}`);
      return null;
    }
    const json = await res.json() as { embedding?: { values?: number[] } };
    const values = json?.embedding?.values;
    if (!Array.isArray(values) || values.length === 0) return null;
    return values;
  } catch (e) {
    console.warn('[ImageCache] embedding 실패:', e instanceof Error ? e.message : String(e));
    return null;
  }
}

// ─── 서비스 ──────────────────────────────────────────────────────────────

export class ImageCacheService {
  private cacheDir: string;
  private indexPath: string;
  private logPath: string;
  private memoRepoPath: string;
  private slug: string;
  private index: ImageCacheIndex | null = null;

  constructor(characterDir: string, memoRepoPath: string, slug: string) {
    this.cacheDir = path.join(characterDir, 'image-cache');
    this.indexPath = path.join(this.cacheDir, 'index.json');
    this.logPath = path.join(this.cacheDir, 'image-log.jsonl');
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

  private writeLog(entry: ImageLogEntry): void {
    this.ensureDir();
    try {
      fs.appendFileSync(this.logPath, JSON.stringify(entry) + '\n', 'utf-8');
    } catch {
      /* ignore log write errors */
    }
  }

  /**
   * 씬과 유사한 캐시 엔트리 검색 (Gemini 임베딩 코사인 유사도).
   *
   * `sceneDesc`를 임베딩해서 저장된 벡터와 비교한다.
   * 임베딩이 없는 엔트리는 건너뛴다.
   * 가장 높은 유사도 엔트리를 반환하며, 임계값 미만이면 null.
   */
  async findSimilar(
    sceneDesc: string,
    threshold = DEFAULT_COSINE_THRESHOLD,
  ): Promise<CacheEntry | null> {
    const index = this.readIndex();
    if (index.scenes.length === 0) return null;

    const apiKey = process.env.GEMINI_API_KEY?.trim();
    if (!apiKey) return null;

    const queryEmbedding = await embedText(sceneDesc, apiKey);
    if (!queryEmbedding) return null;

    let best: CacheEntry | null = null;
    let bestScore = 0;

    for (const entry of index.scenes) {
      if (!entry.embedding || !fs.existsSync(entry.filePath)) continue;
      const score = cosineSimilarity(queryEmbedding, entry.embedding);
      if (score >= threshold && score > bestScore) {
        bestScore = score;
        best = entry;
      }
    }

    if (best) {
      console.log(
        `[ImageCache] 히트: id=${best.id}, 유사도=${bestScore.toFixed(3)}, 태그=${best.tags.join(',')}`,
      );
    }
    return best;
  }

  /** 캐시 히트 횟수 +1 후 저장 + 로그 기록 */
  incrementHit(entry: CacheEntry): void {
    entry.hitCount++;
    this.writeIndex();
    this.writeLog({
      timestamp: new Date().toISOString(),
      type: 'cache_hit',
      id: entry.id,
      tags: entry.tags,
    });
  }

  /**
   * 이미지 버퍼를 파일로 저장하고 인덱스에 추가 + 로그 기록 + git commit.
   *
   * `sceneDesc`가 있으면 Gemini 임베딩을 계산해서 `CacheEntry.embedding`에 저장한다.
   */
  async add(
    tags: string[],
    prompt: string,
    buffer: Buffer,
    mimeType: string,
    modelId?: string,
    sceneDesc?: string,
  ): Promise<CacheEntry> {
    this.ensureDir();
    const id = crypto.randomBytes(8).toString('hex');
    const ext = (mimeType.split('/')[1] || 'png').replace(/[^a-z0-9]/gi, '');
    const filePath = path.join(this.cacheDir, `${id}.${ext}`);
    fs.writeFileSync(filePath, buffer);

    // 임베딩 계산 (실패해도 Jaccard 폴백 가능하므로 non-blocking)
    let embedding: number[] | undefined;
    const apiKey = process.env.GEMINI_API_KEY?.trim();
    if (apiKey && sceneDesc) {
      const vec = await embedText(sceneDesc, apiKey);
      if (vec) embedding = vec;
    }

    const entry: CacheEntry = {
      id,
      tags,
      prompt,
      filePath,
      mimeType,
      createdAt: new Date().toISOString(),
      hitCount: 0,
      ...(embedding ? { embedding } : {}),
    };

    const index = this.readIndex();
    index.scenes.push(entry);
    this.prune(index);
    this.writeIndex();

    const logEntry: ImageLogEntry = {
      timestamp: entry.createdAt,
      type: 'generated',
      id,
      tags,
    };
    if (modelId) {
      logEntry.model = modelId;
      // Imagen 단가 (USD, 2026-04 기준)
      const prices: Record<string, number> = {
        'imagen-4.0-fast-generate-001': 0.02,
        'imagen-4.0-generate-001': 0.04,
        'imagen-4.0-ultra-generate-001': 0.06,
        'imagen-3.0-fast-generate-001': 0.02,
        'imagen-3.0-generate-001': 0.04,
        'imagen-3.0-generate-002': 0.04,
      };
      if (prices[modelId] != null) logEntry.costUsd = prices[modelId];
    }
    this.writeLog(logEntry);

    const embeddingLabel = embedding ? `, embedding=${embedding.length}d` : '';
    console.log(`[ImageCache] 저장: id=${id}, 태그=${tags.join(',')}${embeddingLabel}`);
    this.commitToGit(entry);
    return entry;
  }

  /** 전체 씬 목록 반환 (image-history 조회용) */
  listScenes(): CacheEntry[] {
    return this.readIndex().scenes;
  }

  /** 로그 파일에서 최근 N개 항목 반환 */
  readLog(limit = 50): ImageLogEntry[] {
    try {
      if (!fs.existsSync(this.logPath)) return [];
      const lines = fs.readFileSync(this.logPath, 'utf-8')
        .split('\n')
        .filter(Boolean)
        .slice(-limit);
      return lines.map((l) => JSON.parse(l) as ImageLogEntry).reverse();
    } catch {
      return [];
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

  private commitToGit(entry: CacheEntry): void {
    try {
      const relCacheDir = path.relative(this.memoRepoPath, this.cacheDir).replace(/\\/g, '/');
      const ext = (entry.mimeType.split('/')[1] || 'png').replace(/[^a-z0-9]/gi, '');
      execSync(
        `git -C "${this.memoRepoPath}" add "${relCacheDir}/index.json" "${relCacheDir}/image-log.jsonl" "${relCacheDir}/${entry.id}.${ext}"`,
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
}
