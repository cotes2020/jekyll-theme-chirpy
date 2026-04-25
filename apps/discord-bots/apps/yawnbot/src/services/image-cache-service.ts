/**
 * ImageCacheService — 자동 이미지 생성 캐시
 *
 * 구조:
 *   {characterDir}/image-cache/
 *     index.json        — 씬 메타데이터 목록 (embedding 포함)
 *     image-log.jsonl   — 생성/히트 로그 (JSON Lines)
 *     {id}.png          — 실제 이미지 파일
 *
 * 유사도: Gemini 임베딩 코사인 유사도 (karmolab-ai generateEmbedding)
 *   KARMOLAB_AI_SURFACE=vertex → Vertex text-embedding-004
 *   기본 → AI Studio gemini-embedding-001
 * 캐시 최대: MAX_CACHE_ENTRIES개, 초과 시 hitCount 낮은 것 삭제
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { execSync } from 'child_process';
import { generateEmbedding } from 'karmolab-ai/node';

const MAX_CACHE_ENTRIES = 50;
const DEFAULT_COSINE_THRESHOLD = 0.75;

export interface CacheEntry {
  id: string;
  /** 영어 씬 묘사 (Imagen 프롬프트의 Scene 부분, 임베딩·표시 공통 소스) */
  sceneDesc: string;
  prompt: string;
  filePath: string;
  mimeType: string;
  createdAt: string;
  hitCount: number;
  /** Gemini text-embedding 벡터 (정규화됨). sceneDesc로 생성. */
  embedding?: number[];
}

interface ImageCacheIndex {
  scenes: CacheEntry[];
}

interface ImageLogEntry {
  timestamp: string;
  type: 'generated' | 'cache_hit';
  id: string;
  sceneDesc: string;
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

    let queryEmbedding: number[];
    try {
      queryEmbedding = await generateEmbedding(process.env, sceneDesc);
    } catch (e) {
      console.warn('[ImageCache] embedding 실패:', e instanceof Error ? e.message : String(e));
      return null;
    }

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
        `[ImageCache] 히트: id=${best.id}, 유사도=${bestScore.toFixed(3)}, scene="${best.sceneDesc.slice(0, 60)}"`,
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
      sceneDesc: entry.sceneDesc,
    });
  }

  /**
   * 이미지 버퍼를 파일로 저장하고 인덱스에 추가 + 로그 기록 + git commit.
   * `sceneDesc`로 임베딩을 계산해 저장 (실패 시 embedding 없이 저장).
   */
  async add(
    sceneDesc: string,
    prompt: string,
    buffer: Buffer,
    mimeType: string,
    modelId?: string,
  ): Promise<CacheEntry> {
    this.ensureDir();
    const id = crypto.randomBytes(8).toString('hex');
    const ext = (mimeType.split('/')[1] || 'png').replace(/[^a-z0-9]/gi, '');
    const filePath = path.join(this.cacheDir, `${id}.${ext}`);
    fs.writeFileSync(filePath, buffer);

    let embedding: number[] | undefined;
    try {
      embedding = await generateEmbedding(process.env, sceneDesc);
    } catch (e) {
      console.warn('[ImageCache] 저장 시 embedding 실패 (건너뜀):', e instanceof Error ? e.message : String(e));
    }

    const entry: CacheEntry = {
      id,
      sceneDesc,
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
      sceneDesc,
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
    console.log(`[ImageCache] 저장: id=${id}, scene="${sceneDesc.slice(0, 60)}"${embeddingLabel}`);
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
        `git -C "${this.memoRepoPath}" commit -m "feat(${this.slug}): 씬 이미지 캐시 추가 [${entry.sceneDesc.slice(0, 80)}]"`,
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
