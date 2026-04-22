/**
 * cost.ts — 이미지 생성 비용 대시보드
 *
 * image-log.jsonl (캐릭터 캐시) + image-log/ (수동 /이미지 커맨드) 집계.
 *
 * 집계 항목:
 *   - 총 비용, 생성 횟수, 캐시 히트 횟수
 *   - 모델별 비용/횟수
 *   - 최근 7일 일별 비용
 */
import fs from 'fs';
import path from 'path';
import { EmbedBuilder, MessageFlags } from 'discord.js';
import type { ChatInputCommandInteraction } from 'discord.js';
import type { BotContext } from './bot-context';

interface CacheLogEntry {
  timestamp: string;
  type: 'generated' | 'cache_hit';
  model?: string;
  costUsd?: number;
}

interface ImageLogEntry {
  savedAt: string;
  model?: string;
  prompt?: string;
  character?: string;
}

interface AggStats {
  totalCost: number;
  generated: number;
  cacheHit: number;
  byModel: Record<string, { cost: number; count: number }>;
  byDay: Record<string, number>; // YYYY-MM-DD → cost
}

function toDateStr(isoStr: string): string {
  return isoStr.slice(0, 10);
}

// memo/characters/<slug>/image-cache/image-log.jsonl 전체 집계
function aggregateCacheLog(memoRepoPath: string, stats: AggStats): void {
  const charsDir = path.join(memoRepoPath, 'characters');
  if (!fs.existsSync(charsDir)) return;

  for (const slug of fs.readdirSync(charsDir)) {
    const logPath = path.join(charsDir, slug, 'image-cache', 'image-log.jsonl');
    if (!fs.existsSync(logPath)) continue;

    for (const line of fs.readFileSync(logPath, 'utf-8').split('\n')) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const entry = JSON.parse(trimmed) as CacheLogEntry;
        const day = toDateStr(entry.timestamp);

        if (entry.type === 'cache_hit') {
          stats.cacheHit++;
          continue;
        }

        // generated
        stats.generated++;
        const cost = entry.costUsd ?? 0;
        stats.totalCost += cost;
        stats.byDay[day] = (stats.byDay[day] ?? 0) + cost;

        if (entry.model) {
          const m = stats.byModel[entry.model] ?? { cost: 0, count: 0 };
          m.cost += cost;
          m.count++;
          stats.byModel[entry.model] = m;
        }
      } catch {
        /* 파싱 실패 라인 스킵 */
      }
    }
  }
}

/** memo/image-log/**\/log.jsonl (/이미지 슬래시 커맨드 로그) 집계 */
function aggregateImageLog(memoRepoPath: string, stats: AggStats): void {
  const imageLogDir = path.join(memoRepoPath, 'image-log');
  if (!fs.existsSync(imageLogDir)) return;

  const walkDir = (dir: string) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walkDir(full);
      } else if (entry.name === 'log.jsonl') {
        for (const line of fs.readFileSync(full, 'utf-8').split('\n')) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            const e = JSON.parse(trimmed) as ImageLogEntry;
            const day = toDateStr(e.savedAt);
            stats.generated++;
            // /이미지 커맨드 로그에는 costUsd 없음 — 모델로 추정
            const cost = estimateCostFromModel(e.model ?? '');
            stats.totalCost += cost;
            stats.byDay[day] = (stats.byDay[day] ?? 0) + cost;
            if (e.model) {
              const m = stats.byModel[e.model] ?? { cost: 0, count: 0 };
              m.cost += cost;
              m.count++;
              stats.byModel[e.model] = m;
            }
          } catch {
            /* skip */
          }
        }
      }
    }
  };
  walkDir(imageLogDir);
}

const IMAGEN_PRICE: Record<string, number> = {
  'imagen-4.0-fast-generate-001': 0.02,
  'imagen-4.0-generate-001': 0.04,
  'imagen-4.0-ultra-generate-001': 0.06,
  'imagen-3.0-fast-generate-001': 0.02,
  'imagen-3.0-generate-001': 0.04,
  'imagen-3.0-generate-002': 0.04,
};

function estimateCostFromModel(model: string): number {
  return IMAGEN_PRICE[model] ?? 0;
}

function fmt(n: number): string {
  return `$${n.toFixed(4)}`;
}

function recentDays(n: number): string[] {
  const days: string[] = [];
  for (let i = n - 1; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    days.push(d.toISOString().slice(0, 10));
  }
  return days;
}

export async function handleCost(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  if (!ctx.memoRepoPath) {
    await interaction.reply({ content: 'MEMO_REPO_PATH가 설정되지 않아 사용량을 집계할 수 없어요.', flags: MessageFlags.Ephemeral });
    return;
  }

  await interaction.deferReply({ flags: MessageFlags.Ephemeral });

  const stats: AggStats = { totalCost: 0, generated: 0, cacheHit: 0, byModel: {}, byDay: {} };
  aggregateCacheLog(ctx.memoRepoPath, stats);
  aggregateImageLog(ctx.memoRepoPath, stats);

  const totalImages = stats.generated + stats.cacheHit;
  const hitRate = totalImages > 0 ? Math.round((stats.cacheHit / totalImages) * 100) : 0;

  // 모델별 정렬 (비용 내림차순)
  const modelLines = Object.entries(stats.byModel)
    .sort((a, b) => b[1].cost - a[1].cost)
    .map(([model, { cost, count }]) => `\`${model}\` — ${count}장 / ${fmt(cost)}`)
    .join('\n') || '없음';

  // 최근 7일 일별
  const days7 = recentDays(7);
  const dailyLines = days7
    .map((day) => {
      const cost = stats.byDay[day] ?? 0;
      const bar = cost > 0 ? '█'.repeat(Math.max(1, Math.round(cost / 0.02))) : '·';
      return `\`${day.slice(5)}\` ${bar} ${fmt(cost)}`;
    })
    .join('\n');

  const embed = new EmbedBuilder()
    .setTitle('📊 이미지 비용 대시보드')
    .addFields(
      { name: '총 비용', value: fmt(stats.totalCost), inline: true },
      { name: '생성', value: `${stats.generated}장`, inline: true },
      { name: '캐시 히트', value: `${stats.cacheHit}회 (${hitRate}%)`, inline: true },
      { name: '모델별', value: modelLines },
      { name: '최근 7일', value: dailyLines },
    )
    .setColor(0x7c4dff)
    .setFooter({ text: '* /이미지 커맨드 로그는 모델 단가로 추정. 실제 청구액과 다를 수 있음.' });

  await interaction.editReply({ embeds: [embed] });
}
