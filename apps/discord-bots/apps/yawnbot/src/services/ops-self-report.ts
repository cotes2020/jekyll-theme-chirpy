/**
 * yawnbot 운영 자기보고 (ops-self-report) — TASK-YB-002-D
 *
 * 봇 라이프사이클 (시작 / 종료 / 크래시) + deploy 결과를 Discord 운영 채널에
 * 임베드로 보고. 노트북 prod 머신 GUI 부재 → 데스크탑에서 봇 상태 가시성.
 *
 * 환경:
 *   - DISCORD_TOKEN                  (필수 — 미설정 시 봇 자체가 못 떠서 무관)
 *   - YAWNBOT_OPS_REPORT_CHANNEL_ID  (선택 — 미설정 시 모든 report no-op)
 *   - YAWNBOT_ENV                    (선택 — 'dev' | 'prod' | 미설정 시 'unknown')
 *   - GIT_COMMIT                     (선택 — build 시점 주입. 없으면 'unknown')
 *
 * 모든 send 는 try/catch silent — 자기보고가 봇 자체를 죽이지 않도록.
 */
import os from 'node:os';
import crypto from 'node:crypto';
import { REST, Routes, EmbedBuilder } from 'discord.js';

export interface OpsReportContext {
  token: string;
  channelId: string;
  env: string;
  hostname: string;
  gitCommit: string;
}

/** 환경변수에서 ctx 로드. channelId 없으면 null (= self-report 비활성). */
export function loadOpsReportContext(): OpsReportContext | null {
  const token = process.env.DISCORD_TOKEN?.trim();
  const channelId = process.env.YAWNBOT_OPS_REPORT_CHANNEL_ID?.trim();
  if (!token || !channelId) return null;
  return {
    token,
    channelId,
    env: process.env.YAWNBOT_ENV?.trim() || 'unknown',
    hostname: os.hostname(),
    gitCommit: process.env.GIT_COMMIT?.trim() || 'unknown',
  };
}

/** 같은 stack 의 에러를 5분 내 1회만 send. 메모리 leak 방지: 최대 100 entries. */
const ERROR_DEDUP_TTL_MS = 5 * 60 * 1000;
const ERROR_DEDUP_MAX = 100;
const errorDedupCache = new Map<string, number>();

function shouldReportError(stackHash: string): boolean {
  const now = Date.now();
  const last = errorDedupCache.get(stackHash);
  if (last && now - last < ERROR_DEDUP_TTL_MS) return false;
  errorDedupCache.set(stackHash, now);
  if (errorDedupCache.size > ERROR_DEDUP_MAX) {
    const oldestKey = errorDedupCache.keys().next().value;
    if (oldestKey !== undefined) errorDedupCache.delete(oldestKey);
  }
  return true;
}

function hashStack(stack: string): string {
  return crypto.createHash('sha1').update(stack).digest('hex').slice(0, 12);
}

function buildFooter(ctx: OpsReportContext): string {
  return `${ctx.env} · ${ctx.hostname} · ${ctx.gitCommit}`;
}

async function sendEmbed(ctx: OpsReportContext, embed: EmbedBuilder, timeoutMs = 3000): Promise<void> {
  try {
    const rest = new REST({ timeout: timeoutMs }).setToken(ctx.token);
    await rest.post(Routes.channelMessages(ctx.channelId), {
      body: { embeds: [embed.toJSON()] },
    });
  } catch (e: unknown) {
    console.warn('[OpsReport] send 실패 (silent):', e instanceof Error ? e.message : e);
  }
}

/** 봇 startup 임베드 (녹색). client ready 직후 호출. */
export async function reportStartup(
  ctx: OpsReportContext,
  info: { botTag: string; guilds: number; users: number },
): Promise<void> {
  const embed = new EmbedBuilder()
    .setColor(0x2ecc71)
    .setTitle('🟢 YawnBot 시작')
    .setDescription(
      [
        `**봇**: \`${info.botTag}\``,
        `**서버**: ${info.guilds}개 · **유저**: ${info.users}명`,
      ].join('\n'),
    )
    .setFooter({ text: buildFooter(ctx) })
    .setTimestamp();
  await sendEmbed(ctx, embed);
}

/** 봇 graceful shutdown 임베드 (회색). SIGINT/SIGTERM 직후 호출. */
export async function reportShutdown(ctx: OpsReportContext, reason: string): Promise<void> {
  const embed = new EmbedBuilder()
    .setColor(0x95a5a6)
    .setTitle('⚪ YawnBot 종료')
    .setDescription(`**이유**: \`${reason}\``)
    .setFooter({ text: buildFooter(ctx) })
    .setTimestamp();
  await sendEmbed(ctx, embed);
}

/** 미처리 예외 임베드 (빨강). dedup 적용 — 동일 stack 5분 내 1회만. */
export async function reportError(
  ctx: OpsReportContext,
  err: unknown,
  kind: 'uncaughtException' | 'unhandledRejection',
): Promise<void> {
  const e = err instanceof Error ? err : new Error(String(err));
  const stack = e.stack || e.message;
  const stackHash = hashStack(stack);
  if (!shouldReportError(stackHash)) return;

  const stackPreview = stack.split('\n').slice(0, 6).join('\n').slice(0, 1500);
  const embed = new EmbedBuilder()
    .setColor(0xe74c3c)
    .setTitle(`🔴 YawnBot 에러 (${kind})`)
    .setDescription(`**메시지**: ${e.message.slice(0, 300)}\n\`\`\`\n${stackPreview}\n\`\`\``)
    .setFooter({ text: `${buildFooter(ctx)} · stack:${stackHash}` })
    .setTimestamp();
  await sendEmbed(ctx, embed);
}

/** deploy 결과 임베드 (파랑). deploy-commands.ts main() 끝에 호출. */
export async function reportDeploy(
  ctx: OpsReportContext,
  info: { count: number; target: string },
): Promise<void> {
  const embed = new EmbedBuilder()
    .setColor(0x3498db)
    .setTitle('🔵 YawnBot 슬래시 배포')
    .setDescription(`**${info.count}개** 커맨드 등록 → \`${info.target}\``)
    .setFooter({ text: buildFooter(ctx) })
    .setTimestamp();
  await sendEmbed(ctx, embed);
}
