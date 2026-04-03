import { EmbedBuilder, MessageFlags } from 'discord.js';
import type { ChatInputCommandInteraction } from 'discord.js';
import { truncateEmbedField } from './embeds';

type DeferKind = 'cursor' | 'gemini';

export function buildDeferProgressEmbed(
  kind: DeferKind,
  s: {
    elapsedSec: number;
    spinner: string;
    progressMin?: number;
    requestText?: string;
    modeLabel?: string;
    liveAssistantText?: string;
  },
): EmbedBuilder {
  const mm = Math.floor(s.elapsedSec / 60);
  const ss = String(s.elapsedSec % 60).padStart(2, '0');
  const reqSnippet = s.requestText ? truncateEmbedField(s.requestText, 350) : '';
  const liveSnippet = s.liveAssistantText ? truncateEmbedField(s.liveAssistantText, 450) : '';

  if (kind === 'cursor') {
    const mode = s.modeLabel ? String(s.modeLabel) : 'agent';
    const parts: string[] = [];
    if (reqSnippet) parts.push(`**요청** ${reqSnippet}`);
    if (liveSnippet) parts.push(`**스트림** ${liveSnippet}`);
    const desc = parts.length ? parts.join('\n') : '처리 중…';
    return new EmbedBuilder()
      .setTitle(`${s.spinner} Cursor (${mode})`)
      .setDescription(desc)
      .setColor(0x5865f2)
      .addFields(
        { name: '경과', value: `${mm}:${ss}`, inline: true },
        { name: '한도', value: `~${typeof s.progressMin === 'number' ? s.progressMin : 10}분`, inline: true },
      )
      .setFooter({ text: '완료 시 이 메시지가 결과로 바뀝니다' });
  }

  const desc = reqSnippet ? `**질문** ${reqSnippet}` : 'Gemini 호출 중…';
  return new EmbedBuilder()
    .setTitle(`${s.spinner} Gemini`)
    .setDescription(desc)
    .setColor(0x4285f4)
    .addFields({ name: '경과', value: `${mm}:${ss}`, inline: true })
    .setFooter({ text: '완료 시 이 메시지가 결과로 바뀝니다' });
}

export async function startDeferElapsedTicker(
  interaction: Pick<ChatInputCommandInteraction, 'editReply'>,
  kind: DeferKind,
  extra: {
    progressMin?: number;
    requestText?: string;
    modeLabel?: string;
    liveAssistantText?: string | (() => string);
  } = {},
): Promise<() => Promise<void>> {
  const tickMs = Math.max(1500, parseInt(process.env.DEFER_TICK_MS || '2500', 10));
  const t0 = Date.now();
  const SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  const progressMin = typeof extra.progressMin === 'number' ? extra.progressMin : 10;
  let tick = 0;
  let cancelled = false;
  let inFlight = 0;

  const run = async () => {
    if (cancelled) return;
    const elapsedSec = Math.floor((Date.now() - t0) / 1000);
    const spinner = SPINNER[tick % SPINNER.length];
    tick += 1;
    const liveAssistantText =
      typeof extra.liveAssistantText === 'function' ? extra.liveAssistantText() : extra.liveAssistantText;
    const embed = buildDeferProgressEmbed(kind, {
      elapsedSec,
      spinner,
      progressMin,
      requestText: extra.requestText || '',
      modeLabel: extra.modeLabel || '',
      liveAssistantText: liveAssistantText || '',
    });
    if (cancelled) return;
    inFlight++;
    try {
      if (cancelled) return;
      await interaction.editReply({ content: null as any, embeds: [embed] } as any);
    } catch (e: any) {
      const code = e && typeof e === 'object' && 'code' in e ? (e as any).code : undefined;
      if (code !== 50006) console.error('[defer ticker] editReply 실패:', e?.message ?? e);
    } finally {
      inFlight--;
    }
  };

  const id = setInterval(run, tickMs);
  await run();
  return async () => {
    if (!cancelled) {
      cancelled = true;
      clearInterval(id);
    }
    while (inFlight > 0) {
      await new Promise<void>((r) => setTimeout(r, 50));
    }
  };
}

export async function notifyDeferCompletion(
  interaction: Pick<ChatInputCommandInteraction, 'followUp' | 'user'>,
  { ok, kind }: { ok: boolean; kind: DeferKind },
): Promise<void> {
  const mode = (process.env.DEFER_COMPLETION_NOTIFY || 'ephemeral').toLowerCase().trim();
  if (mode === 'off' || mode === 'false' || mode === '0') return;
  const uid = interaction.user.id;
  const label = kind === 'cursor' ? 'Cursor' : kind === 'gemini' ? 'Gemini' : '작업';
  const line = ok
    ? `✅ **${label}** 작업이 완료되었습니다. 위 응답 메시지를 확인하세요.`
    : `❌ **${label}** 처리가 끝났습니다(실패 또는 오류). 위 메시지를 확인하세요.`;
  try {
    if (mode === 'mention') {
      await interaction.followUp({
        content: `<@${uid}> ${line}`,
        allowedMentions: { users: [uid] },
      } as any);
    } else {
      await interaction.followUp({ content: line, flags: MessageFlags.Ephemeral } as any);
    }
  } catch (e: any) {
    console.error('[notifyDeferCompletion] followUp 실패:', e?.message ?? e);
  }
}

