import { EmbedBuilder, MessageFlags } from 'discord.js';
import type { ChatInputCommandInteraction } from 'discord.js';
import {
  formatCursorAcpRpcSummaryField,
  hasGitWorkingChanges,
  truncateDiscordDescription,
  truncateEmbedField,
  notifyDeferCompletion,
  startDeferElapsedTicker,
} from '@discord-bots/common';
import {
  generateBlobTextFromEnvWithOptions,
  parseGenerativeSurfaceFromEnv,
  type GenerativeSurfaceOverride,
} from 'karmolab-ai/node';
import { discordAnswerCursorQuestion, getCursorMaxPromptChars, runCursorLocalRunner } from '../cursor-local';
import { resolveCursorRepoDirForSlash } from '../../paths';
import type { BotContext } from './bot-context';

const DEFAULT_YAWN_SYSTEM = `시스템: 너는 'YawnBot'이라는 이름의 활기차고 재치 있는 디스코드 봇이야. 사용자의 질문에 친절하고 유머러스하게 대답해줘.`;

function yawnSystemPromptFromEnv(): string {
  const raw = process.env.YAWN_SYSTEM_PROMPT ?? process.env.BOT_YAWN_SYSTEM_PROMPT ?? '';
  const t = String(raw).trim();
  if (!t) return DEFAULT_YAWN_SYSTEM;
  return t.replace(/\\n/g, '\n');
}

function yawnEnvPrecheckError(env: NodeJS.ProcessEnv, surfaceOverride: string): string | null {
  const eff = surfaceOverride === 'inherit' ? parseGenerativeSurfaceFromEnv(env) : surfaceOverride;
  if (eff === 'vertex') {
    if (!env.VERTEX_API_KEY?.trim() || !env.VERTEX_PROJECT_ID?.trim()) {
      return 'Vertex에는 .env의 VERTEX_API_KEY, VERTEX_PROJECT_ID가 필요합니다.';
    }
  } else if (!env.GEMINI_API_KEY?.trim()) {
    return 'AI Studio에는 .env의 GEMINI_API_KEY가 필요합니다.';
  }
  return null;
}

function friendlyYawnErrorMessage(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  const lower = msg.toLowerCase();
  if (lower.includes('429') || lower.includes('resource exhausted') || lower.includes('quota')) {
    return '요청이 많아 잠시 후 다시 시도해 주세요. (한도/속도 제한)';
  }
  if (lower.includes('safety') || lower.includes('blocked') || lower.includes('block_reason')) {
    return '안전 정책 때문에 응답할 수 없습니다. 질문을 바꿔 보세요.';
  }
  if (
    lower.includes('api key') ||
    lower.includes('permission denied') ||
    lower.includes('401') ||
    lower.includes('403')
  ) {
    return 'API 인증에 문제가 있습니다. 서버 `.env`의 키·프로젝트 설정을 확인해 주세요.';
  }
  if (lower.includes('fetch') || lower.includes('network') || lower.includes('econnreset')) {
    return '네트워크 오류로 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.';
  }
  return 'AI 응답을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.';
}

async function buildYawnChannelContext(interaction: ChatInputCommandInteraction, maxMessages: number): Promise<string> {
  if (maxMessages <= 0) return '';
  try {
    const ch = interaction.channel;
    if (!ch || !ch.isTextBased() || ch.isDMBased()) return '';
    const limit = Math.min(maxMessages + 8, 50);
    const fetched = await ch.messages.fetch({ limit });
    const sorted = [...fetched.values()].sort((a, b) => a.createdTimestamp - b.createdTimestamp);
    const lines = sorted
      .filter((m) => !m.author.bot && typeof m.content === 'string' && m.content.trim().length > 0)
      .slice(-maxMessages)
      .map((m) => `${m.author.username}: ${m.content.trim().slice(0, 400)}`);
    if (!lines.length) return '';
    return `\n\n[최근 이 채널 대화(참고용)]\n${lines.join('\n')}`;
  } catch {
    return '';
  }
}

export async function handleCursorEdit(ctx: BotContext, interaction: ChatInputCommandInteraction, userId: string): Promise<void> {
  const { gameData, isAdmin, cursorState } = ctx;
  if (!isAdmin(userId)) {
    await interaction.reply({ content: gameData.getMessage('Admin_AccessDenied_Desc'), flags: MessageFlags.Ephemeral });
    return;
  }
  const repoDir = resolveCursorRepoDirForSlash();
  if (!repoDir) {
    const triedEnv = process.env.CURSOR_LOCAL_REPO_DIR != null && String(process.env.CURSOR_LOCAL_REPO_DIR).trim();
    await interaction.reply({
      content: triedEnv
        ? '`CURSOR_LOCAL_REPO_DIR`이 가리키는 폴더가 없습니다. .env 경로를 고치거나, 비우면 이 레포 루트(자동)를 씁니다.'
        : 'Cursor 작업 폴더를 찾을 수 없습니다. yawnbot이 이 레포의 `apps/discord-bots/apps/yawnbot` 아래에 있어야 하거나, `.env`에 `CURSOR_LOCAL_REPO_DIR`을 직접 지정하세요.',
      flags: MessageFlags.Ephemeral,
    });
    return;
  }
  const promptText = interaction.options.getString('prompt', true);
  const modeOpt = interaction.options.getString('mode');
  const maxChars = getCursorMaxPromptChars();
  if (promptText.length > maxChars) {
    await interaction.reply({ content: `프롬프트가 너무 깁니다. 최대 ${maxChars}자까지입니다.`, flags: MessageFlags.Ephemeral });
    return;
  }
  if (cursorState.inFlight) {
    await interaction.reply({ content: '이미 Cursor 로컬 작업이 실행 중입니다. 잠시 후 다시 시도하세요.', flags: MessageFlags.Ephemeral });
    return;
  }
  await interaction.deferReply();
  const progressMin = Math.ceil(parseInt(process.env.CURSOR_TIMEOUT_MS || '600000', 10) / 60000);
  cursorState.inFlight = true;
  let stopDeferTicker: () => Promise<void> = async () => {};
  try {
    let liveAssistant = '';
    stopDeferTicker = await startDeferElapsedTicker(interaction, 'cursor', {
      progressMin,
      requestText: promptText,
      modeLabel: modeOpt || 'agent',
      liveAssistantText: () => liveAssistant,
    });
    type GitState = { isRepo?: boolean; statusPorcelain?: string; diffStat?: string; diffPreview?: string };
    type CursorResult = { json: { ok: boolean; error?: unknown; stopReason?: unknown; assistantPreview?: string; git?: GitState; acpRpc?: unknown; acpRpcSummary?: unknown; cwd?: string; stderrTail?: string }; code: number; err: string };
    const { json, code, err } = await (runCursorLocalRunner(
      repoDir,
      promptText,
      modeOpt || 'agent',
      (chunk) => {
        liveAssistant += chunk;
        const maxLive = parseInt(process.env.CURSOR_LIVE_PREVIEW_CHARS || '700', 10);
        if (liveAssistant.length > maxLive) liveAssistant = liveAssistant.slice(-maxLive);
      },
      (q) => discordAnswerCursorQuestion(interaction, q),
    ) as Promise<CursorResult>);
    await stopDeferTicker();
    stopDeferTicker = async () => {};
    if (!json.ok) {
      const git: GitState = json.git ?? {};
      const embed = new EmbedBuilder()
        .setTitle('Cursor 로컬 실행 실패')
        .setDescription(
          truncateDiscordDescription(
            `**📝 요청**\n\`\`\`\n${truncateEmbedField(promptText, 900)}\n\`\`\`\n\n` +
              `**오류**\n${String(json.error || 'unknown').slice(0, 2800)}`,
          ),
        )
        .setColor(0xf44336)
        .addFields(
          { name: 'exit', value: String(code), inline: true },
          { name: 'git repo', value: git.isRepo ? 'yes' : 'no', inline: true },
          { name: 'mode', value: `\`${modeOpt || 'agent'}\``, inline: true },
        );
      if (hasGitWorkingChanges(git)) {
        if (String(git.statusPorcelain || '').trim()) {
          embed.addFields({
            name: '변경 파일 (status)',
            value: `\`\`\`\n${String(git.statusPorcelain).slice(0, 900)}\n\`\`\``,
          });
        }
        if (String(git.diffStat || '').trim()) {
          embed.addFields({ name: 'diff --stat', value: `\`\`\`\n${String(git.diffStat).slice(0, 900)}\n\`\`\`` });
        }
        if (String(git.diffPreview || '').trim()) {
          embed.addFields({ name: 'diffPreview', value: `\`\`\`\n${String(git.diffPreview).slice(0, 900)}\n\`\`\`` });
        }
      }
      if (err) embed.setFooter({ text: err.slice(0, 500) });
      await interaction.editReply({ content: null, embeds: [embed] });
      await notifyDeferCompletion(interaction, { ok: false, kind: 'cursor' });
      return;
    }
    const g: GitState = json.git ?? {};
    const descParts = [];
    if (json.stopReason != null) descParts.push(`**stopReason:** ${json.stopReason}`);
    if (json.assistantPreview) descParts.push(String(json.assistantPreview).slice(0, 2600));
    const embed = new EmbedBuilder()
      .setTitle('Cursor 로컬 실행 완료')
      .setDescription(
        truncateDiscordDescription(
          `**📝 요청**\n\`\`\`\n${truncateEmbedField(promptText, 900)}\n\`\`\`\n\n` +
            `**에이전트 응답**\n${descParts.join('\n\n') || '(응답 본문 없음)'}`,
        ),
      )
      .setColor(0x4caf50)
      .addFields(
        { name: '모드', value: `\`${modeOpt || 'agent'}\``, inline: true },
        { name: '작업 경로', value: `\`${String(json.cwd || repoDir).slice(0, 900)}\`` },
        {
          name: '질문·플랜 (ACP)',
          value: formatCursorAcpRpcSummaryField(json.acpRpcSummary),
          inline: false,
        },
      );
    if (hasGitWorkingChanges(g)) {
      if (String(g.statusPorcelain || '').trim()) {
        embed.addFields({
          name: '변경 파일 (status)',
          value: `\`\`\`\n${String(g.statusPorcelain).slice(0, 900)}\n\`\`\``,
        });
      }
      if (String(g.diffStat || '').trim()) {
        embed.addFields({ name: 'diff --stat', value: `\`\`\`\n${String(g.diffStat).slice(0, 900)}\n\`\`\`` });
      }
      if (String(g.diffPreview || '').trim()) {
        embed.addFields({ name: 'diffPreview', value: `\`\`\`\n${String(g.diffPreview).slice(0, 900)}\n\`\`\`` });
      }
    }
    if (json.stderrTail) embed.addFields({ name: 'agent stderr (tail)', value: `\`\`\`\n${String(json.stderrTail).slice(0, 800)}\n\`\`\`` });
    await interaction.editReply({ content: null, embeds: [embed] });
    await notifyDeferCompletion(interaction, { ok: true, kind: 'cursor' });
  } catch (e) {
    await stopDeferTicker();
    stopDeferTicker = async () => {};
    await interaction.editReply({
      content: null,
      embeds: [
        new EmbedBuilder()
          .setTitle('Cursor 로컬 실행 예외')
          .setDescription(
            truncateDiscordDescription(
              `**📝 요청**\n\`\`\`\n${truncateEmbedField(promptText, 900)}\n\`\`\`\n\n` + `**오류**\n${String(e instanceof Error ? e.message : e).slice(0, 1800)}`,
            ),
          )
          .setColor(0xf44336),
      ],
    });
    await notifyDeferCompletion(interaction, { ok: false, kind: 'cursor' });
  } finally {
    await stopDeferTicker();
    cursorState.inFlight = false;
  }
}

export async function handleYawn(ctx: BotContext, interaction: ChatInputCommandInteraction): Promise<void> {
  const rawPrompt = interaction.options.getString('질문');
  const prompt = (rawPrompt ?? '').trim();
  if (!prompt) {
    await interaction.reply({ content: '질문 내용을 입력해 주세요.', flags: MessageFlags.Ephemeral });
    return;
  }

  const apiRaw = interaction.options.getString('api');
  let surfaceOverride: GenerativeSurfaceOverride = 'inherit';
  if (apiRaw === 'ai_studio') surfaceOverride = 'aiStudio';
  else if (apiRaw === 'vertex') surfaceOverride = 'vertex';

  const rawModelOpt = interaction.options.getString('model');
  let modelOpt = null;
  if (rawModelOpt != null && String(rawModelOpt).trim()) {
    const t = String(rawModelOpt).trim();
    if (t.length > 64 || !/^[a-zA-Z0-9._-]+$/.test(t)) {
      await interaction.reply({
        content: '`model` 옵션은 `a-zA-Z0-9._-` 만 허용, 최대 64자입니다.',
        flags: MessageFlags.Ephemeral,
      });
      return;
    }
    modelOpt = t;
  }

  const envErr = yawnEnvPrecheckError(process.env, surfaceOverride);
  if (envErr) {
    await interaction.reply({ content: envErr, flags: MessageFlags.Ephemeral });
    return;
  }

  const maxQ = parseInt(process.env.YAWN_MAX_QUESTION_CHARS || '1500', 10);
  const maxQuestion = Math.min(Math.max(200, Number.isFinite(maxQ) ? maxQ : 1500), 8000);
  if (prompt.length > maxQuestion) {
    await interaction.reply({
      content: `질문이 너무 깁니다. 최대 **${maxQuestion}**자까지입니다. (조절: \`YAWN_MAX_QUESTION_CHARS\`)`,
      flags: MessageFlags.Ephemeral,
    });
    return;
  }

  const ctxN = parseInt(process.env.YAWN_CONTEXT_MESSAGES || '10', 10);
  const contextCount = Math.min(Math.max(0, Number.isFinite(ctxN) ? ctxN : 10), 30);

  await interaction.deferReply();
  let stopGeminiTicker: () => Promise<void> = async () => {};
  try {
    stopGeminiTicker = await startDeferElapsedTicker(interaction, 'gemini', { requestText: prompt });
    const contextBlock = await buildYawnChannelContext(interaction, contextCount);
    const systemInstruction = yawnSystemPromptFromEnv();
    const userBlock = `사용자 질문:\n${prompt}`;
    let blobPrompt = `${contextBlock}\n\n${userBlock}`.trim();
    const maxFull = parseInt(process.env.YAWN_MAX_PROMPT_CHARS || '12000', 10);
    const maxFullClamped = Math.min(Math.max(2000, Number.isFinite(maxFull) ? maxFull : 12000), 32000);
    if (blobPrompt.length > maxFullClamped) {
      blobPrompt = blobPrompt.slice(0, maxFullClamped) + '\n\n…(앞부분·맥락이 잘렸습니다)';
    }
    const { text: response, surface: usedSurface, modelId: usedModelId } =
      await generateBlobTextFromEnvWithOptions(process.env, blobPrompt, {
        surface: surfaceOverride,
        modelId: modelOpt,
        systemInstruction,
      });
    await stopGeminiTicker();
    stopGeminiTicker = async () => {};
    const apiLabel = usedSurface === 'vertex' ? 'Vertex AI' : 'Google AI Studio';
    const embed = new EmbedBuilder()
      .setTitle('YawnBot AI Response')
      .setDescription(
        truncateDiscordDescription(
          `**📝 질문**\n\`\`\`\n${truncateEmbedField(prompt, 800)}\n\`\`\`\n\n` + `**💬 답변**\n${response.slice(0, 3000)}`,
        ),
      )
      .setColor(0x4285f4)
      .setFooter({ text: `${apiLabel} · ${usedModelId}` })
      .setTimestamp();
    await interaction.editReply({ content: null, embeds: [embed] });
    await notifyDeferCompletion(interaction, { ok: true, kind: 'gemini' });
  } catch (e) {
    console.warn('[yawn]', e instanceof Error ? e.message : e);
    await stopGeminiTicker();
    stopGeminiTicker = async () => {};
    const shortMsg = friendlyYawnErrorMessage(e);
    await interaction.editReply({
      content: null,
      embeds: [
        new EmbedBuilder()
          .setTitle('AI 응답 실패')
          .setDescription(
            truncateDiscordDescription(
              `**📝 질문**\n\`\`\`\n${truncateEmbedField(prompt, 800)}\n\`\`\`\n\n**안내**\n${shortMsg}`,
            ),
          )
          .setColor(0xf44336),
      ],
    });
    await notifyDeferCompletion(interaction, { ok: false, kind: 'gemini' });
  } finally {
    await stopGeminiTicker();
  }
}

