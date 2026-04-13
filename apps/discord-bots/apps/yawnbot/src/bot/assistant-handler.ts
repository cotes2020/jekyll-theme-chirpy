/**
 * AssistantHandler — DM 및 전용 채널에서 AI 비서와 대화
 * - ASSISTANT_USER_ID의 메시지에만 응답
 * - 컨텍스트: user.md + self.md + 주간요약 + 어제요약 + 오늘 전체 로그
 * - 메시지는 logs/에 즉시 기록 (손실 없음)
 */
import { Message, DMChannel, TextChannel } from 'discord.js';
import { generateAssistantText } from 'karmolab-ai/node';
import type { MemoryService, ConversationEntry } from '../services/memory-service';

const MAX_RESPONSE_LENGTH = 1900;
const MAX_PROMPT_CHARS = parseInt(process.env.ASSISTANT_MAX_PROMPT_CHARS || '12000', 10);

async function detectAndSaveHotMemory(
  memory: MemoryService,
  userMessage: string,
): Promise<void> {
  try {
    const { text } = await generateAssistantText(
      process.env,
      `다음 메시지에서 user.md에 즉시 저장할 중요한 사실이 있으면 한 줄로 작성해줘.\n` +
        `중요한 사실: 새로운 프로젝트/계획, 감정 변화, 중요한 결정, 인생 이벤트.\n` +
        `없으면 정확히 "SKIP"이라고만 반환.\n\n` +
        `메시지: "${userMessage}"`,
    );
    const trimmed = text.trim();
    if (trimmed === 'SKIP' || !trimmed) return;
    memory.appendHotMemory(trimmed);
  } catch {
    /* 실패해도 무시 */
  }
}

function buildSystemPrompt(channelType: 'dm' | 'public'): string {
  const channelDesc =
    channelType === 'dm' ? '지금은 DM으로 사적인 대화 중이야.' : '지금은 공개 채널에서 대화 중이야.';
  return (
    `너는 mascari4615의 개인 AI 비서야.\n` +
    `한국어로 대화해. 짧고 자연스럽게, 딱딱하지 않게.\n` +
    `사용자가 힘들 때 공감해주고, 기쁠 때 같이 기뻐해줘.\n` +
    `이전 대화와 기억을 바탕으로 자연스럽게 연결해줘.\n` +
    `${channelDesc}`
  );
}

function buildFullPrompt(
  memory: MemoryService,
  channelType: 'dm' | 'public',
  userMessage: string,
): string {
  const system = buildSystemPrompt(channelType);
  const context = memory.buildContext();

  const contextBlock = context ? `\n\n${context}` : '';
  let full = `${system}${contextBlock}\n\n나: ${userMessage}`;

  if (full.length > MAX_PROMPT_CHARS) {
    // 시스템 프롬프트는 보존, 컨텍스트 앞부분을 잘라냄
    const budget = MAX_PROMPT_CHARS - system.length - userMessage.length - 50;
    const trimmedContext = budget > 0 ? context.slice(-budget) : '';
    full = `${system}\n\n${trimmedContext}\n\n나: ${userMessage}`;
  }

  return full;
}

function friendlyError(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  const lower = msg.toLowerCase();
  if (lower.includes('429') || lower.includes('quota') || lower.includes('resource exhausted')) {
    return '잠깐, 요청이 너무 많아서 잠시 후에 다시 말해줘요.';
  }
  if (lower.includes('safety') || lower.includes('blocked')) {
    return '그 질문은 대답하기 어렵네요. 다르게 물어봐줄 수 있어요?';
  }
  if (lower.includes('api key') || lower.includes('401') || lower.includes('403')) {
    return 'API 인증 문제가 생겼어요. 잠시 후 다시 시도해줘요.';
  }
  return '답변을 가져오는 데 실패했어요. 잠시 후 다시 시도해줘요.';
}

export async function handleAssistantMessage(
  message: Message,
  memory: MemoryService,
): Promise<void> {
  const assistantUserId = process.env.ASSISTANT_USER_ID?.trim();
  const publicChannelId = process.env.ASSISTANT_PUBLIC_CHANNEL_ID?.trim();

  if (!assistantUserId) return;
  if (message.author.id !== assistantUserId) return;
  if (message.author.bot) return;

  const isDM = message.channel instanceof DMChannel || message.channel.isDMBased();
  const isPublicChannel = !isDM && !!publicChannelId && message.channel.id === publicChannelId;

  if (!isDM && !isPublicChannel) return;

  const channelType: 'dm' | 'public' = isDM ? 'dm' : 'public';
  const userContent = message.content.trim();
  if (!userContent) return;

  const provider = (process.env.ASSISTANT_AI_PROVIDER || 'gemini').toLowerCase();
  if (provider !== 'claude-cli' && !process.env.GEMINI_API_KEY?.trim()) {
    await message.reply('GEMINI_API_KEY가 설정되지 않아서 대화할 수 없어요.');
    return;
  }

  // 사용자 메시지 즉시 로그에 기록
  const userEntry: ConversationEntry = {
    timestamp: new Date().toISOString(),
    role: 'user',
    content: userContent,
    channel: channelType,
  };
  memory.appendToLog(userEntry);

  // 오늘 처음 대화라면 어제/지난주 요약 생성 (비동기, 응답 막지 않음)
  memory.checkAndGenerateSummaries().catch((e) =>
    console.error('[Assistant] 요약 생성 오류:', e instanceof Error ? e.message : e),
  );

  // 타이핑 표시
  try {
    if ('sendTyping' in message.channel) {
      await (message.channel as TextChannel | DMChannel).sendTyping();
    }
  } catch { /* ignore */ }

  const fullPrompt = buildFullPrompt(memory, channelType, userContent);

  try {
    const { text: response } = await generateAssistantText(process.env, fullPrompt);

    const reply = response.trim().slice(0, MAX_RESPONSE_LENGTH);

    // 봇 응답도 즉시 로그에 기록
    memory.appendToLog({
      timestamp: new Date().toISOString(),
      role: 'assistant',
      content: reply,
      channel: channelType,
    });

    // hot path — 중요 정보 즉시 저장 (비동기, 응답 블로킹 없음)
    detectAndSaveHotMemory(memory, userContent).catch(() => {});

    await message.reply(reply);
  } catch (e: unknown) {
    console.error('[Assistant] 응답 실패:', e instanceof Error ? e.message : e);
    await message.reply(friendlyError(e));
  }
}
