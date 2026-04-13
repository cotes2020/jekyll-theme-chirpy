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
    console.log(`[Assistant] 핫메모리 감지 중...`);
    const { text } = await generateAssistantText(
      process.env,
      `다음 메시지에서 user.md에 즉시 저장할 중요한 사실이 있으면 한 줄로 작성해줘.\n` +
        `중요한 사실: 새로운 프로젝트/계획, 감정 변화, 중요한 결정, 인생 이벤트.\n` +
        `없으면 정확히 "SKIP"이라고만 반환.\n\n` +
        `메시지: "${userMessage}"`,
    );
    const trimmed = text.trim();
    if (trimmed === 'SKIP' || !trimmed) {
      console.log(`[Assistant] 핫메모리: 저장할 정보 없음 (SKIP)`);
      return;
    }
    memory.appendHotMemory(trimmed);
    console.log(`[Assistant] 핫메모리 저장: ${trimmed.slice(0, 60)}...`);
  } catch (e) {
    console.error(
      `[Assistant] 핫메모리 감지 실패:`,
      e instanceof Error ? e.message : String(e),
    );
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
  const budget = MAX_PROMPT_CHARS - system.length - userMessage.length - 50;
  const contextBudget = Math.max(2000, budget);
  const context = memory.buildContext(contextBudget);

  const contextSize = Buffer.byteLength(context, 'utf-8');
  console.log(
    `[Assistant] 컨텍스트 빌드: ${contextSize}바이트 (할당: ${contextBudget}자, 사용: ${context.length}자)`,
  );

  const contextBlock = context ? `\n\n${context}` : '';
  return `${system}${contextBlock}\n\n나: ${userMessage}`;
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

  console.log(`[Assistant] 메시지 수신 [${channelType}] (${userContent.length}자): ${userContent.slice(0, 50)}`);

  const provider = (process.env.ASSISTANT_AI_PROVIDER || 'gemini').toLowerCase();
  if (provider !== 'claude-cli' && !process.env.GEMINI_API_KEY?.trim()) {
    console.warn('[Assistant] API 키 없음:', provider);
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

  const startTime = Date.now();
  console.log(`[Assistant] 프롬프트 빌드 시작...`);
  const fullPrompt = buildFullPrompt(memory, channelType, userContent);
  const promptSize = Buffer.byteLength(fullPrompt, 'utf-8');
  console.log(
    `[Assistant] 프롬프트 준비 완료: ${promptSize}바이트, ${fullPrompt.length}자 (빌드 소요: ${Date.now() - startTime}ms)`,
  );

  try {
    const aiStartTime = Date.now();
    let response: string | null = null;
    let lastError: Error | null = null;

    // 재시도 로직: 최대 2회 시도 (총 3회)
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        console.log(`[Assistant] AI 호출 시작 (${provider}, 시도 ${attempt}/3, 타임아웃 20초)...`);
        const { text } = await generateAssistantText(process.env, fullPrompt, {
          timeoutMs: 20000, // 20초
        });
        response = text;
        const aiDuration = Date.now() - aiStartTime;
        console.log(
          `[Assistant] AI 응답 수신 (${aiDuration}ms, 시도 ${attempt}/3): ${response.length}자 -> 응답 중...`,
        );
        break; // 성공하면 루프 탈출
      } catch (e: unknown) {
        lastError = e instanceof Error ? e : new Error(String(e));
        const attemptDuration = Date.now() - aiStartTime;
        console.warn(
          `[Assistant] 시도 ${attempt}/3 실패 (${attemptDuration}ms): ${lastError.message}`,
        );

        if (attempt < 3) {
          const waitMs = Math.min(1000 * attempt, 3000); // 1s, 2s, 3s
          console.log(`[Assistant] ${waitMs}ms 후 재시도...`);
          await new Promise((resolve) => setTimeout(resolve, waitMs));
        }
      }
    }

    if (!response) {
      throw (
        lastError || new Error('[Assistant] 최대 재시도 횟수 초과 (응답 없음)')
      );
    }

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

    const sentTime = Date.now();
    await message.reply(reply);
    const totalDuration = Date.now() - startTime;
    console.log(`[Assistant] 완료 (총 ${totalDuration}ms, 응답크기: ${reply.length}자)`);
  } catch (e: unknown) {
    const totalDuration = Date.now() - startTime;
    console.error(
      `[Assistant] 응답 실패 (${totalDuration}ms):`,
      e instanceof Error ? `${e.name}: ${e.message}` : String(e),
    );
    await message.reply(friendlyError(e));
  }
}
