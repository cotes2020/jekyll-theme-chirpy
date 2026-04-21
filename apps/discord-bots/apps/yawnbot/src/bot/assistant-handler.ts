/**
 * AssistantHandler — DM 및 전용 채널에서 AI 비서와 대화
 *
 * - ASSISTANT_USER_ID의 메시지에만 응답
 * - 채널/DM의 활성 캐릭터(card.md 본문)를 시스템 프롬프트로 사용
 * - 컨텍스트: card 본문 + user.md + self.md + 주간요약 + 어제요약 + 오늘 로그
 * - 메시지는 슬러그별 logs/에 즉시 기록 (손실 없음)
 */
import { Message, DMChannel, TextChannel } from 'discord.js';
import { generateAssistantText } from 'karmolab-ai/node';
import type { MemoryService, ConversationEntry } from '../services/memory-service';
import { CharacterService, type CharacterCard } from '../services/character-service';

const MAX_RESPONSE_LENGTH = 1900;
const MAX_PROMPT_CHARS = parseInt(process.env.ASSISTANT_MAX_PROMPT_CHARS || '12000', 10);

async function detectAndSaveHotMemory(
  memory: MemoryService,
  userMessage: string,
): Promise<void> {
  try {
    console.log(`[Assistant:${memory.slug}] 핫메모리 감지 중...`);
    const { text } = await generateAssistantText(
      process.env,
      `다음 메시지에서 user.md에 즉시 저장할 중요한 사실이 있으면 한 줄로 작성해줘.\n` +
        `중요한 사실: 새로운 프로젝트/계획, 감정 변화, 중요한 결정, 인생 이벤트.\n` +
        `없으면 정확히 "SKIP"이라고만 반환.\n\n` +
        `메시지: "${userMessage}"`,
    );
    const trimmed = text.trim();
    if (trimmed === 'SKIP' || !trimmed) {
      console.log(`[Assistant:${memory.slug}] 핫메모리: 저장할 정보 없음 (SKIP)`);
      return;
    }
    memory.appendHotMemory(trimmed);
    console.log(`[Assistant:${memory.slug}] 핫메모리 저장: ${trimmed.slice(0, 60)}...`);
  } catch (e) {
    console.error(
      `[Assistant:${memory.slug}] 핫메모리 감지 실패:`,
      e instanceof Error ? e.message : String(e),
    );
  }
}

/**
 * card.md 본문 자체가 시스템 프롬프트. 채널 타입 한 줄만 덧붙인다.
 */
function buildSystemPrompt(card: CharacterCard, channelType: 'dm' | 'public'): string {
  const channelDesc =
    channelType === 'dm'
      ? '지금은 DM으로 사적인 대화 중이야.'
      : '지금은 공개 채널에서 대화 중이야.';
  return `${card.body}\n\n${channelDesc}`;
}

function buildFullPrompt(
  card: CharacterCard,
  memory: MemoryService,
  channelType: 'dm' | 'public',
  userMessage: string,
): string {
  const system = buildSystemPrompt(card, channelType);
  const budget = MAX_PROMPT_CHARS - system.length - userMessage.length - 50;
  const contextBudget = Math.max(2000, budget);
  const context = memory.buildContext(contextBudget);

  const contextSize = Buffer.byteLength(context, 'utf-8');
  console.log(
    `[Assistant:${memory.slug}] 컨텍스트 빌드: ${contextSize}바이트 (할당: ${contextBudget}자, 사용: ${context.length}자)`,
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
  characterService: CharacterService,
  getMemory: (slug: string) => MemoryService,
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

  // 채널/DM 단위 활성 캐릭터 해석
  const channelKey = CharacterService.channelKey({
    isDM,
    userId: message.author.id,
    channelId: message.channel.id,
  });
  const card = characterService.resolveCard(channelKey);
  if (!card) {
    console.warn('[Assistant] 활성 캐릭터 카드 없음 — 응답 스킵');
    await message.reply('활성 캐릭터 카드가 없어서 대답할 수 없어요. `/character list` 로 확인해봐요.');
    return;
  }
  const memory = getMemory(card.slug);

  console.log(
    `[Assistant:${card.slug}] 메시지 수신 [${channelType}] (${userContent.length}자): ${userContent.slice(0, 50)}`,
  );

  const provider = (process.env.ASSISTANT_AI_PROVIDER || 'gemini').toLowerCase();
  if (provider !== 'claude-cli' && !process.env.GEMINI_API_KEY?.trim()) {
    console.warn(`[Assistant:${card.slug}] API 키 없음:`, provider);
    await message.reply('GEMINI_API_KEY가 설정되지 않아서 대화할 수 없어요.');
    return;
  }

  const userEntry: ConversationEntry = {
    timestamp: new Date().toISOString(),
    role: 'user',
    content: userContent,
    channel: channelType,
  };
  memory.appendToLog(userEntry);

  memory.checkAndGenerateSummaries().catch((e) =>
    console.error(
      `[Assistant:${card.slug}] 요약 생성 오류:`,
      e instanceof Error ? e.message : e,
    ),
  );

  try {
    if ('sendTyping' in message.channel) {
      await (message.channel as TextChannel | DMChannel).sendTyping();
    }
  } catch {
    /* ignore */
  }

  const startTime = Date.now();
  console.log(`[Assistant:${card.slug}] 프롬프트 빌드 시작...`);
  const fullPrompt = buildFullPrompt(card, memory, channelType, userContent);
  const promptSize = Buffer.byteLength(fullPrompt, 'utf-8');
  console.log(
    `[Assistant:${card.slug}] 프롬프트 준비 완료: ${promptSize}바이트, ${fullPrompt.length}자 (빌드 소요: ${Date.now() - startTime}ms)`,
  );

  try {
    const aiStartTime = Date.now();
    let response: string | null = null;
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        console.log(
          `[Assistant:${card.slug}] AI 호출 시작 (${provider}, 시도 ${attempt}/3, 타임아웃 20초)...`,
        );
        const { text } = await generateAssistantText(process.env, fullPrompt, {
          timeoutMs: 20000,
        });
        response = text;
        const aiDuration = Date.now() - aiStartTime;
        console.log(
          `[Assistant:${card.slug}] AI 응답 수신 (${aiDuration}ms, 시도 ${attempt}/3): ${response.length}자 -> 응답 중...`,
        );
        break;
      } catch (e: unknown) {
        lastError = e instanceof Error ? e : new Error(String(e));
        const attemptDuration = Date.now() - aiStartTime;
        console.warn(
          `[Assistant:${card.slug}] 시도 ${attempt}/3 실패 (${attemptDuration}ms): ${lastError.message}`,
        );

        if (attempt < 3) {
          const waitMs = Math.min(1000 * attempt, 3000);
          console.log(`[Assistant:${card.slug}] ${waitMs}ms 후 재시도...`);
          await new Promise((resolve) => setTimeout(resolve, waitMs));
        }
      }
    }

    if (!response) {
      throw lastError || new Error('[Assistant] 최대 재시도 횟수 초과 (응답 없음)');
    }

    const reply = response.trim().slice(0, MAX_RESPONSE_LENGTH);

    memory.appendToLog({
      timestamp: new Date().toISOString(),
      role: 'assistant',
      content: reply,
      channel: channelType,
    });

    detectAndSaveHotMemory(memory, userContent).catch(() => {});

    await message.reply(reply);
    const totalDuration = Date.now() - startTime;
    console.log(
      `[Assistant:${card.slug}] 완료 (총 ${totalDuration}ms, 응답크기: ${reply.length}자)`,
    );
  } catch (e: unknown) {
    const totalDuration = Date.now() - startTime;
    console.error(
      `[Assistant:${card.slug}] 응답 실패 (${totalDuration}ms):`,
      e instanceof Error ? `${e.name}: ${e.message}` : String(e),
    );
    await message.reply(friendlyError(e));
  }
}
