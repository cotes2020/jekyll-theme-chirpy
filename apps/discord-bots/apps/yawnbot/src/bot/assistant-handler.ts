/**
 * AssistantHandler — DM 및 전용 채널에서 AI 비서와 대화
 * - ASSISTANT_USER_ID의 메시지에만 응답
 * - DM: 사적 대화
 * - ASSISTANT_PUBLIC_CHANNEL_ID: 공개 대화
 * - 인메모리 최근 20턴 + diary 컨텍스트를 Gemini에 전달
 */
import { Message, DMChannel, TextChannel } from 'discord.js';
import { generateBlobTextFromEnvWithOptions } from 'karmolab-ai/node';
import type { MemoryService, ConversationEntry } from '../services/memory-service';

const MAX_HISTORY_TURNS = 20;
const MAX_RESPONSE_LENGTH = 1900;

interface HistoryEntry {
  role: 'user' | 'assistant';
  content: string;
}

// 채널별 인메모리 히스토리 (DM channelId → 배열, public channelId → 배열)
const historyMap = new Map<string, HistoryEntry[]>();

function getHistory(channelId: string): HistoryEntry[] {
  if (!historyMap.has(channelId)) {
    historyMap.set(channelId, []);
  }
  return historyMap.get(channelId)!;
}

function pushHistory(channelId: string, entry: HistoryEntry): void {
  const hist = getHistory(channelId);
  hist.push(entry);
  if (hist.length > MAX_HISTORY_TURNS * 2) {
    hist.splice(0, 2);
  }
}

function buildSystemPrompt(memory: MemoryService, channelType: 'dm' | 'public'): string {
  const profile = memory.getProfile();
  const recentDiaries = memory.getRecentDiaries(3);

  const channelDesc = channelType === 'dm'
    ? '지금은 DM으로 사적인 대화 중이야.'
    : '지금은 공개 채널에서 대화 중이야.';

  const profileSection = profile
    ? `\n\n[사용자 정보]\n${profile}`
    : '';

  const diarySection = recentDiaries
    ? `\n\n[최근 대화 기록 (참고)]\n${recentDiaries}`
    : '';

  return (
    `너는 mascari4615의 개인 AI 비서야. 이름은 아직 없어.\n` +
    `한국어로 대화해. 짧고 자연스럽게 말해. 딱딱하지 않게.\n` +
    `사용자가 힘들 때 공감해주고, 기쁠 때 같이 기뻐해줘.\n` +
    `중요한 정보는 기억해뒀다가 나중에 자연스럽게 언급해줘.\n` +
    `${channelDesc}` +
    profileSection +
    diarySection
  );
}

function buildFullPrompt(
  memory: MemoryService,
  channelId: string,
  channelType: 'dm' | 'public',
  userMessage: string,
): string {
  const system = buildSystemPrompt(memory, channelType);
  const history = getHistory(channelId);

  let historyBlock = '';
  if (history.length > 0) {
    const lines = history.map((h) => {
      const label = h.role === 'user' ? '나' : 'YawnBot';
      return `${label}: ${h.content}`;
    });
    historyBlock = `\n\n[이번 대화]\n${lines.join('\n')}`;
  }

  const maxTotal = parseInt(process.env.ASSISTANT_MAX_PROMPT_CHARS || '12000', 10);
  let full = `${system}${historyBlock}\n\n나: ${userMessage}`;

  if (full.length > maxTotal) {
    full = full.slice(-maxTotal) + '\n(앞부분 잘림)';
  }

  return full;
}

function friendlyError(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  const lower = msg.toLowerCase();
  if (lower.includes('429') || lower.includes('quota') || lower.includes('resource exhausted')) {
    return '잠깐, 요청이 너무 많아서 잠시 후에 다시 말해줘요. (할당량 초과)';
  }
  if (lower.includes('safety') || lower.includes('blocked')) {
    return '그 질문은 대답하기 어렵네요. 다른 방식으로 물어봐 줄 수 있어요?';
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
  const isPublicChannel = !isDM && publicChannelId && message.channel.id === publicChannelId;

  if (!isDM && !isPublicChannel) return;

  const channelType: 'dm' | 'public' = isDM ? 'dm' : 'public';
  const channelId = message.channel.id;
  const userContent = message.content.trim();

  if (!userContent) return;

  if (!process.env.GEMINI_API_KEY?.trim()) {
    await message.reply('GEMINI_API_KEY가 설정되지 않아서 대화할 수 없어요.');
    return;
  }

  // 사용자 메시지 히스토리에 추가
  pushHistory(channelId, { role: 'user', content: userContent });

  // 기억 서비스에 저장
  memory.addEntry({
    timestamp: new Date().toISOString(),
    role: 'user',
    content: userContent,
    channel: channelType,
  });

  // 타이핑 표시
  try {
    if ('sendTyping' in message.channel) {
      await (message.channel as TextChannel | DMChannel).sendTyping();
    }
  } catch { /* ignore */ }

  const fullPrompt = buildFullPrompt(memory, channelId, channelType, userContent);

  try {
    const { text: response } = await generateBlobTextFromEnvWithOptions(
      process.env,
      fullPrompt,
      { surface: 'inherit' },
    );

    const reply = response.slice(0, MAX_RESPONSE_LENGTH);

    // 응답 히스토리에 추가
    pushHistory(channelId, { role: 'assistant', content: reply });

    // 기억 서비스에 저장
    memory.addEntry({
      timestamp: new Date().toISOString(),
      role: 'assistant',
      content: reply,
      channel: channelType,
    });

    await message.reply(reply);
  } catch (e: unknown) {
    console.error('[Assistant] 응답 실패:', e instanceof Error ? e.message : e);
    await message.reply(friendlyError(e));
  }
}
