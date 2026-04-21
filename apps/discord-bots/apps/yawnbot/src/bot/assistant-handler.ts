/**
 * AssistantHandler — DM 및 전용 채널에서 AI 비서와 대화
 *
 * - ASSISTANT_USER_ID의 메시지에만 응답
 * - 채널/DM의 활성 캐릭터(card.md 본문)를 시스템 프롬프트로 사용
 * - 컨텍스트: card 본문 + user.md + self.md + 주간요약 + 어제요약 + 오늘 로그
 * - 메시지는 슬러그별 logs/에 즉시 기록 (손실 없음)
 */
import fs from 'fs';
import { Message, DMChannel, TextChannel, AttachmentBuilder } from 'discord.js';
import { generateAssistantText, generateImageFromEnvWithOptions } from 'karmolab-ai/node';
import type { ChatContent } from 'karmolab-ai/node';
import type { MemoryService, ConversationEntry } from '../services/memory-service';
import { CharacterService, type CharacterCard } from '../services/character-service';
import { ImageCacheService } from '../services/image-cache-service';
import { buildCharacterImagePrompt } from './slash/image';

const MAX_RESPONSE_LENGTH = 1900;
const MAX_PROMPT_CHARS = parseInt(process.env.ASSISTANT_MAX_PROMPT_CHARS || '12000', 10);
const MAX_HISTORY_TURNS = 20;
const IMAGE_COOLDOWN_MS = (() => {
  const v = parseInt(process.env.ASSISTANT_IMAGE_COOLDOWN_MS || '', 10);
  return Number.isFinite(v) && v >= 0 ? v : 2 * 60 * 1000;
})();

const chatHistories = new Map<string, ChatContent[]>();
const imageCacheServices = new Map<string, ImageCacheService>();
const lastImageAt = new Map<string, number>();
const lastSentImageId = new Map<string, string>();

function getImageCacheService(card: CharacterCard): ImageCacheService {
  if (!imageCacheServices.has(card.slug)) {
    const memoRepoPath = process.env.MEMO_REPO_PATH?.trim() || '';
    imageCacheServices.set(card.slug, new ImageCacheService(card.dir, memoRepoPath, card.slug));
  }
  return imageCacheServices.get(card.slug)!;
}

function getHistory(slug: string): ChatContent[] {
  if (!chatHistories.has(slug)) chatHistories.set(slug, []);
  return chatHistories.get(slug)!;
}

function appendHistory(slug: string, userMsg: string, assistantMsg: string): void {
  const history = getHistory(slug);
  history.push({ role: 'user', parts: [{ text: userMsg }] });
  history.push({ role: 'model', parts: [{ text: assistantMsg }] });
  if (history.length > MAX_HISTORY_TURNS * 2) {
    history.splice(0, history.length - MAX_HISTORY_TURNS * 2);
  }
}

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
 * 대화 맥락에서 이미지 생성이 필요한지 판단 후 영어 태그 반환.
 * LLM이 "SKIP" 반환하거나 에러나면 null.
 */
async function detectSceneTags(
  slug: string,
  userMsg: string,
  aiResponse: string,
): Promise<string[] | null> {
  try {
    console.log(`[Assistant:${slug}] 씬 감지 중...`);
    const { text } = await generateAssistantText(
      process.env,
      `다음 대화 장면에서 캐릭터 이미지를 자동 생성해야 하는지 판단해줘.\n` +
        `생성해야 한다면: 장면을 설명하는 영어 태그 5개 이내를 쉼표로 구분해서만 반환 (예: "smiling, outdoor, casual")\n` +
        `생성하지 않아도 된다면: 정확히 "SKIP" 반환.\n\n` +
        `기준:\n` +
        `- 감정 변화나 상황 전환이 뚜렷하면 생성\n` +
        `- 대화가 평범하게 이어지는 경우 SKIP\n\n` +
        `유저: "${userMsg}"\n` +
        `캐릭터: "${aiResponse}"`,
    );
    const trimmed = text.trim();
    if (!trimmed || trimmed.toUpperCase() === 'SKIP') {
      console.log(`[Assistant:${slug}] 씬 감지: SKIP`);
      return null;
    }
    const tags = trimmed
      .split(',')
      .map((t) => t.trim().toLowerCase())
      .filter(Boolean)
      .slice(0, 5);
    console.log(`[Assistant:${slug}] 씬 감지: 태그=[${tags.join(', ')}]`);
    return tags.length > 0 ? tags : null;
  } catch (e) {
    console.error(
      `[Assistant:${slug}] 씬 감지 실패:`,
      e instanceof Error ? e.message : String(e),
    );
    return null;
  }
}

interface SceneImage {
  id: string;
  tags: string[];
  buffer: Buffer;
  mimeType: string;
}

/**
 * 씬 감지 → 캐시 조회 → 필요 시 이미지 생성까지 모두 수행.
 * 이미지가 있으면 { tags, buffer, mimeType } 반환, 없으면 null.
 */
async function resolveSceneImage(
  card: CharacterCard,
  userMsg: string,
  aiResponse: string,
): Promise<SceneImage | null> {
  const slug = card.slug;

  const last = lastImageAt.get(slug) ?? 0;
  const remaining = IMAGE_COOLDOWN_MS - (Date.now() - last);
  if (remaining > 0) {
    console.log(`[Assistant:${slug}] 자동 이미지 쿨다운 (${Math.round(remaining / 1000)}초 남음)`);
    return null;
  }

  const tags = await detectSceneTags(slug, userMsg, aiResponse);
  if (!tags) return null;

  const cacheService = getImageCacheService(card);
  const cached = cacheService.findSimilar(tags);

  if (cached) {
    if (lastSentImageId.get(slug) === cached.id) {
      console.log(`[Assistant:${slug}] 자동 이미지: 직전과 동일 이미지 (id=${cached.id}), 텍스트만 전송`);
      return null;
    }
    cacheService.incrementHit(cached);
    const buffer = fs.readFileSync(cached.filePath);
    console.log(`[Assistant:${slug}] 자동 이미지: 캐시 히트 (id=${cached.id})`);
    return { tags, buffer, mimeType: cached.mimeType, id: cached.id };
  }

  // 캐시 미스 → 새 이미지 생성 (이 시점에만 쿨다운 소모)
  lastImageAt.set(slug, Date.now());
  const finalPrompt = buildCharacterImagePrompt(card, tags.join(', '));
  console.log(`[Assistant:${slug}] 자동 이미지: 생성 시작 (태그=[${tags.join(', ')}])`);

  const { images, modelId } = await generateImageFromEnvWithOptions(process.env, finalPrompt, {
    sampleCount: 1,
  });

  if (images.length === 0) return null;
  const img = images[0];
  const entry = cacheService.add(tags, finalPrompt, img.buffer, img.mimeType, modelId);
  console.log(`[Assistant:${slug}] 자동 이미지: 완료 (id=${entry.id}, 모델=${modelId})`);
  return { id: entry.id, tags, buffer: img.buffer, mimeType: img.mimeType };
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
  const context = memory.buildContext(Math.max(2000, budget));
  const nowKST = new Date().toLocaleString('ko-KR', {
    timeZone: 'Asia/Seoul',
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  });
  const contextBlock = context ? `\n\n${context}` : '';
  return `${system}\n\n[현재 시각] ${nowKST}${contextBlock}\n\n나: ${userMessage}`;
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

  const isGemini = provider !== 'claude-cli';
  let systemInstruction: string | undefined;
  let fullPrompt: string;
  const history = isGemini ? getHistory(card.slug) : undefined;

  if (isGemini) {
    // systemInstruction = 캐릭터 카드 + 채널 타입만 (안정적 → Gemini implicit cache 활성화)
    systemInstruction = buildSystemPrompt(card, channelType);
    // 가변 부분(시각, 메모리 컨텍스트, 오늘 로그)은 user message에 포함
    const nowKST = new Date().toLocaleString('ko-KR', {
      timeZone: 'Asia/Seoul',
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', hour12: false,
    });
    const budget = MAX_PROMPT_CHARS - systemInstruction.length - userContent.length - 50;
    const contextBudget = Math.max(2000, budget);
    const context = memory.buildContext(contextBudget);
    const contextSize = Buffer.byteLength(context, 'utf-8');
    console.log(
      `[Assistant:${memory.slug}] 컨텍스트 빌드: ${contextSize}바이트 (할당: ${contextBudget}자, 사용: ${context.length}자)`,
    );
    const contextBlock = context ? `\n\n${context}` : '';
    fullPrompt = `[현재 시각] ${nowKST}${contextBlock}\n\n나: ${userContent}`;
  } else {
    fullPrompt = buildFullPrompt(card, memory, channelType, userContent);
  }

  const promptSize = Buffer.byteLength(fullPrompt, 'utf-8');
  if (isGemini && systemInstruction != null) {
    const sysSize = Buffer.byteLength(systemInstruction, 'utf-8');
    const histSize = history
      ? history.reduce((sum, e) => sum + Buffer.byteLength(JSON.stringify(e), 'utf-8'), 0)
      : 0;
    console.log(
      `[Assistant:${card.slug}] 프롬프트 준비 완료: 메시지 ${promptSize}B / systemInstruction ${sysSize}B / history ${histSize}B / 합계 ${promptSize + sysSize + histSize}B / 히스토리 ${(history?.length ?? 0) / 2}턴 (빌드 소요: ${Date.now() - startTime}ms)`,
    );
  } else {
    console.log(
      `[Assistant:${card.slug}] 프롬프트 준비 완료: ${promptSize}바이트, ${fullPrompt.length}자 (빌드 소요: ${Date.now() - startTime}ms)`,
    );
  }

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
          history,
          systemInstruction,
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

    if (isGemini) appendHistory(card.slug, userContent, reply);

    detectAndSaveHotMemory(memory, userContent).catch(() => {});

    // 씬 감지 → 캐시 조회 → 이미지 생성까지 완료 후 텍스트와 함께 전송
    const sceneImage = await resolveSceneImage(card, userContent, reply).catch((e) => {
      console.error(`[Assistant:${card.slug}] 자동 이미지 실패:`, e instanceof Error ? e.message : String(e));
      return null;
    });

    if (sceneImage) {
      const ext = (sceneImage.mimeType.split('/')[1] || 'png').replace(/[^a-z0-9]/gi, '');
      const attachment = new AttachmentBuilder(sceneImage.buffer, { name: `scene.${ext}` });
      await message.reply({ content: reply, files: [attachment] });
      lastSentImageId.set(card.slug, sceneImage.id);
    } else {
      await message.reply(reply);
    }

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
