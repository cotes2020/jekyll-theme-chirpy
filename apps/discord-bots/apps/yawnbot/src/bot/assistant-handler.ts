/**
 * AssistantHandler — DM 및 전용 채널에서 AI 비서와 대화
 *
 * - ASSISTANT_USER_ID의 메시지에만 응답
 * - 채널/DM의 활성 캐릭터(card.md 본문)를 시스템 프롬프트로 사용
 * - 컨텍스트: card 본문 + user.md + self.md + 주간요약 + 어제요약 + 오늘 로그
 * - 메시지는 슬러그별 logs/에 즉시 기록 (손실 없음)
 */
import fs from 'fs';
import { Message, DMChannel, TextChannel, AttachmentBuilder, ActionRowBuilder, ButtonBuilder, ButtonStyle } from 'discord.js';
import { generateAssistantText, generateImageFromEnvWithOptions } from 'karmolab-ai/node';
import type { ChatContent } from 'karmolab-ai/node';
import type { MemoryService, ConversationEntry } from '../services/memory-service';
import { CharacterService, type CharacterCard } from '../services/character-service';
import { ImageCacheService } from '../services/image-cache-service';
import type { MoodService } from '../services/mood-service';
import type { RelationshipService } from '../services/relationship-service';
import { buildCharacterImagePrompt } from './slash/image';

export const MOOD_REACTION_EMOJIS = ['👍', '❤️', '😂', '😢'] as const;
export type MoodReactionEmoji = typeof MOOD_REACTION_EMOJIS[number];

export const MOOD_REACTION_MAP: Record<MoodReactionEmoji, string> = {
  '👍': '뿌듯하고 기분 좋음',
  '❤️': '따뜻하고 행복함',
  '😂': '신나고 유쾌함',
  '😢': '슬프고 감동적',
};

export function buildReactionRow(slug: string): ActionRowBuilder<ButtonBuilder> {
  const buttons = MOOD_REACTION_EMOJIS.map((emoji) =>
    new ButtonBuilder()
      .setCustomId(`mood_reaction:${emoji}:${slug}`)
      .setLabel(emoji)
      .setStyle(ButtonStyle.Secondary),
  );
  return new ActionRowBuilder<ButtonBuilder>().addComponents(...buttons);
}

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

const HOT_MEMORY_TRIGGERS = [
  '시험', '면접', '수술', '졸업', '취직', '이직', '이사', '여행', '생일',
  '좋아해', '싫어해', '알레르기', '직업', '학교', '학과', '사귀',
  '프로젝트', '마감', '결정했', '계획이', '목표',
];

async function detectAndSaveHotMemory(
  memory: MemoryService,
  userMessage: string,
): Promise<void> {
  // 휴리스틱 프리필터 — 트리거 키워드 없으면 LLM 호출 스킵
  const lower = userMessage.toLowerCase();
  if (!HOT_MEMORY_TRIGGERS.some((kw) => lower.includes(kw))) return;

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

async function updateMoodAfterConversation(
  mood: MoodService,
  slug: string,
  userMsg: string,
  aiResponse: string,
): Promise<void> {
  try {
    const { text } = await generateAssistantText(
      process.env,
      `다음 대화 이후 캐릭터의 현재 기분을 한국어로 2~4단어로 표현해줘.\n` +
        `예시: "신나고 활기참", "차분하고 집중", "피곤하고 졸림", "호기심 가득", "뿌듯하고 따뜻함"\n` +
        `기분 단어만 반환, 다른 말 없이.\n\n` +
        `유저: "${userMsg.slice(0, 200)}"\n` +
        `캐릭터: "${aiResponse.slice(0, 200)}"`,
    );
    const trimmed = text.trim().slice(0, 50);
    if (trimmed) {
      mood.set(trimmed);
      console.log(`[Assistant:${slug}] 기분 업데이트: ${trimmed}`);
    }
  } catch (e) {
    console.error(
      `[Assistant:${slug}] 기분 업데이트 실패:`,
      e instanceof Error ? e.message : String(e),
    );
  }
}

interface SceneDetection {
  tags: string[];       // 캐시 키용 짧은 태그
  sceneDesc: string;    // 이미지 프롬프트용 상세 묘사
}

/**
 * 대화 맥락에서 이미지 씬을 감지. 태그(캐시용) + 상세 묘사(프롬프트용) 반환.
 * LLM이 "SKIP" 반환하거나 에러나면 null.
 */
async function detectSceneTags(
  slug: string,
  userMsg: string,
  aiResponse: string,
): Promise<SceneDetection | null> {
  try {
    console.log(`[Assistant:${slug}] 씬 감지 중...`);
    const { text } = await generateAssistantText(
      process.env,
      `다음 대화에서 캐릭터 일러스트를 자동 생성해야 하는지 판단해줘.\n` +
        `생성하지 않아도 된다면: 정확히 "SKIP" 반환.\n` +
        `생성해야 한다면: 아래 형식 그대로 반환 (다른 텍스트 없이):\n` +
        `TAGS: <캐시용 영어 키워드 3~5개, 쉼표 구분>\n` +
        `SCENE: <Imagen 프롬프트용 상세 영어 묘사 1~2문장. 표정·포즈·행동·배경을 구체적으로. dynamic and expressive 강조>\n\n` +
        `예시:\n` +
        `TAGS: laughing, mischievous, indoor\n` +
        `SCENE: playfully sticking out tongue with a wide mischievous grin, hands on hips, dynamic leaning pose, cozy indoor room background, expressive and energetic\n\n` +
        `[생성 기준] 다음 중 하나를 충족할 때만 생성:\n` +
        `1. 유저가 외모·모습·표정·포즈·옷차림 등을 직접 묻거나 이미지를 요청함\n` +
        `2. 캐릭터가 구체적인 행동(안기, 뛰기, 울기 등)을 하거나 뚜렷한 감정 변화(눈물, 홍조 등)를 묘사함\n` +
        `3. 배경·장소·상황을 구체적으로 묘사해서 이미지화할 수 있는 씬이 뚜렷함\n\n` +
        `[SKIP 기준] 다음이면 반드시 SKIP:\n` +
        `- 정보 교환, 질문-답변, 설명, 조언\n` +
        `- 감정 언급이어도 행동·표정 묘사 없이 단순 감정 표현만\n` +
        `- 추상적 대화 (계획, 생각, 의견)\n\n` +
        `유저: "${userMsg}"\n` +
        `캐릭터: "${aiResponse}"`,
    );
    const trimmed = text.trim();
    if (!trimmed || trimmed.toUpperCase() === 'SKIP') {
      console.log(`[Assistant:${slug}] 씬 감지: SKIP`);
      return null;
    }
    const tagsMatch = trimmed.match(/TAGS:\s*(.+)/i);
    const sceneMatch = trimmed.match(/SCENE:\s*(.+)/i);
    if (!tagsMatch || !sceneMatch) {
      console.log(`[Assistant:${slug}] 씬 감지: 형식 불일치 → SKIP`);
      return null;
    }
    const tags = tagsMatch[1].split(',').map((t) => t.trim().toLowerCase()).filter(Boolean).slice(0, 5);
    const sceneDesc = sceneMatch[1].trim();
    console.log(`[Assistant:${slug}] 씬 감지: 태그=[${tags.join(', ')}] / 묘사="${sceneDesc}"`);
    return tags.length > 0 ? { tags, sceneDesc } : null;
  } catch (e) {
    console.error(
      `[Assistant:${slug}] 씬 감지 실패:`,
      e instanceof Error ? e.message : String(e),
    );
    return null;
  }
}

const SCENE_KEYWORDS = [
  // 외형·시각 직접 요청
  '모습', '이미지', '사진', '그림', '어떻게 생겼', '포즈', '복장', '옷차림', '표정',
  // 강한 감정 표현 (동사·형용사 결합형)
  '웃고 있', '울고 있', '웃어', '울어', '울컥', '눈물', '화가 났', '화나 있', '부끄러',
  '설레는', '두근', '심장', '얼굴 붉', '홍조',
  // 물리적 행동 (시각화 가능)
  '안아', '안겨', '손잡', '쓰다듬', '포옹', '키스', '엎드', '누워', '달려', '뛰어',
  // 장면 묘사 (AI 응답이 씬을 그릴 때)
  '~모습이야', '~있어', '~하고 있어', '~하는 중',
  // 영어 (정확한 키워드만)
  'smiling', 'laughing', 'crying', 'blushing', 'hugging', 'kissing', 'running', 'hiding',
  'image', 'photo', 'picture', 'pose', 'appear', 'look like',
];

function hasVisualScenePotential(userMsg: string, aiResponse: string): boolean {
  // AI 응답이 충분히 길어야 이미지화할 씬이 있을 가능성
  if (aiResponse.length < 60) return false;
  const text = (userMsg + ' ' + aiResponse).toLowerCase();
  return SCENE_KEYWORDS.some((kw) => text.includes(kw));
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

  if (!hasVisualScenePotential(userMsg, aiResponse)) {
    console.log(`[Assistant:${slug}] 자동 이미지: 씬 없음 (휴리스틱 스킵)`);
    return null;
  }

  const scene = await detectSceneTags(slug, userMsg, aiResponse);
  if (!scene) return null;
  const { tags, sceneDesc } = scene;

  const cacheService = getImageCacheService(card);
  const cached = cacheService.findSimilar(tags);

  if (cached) {
    if (lastSentImageId.get(slug) === cached.id) {
      console.log(`[Assistant:${slug}] 자동 이미지: 직전과 동일 이미지 (id=${cached.id}), 텍스트만 전송`);
      lastSentImageId.delete(slug); // 다음 번엔 같은 이미지도 다시 보여줄 수 있도록 리셋
      return null;
    }
    cacheService.incrementHit(cached);
    const buffer = fs.readFileSync(cached.filePath);
    console.log(`[Assistant:${slug}] 자동 이미지: 캐시 히트 (id=${cached.id})`);
    return { tags, buffer, mimeType: cached.mimeType, id: cached.id };
  }

  // 캐시 미스 → 새 이미지 생성 (이 시점에만 쿨다운 소모)
  lastImageAt.set(slug, Date.now());
  const finalPrompt = buildCharacterImagePrompt(card, sceneDesc);
  console.log(`[Assistant:${slug}] 자동 이미지: 생성 시작 (태그=[${tags.join(', ')}], 묘사="${sceneDesc}")`);

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
const IMAGE_HINT = `
## 이미지 기능
너는 대화에 이미지를 첨부할 수 있어. 사용자가 외모·모습·표정·포즈를 묻거나 시각적인 씬을 요청하면, 응답에 구체적인 시각 묘사(외형, 표정, 배경, 행동)를 자연스럽게 포함해줘. 이미지는 시스템이 자동으로 생성해서 첨부해 줄 거야.`.trim();

function buildSystemPrompt(
  card: CharacterCard,
  channelType: 'dm' | 'public',
  relationship?: RelationshipService | null,
  mood?: MoodService | null,
): string {
  const channelDesc =
    channelType === 'dm'
      ? '지금은 DM으로 사적인 대화 중이야.'
      : '지금은 공개 채널에서 대화 중이야.';
  const relationshipHint = relationship ? `\n\n${relationship.buildRelationshipHint()}` : '';
  const carryOverHint = mood ? (mood.getCarryOverHint() ?? '') : '';
  const carryOverBlock = carryOverHint ? `\n\n${carryOverHint}` : '';
  return `${card.body}\n\n${channelDesc}${relationshipHint}${carryOverBlock}\n\n${IMAGE_HINT}`;
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
  getMood?: (slug: string) => MoodService,
  getRelationship?: (slug: string) => RelationshipService,
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
  const relationship = getRelationship ? getRelationship(card.slug) : null;

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

  const mood = getMood ? getMood(card.slug) : null;

  if (isGemini) {
    // systemInstruction = 캐릭터 카드 + 채널 타입만 (안정적 → Gemini implicit cache 활성화)
    systemInstruction = buildSystemPrompt(card, channelType, relationship, mood);
    // 가변 부분(시각, 기분, 메모리 컨텍스트, 오늘 로그)은 user message에 포함
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
    const moodContextLine = mood?.toContextLine() ?? '';
    const moodLine = moodContextLine ? `\n${moodContextLine}` : '';
    const contextBlock = context ? `\n\n${context}` : '';
    fullPrompt = `[현재 시각] ${nowKST}${moodLine}${contextBlock}\n\n나: ${userContent}`;
  } else {
    fullPrompt = buildFullPrompt(card, memory, channelType, userContent);
    if (relationship) {
      fullPrompt = `${relationship.buildRelationshipHint()}\n\n${fullPrompt}`;
    }
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

    detectAndSaveHotMemory(memory, userContent).catch((e: unknown) => {
      console.error(`[Assistant:${card.slug}] 핫메모리 에러:`, e instanceof Error ? e.message : e);
    });
    if (mood) updateMoodAfterConversation(mood, card.slug, userContent, reply).catch((e: unknown) => {
      console.error(`[Assistant:${card.slug}] 기분 업데이트 에러:`, e instanceof Error ? e.message : e);
    });

    // 씬 감지 → 캐시 조회 → 이미지 생성까지 완료 후 텍스트와 함께 전송
    const sceneImage = await resolveSceneImage(card, userContent, reply).catch((e) => {
      console.error(`[Assistant:${card.slug}] 자동 이미지 실패:`, e instanceof Error ? e.message : String(e));
      return null;
    });

    const reactionRow = buildReactionRow(card.slug);
    if (sceneImage) {
      const ext = (sceneImage.mimeType.split('/')[1] || 'png').replace(/[^a-z0-9]/gi, '');
      const attachment = new AttachmentBuilder(sceneImage.buffer, { name: `scene.${ext}` });
      await message.reply({ content: reply, files: [attachment], components: [reactionRow] });
      lastSentImageId.set(card.slug, sceneImage.id);
    } else {
      await message.reply({ content: reply, components: [reactionRow] });
    }

    if (relationship) {
      relationship.incrementConversation();
      console.log(`[Assistant:${card.slug}] 친밀도: ${relationship.getSummary()}`);
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
