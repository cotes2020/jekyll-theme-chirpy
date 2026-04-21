/**
 * proactive.ts — 먼저 말 걸기
 * - 봇 시작 시 DM으로 기상 메시지 (해당 DM의 활성 캐릭터 카드로)
 * - 매일 아침 ASSISTANT_MORNING_HOUR 시 (KST) DM으로 인사
 * - 매일 저녁 ASSISTANT_EVENING_HOUR 시 (KST) 체크인
 * - 자발적 메시지: 랜덤 인터벌로 활성 시간대에 먼저 말 걸기
 *
 * 시스템 프롬프트는 캐릭터 카드 본문. 페르소나 하드코딩 없음.
 */
import type { Client, DMChannel } from 'discord.js';
import { generateAssistantText } from 'karmolab-ai/node';
import type { CharacterService, CharacterCard } from '../services/character-service';
import { CharacterService as CSHelper } from '../services/character-service';
import type { MemoryService } from '../services/memory-service';
import type { ScheduleService } from '../services/schedule-service';
import type { MoodService } from '../services/mood-service';

let morningTimer: ReturnType<typeof setTimeout> | null = null;
let eveningTimer: ReturnType<typeof setTimeout> | null = null;
let reminderTimer: ReturnType<typeof setInterval> | null = null;
let spontaneousTimer: ReturnType<typeof setTimeout> | null = null;

function msUntilNextKSTHour(targetHour: number): number {
  const now = new Date();
  const kst = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
  const kstHour = kst.getHours();
  const kstMin = kst.getMinutes();
  const kstSec = kst.getSeconds();

  let hoursUntil = targetHour - kstHour;
  if (hoursUntil <= 0) hoursUntil += 24;

  const minsUntil = hoursUntil * 60 - kstMin;
  const secsUntil = minsUntil * 60 - kstSec;
  return secsUntil * 1000;
}

/**
 * 랜덤 인터벌 후 목표 시각이 KST 활성 시간대(activeStart~activeEnd)에 들어오도록 ms를 반환.
 * 활성 시간대 밖으로 벗어나면 다음 activeStart + 랜덤 오프셋으로 밀어줌.
 */
function msUntilNextSpontaneous(
  activeStart: number,
  activeEnd: number,
  minMs: number,
  maxMs: number,
): number {
  const randomMs = minMs + Math.random() * (maxMs - minMs);
  const targetUTC = new Date(Date.now() + randomMs);
  const targetKST = new Date(targetUTC.toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
  const targetHour = targetKST.getHours();

  if (targetHour >= activeStart && targetHour < activeEnd) {
    return randomMs;
  }

  // 활성 시간대 밖 → 다음 activeStart까지 남은 ms + 활성 창 안 랜덤 오프셋
  const kstNow = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
  let hoursUntilActive = activeStart - kstNow.getHours();
  if (hoursUntilActive <= 0) hoursUntilActive += 24;

  const msUntilActive =
    hoursUntilActive * 3600_000 -
    kstNow.getMinutes() * 60_000 -
    kstNow.getSeconds() * 1000;

  const windowMs = (activeEnd - activeStart) * 3600_000;
  const offsetMs = Math.random() * windowMs;

  return msUntilActive + offsetMs;
}

function dmChannelKey(userId: string): string {
  return CSHelper.channelKey({ isDM: true, userId, channelId: '' });
}

function formatKSTDate(): { dateStr: string; dayStr: string; timeStr: string } {
  const kstNow = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
  const dateStr = `${kstNow.getFullYear()}년 ${kstNow.getMonth() + 1}월 ${kstNow.getDate()}일`;
  const dayNames = ['일', '월', '화', '수', '목', '금', '토'];
  const dayStr = dayNames[kstNow.getDay()] + '요일';
  const timeStr = `${String(kstNow.getHours()).padStart(2, '0')}:${String(kstNow.getMinutes()).padStart(2, '0')}`;
  return { dateStr, dayStr, timeStr };
}

function resolveDMCard(
  characterService: CharacterService,
  userId: string,
): CharacterCard | null {
  return characterService.resolveCard(dmChannelKey(userId));
}

async function sendMorningGreeting(
  client: Client,
  characterService: CharacterService,
): Promise<void> {
  const userId = process.env.ASSISTANT_USER_ID?.trim();
  if (!userId) return;
  if (!process.env.GEMINI_API_KEY?.trim()) return;

  const card = resolveDMCard(characterService, userId);
  if (!card) {
    console.warn('[Proactive] 활성 캐릭터 카드 없음 — 아침 인사 스킵');
    return;
  }

  try {
    const user = await client.users.fetch(userId);
    const dmChannel = (await user.createDM()) as DMChannel;

    const { dateStr, dayStr } = formatKSTDate();

    const prompt =
      `오늘은 ${dateStr} ${dayStr}이야.\n` +
      `아침 인사 메시지를 한국어로, 짧고 따뜻하게 보내줘.\n` +
      `날짜와 요일을 자연스럽게 언급하고, 오늘 하루도 잘 보내길 바란다는 마음을 담아줘.\n` +
      `2-3문장 이내로.`;

    const { text } = await generateAssistantText(process.env, prompt, { systemInstruction: card.body });

    await dmChannel.send(text.slice(0, 1900));
    console.log(`[Proactive:${card.slug}] 아침 인사 전송 완료`);
  } catch (e: unknown) {
    console.error(
      `[Proactive:${card.slug}] 아침 인사 실패:`,
      e instanceof Error ? e.message : e,
    );
  }
}

function scheduleMorning(
  client: Client,
  characterService: CharacterService,
  targetHour: number,
): void {
  const delay = msUntilNextKSTHour(targetHour);
  console.log(`[Proactive] 다음 아침 인사까지 ${Math.round(delay / 60000)}분 대기`);

  morningTimer = setTimeout(async () => {
    await sendMorningGreeting(client, characterService);
    scheduleMorning(client, characterService, targetHour);
  }, delay);
}

async function sendEveningCheckin(
  client: Client,
  characterService: CharacterService,
  getMemory: (slug: string) => MemoryService,
): Promise<void> {
  const userId = process.env.ASSISTANT_USER_ID?.trim();
  if (!userId) return;

  const card = resolveDMCard(characterService, userId);
  if (!card) {
    console.warn('[Proactive] 활성 캐릭터 카드 없음 — 저녁 체크인 스킵');
    return;
  }

  try {
    const user = await client.users.fetch(userId);
    const dmChannel = (await user.createDM()) as DMChannel;

    const memory = getMemory(card.slug);
    const context = memory.buildContext(3000);
    const { dateStr, dayStr } = formatKSTDate();

    const prompt =
      `오늘은 ${dateStr} ${dayStr}이야.\n` +
      `저녁 체크인 시간이야. 오늘 하루 어땠는지 가볍게 물어봐줘.\n` +
      `따뜻하고 편안한 말투로, 2-3문장 이내로. 한국어로.\n` +
      (context ? `\n[오늘 기억]\n${context}` : '');

    const { text } = await generateAssistantText(process.env, prompt, { systemInstruction: card.body });
    await dmChannel.send(text.slice(0, 1900));
    console.log(`[Proactive:${card.slug}] 저녁 체크인 전송 완료`);
  } catch (e: unknown) {
    console.error(
      `[Proactive] 저녁 체크인 실패:`,
      e instanceof Error ? e.message : e,
    );
  }
}

function scheduleEvening(
  client: Client,
  characterService: CharacterService,
  getMemory: (slug: string) => MemoryService,
  targetHour: number,
): void {
  const delay = msUntilNextKSTHour(targetHour);
  console.log(`[Proactive] 다음 저녁 체크인까지 ${Math.round(delay / 60000)}분 대기`);

  eveningTimer = setTimeout(async () => {
    await sendEveningCheckin(client, characterService, getMemory);
    scheduleEvening(client, characterService, getMemory, targetHour);
  }, delay);
}

export async function sendStartupGreeting(
  client: Client,
  characterService: CharacterService,
  getMemory: (slug: string) => MemoryService,
): Promise<void> {
  const userId = process.env.ASSISTANT_USER_ID?.trim();
  if (!userId) return;

  const card = resolveDMCard(characterService, userId);
  if (!card) {
    console.warn('[Proactive] 활성 캐릭터 카드 없음 — 기상 메시지 스킵');
    return;
  }

  try {
    const user = await client.users.fetch(userId);
    const dmChannel = (await user.createDM()) as DMChannel;

    const memory = getMemory(card.slug);
    const context = memory.buildContext();
    const { dateStr, dayStr, timeStr } = formatKSTDate();

    const prompt =
      `방금 봇이 시작됐어. ${dateStr} ${dayStr} ${timeStr}.\n` +
      `마치 잠에서 깨어난 것처럼, 짧고 자연스럽게 기상 메시지를 보내줘.\n` +
      `2-3문장 이내로. 한국어로.\n` +
      (context ? `\n[기억]\n${context}` : '');

    const { text } = await generateAssistantText(process.env, prompt, { systemInstruction: card.body });
    await dmChannel.send(text.slice(0, 1900));
    console.log(`[Proactive:${card.slug}] 기상 메시지 전송 완료`);
  } catch (e: unknown) {
    console.error(
      `[Proactive:${card.slug}] 기상 메시지 실패:`,
      e instanceof Error ? e.message : e,
    );
  }
}

export function startProactive(
  client: Client,
  characterService: CharacterService,
  getMemory: (slug: string) => MemoryService,
): void {
  const userId = process.env.ASSISTANT_USER_ID?.trim();
  if (!userId) {
    console.warn('[Proactive] ASSISTANT_USER_ID 미설정 — 아침/저녁 인사 비활성화');
    return;
  }

  const morningHour = parseInt(process.env.ASSISTANT_MORNING_HOUR || '8', 10);
  scheduleMorning(client, characterService, morningHour);

  const eveningHour = parseInt(process.env.ASSISTANT_EVENING_HOUR || '21', 10);
  scheduleEvening(client, characterService, getMemory, eveningHour);
}

export function startScheduleReminder(
  client: Client,
  characterService: CharacterService,
  getSchedule: (slug: string) => ScheduleService,
): void {
  const userId = process.env.ASSISTANT_USER_ID?.trim();
  if (!userId) return;

  reminderTimer = setInterval(async () => {
    const slugs = characterService.listCharacters();
    for (const slug of slugs) {
      try {
        const schedule = getSchedule(slug);
        const pending = schedule.getPendingReminders();
        for (const entry of pending) {
          schedule.markNotified(entry.id);
          const unixSec = Math.floor(new Date(entry.datetime).getTime() / 1000);
          const user = await client.users.fetch(userId);
          const dm = (await user.createDM()) as DMChannel;
          await dm.send(`⏰ **일정 알림**\n**${entry.title}** — <t:${unixSec}:f>`);
          console.log(`[Reminder:${slug}] 알림 전송: ${entry.title}`);
        }
      } catch (e: unknown) {
        console.error(`[Reminder:${slug}]`, e instanceof Error ? e.message : e);
      }
    }
  }, 60_000);
}

function buildTimeOfDayHint(kstHour: number): string {
  if (kstHour < 12) return '오전이라 조금 맑은 기분이야.';
  if (kstHour < 15) return '점심 먹고 약간 나른한 시간이야.';
  if (kstHour < 18) return '오후 시간, 하루가 슬슬 기울어 가고 있어.';
  if (kstHour < 21) return '저녁 시간이야. 하루가 거의 끝나가고 있어.';
  return '밤이 깊어가고 있어. 조용하고 느긋한 시간이야.';
}

async function sendSpontaneousMessage(
  client: Client,
  characterService: CharacterService,
  getMemory: (slug: string) => MemoryService,
  getMood?: (slug: string) => MoodService,
  getSchedule?: (slug: string) => ScheduleService,
): Promise<void> {
  const userId = process.env.ASSISTANT_USER_ID?.trim();
  if (!userId) return;

  const card = resolveDMCard(characterService, userId);
  if (!card) return;

  try {
    const user = await client.users.fetch(userId);
    const dmChannel = (await user.createDM()) as DMChannel;

    const memory = getMemory(card.slug);
    const context = memory.buildContext(2000);
    const { dateStr, dayStr, timeStr } = formatKSTDate();
    const kstHour = parseInt(timeStr.split(':')[0], 10);
    const timeHint = buildTimeOfDayHint(kstHour);

    // 기분 라인
    const moodLine = getMood ? (getMood(card.slug).toContextLine() || '') : '';

    // 다가오는 일정 (24시간 이내)
    let scheduleHint = '';
    if (getSchedule) {
      try {
        const pending = getSchedule(card.slug).getPendingReminders();
        if (pending.length > 0) {
          const next = pending[0];
          const unixSec = Math.floor(new Date(next.datetime).getTime() / 1000);
          scheduleHint = `\n곧 일정이 있어: "${next.title}" (<t:${unixSec}:R>)`;
        }
      } catch { /* ignore */ }
    }

    const prompt =
      `지금은 ${dateStr} ${dayStr} ${timeStr}이야. ${timeHint}\n` +
      (moodLine ? `${moodLine}\n` : '') +
      `너는 갑자기 말을 걸고 싶어졌어.\n` +
      `최근 기억을 참고해서 자연스럽게 한마디 건네줘 — 최근 대화 주제 후속, 오늘 하루 어떤지, 문득 떠오른 생각, 궁금한 것 등.\n` +
      `1-2문장, 한국어로.\n` +
      (scheduleHint ? `${scheduleHint}\n` : '') +
      (context ? `\n[최근 기억]\n${context}` : '');

    const { text } = await generateAssistantText(process.env, prompt, { systemInstruction: card.body });
    await dmChannel.send(text.slice(0, 1900));
    console.log(`[Proactive:${card.slug}] 자발적 메시지 전송 완료`);
  } catch (e: unknown) {
    console.error(
      `[Proactive] 자발적 메시지 실패:`,
      e instanceof Error ? e.message : e,
    );
  }
}

function scheduleSpontaneous(
  client: Client,
  characterService: CharacterService,
  getMemory: (slug: string) => MemoryService,
  activeStart: number,
  activeEnd: number,
  minMs: number,
  maxMs: number,
  getMood?: (slug: string) => MoodService,
  getSchedule?: (slug: string) => ScheduleService,
): void {
  const delay = msUntilNextSpontaneous(activeStart, activeEnd, minMs, maxMs);
  console.log(`[Proactive] 다음 자발적 메시지까지 ${Math.round(delay / 60000)}분 대기`);

  spontaneousTimer = setTimeout(async () => {
    await sendSpontaneousMessage(client, characterService, getMemory, getMood, getSchedule);
    scheduleSpontaneous(client, characterService, getMemory, activeStart, activeEnd, minMs, maxMs, getMood, getSchedule);
  }, delay);
}

export function startSpontaneous(
  client: Client,
  characterService: CharacterService,
  getMemory: (slug: string) => MemoryService,
  getMood?: (slug: string) => MoodService,
  getSchedule?: (slug: string) => ScheduleService,
): void {
  const userId = process.env.ASSISTANT_USER_ID?.trim();
  if (!userId) return;

  const enabled = process.env.ASSISTANT_SPONTANEOUS_ENABLED;
  if (enabled === '0' || enabled === '끔') return;

  const activeStart = parseInt(process.env.ASSISTANT_SPONTANEOUS_ACTIVE_START || '10', 10);
  const activeEnd = parseInt(process.env.ASSISTANT_SPONTANEOUS_ACTIVE_END || '22', 10);
  const minMs = parseInt(
    process.env.ASSISTANT_SPONTANEOUS_MIN_MS || String(3 * 3600_000),
    10,
  );
  const maxMs = parseInt(
    process.env.ASSISTANT_SPONTANEOUS_MAX_MS || String(6 * 3600_000),
    10,
  );

  scheduleSpontaneous(client, characterService, getMemory, activeStart, activeEnd, minMs, maxMs, getMood, getSchedule);
}

export function stopProactive(): void {
  if (morningTimer) { clearTimeout(morningTimer); morningTimer = null; }
  if (eveningTimer) { clearTimeout(eveningTimer); eveningTimer = null; }
  if (reminderTimer) { clearInterval(reminderTimer); reminderTimer = null; }
  if (spontaneousTimer) { clearTimeout(spontaneousTimer); spontaneousTimer = null; }
}
