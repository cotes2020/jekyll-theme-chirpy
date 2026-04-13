/**
 * proactive.ts — 먼저 말 걸기
 * - 매일 아침 ASSISTANT_MORNING_HOUR 시 (KST) DM으로 인사
 */
import type { Client, DMChannel } from 'discord.js';
import { generateBlobTextFromEnvWithOptions } from 'karmolab-ai/node';

let morningTimer: ReturnType<typeof setTimeout> | null = null;

function getKSTHour(): number {
  const kst = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
  return kst.getHours();
}

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

async function sendMorningGreeting(client: Client): Promise<void> {
  const userId = process.env.ASSISTANT_USER_ID?.trim();
  if (!userId) return;
  if (!process.env.GEMINI_API_KEY?.trim()) return;

  try {
    const user = await client.users.fetch(userId);
    const dmChannel = await user.createDM() as DMChannel;

    const kstNow = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
    const dateStr = `${kstNow.getFullYear()}년 ${kstNow.getMonth() + 1}월 ${kstNow.getDate()}일`;
    const dayNames = ['일', '월', '화', '수', '목', '금', '토'];
    const dayStr = dayNames[kstNow.getDay()] + '요일';

    const prompt =
      `너는 mascari4615의 개인 AI 비서야.\n` +
      `오늘은 ${dateStr} ${dayStr}이야.\n` +
      `아침 인사 메시지를 한국어로, 짧고 따뜻하게 보내줘.\n` +
      `날짜와 요일을 자연스럽게 언급하고, 오늘 하루도 잘 보내길 바란다는 마음을 담아줘.\n` +
      `2-3문장 이내로.`;

    const { text } = await generateBlobTextFromEnvWithOptions(process.env, prompt, {
      surface: 'inherit',
    });

    await dmChannel.send(text.slice(0, 1900));
    console.log('[Proactive] 아침 인사 전송 완료');
  } catch (e: unknown) {
    console.error('[Proactive] 아침 인사 실패:', e instanceof Error ? e.message : e);
  }
}

function scheduleMorning(client: Client, targetHour: number): void {
  const delay = msUntilNextKSTHour(targetHour);
  console.log(`[Proactive] 다음 아침 인사까지 ${Math.round(delay / 60000)}분 대기`);

  morningTimer = setTimeout(async () => {
    await sendMorningGreeting(client);
    // 다음 날 같은 시각 예약
    scheduleMorning(client, targetHour);
  }, delay);
}

export function startProactive(client: Client): void {
  const userId = process.env.ASSISTANT_USER_ID?.trim();
  if (!userId) {
    console.warn('[Proactive] ASSISTANT_USER_ID 미설정 — 아침 인사 비활성화');
    return;
  }

  const morningHour = parseInt(process.env.ASSISTANT_MORNING_HOUR || '8', 10);
  scheduleMorning(client, morningHour);
}

export function stopProactive(): void {
  if (morningTimer) {
    clearTimeout(morningTimer);
    morningTimer = null;
  }
}
