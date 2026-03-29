import type { StreakState, ToolboxUserData } from './types';

/** 로컬 캘린더 기준 YYYY-MM-DD */
export function localDateString(d: Date = new Date()): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function parseLocalDate(s: string): Date {
  const [y, m, d] = s.split('-').map(Number);
  return new Date(y, m - 1, d);
}

/** prev가 today의 전날(어제)인지 */
function isYesterday(prev: string, today: string): boolean {
  const a = parseLocalDate(prev);
  const b = parseLocalDate(today);
  const diff = (b.getTime() - a.getTime()) / 86400000;
  return diff === 1;
}

function emptyStreak(): StreakState {
  return { current: 0, longest: 0, lastActivityDate: '' };
}

export function hadAnyStreakActivity(data: ToolboxUserData): boolean {
  return Object.values(data.streaks).some((v) => v.lastActivityDate);
}

/**
 * 하루에 한 번만 streak 증가. 같은 날 재호출 시 changed: false.
 */
export function recordActivity(
  data: ToolboxUserData,
  trackId: string,
  activityDate?: string
): { data: ToolboxUserData; changed: boolean; newState?: StreakState; previousState?: StreakState } {
  const today = activityDate ?? localDateString(new Date());
  const prev = data.streaks[trackId] ?? emptyStreak();

  if (prev.lastActivityDate === today) {
    return { data, changed: false };
  }

  const streaks = { ...data.streaks };
  let current: number;
  let longest: number;

  if (!prev.lastActivityDate) {
    current = 1;
    longest = Math.max(prev.longest, 1);
  } else if (isYesterday(prev.lastActivityDate, today)) {
    current = prev.current + 1;
    longest = Math.max(prev.longest, current);
  } else {
    current = 1;
    longest = Math.max(prev.longest, 1);
  }

  const newState: StreakState = {
    current,
    longest,
    lastActivityDate: today,
  };
  streaks[trackId] = newState;

  return {
    data: { ...data, streaks },
    changed: true,
    newState,
    previousState: prev.lastActivityDate ? prev : undefined,
  };
}
