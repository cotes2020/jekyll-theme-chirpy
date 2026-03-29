import type { StreakState, ToolboxUserData } from './types';

export const USER_DATA_KEY = 'toolbox_user_data';

function emptyUserData(): ToolboxUserData {
  return {
    achievements: [],
    badges: [],
    progress: {},
    streaks: {},
    totalExp: 0,
    level: 0,
  };
}

function normalizeStreakState(raw: unknown): StreakState | null {
  if (!raw || typeof raw !== 'object') return null;
  const o = raw as Record<string, unknown>;
  const current = Number(o.current);
  const longest = Number(o.longest);
  const lastActivityDate = typeof o.lastActivityDate === 'string' ? o.lastActivityDate : '';
  if (!Number.isFinite(current) || !Number.isFinite(longest)) return null;
  return { current, longest, lastActivityDate };
}

function mergeUserData(parsed: Partial<ToolboxUserData>): ToolboxUserData {
  const d = emptyUserData();
  return {
    achievements: Array.isArray(parsed.achievements) ? [...parsed.achievements] : d.achievements,
    badges: Array.isArray(parsed.badges) ? [...parsed.badges] : d.badges,
    progress:
      parsed.progress && typeof parsed.progress === 'object' && !Array.isArray(parsed.progress)
        ? { ...parsed.progress }
        : d.progress,
    streaks: (() => {
      if (!parsed.streaks || typeof parsed.streaks !== 'object' || Array.isArray(parsed.streaks)) {
        return d.streaks;
      }
      const out: Record<string, StreakState> = {};
      for (const [k, v] of Object.entries(parsed.streaks)) {
        const n = normalizeStreakState(v);
        if (n) out[k] = n;
      }
      return out;
    })(),
    totalExp: typeof parsed.totalExp === 'number' ? parsed.totalExp : 0,
    level: typeof parsed.level === 'number' ? parsed.level : 0,
  };
}

export function loadUserData(): ToolboxUserData {
  try {
    const raw = localStorage.getItem(USER_DATA_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<ToolboxUserData>;
      return mergeUserData(parsed);
    }
  } catch {
    /* ignore */
  }
  return emptyUserData();
}

export function saveUserData(data: ToolboxUserData): void {
  try {
    localStorage.setItem(USER_DATA_KEY, JSON.stringify(data));
  } catch {
    /* ignore */
  }
}

/** EXP를 더하고 저장, 레벨업 여부 반환 */
export function addExp(amount: number): { newLevel: number; leveledUp: boolean } {
  const data = loadUserData();
  const oldLevel = data.level;
  data.totalExp = (data.totalExp || 0) + amount;
  // level = floor(sqrt(totalExp / 50))
  data.level = Math.floor(Math.sqrt(data.totalExp / 50));
  saveUserData(data);
  return { newLevel: data.level, leveledUp: data.level > oldLevel };
}
