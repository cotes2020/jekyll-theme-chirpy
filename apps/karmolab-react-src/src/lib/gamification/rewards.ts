import {
  STREAK_ACHIEVEMENT_IDS,
  STREAK_ACHIEVEMENT_META,
  type StreakAchievementId,
} from './constants';
import { hadAnyStreakActivity } from './streakEngine';
import type { StreakState, ToolboxUserData } from './types';

/**
 * 스트릭 갱신 직후 마일스톤 도전과제 반영 (achievements 배열에 id 추가)
 */
export function applyStreakMilestones(
  dataBeforeRecord: ToolboxUserData,
  newState: StreakState
): { data: ToolboxUserData; unlocked: StreakAchievementId[] } {
  const unlocked: StreakAchievementId[] = [];
  const achievements = [...dataBeforeRecord.achievements];

  const push = (id: StreakAchievementId) => {
    if (!achievements.includes(id)) {
      achievements.push(id);
      unlocked.push(id);
    }
  };

  const firstEver = !hadAnyStreakActivity(dataBeforeRecord);
  if (firstEver && newState.current === 1) {
    push(STREAK_ACHIEVEMENT_IDS.streak_first);
  }
  if (newState.current === 7) push(STREAK_ACHIEVEMENT_IDS.streak_7);
  if (newState.current === 30) push(STREAK_ACHIEVEMENT_IDS.streak_30);
  if (newState.current === 100) push(STREAK_ACHIEVEMENT_IDS.streak_100);

  return { data: { ...dataBeforeRecord, achievements }, unlocked };
}

export function toastUnlockedAchievements(unlocked: StreakAchievementId[]): void {
  const tb = typeof window !== 'undefined' ? window.Toolbox : undefined;
  for (const id of unlocked) {
    const title = STREAK_ACHIEVEMENT_META[id]?.title ?? id;
    tb?.showToast?.(`도전과제 달성: ${title}`, 'success');
  }
}
