export type { StreakState, ToolboxUserData, TrackMeta } from './types';
export {
  DEFAULT_TRACKS,
  STREAK_ACHIEVEMENT_IDS,
  STREAK_ACHIEVEMENT_META,
} from './constants';
export { loadUserData, saveUserData, addExp, USER_DATA_KEY } from './storage';
export { localDateString, recordActivity, hadAnyStreakActivity } from './streakEngine';
export { applyStreakMilestones, toastUnlockedAchievements } from './rewards';
export { calcLevel, getLevelRange, getLevelProgress, getLevelTitle, EXP_REWARDS } from './expEngine';

import { applyStreakMilestones, toastUnlockedAchievements } from './rewards';
import { loadUserData, saveUserData, addExp } from './storage';
import { recordActivity as applyRecordActivity } from './streakEngine';
import { EXP_REWARDS } from './expEngine';
import type { ToolboxUserData } from './types';

/**
 * 플래너·칸반·캘린더에서 호출: 트랙별 오늘 활동 기록 → streak 갱신 → 마일스톤 도전과제 → 저장 → 토스트
 */
export function recordStreakActivity(trackId: string, activityDate?: string): void {
  const before = loadUserData();
  const result = applyRecordActivity(before, trackId, activityDate);
  if (!result.changed || !result.newState) return;

  const { data: withAch, unlocked } = applyStreakMilestones(before, result.newState);
  const currentStreak = result.newState.current ?? 1;
  const merged: ToolboxUserData = {
    ...withAch,
    streaks: result.data.streaks,
  };
  saveUserData(merged);
  toastUnlockedAchievements(unlocked);

  // EXP 보상
  const bonusExp = EXP_REWARDS.STREAK_BONUS_PER_DAY * Math.min(currentStreak, 10); // 최대 10일 보너스
  const { leveledUp, newLevel } = addExp(EXP_REWARDS.STREAK_COMPLETE + bonusExp);
  if (leveledUp) {
    setTimeout(() => {
      alert(`🎉 레벨업! 레벨 ${newLevel}이 되었습니다!`);
    }, 100);
  }
}
