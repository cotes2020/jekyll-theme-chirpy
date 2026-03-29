import type { TrackMeta } from './types';

/**
 * 스트릭 마일스톤 도전과제 id — [apps/karmolab/js/widgets/user.js] DEFS.achievements와 동일 문자열 유지
 */
export const STREAK_ACHIEVEMENT_IDS = {
  streak_first: 'streak_first',
  streak_7: 'streak_7',
  streak_30: 'streak_30',
  streak_100: 'streak_100',
} as const;

export type StreakAchievementId =
  (typeof STREAK_ACHIEVEMENT_IDS)[keyof typeof STREAK_ACHIEVEMENT_IDS];

export const STREAK_ACHIEVEMENT_META: Record<StreakAchievementId, { title: string }> = {
  streak_first: { title: '첫 줄기' },
  streak_7: { title: '7일 연속' },
  streak_30: { title: '30일 연속' },
  streak_100: { title: '100일 연속' },
};

/** 데모·플래너 연동용 기본 트랙 */
export const DEFAULT_TRACKS: readonly TrackMeta[] = [
  { id: 'daily_review', label: '일일 리뷰' },
  { id: 'exercise', label: '운동' },
] as const;
