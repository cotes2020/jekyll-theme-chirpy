/**
 * EXP/Level 계산 엔진
 * 레벨 공식: level = floor(sqrt(totalExp / 50))
 * EXP 보상:
 *   - 스트릭 완료: 30 EXP (+ 5 * streak_days bonus)
 *   - 칸반 태스크 완료: 20 EXP
 *   - 칸반 태스크 진행 중으로 이동: 10 EXP
 */

export const EXP_REWARDS = {
  STREAK_COMPLETE: 30,
  STREAK_BONUS_PER_DAY: 5,
  TASK_COMPLETE: 20,
  TASK_IN_PROGRESS: 10,
} as const;

/** EXP으로 레벨 계산 */
export function calcLevel(totalExp: number): number {
  return Math.floor(Math.sqrt(totalExp / 50));
}

/** 현재 레벨에서 다음 레벨까지 필요한 EXP 범위 */
export function getLevelRange(level: number): { min: number; max: number } {
  const min = level * level * 50;
  const max = (level + 1) * (level + 1) * 50;
  return { min, max };
}

/** 현재 레벨 EXP 진행률 (0~1) */
export function getLevelProgress(totalExp: number): number {
  const level = calcLevel(totalExp);
  const { min, max } = getLevelRange(level);
  if (max === min) return 1;
  return (totalExp - min) / (max - min);
}

/** 레벨 이름 / 칭호 */
export function getLevelTitle(level: number): string {
  if (level < 5) return '🌱 모험의 시작';
  if (level < 10) return '⚔️ 훈련생';
  if (level < 20) return '🛡️ 전사';
  if (level < 35) return '🔥 영웅';
  if (level < 50) return '💎 전설';
  return '👑 신화';
}
