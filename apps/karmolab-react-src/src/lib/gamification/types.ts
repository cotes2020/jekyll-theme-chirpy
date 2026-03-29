/** toolbox_user_data.streaks 항목 (로컬 캘린더 YYYY-MM-DD) */
export interface StreakState {
  current: number;
  longest: number;
  lastActivityDate: string;
}

/** localStorage toolbox_user_data 전체 형태 (기존 필드 유지) */
export interface ToolboxUserData {
  achievements: string[];
  badges: string[];
  progress: Record<string, number>;
  streaks: Record<string, StreakState>;
  totalExp: number;
  level: number;
}

/** 향후 트랙 메타(이름·색) 확장용 */
export interface TrackMeta {
  id: string;
  label: string;
  color?: string;
}
