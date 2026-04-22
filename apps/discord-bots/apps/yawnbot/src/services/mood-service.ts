/**
 * MoodService — 캐릭터별 기분/상태 관리
 *
 * 저장 위치: characters/<slug>/memory/mood.json
 * 구조: { mood: string, updatedAt: string }
 *
 * 대화 후 경량 LLM 호출로 자동 업데이트.
 * 시스템 컨텍스트에 주입돼 AI 응답 톤에 반영됨.
 */
import fs from 'fs';
import path from 'path';

export interface MoodState {
  mood: string;
  updatedAt: string;
}

export class MoodService {
  readonly slug: string;
  private filePath: string;
  private state: MoodState | null = null;

  constructor(memoRepoPath: string, slug: string) {
    this.slug = slug;
    this.filePath = path.join(memoRepoPath, 'characters', slug, 'memory', 'mood.json');
    this._load();
  }

  private _load(): void {
    try {
      if (fs.existsSync(this.filePath)) {
        this.state = JSON.parse(fs.readFileSync(this.filePath, 'utf-8')) as MoodState;
      }
    } catch {
      this.state = null;
    }
  }

  private _save(): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(this.state, null, 2), 'utf-8');
  }

  get(): MoodState | null {
    return this.state;
  }

  set(mood: string): void {
    this.state = { mood: mood.trim(), updatedAt: new Date().toISOString() };
    this._save();
  }

  /** 시스템 컨텍스트용 한 줄 요약. 기분 없으면 빈 문자열. */
  toContextLine(): string {
    if (!this.state?.mood) return '';
    return `[현재 기분: ${this.state.mood}]`;
  }

  /**
   * 어제 기분이 남아 있으면 carry-over 힌트 반환.
   * 오늘 이미 기분이 업데이트됐거나 48시간 이상 지났으면 null.
   */
  /**
   * 어제 기분 캐리오버. 오늘 이미 업데이트됐거나 48시간 초과면 null.
   * - `systemHint`: system prompt에 주입할 문자열
   * - `displayMood`: /프로필 등 UI 표시용 기분 단어만
   */
  getCarryOver(): { systemHint: string; displayMood: string } | null {
    if (!this.state?.mood || !this.state?.updatedAt) return null;
    const KST_OFFSET = 9 * 60 * 60 * 1000;
    const todayStr = new Date(Date.now() + KST_OFFSET).toISOString().slice(0, 10);
    const updatedStr = new Date(new Date(this.state.updatedAt).getTime() + KST_OFFSET).toISOString().slice(0, 10);
    if (updatedStr === todayStr) return null;
    const diffMs = Date.now() - new Date(this.state.updatedAt).getTime();
    if (diffMs > 48 * 60 * 60 * 1000) return null;
    return {
      systemHint: `[어제 기분: ${this.state.mood}] 어제의 분위기가 오늘 대화 초반에 미묘하게 남아 있어.`,
      displayMood: this.state.mood,
    };
  }

  /** @deprecated `getCarryOver().systemHint` 사용 */
  getCarryOverHint(): string | null {
    return this.getCarryOver()?.systemHint ?? null;
  }
}
