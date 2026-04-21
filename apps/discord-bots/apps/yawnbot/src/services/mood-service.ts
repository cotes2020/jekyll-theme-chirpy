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
}
