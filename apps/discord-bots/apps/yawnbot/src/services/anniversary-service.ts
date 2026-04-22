/**
 * AnniversaryService — 캐릭터별 기념일 관리
 *
 * 저장 위치: characters/<slug>/memory/anniversaries.json
 * 형식: [{ id, label, month, day, year? }]
 *   - year 있으면 "N주년" 계산 가능
 *   - year 없으면 매년 반복 기념일
 *
 * 봇이 자동 추가하는 항목:
 *   - "첫 대화" — 캐릭터와 처음 대화한 날 (첫 로그 파일 날짜 기준)
 */
import fs from 'fs';
import path from 'path';

export interface Anniversary {
  id: string;
  label: string;
  month: number;  // 1-12
  day: number;    // 1-31
  year?: number;  // 기준 연도 (없으면 매년)
}

export class AnniversaryService {
  private filePath: string;
  private logsDir: string;
  private entries: Anniversary[] = [];

  constructor(memoRepoPath: string, slug: string) {
    const memoryDir = path.join(memoRepoPath, 'characters', slug, 'memory');
    this.filePath = path.join(memoryDir, 'anniversaries.json');
    this.logsDir = path.join(memoryDir, 'logs');
    this._load();
    this._ensureFirstConversation();
  }

  private _load(): void {
    try {
      if (fs.existsSync(this.filePath)) {
        this.entries = JSON.parse(fs.readFileSync(this.filePath, 'utf-8')) as Anniversary[];
      }
    } catch {
      this.entries = [];
    }
  }

  private _save(): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(this.entries, null, 2), 'utf-8');
  }

  /** 첫 로그 파일 날짜를 "첫 대화" 기념일로 자동 등록 (이미 있으면 스킵) */
  private _ensureFirstConversation(): void {
    if (this.entries.some((e) => e.id === '__first_conversation__')) return;
    if (!fs.existsSync(this.logsDir)) return;

    const files = fs.readdirSync(this.logsDir)
      .filter((f) => /^\d{4}-\d{2}-\d{2}\.md$/.test(f))
      .sort();
    if (!files.length) return;

    const [yearStr, monthStr, dayStr] = files[0].replace('.md', '').split('-');
    this.entries.push({
      id: '__first_conversation__',
      label: '첫 대화',
      month: parseInt(monthStr, 10),
      day: parseInt(dayStr, 10),
      year: parseInt(yearStr, 10),
    });
    this._save();
  }

  list(): Anniversary[] {
    return [...this.entries];
  }

  add(label: string, month: number, day: number, year?: number): Anniversary {
    const id = `${Date.now()}`;
    const entry: Anniversary = { id, label, month, day, ...(year != null ? { year } : {}) };
    this.entries.push(entry);
    this._save();
    return entry;
  }

  remove(id: string): boolean {
    const before = this.entries.length;
    this.entries = this.entries.filter((e) => e.id !== id);
    if (this.entries.length !== before) { this._save(); return true; }
    return false;
  }

  /**
   * 오늘 KST 기준 앞으로 N일 이내 다가오는 기념일 목록 반환 (D-day, 연 계산 포함).
   * 오늘 포함. 정렬: 가까운 것 먼저.
   */
  getUpcoming(withinDays = 30): Array<Anniversary & { dDay: number; years?: number }> {
    const kstNow = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
    const year = kstNow.getFullYear();
    const todayMs = new Date(year, kstNow.getMonth(), kstNow.getDate()).getTime();

    return this.entries
      .map((e) => {
        // 올해 기준 날짜 계산 (12월 기념일이 1월 이전이면 내년으로)
        let targetDate = new Date(year, e.month - 1, e.day);
        if (targetDate.getTime() < todayMs) {
          targetDate = new Date(year + 1, e.month - 1, e.day);
        }
        const dDay = Math.round((targetDate.getTime() - todayMs) / (1000 * 60 * 60 * 24));
        if (dDay > withinDays) return null;
        const targetYear = targetDate.getFullYear();
        return {
          ...e,
          dDay,
          ...(e.year != null ? { years: targetYear - e.year } : {}),
        };
      })
      .filter((x): x is NonNullable<typeof x> => x !== null)
      .sort((a, b) => a.dDay - b.dDay);
  }

  /** 오늘 KST 기준 해당하는 기념일 목록 반환 */
  getTodayAnniversaries(): Array<Anniversary & { years?: number }> {
    const kst = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
    const todayMonth = kst.getMonth() + 1;
    const todayDay = kst.getDate();
    const todayYear = kst.getFullYear();

    return this.entries
      .filter((e) => e.month === todayMonth && e.day === todayDay)
      .map((e) => ({
        ...e,
        ...(e.year != null ? { years: todayYear - e.year } : {}),
      }));
  }
}
