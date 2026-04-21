/**
 * ScheduleService — 캐릭터별 일정 관리
 *
 * 저장 위치: characters/<slug>/memory/schedule.json
 * 구조: ScheduleEntry[]
 */
import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';

export interface ScheduleEntry {
  id: string;
  title: string;
  datetime: string; // UTC ISO 8601
  notifyMinutes: number;
  notified: boolean;
}

export class ScheduleService {
  readonly slug: string;
  private filePath: string;
  private entries: ScheduleEntry[] = [];

  constructor(memoRepoPath: string, slug: string) {
    this.slug = slug;
    this.filePath = path.join(memoRepoPath, 'characters', slug, 'memory', 'schedule.json');
    this._load();
  }

  private _load(): void {
    try {
      if (fs.existsSync(this.filePath)) {
        this.entries = JSON.parse(fs.readFileSync(this.filePath, 'utf-8'));
      }
    } catch {
      this.entries = [];
    }
  }

  private _save(): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(this.entries, null, 2), 'utf-8');
  }

  add(title: string, isoDatetime: string, notifyMinutes = 10): ScheduleEntry {
    const entry: ScheduleEntry = {
      id: randomUUID().slice(0, 8),
      title,
      datetime: isoDatetime,
      notifyMinutes,
      notified: false,
    };
    this.entries.push(entry);
    this._save();
    return entry;
  }

  list(): ScheduleEntry[] {
    return [...this.entries].sort(
      (a, b) => new Date(a.datetime).getTime() - new Date(b.datetime).getTime(),
    );
  }

  remove(id: string): boolean {
    const before = this.entries.length;
    this.entries = this.entries.filter((e) => e.id !== id);
    if (this.entries.length < before) {
      this._save();
      return true;
    }
    return false;
  }

  /** 알림을 보내야 할 항목. notified=false이고 notifyMinutes 이내 또는 지난 것 (24시간 이내). */
  getPendingReminders(): ScheduleEntry[] {
    const now = Date.now();
    return this.entries.filter((e) => {
      if (e.notified) return false;
      const eventTime = new Date(e.datetime).getTime();
      if (now > eventTime + 24 * 60 * 60 * 1000) return false;
      return now >= eventTime - e.notifyMinutes * 60 * 1000;
    });
  }

  markNotified(id: string): void {
    const entry = this.entries.find((e) => e.id === id);
    if (entry) {
      entry.notified = true;
      this._save();
    }
  }
}
