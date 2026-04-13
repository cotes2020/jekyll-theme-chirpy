/**
 * MemoryService — memo 레포에 대화 기록을 저장하고 커밋
 * 1시간마다 자동 커밋 (push 없음)
 */
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

export interface ConversationEntry {
  timestamp: string;   // ISO string
  role: 'user' | 'assistant';
  content: string;
  channel: 'dm' | 'public';
}

function kstDateString(date: Date = new Date()): string {
  const kst = new Date(date.toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
  const y = kst.getFullYear();
  const m = String(kst.getMonth() + 1).padStart(2, '0');
  const d = String(kst.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function kstTimeString(date: Date = new Date()): string {
  const kst = new Date(date.toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
  const h = String(kst.getHours()).padStart(2, '0');
  const m = String(kst.getMinutes()).padStart(2, '0');
  return `${h}:${m}`;
}

export class MemoryService {
  private memoRepoPath: string;
  private diaryDir: string;
  private pendingEntries: ConversationEntry[] = [];
  private commitTimer: ReturnType<typeof setInterval> | null = null;
  private dirty = false;

  constructor(memoRepoPath: string) {
    this.memoRepoPath = memoRepoPath;
    this.diaryDir = path.join(memoRepoPath, 'assistant', 'diary');
  }

  initialize(): void {
    if (!fs.existsSync(this.diaryDir)) {
      fs.mkdirSync(this.diaryDir, { recursive: true });
    }
    const intervalMs = parseInt(process.env.ASSISTANT_MEMORY_COMMIT_INTERVAL_MS || '3600000', 10);
    this.commitTimer = setInterval(() => this.commitIfDirty(), intervalMs);
    console.log(`[Memory] 초기화 완료 (커밋 주기: ${intervalMs / 60000}분, 레포: ${this.memoRepoPath})`);
  }

  addEntry(entry: ConversationEntry): void {
    this.pendingEntries.push(entry);
    this.dirty = true;
  }

  /** 최근 N일치 diary 내용을 문자열로 반환 */
  getRecentDiaries(days: number = 3): string {
    const parts: string[] = [];
    for (let i = 0; i < days; i++) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      const dateStr = kstDateString(d);
      const filePath = path.join(this.diaryDir, `${dateStr}.md`);
      if (fs.existsSync(filePath)) {
        try {
          parts.push(fs.readFileSync(filePath, 'utf-8').trim());
        } catch {
          /* ignore */
        }
      }
    }
    return parts.join('\n\n---\n\n');
  }

  /** profile.md 내용 반환 */
  getProfile(): string {
    const profilePath = path.join(this.memoRepoPath, 'assistant', 'profile.md');
    if (fs.existsSync(profilePath)) {
      try {
        return fs.readFileSync(profilePath, 'utf-8').trim();
      } catch {
        return '';
      }
    }
    return '';
  }

  commitIfDirty(): void {
    if (!this.dirty || this.pendingEntries.length === 0) return;

    const today = kstDateString();
    const filePath = path.join(this.diaryDir, `${today}.md`);

    try {
      let existing = '';
      if (fs.existsSync(filePath)) {
        existing = fs.readFileSync(filePath, 'utf-8');
        if (!existing.endsWith('\n')) existing += '\n';
        existing += '\n';
      } else {
        existing = `# ${today} 대화 기록\n\n`;
      }

      const lines: string[] = [];
      for (const entry of this.pendingEntries) {
        const timeStr = kstTimeString(new Date(entry.timestamp));
        const who = entry.role === 'user' ? '나' : 'YawnBot';
        const ch = entry.channel === 'dm' ? 'DM' : '채널';
        lines.push(`[${timeStr} ${ch}] **${who}**: ${entry.content}`);
      }
      const appended = existing + lines.join('\n') + '\n';

      fs.writeFileSync(filePath, appended, 'utf-8');
      this.pendingEntries = [];
      this.dirty = false;

      execSync(`git -C "${this.memoRepoPath}" add assistant/diary/`, { stdio: 'pipe' });
      execSync(
        `git -C "${this.memoRepoPath}" commit -m "chore: 대화 기록 자동 저장 ${today}"`,
        { stdio: 'pipe' },
      );
      console.log(`[Memory] ${today} 대화 기록 커밋 완료`);
    } catch (e: unknown) {
      console.error('[Memory] 커밋 실패:', e instanceof Error ? e.message : e);
    }
  }

  destroy(): void {
    if (this.commitTimer) clearInterval(this.commitTimer);
    this.commitIfDirty();
  }
}
