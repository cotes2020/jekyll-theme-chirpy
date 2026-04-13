/**
 * MemoryService — 계층형 메모리 시스템
 *
 * logs/YYYY-MM-DD.md      : 전체 대화 원본 (즉시 기록)
 * memory/daily/YYYY-MM-DD : 하루 요약 (AI 생성, 다음날 첫 대화 때)
 * memory/weekly/YYYY-WNN  : 주간 요약 (AI 생성, 다음주 첫 대화 때)
 * memory/user.md          : 나에 대한 누적 정보
 * memory/self.md          : 봇 자신에 대한 누적 정보
 *
 * git commit: 1시간마다 자동 (push 없음)
 */
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { generateAssistantText } from 'karmolab-ai/node';

export interface ConversationEntry {
  timestamp: string;
  role: 'user' | 'assistant';
  content: string;
  channel: 'dm' | 'public';
}

// ── 날짜 유틸 ──────────────────────────────────────────────────────────────

function toKST(date: Date = new Date()): Date {
  return new Date(date.toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
}

function kstDateStr(date: Date = new Date()): string {
  const k = toKST(date);
  return `${k.getFullYear()}-${pad(k.getMonth() + 1)}-${pad(k.getDate())}`;
}

function kstTimeStr(date: Date = new Date()): string {
  const k = toKST(date);
  return `${pad(k.getHours())}:${pad(k.getMinutes())}`;
}

function isoWeekKey(date: Date = new Date()): string {
  const k = toKST(date);
  const jan4 = new Date(k.getFullYear(), 0, 4);
  const startOfWeek1 = new Date(jan4);
  startOfWeek1.setDate(jan4.getDate() - ((jan4.getDay() + 6) % 7));
  const diff = k.getTime() - startOfWeek1.getTime();
  const week = Math.floor(diff / (7 * 24 * 60 * 60 * 1000)) + 1;
  return `${k.getFullYear()}-W${pad(week)}`;
}

function daysAgo(n: number): Date {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return d;
}

function pad(n: number): string {
  return String(n).padStart(2, '0');
}

// ── MemoryService ──────────────────────────────────────────────────────────

export class MemoryService {
  private memoRepoPath: string;
  private logsDir: string;
  private memoryDir: string;
  private commitTimer: ReturnType<typeof setInterval> | null = null;
  private dirty = false;

  constructor(memoRepoPath: string) {
    this.memoRepoPath = memoRepoPath;
    this.logsDir = path.join(memoRepoPath, 'assistant', 'logs');
    this.memoryDir = path.join(memoRepoPath, 'assistant', 'memory');
  }

  initialize(): void {
    for (const dir of [
      this.logsDir,
      path.join(this.memoryDir, 'daily'),
      path.join(this.memoryDir, 'weekly'),
    ]) {
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    }

    // user.md / self.md 없으면 초기 파일 생성
    this._initFile(
      path.join(this.memoryDir, 'user.md'),
      '# 나에 대한 정보\n\n(아직 기록된 정보 없음)\n',
    );
    this._initFile(
      path.join(this.memoryDir, 'self.md'),
      '# 봇 자신에 대한 정보\n\n(아직 기록된 정보 없음)\n',
    );

    this._ensureAgentClaudeMd();

    const intervalMs = parseInt(
      process.env.ASSISTANT_MEMORY_COMMIT_INTERVAL_MS || '3600000',
      10,
    );
    this.commitTimer = setInterval(() => this.commitIfDirty(), intervalMs);
    console.log(`[Memory] 초기화 완료 (커밋 주기: ${intervalMs / 60000}분)`);
  }

  /**
   * ASSISTANT_AGENT_REPO_PATH 루트에 CLAUDE.md 심볼릭 링크가 없으면 생성.
   * memo/CLAUDE-karmoddrine.md → {agentRepoPath}/CLAUDE.md
   * Windows Developer Mode 또는 관리자 권한이 있을 때만 성공.
   */
  private _ensureAgentClaudeMd(): void {
    const agentPath = process.env.ASSISTANT_AGENT_REPO_PATH?.trim();
    if (!agentPath) return;

    const linkPath = path.join(agentPath, 'CLAUDE.md');
    if (fs.existsSync(linkPath)) return;

    const sourcePath = path.join(this.memoRepoPath, 'CLAUDE-karmoddrine.md');
    if (!fs.existsSync(sourcePath)) {
      console.warn('[Memory] CLAUDE-karmoddrine.md 없음 — 에이전트 컨텍스트 파일 생성 건너뜀');
      return;
    }

    try {
      fs.symlinkSync(sourcePath, linkPath);
      console.log(`[Memory] CLAUDE.md 심볼릭 링크 생성: ${linkPath}`);
    } catch (e: unknown) {
      console.warn(`[Memory] CLAUDE.md 심볼릭 링크 생성 실패 (Developer Mode 또는 관리자 권한 필요): ${e instanceof Error ? e.message : e}`);
    }
  }

  private _initFile(filePath: string, content: string): void {
    if (!fs.existsSync(filePath)) {
      fs.writeFileSync(filePath, content, 'utf-8');
      this.dirty = true;
    }
  }

  // ── 로그 즉시 기록 ────────────────────────────────────────────────────────

  appendToLog(entry: ConversationEntry): void {
    const today = kstDateStr();
    const filePath = path.join(this.logsDir, `${today}.md`);

    let content = '';
    if (!fs.existsSync(filePath)) {
      content = `# ${today} 대화 로그\n\n`;
    }

    const time = kstTimeStr(new Date(entry.timestamp));
    const who = entry.role === 'user' ? '나' : 'YawnBot';
    const ch = entry.channel === 'dm' ? 'DM' : '채널';
    content += `[${time} ${ch}] **${who}**: ${entry.content}\n`;

    fs.appendFileSync(filePath, content, 'utf-8');
    this.dirty = true;
  }

  appendHotMemory(fact: string): void {
    const userMdPath = path.join(this.memoryDir, 'user.md');
    const today = kstDateStr();
    const line = `\n- [${today}] ${fact}`;
    try {
      fs.appendFileSync(userMdPath, line, 'utf-8');
      this.dirty = true;
      console.log(`[Memory] Hot memory 저장: ${fact}`);
    } catch (e: unknown) {
      console.error('[Memory] hot memory 저장 실패:', e instanceof Error ? e.message : e);
    }
  }

  getUserMd(): string {
    return this._read(path.join(this.memoryDir, 'user.md'));
  }

  getSelfMd(): string {
    return this._read(path.join(this.memoryDir, 'self.md'));
  }

  getHotMemoryLog(limit: number = 20): string {
    const userMd = this.getUserMd();
    const lines = userMd.split('\n');
    const hotMemories: string[] = [];

    for (const line of lines) {
      const match = line.match(/^- \[(\d{4}-\d{2}-\d{2})\]\s+(.+)$/);
      if (match) {
        hotMemories.push(line.trim());
      }
      if (hotMemories.length >= limit) break;
    }

    return hotMemories.length > 0 ? hotMemories.join('\n') : '(기록 없음)';
  }

  // ── 요약 생성 (필요할 때만) ───────────────────────────────────────────────

  async checkAndGenerateSummaries(): Promise<void> {
    await this._generateDailySummaryIfNeeded();
    await this._generateWeeklySummaryIfNeeded();
    await this._updateUserAndSelfMemoryIfNeeded();
    this._cleanupOldMemories();
  }

  private async _generateDailySummaryIfNeeded(): Promise<void> {
    const yesterday = kstDateStr(daysAgo(1));
    const summaryPath = path.join(this.memoryDir, 'daily', `${yesterday}.md`);
    if (fs.existsSync(summaryPath)) return;

    const logPath = path.join(this.logsDir, `${yesterday}.md`);
    if (!fs.existsSync(logPath)) return;

    try {
      const log = fs.readFileSync(logPath, 'utf-8').trim();
      if (!log) return;

      console.log(`[Memory] ${yesterday} 일간 요약 생성 중...`);
      const { text } = await generateAssistantText(
        process.env,
        `다음은 ${yesterday}의 대화 기록이야. 핵심 내용을 간결하게 요약해줘. ` +
          `어떤 주제로 대화했는지, 중요한 정보나 감정, 결정된 것들 위주로:\n\n${log.slice(0, 8000)}`,
      );

      fs.writeFileSync(summaryPath, `# ${yesterday} 일간 요약\n\n${text.trim()}\n`, 'utf-8');
      this.dirty = true;
      console.log(`[Memory] ${yesterday} 일간 요약 저장 완료`);
    } catch (e: unknown) {
      console.error('[Memory] 일간 요약 생성 실패:', e instanceof Error ? e.message : e);
    }
  }

  private async _generateWeeklySummaryIfNeeded(): Promise<void> {
    // 이번 주가 아닌 지난 주 키
    const lastWeekDate = daysAgo(7);
    const weekKey = isoWeekKey(lastWeekDate);
    const summaryPath = path.join(this.memoryDir, 'weekly', `${weekKey}.md`);
    if (fs.existsSync(summaryPath)) return;

    // 지난 주 일간 요약들 수집
    const dailies: string[] = [];
    for (let i = 7; i <= 13; i++) {
      const d = kstDateStr(daysAgo(i));
      const p = path.join(this.memoryDir, 'daily', `${d}.md`);
      if (fs.existsSync(p)) {
        dailies.push(fs.readFileSync(p, 'utf-8').trim());
      }
    }
    if (dailies.length === 0) return;

    try {
      console.log(`[Memory] ${weekKey} 주간 요약 생성 중...`);
      const { text } = await generateAssistantText(
        process.env,
        `다음은 ${weekKey} 한 주간의 일별 대화 요약이야. ` +
          `일주일 전체를 아우르는 주간 요약을 만들어줘. ` +
          `반복된 주제, 중요한 변화, 감정 흐름, 결정된 것들 위주로:\n\n` +
          dailies.join('\n\n---\n\n').slice(0, 10000),
      );

      fs.writeFileSync(summaryPath, `# ${weekKey} 주간 요약\n\n${text.trim()}\n`, 'utf-8');
      this.dirty = true;
      console.log(`[Memory] ${weekKey} 주간 요약 저장 완료`);
    } catch (e: unknown) {
      console.error('[Memory] 주간 요약 생성 실패:', e instanceof Error ? e.message : e);
    }
  }

  private async _updateUserAndSelfMemoryIfNeeded(): Promise<void> {
    // 오늘 이미 갱신했으면 스킵 (중복 방지)
    const markerPath = path.join(this.memoryDir, '.user-self-updated');
    const today = kstDateStr();
    if (fs.existsSync(markerPath)) {
      const lastUpdated = fs.readFileSync(markerPath, 'utf-8').trim();
      if (lastUpdated === today) return;
    }

    const yesterday = kstDateStr(daysAgo(1));
    const dailySummaryPath = path.join(this.memoryDir, 'daily', `${yesterday}.md`);

    // 어제 요약이 없으면 갱신 스킵
    if (!fs.existsSync(dailySummaryPath)) return;

    try {
      const dailySummary = fs.readFileSync(dailySummaryPath, 'utf-8').trim();
      const currentUserMd = this._read(path.join(this.memoryDir, 'user.md'));
      const currentSelfMd = this._read(path.join(this.memoryDir, 'self.md'));

      console.log(`[Memory] user.md / self.md 갱신 중...`);
      const { text: updatedMemory } = await generateAssistantText(
        process.env,
        `너는 mascari4615의 개인 AI 비서야.\n` +
          `다음은 어제(${yesterday})의 대화 요약이야:\n\n${dailySummary}\n\n` +
          `이를 바탕으로 두 가지를 작성해줘:\n\n` +
          `## [나에 대한 정보]\n` +
          `mascari4615의 성격, 특징, 최근 상태, 관심사, 감정 등을 누적으로 정리해줘.\n` +
          `(기존 정보가 있으면 유지하면서 어제 대화로부터 새로운 정보를 추가/갱신)\n` +
          `기존 정보:\n${currentUserMd || '(아직 기록 없음)'}\n\n` +
          `## [봇 자신에 대한 정보]\n` +
          `어제 대화를 통해 내가 얼마나 도움이 되었는지, 어떤 역할을 하고 있는지 정리해줘.\n` +
          `(기존 정보가 있으면 유지하면서 어제 상황을 반영)\n` +
          `기존 정보:\n${currentSelfMd || '(아직 기록 없음)'}\n\n` +
          `마크다운 형식으로, 간결하게 작성해줘.`,
      );

      // "## [나에 대한 정보]"와 "## [봇 자신에 대한 정보]" 섹션으로 분리
      const userMatch = updatedMemory.match(/##\s*\[나에\s*대한\s*정보\]([\s\S]*?)(?=##\s*\[봇|$)/);
      const selfMatch = updatedMemory.match(/##\s*\[봇\s*자신에\s*대한\s*정보\]([\s\S]*?)$/);

      if (userMatch) {
        const userContent = userMatch[1].trim();
        fs.writeFileSync(
          path.join(this.memoryDir, 'user.md'),
          `# 나에 대한 정보\n\n${userContent}\n`,
          'utf-8',
        );
        this.dirty = true;
      }

      if (selfMatch) {
        const selfContent = selfMatch[1].trim();
        fs.writeFileSync(
          path.join(this.memoryDir, 'self.md'),
          `# 봇 자신에 대한 정보\n\n${selfContent}\n`,
          'utf-8',
        );
        this.dirty = true;
      }

      // 중복 방지 마커 파일 기록
      fs.writeFileSync(markerPath, today, 'utf-8');

      console.log(`[Memory] user.md / self.md 갱신 완료`);
    } catch (e: unknown) {
      console.error('[Memory] user/self 메모리 갱신 실패:', e instanceof Error ? e.message : e);
    }
  }

  private _cleanupOldMemories(): void {
    const cutoffs = [
      { dir: this.logsDir, days: 60 },
      { dir: path.join(this.memoryDir, 'daily'), days: 30 },
      { dir: path.join(this.memoryDir, 'weekly'), days: 84 },
    ];
    for (const { dir, days } of cutoffs) {
      if (!fs.existsSync(dir)) continue;
      try {
        const files = fs.readdirSync(dir).filter((f) => f.endsWith('.md'));
        for (const file of files) {
          const filePath = path.join(dir, file);
          const stat = fs.statSync(filePath);
          const ageMs = Date.now() - stat.mtimeMs;
          if (ageMs > days * 24 * 60 * 60 * 1000) {
            fs.unlinkSync(filePath);
            console.log(`[Memory] 오래된 파일 삭제: ${file}`);
            this.dirty = true;
          }
        }
      } catch (e: unknown) {
        console.warn(`[Memory] cleanup 실패 (${dir}):`, e instanceof Error ? e.message : e);
      }
    }
  }

  // ── 컨텍스트 빌드 ─────────────────────────────────────────────────────────

  buildContext(maxChars = 8000): string {
    // 반드시 포함 (핵심 프로필)
    const fixed: string[] = [];

    const userMd = this._read(path.join(this.memoryDir, 'user.md'));
    if (userMd) fixed.push(`[나에 대한 정보]\n${userMd}`);

    const selfMd = this._read(path.join(this.memoryDir, 'self.md'));
    if (selfMd) fixed.push(`[봇 자신에 대한 정보]\n${selfMd}`);

    // 선택적 포함 (최신 → 오래된 순으로 우선순위)
    const optional: string[] = [];

    const todayLog = this._read(path.join(this.logsDir, `${kstDateStr()}.md`));
    if (todayLog) optional.push(`[오늘 대화 기록]\n${todayLog}`);

    const yesterdaySummary = this._read(
      path.join(this.memoryDir, 'daily', `${kstDateStr(daysAgo(1))}.md`),
    );
    if (yesterdaySummary) optional.push(`[어제 요약]\n${yesterdaySummary}`);

    const weeklyDir = path.join(this.memoryDir, 'weekly');
    const latestWeekly = this._latestFile(weeklyDir);
    if (latestWeekly) optional.push(`[최근 주간 요약]\n${latestWeekly}`);

    // 합치기: fixed는 무조건 포함, optional은 길이 여유 있을 때만
    let result = fixed.join('\n\n');
    for (const part of optional) {
      const candidate = result ? result + '\n\n' + part : part;
      if (candidate.length <= maxChars) {
        result = candidate;
      }
    }

    return result;
  }

  private _read(filePath: string): string {
    if (!fs.existsSync(filePath)) return '';
    try {
      return fs.readFileSync(filePath, 'utf-8').trim();
    } catch {
      return '';
    }
  }

  private _latestFile(dir: string): string {
    if (!fs.existsSync(dir)) return '';
    try {
      const files = fs.readdirSync(dir).filter((f) => f.endsWith('.md')).sort().reverse();
      if (!files.length) return '';
      return fs.readFileSync(path.join(dir, files[0]), 'utf-8').trim();
    } catch {
      return '';
    }
  }

  // ── Git 커밋 ──────────────────────────────────────────────────────────────

  commitIfDirty(): void {
    if (!this.dirty) return;
    const today = kstDateStr();
    try {
      execSync(`git -C "${this.memoRepoPath}" add assistant/`, { stdio: 'pipe' });
      execSync(
        `git -C "${this.memoRepoPath}" commit -m "chore: 대화 기록 자동 저장 ${today}"`,
        { stdio: 'pipe' },
      );
      this.dirty = false;
      console.log(`[Memory] git commit 완료 (${today})`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      if (!msg.includes('nothing to commit')) {
        console.error('[Memory] commit 실패:', msg);
      }
    }
  }

  destroy(): void {
    if (this.commitTimer) clearInterval(this.commitTimer);
    this.commitIfDirty();
  }
}
