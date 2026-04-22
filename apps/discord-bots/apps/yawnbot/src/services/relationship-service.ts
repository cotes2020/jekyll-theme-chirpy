/**
 * RelationshipService — 대화 횟수/감정 누적으로 친밀도 수치화
 *
 * - 대화할 때마다 conversationCount 증가 → level 자동 갱신
 * - 리액션 이모지로 moodScore 누적 (긍정 +, 부정 -)
 * - 친밀도 레벨별 말투 힌트를 system prompt에 주입
 * - 저장 위치: {characterDir}/relationship.json
 */
import fs from 'fs';
import path from 'path';

export interface RelationshipData {
  conversationCount: number;
  moodScore: number;
  level: number;
  lastUpdated: string;
}

export const RELATIONSHIP_LEVELS = [
  { level: 0, label: '낯선사람', threshold: 0,   hint: '유저와 처음 만난 상태야. 정중하고 조심스럽게 대해줘.' },
  { level: 1, label: '지인',     threshold: 20,  hint: '유저와 어느 정도 아는 사이야. 조금 편하게 대화해줘.' },
  { level: 2, label: '친구',     threshold: 100, hint: '유저와 친구야. 편하고 자연스럽게 대화해줘.' },
  { level: 3, label: '절친',     threshold: 300, hint: '유저와 절친이야. 아주 편하고 개인적인 이야기도 나눌 수 있어. 서로를 잘 알아.' },
  { level: 4, label: '베프',     threshold: 800, hint: '유저와 베스트 프렌드야. 완전히 편하게, 서로를 깊이 이해하는 사이로 대화해줘. 오래 함께한 기억이 있어.' },
] as const;

export const MOOD_SCORE_MAP: Record<string, number> = {
  '👍': 1,
  '❤️': 2,
  '😂': 1,
  '😢': -1,
};

function calcLevel(count: number): number {
  let level = 0;
  for (const l of RELATIONSHIP_LEVELS) {
    if (count >= l.threshold) level = l.level;
  }
  return level;
}

export class RelationshipService {
  private filePath: string;
  private data: RelationshipData;

  constructor(characterDir: string) {
    this.filePath = path.join(characterDir, 'relationship.json');
    this.data = this._load();
  }

  private _load(): RelationshipData {
    try {
      if (fs.existsSync(this.filePath)) {
        const raw = fs.readFileSync(this.filePath, 'utf-8');
        return JSON.parse(raw) as RelationshipData;
      }
    } catch {
      /* ignore parse errors */
    }
    return { conversationCount: 0, moodScore: 0, level: 0, lastUpdated: new Date().toISOString() };
  }

  private _save(): void {
    try {
      fs.writeFileSync(this.filePath, JSON.stringify(this.data, null, 2), 'utf-8');
    } catch (e) {
      console.warn('[RelationshipService] 저장 실패:', e instanceof Error ? e.message : String(e));
    }
  }

  /** 대화 1턴 완료 후 호출. count 증가 + level 재계산. */
  incrementConversation(): void {
    this.data.conversationCount += 1;
    this.data.level = calcLevel(this.data.conversationCount);
    this.data.lastUpdated = new Date().toISOString();
    this._save();
  }

  /** 리액션 이모지 → moodScore 조정. */
  addMoodReaction(emoji: string): void {
    const delta = MOOD_SCORE_MAP[emoji] ?? 0;
    if (delta === 0) return;
    this.data.moodScore += delta;
    this.data.lastUpdated = new Date().toISOString();
    this._save();
    console.log(`[RelationshipService] moodScore ${this.data.moodScore - delta} → ${this.data.moodScore} (${emoji})`);
  }

  get conversationCount(): number { return this.data.conversationCount; }
  get moodScore(): number { return this.data.moodScore; }
  get level(): number { return this.data.level; }

  getLevelInfo() {
    return RELATIONSHIP_LEVELS.find((l) => l.level === this.data.level) ?? RELATIONSHIP_LEVELS[0];
  }

  /** system prompt에 주입할 친밀도 힌트 문자열 */
  buildRelationshipHint(): string {
    const info = this.getLevelInfo();
    const moodAdj = this.data.moodScore >= 10
      ? ' 유저가 너에게 매우 긍정적인 감정을 가지고 있어.'
      : this.data.moodScore <= -5
      ? ' 유저와 최근 감정적으로 약간 어색한 상태야.'
      : '';
    return `[친밀도 Lv.${info.level} ${info.label}] ${info.hint}${moodAdj}`;
  }

  getSummary(): string {
    const info = this.getLevelInfo();
    return `Lv.${info.level} ${info.label} | 대화 ${this.data.conversationCount}회 | 호감도 ${this.data.moodScore >= 0 ? '+' : ''}${this.data.moodScore}`;
  }
}
