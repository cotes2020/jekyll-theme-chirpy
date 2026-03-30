/**
 * RaidService — C# RaidService → Node.js 이식
 * 레이드 보스 관리 (소환, 공격, 보상)
 */
import { GameDataService } from './gamedata';

export const BOSSES = [
    { name: '고대 드래곤', hp: 50000, reward: 10000, emoji: '🐉' },
    { name: '마왕 데스나이트', hp: 100000, reward: 25000, emoji: '💀' },
    { name: '어둠의 군주', hp: 200000, reward: 50000, emoji: '👹' },
    { name: '혼돈의 신', hp: 500000, reward: 100000, emoji: '🌑' },
];

interface RaidState {
    boss: string;
    emoji: string;
    currentHp: number;
    maxHp: number;
    reward: number;
    participants: Record<string, number>;
}

export class RaidService {
    gameData: GameDataService;
    currentRaid: RaidState | null = null;

    constructor(gameData: GameDataService) {
        this.gameData = gameData;
    }

    spawnRaid(bossIndex?: number) {
        const idx = bossIndex ?? Math.floor(Math.random() * BOSSES.length);
        const boss = BOSSES[idx] || BOSSES[0];
        this.currentRaid = {
            boss: boss.name,
            emoji: boss.emoji,
            currentHp: boss.hp,
            maxHp: boss.hp,
            reward: boss.reward,
            participants: {},
        };
        return this.currentRaid;
    }

    attack(userId: string) {
        if (!this.currentRaid) return { type: 'no_raid' as const };

        const user = this.gameData.getUser(userId);
        const baseDamage = 500;
        const levelBonus = user.sword.level * 200;
        const crit = Math.random() < 0.15 ? 2 : 1;
        const damage = Math.round((baseDamage + levelBonus + Math.floor(Math.random() * 300)) * crit);

        this.currentRaid.currentHp -= damage;
        if (!this.currentRaid.participants[userId]) this.currentRaid.participants[userId] = 0;
        this.currentRaid.participants[userId] += damage;

        if (this.currentRaid.currentHp <= 0) {
            this.currentRaid.currentHp = 0;
            const rewards = this._distributeRewards();
            const result = {
                type: 'cleared' as const,
                damage,
                crit: crit > 1,
                boss: this.currentRaid.boss,
                emoji: this.currentRaid.emoji,
                rewards,
            };
            this.currentRaid = null;
            return result;
        }

        return {
            type: 'hit' as const,
            damage,
            crit: crit > 1,
            boss: this.currentRaid.boss,
            emoji: this.currentRaid.emoji,
            currentHp: this.currentRaid.currentHp,
            maxHp: this.currentRaid.maxHp,
            hpPct: ((this.currentRaid.currentHp / this.currentRaid.maxHp) * 100).toFixed(1),
        };
    }

    private _distributeRewards(): { userId: string; damage: number; reward: number; share: number }[] {
        const raid = this.currentRaid!;
        const totalDamage = Object.values(raid.participants).reduce((s, d) => s + d, 0);
        const rewards: { userId: string; damage: number; reward: number; share: number }[] = [];

        const sorted = Object.entries(raid.participants).sort((a, b) => b[1] - a[1]);
        for (const [uid, dmg] of sorted) {
            const share = totalDamage > 0 ? dmg / totalDamage : 1 / sorted.length;
            const reward = Math.round(raid.reward * share);
            const user = this.gameData.getUser(uid);
            user.money += reward;
            rewards.push({ userId: uid, damage: dmg, reward, share });
        }
        this.gameData.saveGameData();
        return rewards;
    }

    getRaidInfo() {
        if (!this.currentRaid) return null;
        return {
            boss: this.currentRaid.boss,
            emoji: this.currentRaid.emoji,
            currentHp: this.currentRaid.currentHp,
            maxHp: this.currentRaid.maxHp,
            hpPct: ((this.currentRaid.currentHp / this.currentRaid.maxHp) * 100).toFixed(1),
            participantCount: Object.keys(this.currentRaid.participants).length,
        };
    }
}
