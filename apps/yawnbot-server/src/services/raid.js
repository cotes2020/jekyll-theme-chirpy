/**
 * RaidService — C# RaidService → Node.js 이식
 * 레이드 보스 관리 (소환, 공격, 보상)
 */
const { formatMoney } = require('./gamedata');

const BOSSES = [
    { name: '고대 드래곤', hp: 50000, reward: 10000, emoji: '🐉' },
    { name: '마왕 데스나이트', hp: 100000, reward: 25000, emoji: '💀' },
    { name: '어둠의 군주', hp: 200000, reward: 50000, emoji: '👹' },
    { name: '혼돈의 신', hp: 500000, reward: 100000, emoji: '🌑' },
];

class RaidService {
    /** @param {import('./gamedata').GameDataService} gameData */
    constructor(gameData) {
        this.gameData = gameData;
        this.currentRaid = null; // { boss, currentHp, maxHp, reward, participants: { userId: damage } }
    }

    /** 레이드 소환 */
    spawnRaid(bossIndex) {
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

    /** 공격 */
    attack(userId) {
        if (!this.currentRaid) return { type: 'no_raid' };

        const user = this.gameData.getUser(userId);
        const baseDamage = 500;
        const levelBonus = user.sword.level * 200;
        const crit = Math.random() < 0.15 ? 2 : 1;
        const damage = Math.round((baseDamage + levelBonus + Math.floor(Math.random() * 300)) * crit);

        this.currentRaid.currentHp -= damage;
        if (!this.currentRaid.participants[userId]) this.currentRaid.participants[userId] = 0;
        this.currentRaid.participants[userId] += damage;

        if (this.currentRaid.currentHp <= 0) {
            // 보스 처치 → 보상 배분
            this.currentRaid.currentHp = 0;
            const rewards = this._distributeRewards();
            const result = {
                type: 'cleared', damage, crit: crit > 1,
                boss: this.currentRaid.boss, emoji: this.currentRaid.emoji,
                rewards,
            };
            this.currentRaid = null;
            return result;
        }

        return {
            type: 'hit', damage, crit: crit > 1,
            boss: this.currentRaid.boss, emoji: this.currentRaid.emoji,
            currentHp: this.currentRaid.currentHp,
            maxHp: this.currentRaid.maxHp,
            hpPct: ((this.currentRaid.currentHp / this.currentRaid.maxHp) * 100).toFixed(1),
        };
    }

    /** 보상 배분 (기여도 비례) */
    _distributeRewards() {
        const raid = this.currentRaid;
        const totalDamage = Object.values(raid.participants).reduce((s, d) => s + d, 0);
        const rewards = [];

        const sorted = Object.entries(raid.participants).sort((a, b) => b[1] - a[1]);
        for (const [userId, damage] of sorted) {
            const share = totalDamage > 0 ? (damage / totalDamage) : (1 / sorted.length);
            const reward = Math.round(raid.reward * share);
            const user = this.gameData.getUser(userId);
            user.money += reward;
            rewards.push({ userId, damage, reward, share });
        }
        this.gameData.saveGameData();
        return rewards;
    }

    /** 레이드 정보 */
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

module.exports = { RaidService, BOSSES };
