/**
 * EnhancementService — C# EnhancementService → Node.js 이식
 * 강화, 판매, 미니게임(슬롯/홀짝/가위바위보/배틀), 출첵, 용돈
 */
import {
    GameDataService,
    UserData,
    getRandomImage,
    getRandomWeaponImage,
    getWeaponLore,
} from './gamedata';

const GREAT_SUCCESS_PROB = 0.5;
const PROTECT_PROB = 35.0;

export class EnhancementService {
    gameData: GameDataService;

    constructor(gameData: GameDataService) {
        this.gameData = gameData;
    }

    /** 유저 검사 (마이그레이션) — index 등 외부에서도 호출 */
    ensureSword(user: UserData): void {
        if (!user.sword.weaponType || !user.sword.imageName) {
            const r = getRandomWeaponImage(user.sword.level, user.sword.weaponType, this.gameData);
            user.sword.weaponType = r.weaponType;
            user.sword.imageName = r.imageName ?? '';
            if (user.sword.level === 0) user.sword.name = r.name;
        }
    }

    enhance(userId: string) {
        const user = this.gameData.getUser(userId);
        this.ensureSword(user);
        const level = user.sword.level;
        if (level >= 15) return { type: 'max' as const };

        const prob = this.gameData.getProbability(level);
        if (!prob) return { type: 'error' as const, level };

        if (user.money < prob.cost) return { type: 'no_money' as const, cost: prob.cost, balance: user.money };

        user.money -= prob.cost;
        user.totalEnhanceCount = (user.totalEnhanceCount || 0) + 1;
        user.totalMoneySpent = (user.totalMoneySpent || 0) + prob.cost;

        const roll = Math.random() * 100;
        const isGreat = Math.random() * 100 <= GREAT_SUCCESS_PROB;

        if (isGreat) {
            const oldLevel = user.sword.level;
            user.sword.level = Math.min(user.sword.level + 3, 15);
            const w = getRandomWeaponImage(user.sword.level, user.sword.weaponType, this.gameData);
            user.sword.imageName = w.imageName ?? '';
            user.sword.name = w.name;
            user.sword.weaponType = w.weaponType;

            if (user.sword.level > user.maxLevel) user.maxLevel = user.sword.level;
            this.gameData.saveGameData();
            return {
                type: 'great_success' as const,
                oldLevel,
                newLevel: user.sword.level,
                increase: user.sword.level - oldLevel,
                sword: { ...user.sword },
                cost: prob.cost,
                balance: user.money,
                chat: this.gameData.getChat(user.sword.level, 'success'),
                lore: getWeaponLore(user.sword.weaponType, user.sword.level, this.gameData),
            };
        }

        if (roll <= prob.success) {
            const oldLevel = user.sword.level;
            user.sword.level++;
            const w = getRandomWeaponImage(user.sword.level, user.sword.weaponType, this.gameData);
            user.sword.imageName = w.imageName ?? '';
            user.sword.name = w.name;
            user.sword.weaponType = w.weaponType;

            if (user.sword.level > user.maxLevel) user.maxLevel = user.sword.level;
            this.gameData.saveGameData();
            return {
                type: 'success' as const,
                oldLevel,
                newLevel: user.sword.level,
                sword: { ...user.sword },
                cost: prob.cost,
                balance: user.money,
                chat: this.gameData.getChat(user.sword.level, 'success'),
                lore: getWeaponLore(user.sword.weaponType, user.sword.level, this.gameData),
            };
        }

        if (Math.random() * 100 <= PROTECT_PROB) {
            const maintainImage = getRandomImage('강화_유지_');
            this.gameData.saveGameData();
            return {
                type: 'protected' as const,
                level: user.sword.level,
                sword: { ...user.sword },
                cost: prob.cost,
                balance: user.money,
                chat: this.gameData.getChat(user.sword.level, 'maintain'),
                imageOverride: maintainImage,
            };
        }

        const destroyed = { level: user.sword.level, name: user.sword.name };
        user.sword.level = 0;
        const w = getRandomWeaponImage(0, null, this.gameData);
        user.sword.imageName = w.imageName ?? '';
        user.sword.name = w.name;
        user.sword.weaponType = w.weaponType;

        user.totalDestroyCount = (user.totalDestroyCount || 0) + 1;
        this.gameData.saveGameData();
        return {
            type: 'destroy' as const,
            destroyedLevel: destroyed.level,
            destroyedName: destroyed.name,
            sword: { ...user.sword },
            cost: prob.cost,
            balance: user.money,
            chat: this.gameData.getChat(Math.min(destroyed.level + 1, 15), 'fail'),
            imageOverride: getRandomImage('강화_실패_'),
        };
    }

    sell(userId: string) {
        const user = this.gameData.getUser(userId);
        this.ensureSword(user);
        if (user.sword.level === 0) return { type: 'no_sword' as const };

        const prob = this.gameData.getProbability(user.sword.level);
        const basePrice = prob ? prob.sellPrice : user.sword.level * 10000;
        const multiplier = 0.9 + Math.random() * 0.6;
        let finalPrice = Math.round(basePrice * multiplier) + Math.floor(Math.random() * 101) - 50;
        if (finalPrice < 0) finalPrice = 0;

        const soldName = user.sword.name;
        const soldLevel = user.sword.level;
        user.money += finalPrice;
        user.sword.level = 0;
        const w = getRandomWeaponImage(0, null, this.gameData);
        user.sword.imageName = w.imageName ?? '';
        user.sword.name = w.name;
        user.sword.weaponType = w.weaponType;

        const comment =
            multiplier >= 1.3
                ? '상태가 아주 훌륭하군요! 값을 더 쳐드리겠습니다.'
                : multiplier <= 1.0
                  ? '흠... 흠집이 좀 있네요. 많이는 못 드립니다.'
                  : '적당한 물건이군요. 시세대로 드리겠습니다.';

        this.gameData.saveGameData();
        return {
            type: 'sold' as const,
            soldName,
            soldLevel,
            basePrice,
            finalPrice,
            multiplier,
            comment,
            balance: user.money,
        };
    }

    checkAttendance(userId: string) {
        const user = this.gameData.getUser(userId);
        const now = Date.now();
        const last = user.lastAttendance ? new Date(user.lastAttendance).getTime() : 0;
        const hourMs = 3600000;
        if (now - last >= hourMs) {
            const reward = 1000;
            user.money += reward;
            user.lastAttendance = new Date().toISOString();
            this.gameData.saveGameData();
            return { type: 'ok' as const, reward, balance: user.money };
        }
        const remaining = hourMs - (now - last);
        return {
            type: 'wait' as const,
            min: Math.floor(remaining / 60000),
            sec: Math.floor((remaining % 60000) / 1000),
        };
    }

    giveMeMoney(userId: string) {
        const user = this.gameData.getUser(userId);
        const amount = Math.floor(Math.random() * 2500) + 1;
        user.money += amount;
        this.gameData.saveGameData();
        return { amount, balance: user.money };
    }

    slot(userId: string, bet: number) {
        const user = this.gameData.getUser(userId);
        if (bet <= 0) return { type: 'error' as const, msg: '배팅 금액은 0보다 커야 합니다.' };
        if (user.money < bet) return { type: 'no_money' as const, balance: user.money };

        user.money -= bet;
        const SYMS = ['🍒', '🍋', '🍇', '💎', '7️⃣'];
        const s = [0, 1, 2].map(() => SYMS[Math.floor(Math.random() * SYMS.length)]);
        let payout = 0;
        let msg = '꽝!';
        let resultType: string = 'lose';

        if (s[0] === '7️⃣' && s[1] === '7️⃣' && s[2] === '7️⃣') {
            payout = bet * 77;
            msg = 'Jackpot! (77배)';
            resultType = 'jackpot';
        } else if (s[0] === '💎' && s[1] === '💎' && s[2] === '💎') {
            payout = bet * 50;
            msg = 'Diamond! (50배)';
            resultType = 'diamond';
        } else if (s[0] === s[1] && s[1] === s[2]) {
            payout = bet * 10;
            msg = 'Triple! (10배)';
            resultType = 'triple';
        } else if (s[0] === s[1] || s[1] === s[2] || s[0] === s[2]) {
            payout = bet * 2;
            msg = 'Double! (2배)';
            resultType = 'double';
        }

        if (payout > 0) user.money += payout;
        this.gameData.saveGameData();
        return { type: resultType, symbols: s, bet, payout, msg, balance: user.money };
    }

    oddEven(userId: string, choice: string, bet: number) {
        const user = this.gameData.getUser(userId);
        if (bet <= 0) return { type: 'error' as const, msg: '배팅 금액은 0보다 커야 합니다.' };
        if (user.money < bet) return { type: 'no_money' as const, balance: user.money };

        user.money -= bet;
        const isOdd = Math.random() < 0.5;
        const result = isOdd ? '홀' : '짝';
        const dice = isOdd ? [1, 3, 5][Math.floor(Math.random() * 3)] : [2, 4, 6][Math.floor(Math.random() * 3)];
        const win = choice === result;
        let payout = 0;
        if (win) {
            payout = bet * 2;
            user.money += payout;
        }
        this.gameData.saveGameData();
        return { type: win ? ('win' as const) : ('lose' as const), choice, result, dice, bet, payout, balance: user.money };
    }

    rps(userId: string, choice: string, bet: number) {
        const RPS = ['가위', '바위', '보'];
        const user = this.gameData.getUser(userId);
        if (bet <= 0) return { type: 'error' as const, msg: '배팅 금액은 0보다 커야 합니다.' };
        if (!RPS.includes(choice)) return { type: 'error' as const, msg: '가위, 바위, 보 중 하나를 선택하세요.' };
        if (user.money < bet) return { type: 'no_money' as const, balance: user.money };

        user.money -= bet;
        const botChoice = RPS[Math.floor(Math.random() * 3)];
        const r = (RPS.indexOf(choice) - RPS.indexOf(botChoice) + 3) % 3;
        let payout = 0;
        let resultType = '';
        if (r === 0) {
            payout = bet;
            user.money += payout;
            resultType = 'draw';
        } else if (r === 1) {
            payout = bet * 2;
            user.money += payout;
            resultType = 'win';
        } else {
            resultType = 'lose';
        }
        this.gameData.saveGameData();
        return { type: resultType, userChoice: choice, botChoice, bet, payout, balance: user.money };
    }

    battle(userId: string, targetUserId: string) {
        const user = this.gameData.getUser(userId);
        this.ensureSword(user);
        const today = new Date().toDateString();
        if (user.dailyBattleDate !== today) {
            user.dailyBattleCount = 0;
            user.dailyBattleDate = today;
        }
        if (user.dailyBattleCount >= 10) return { type: 'limit' as const, remaining: 0 };

        const target = this.gameData.getUser(targetUserId);
        const myLevel = user.sword.level;
        const targetLevel = target.sword.level;
        const winChance = Math.max(5, Math.min(95, 50 + (myLevel - targetLevel) * 5));
        const isWin = Math.random() * 100 < winChance;

        user.dailyBattleCount++;
        const remaining = 10 - user.dailyBattleCount;
        let reward = 0;
        if (isWin) {
            reward = 300 + targetLevel * 100;
            user.money += reward;
        }
        this.gameData.saveGameData();
        return {
            type: isWin ? ('win' as const) : ('lose' as const),
            myLevel,
            mySword: user.sword.name,
            targetLevel,
            targetSword: target.sword.name,
            winChance,
            reward,
            remaining,
            balance: user.money,
            battleImage: getRandomImage('배틀_시작_'),
        };
    }
}
