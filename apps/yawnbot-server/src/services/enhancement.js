/**
 * EnhancementService — C# EnhancementService → Node.js 이식
 * 강화, 판매, 미니게임(슬롯/홀짝/가위바위보/배틀), 출첵, 용돈
 */
const { generateWeaponName, randomWeaponType, getRandomWeaponImage, getRandomImage, getWeaponLore, formatMoney, getLevelColor } = require('./gamedata');

const GREAT_SUCCESS_PROB = 0.5;  // %
const PROTECT_PROB = 35.0;       // %

class EnhancementService {
    /** @param {import('./gamedata').GameDataService} gameData */
    constructor(gameData) {
        this.gameData = gameData;
    }

    /* ═══════ 유저 검사 (마이그레이션) ═══════ */
    _ensureSword(user) {
        if (!user.sword.weaponType || !user.sword.imageName) {
            const r = getRandomWeaponImage(user.sword.level, user.sword.weaponType, this.gameData);
            user.sword.weaponType = r.weaponType;
            user.sword.imageName = r.imageName;
            if (user.sword.level === 0) user.sword.name = r.name;
        }
    }

    /* ═══════ 강화 ═══════ */
    enhance(userId) {
        const user = this.gameData.getUser(userId);
        this._ensureSword(user);
        const level = user.sword.level;
        if (level >= 15) return { type: 'max' };

        const prob = this.gameData.getProbability(level);
        if (!prob) return { type: 'error', level };

        if (user.money < prob.cost) return { type: 'no_money', cost: prob.cost, balance: user.money };

        user.money -= prob.cost;
        user.totalEnhanceCount = (user.totalEnhanceCount || 0) + 1;
        user.totalMoneySpent = (user.totalMoneySpent || 0) + prob.cost;

        const roll = Math.random() * 100;
        const isGreat = Math.random() * 100 <= GREAT_SUCCESS_PROB;

        // 대성공
        if (isGreat) {
            const oldLevel = user.sword.level;
            user.sword.level = Math.min(user.sword.level + 3, 15);
            const w = getRandomWeaponImage(user.sword.level, user.sword.weaponType, this.gameData);
            user.sword.imageName = w.imageName;
            user.sword.name = w.name;
            user.sword.weaponType = w.weaponType;

            if (user.sword.level > user.maxLevel) user.maxLevel = user.sword.level;
            this.gameData.saveGameData();
            return {
                type: 'great_success', oldLevel, newLevel: user.sword.level,
                increase: user.sword.level - oldLevel, sword: { ...user.sword },
                cost: prob.cost, balance: user.money,
                chat: this.gameData.getChat(user.sword.level, 'success'),
                lore: getWeaponLore(user.sword.weaponType, user.sword.level, this.gameData)
            };
        }

        // 성공
        if (roll <= prob.success) {
            const oldLevel = user.sword.level;
            user.sword.level++;
            const w = getRandomWeaponImage(user.sword.level, user.sword.weaponType, this.gameData);
            user.sword.imageName = w.imageName;
            user.sword.name = w.name;
            user.sword.weaponType = w.weaponType;

            if (user.sword.level > user.maxLevel) user.maxLevel = user.sword.level;
            this.gameData.saveGameData();
            return {
                type: 'success', oldLevel, newLevel: user.sword.level,
                sword: { ...user.sword }, cost: prob.cost, balance: user.money,
                chat: this.gameData.getChat(user.sword.level, 'success'),
                lore: getWeaponLore(user.sword.weaponType, user.sword.level, this.gameData)
            };
        }

        // 실패 → 파괴방어 체크
        if (Math.random() * 100 <= PROTECT_PROB) {
            const maintainImage = getRandomImage('강화_유지_');
            this.gameData.saveGameData();
            return {
                type: 'protected', level: user.sword.level, sword: { ...user.sword },
                cost: prob.cost, balance: user.money,
                chat: this.gameData.getChat(user.sword.level, 'maintain'),
                imageOverride: maintainImage
            };
        }

        // 파괴
        const destroyed = { level: user.sword.level, name: user.sword.name };
        user.sword.level = 0;
        const w = getRandomWeaponImage(0, null, this.gameData);
        user.sword.imageName = w.imageName;
        user.sword.name = w.name;
        user.sword.weaponType = w.weaponType;

        user.totalDestroyCount = (user.totalDestroyCount || 0) + 1;
        this.gameData.saveGameData();
        return {
            type: 'destroy', destroyedLevel: destroyed.level, destroyedName: destroyed.name,
            sword: { ...user.sword }, cost: prob.cost, balance: user.money,
            chat: this.gameData.getChat(Math.min(destroyed.level + 1, 15), 'fail'),
            imageOverride: getRandomImage('강화_실패_')
        };
    }

    /* ═══════ 판매 ═══════ */
    sell(userId) {
        const user = this.gameData.getUser(userId);
        this._ensureSword(user);
        if (user.sword.level === 0) return { type: 'no_sword' };

        const prob = this.gameData.getProbability(user.sword.level);
        const basePrice = prob ? prob.sellPrice : user.sword.level * 10000;
        const multiplier = 0.9 + Math.random() * 0.6;
        let finalPrice = Math.round(basePrice * multiplier) + Math.floor(Math.random() * 101) - 50;
        if (finalPrice < 0) finalPrice = 0;

        const soldName = user.sword.name, soldLevel = user.sword.level;
        user.money += finalPrice;
        user.sword.level = 0;
        const w = getRandomWeaponImage(0, null, this.gameData);
        user.sword.imageName = w.imageName;
        user.sword.name = w.name;
        user.sword.weaponType = w.weaponType;

        const comment = multiplier >= 1.3 ? '상태가 아주 훌륭하군요! 값을 더 쳐드리겠습니다.' :
            multiplier <= 1.0 ? '흠... 흠집이 좀 있네요. 많이는 못 드립니다.' :
            '적당한 물건이군요. 시세대로 드리겠습니다.';

        this.gameData.saveGameData();
        return { type: 'sold', soldName, soldLevel, basePrice, finalPrice, multiplier, comment, balance: user.money };
    }

    /* ═══════ 출석체크 ═══════ */
    checkAttendance(userId) {
        const user = this.gameData.getUser(userId);
        const now = Date.now();
        const last = user.lastAttendance ? new Date(user.lastAttendance).getTime() : 0;
        const hourMs = 3600000;
        if (now - last >= hourMs) {
            const reward = 1000;
            user.money += reward;
            user.lastAttendance = new Date().toISOString();
            this.gameData.saveGameData();
            return { type: 'ok', reward, balance: user.money };
        }
        const remaining = hourMs - (now - last);
        return { type: 'wait', min: Math.floor(remaining / 60000), sec: Math.floor((remaining % 60000) / 1000) };
    }

    /* ═══════ 돈내놔 ═══════ */
    giveMeMoney(userId) {
        const user = this.gameData.getUser(userId);
        const amount = Math.floor(Math.random() * 2500) + 1;
        user.money += amount;
        this.gameData.saveGameData();
        return { amount, balance: user.money };
    }

    /* ═══════ 슬롯 머신 ═══════ */
    slot(userId, bet) {
        const user = this.gameData.getUser(userId);
        if (bet <= 0) return { type: 'error', msg: '배팅 금액은 0보다 커야 합니다.' };
        if (user.money < bet) return { type: 'no_money', balance: user.money };

        user.money -= bet;
        const SYMS = ['🍒', '🍋', '🍇', '💎', '7️⃣'];
        const s = [0, 1, 2].map(() => SYMS[Math.floor(Math.random() * SYMS.length)]);
        let payout = 0, msg = '꽝!', resultType = 'lose';

        if (s[0] === '7️⃣' && s[1] === '7️⃣' && s[2] === '7️⃣') { payout = bet * 77; msg = 'Jackpot! (77배)'; resultType = 'jackpot'; }
        else if (s[0] === '💎' && s[1] === '💎' && s[2] === '💎') { payout = bet * 50; msg = 'Diamond! (50배)'; resultType = 'diamond'; }
        else if (s[0] === s[1] && s[1] === s[2]) { payout = bet * 10; msg = 'Triple! (10배)'; resultType = 'triple'; }
        else if (s[0] === s[1] || s[1] === s[2] || s[0] === s[2]) { payout = bet * 2; msg = 'Double! (2배)'; resultType = 'double'; }

        if (payout > 0) user.money += payout;
        this.gameData.saveGameData();
        return { type: resultType, symbols: s, bet, payout, msg, balance: user.money };
    }

    /* ═══════ 홀짝 ═══════ */
    oddEven(userId, choice, bet) {
        const user = this.gameData.getUser(userId);
        if (bet <= 0) return { type: 'error', msg: '배팅 금액은 0보다 커야 합니다.' };
        if (user.money < bet) return { type: 'no_money', balance: user.money };

        user.money -= bet;
        const isOdd = Math.random() < 0.5;
        const result = isOdd ? '홀' : '짝';
        const dice = isOdd ? [1, 3, 5][Math.floor(Math.random() * 3)] : [2, 4, 6][Math.floor(Math.random() * 3)];
        const win = choice === result;
        let payout = 0;
        if (win) { payout = bet * 2; user.money += payout; }
        this.gameData.saveGameData();
        return { type: win ? 'win' : 'lose', choice, result, dice, bet, payout, balance: user.money };
    }

    /* ═══════ 가위바위보 ═══════ */
    rps(userId, choice, bet) {
        const RPS = ['가위', '바위', '보'];
        const user = this.gameData.getUser(userId);
        if (bet <= 0) return { type: 'error', msg: '배팅 금액은 0보다 커야 합니다.' };
        if (!RPS.includes(choice)) return { type: 'error', msg: '가위, 바위, 보 중 하나를 선택하세요.' };
        if (user.money < bet) return { type: 'no_money', balance: user.money };

        user.money -= bet;
        const botChoice = RPS[Math.floor(Math.random() * 3)];
        const r = (RPS.indexOf(choice) - RPS.indexOf(botChoice) + 3) % 3;
        let payout = 0, resultType = '';
        if (r === 0) { payout = bet; user.money += payout; resultType = 'draw'; }
        else if (r === 1) { payout = bet * 2; user.money += payout; resultType = 'win'; }
        else { resultType = 'lose'; }
        this.gameData.saveGameData();
        return { type: resultType, userChoice: choice, botChoice, bet, payout, balance: user.money };
    }

    /* ═══════ 배틀 ═══════ */
    battle(userId, targetUserId) {
        const user = this.gameData.getUser(userId);
        this._ensureSword(user);
        const today = new Date().toDateString();
        if (user.dailyBattleDate !== today) { user.dailyBattleCount = 0; user.dailyBattleDate = today; }
        if (user.dailyBattleCount >= 10) return { type: 'limit', remaining: 0 };

        const target = this.gameData.getUser(targetUserId);
        const myLevel = user.sword.level;
        const targetLevel = target.sword.level;
        let winChance = Math.max(5, Math.min(95, 50 + (myLevel - targetLevel) * 5));
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
            type: isWin ? 'win' : 'lose',
            myLevel, mySword: user.sword.name,
            targetLevel, targetSword: target.sword.name,
            winChance, reward, remaining, balance: user.money,
            battleImage: getRandomImage('배틀_시작_')
        };
    }
}

module.exports = { EnhancementService };
