/**
 * GameDataService — C# GameDataService → Node.js 이식
 * JSON 파일 기반 게임 데이터 관리 (유저 데이터, 확률, 대사, 메시지)
 */
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', '..', 'data');
const GAMEDATA_PATH = path.join(DATA_DIR, 'gamedata.json');
const PROBABILITIES_PATH = path.join(DATA_DIR, 'probabilities.json');
const CHAT_PATH = path.join(DATA_DIR, 'chat.json');
const MESSAGES_PATH = path.join(DATA_DIR, 'bot_messages.json');

class GameDataService {
    constructor() {
        /** @type {Object<string, import('./models').UserData>} */
        this.users = {};
        this.probabilities = [];
        this.chatData = {};
        this.messages = {};
        this.weaponLores = {}; // 무기 설명 데이터 추가
        this._saveTimer = null;
    }

    async initialize() {
        // Load probabilities
        try {
            const rawProb = JSON.parse(fs.readFileSync(PROBABILITIES_PATH, 'utf-8'));
            this.probabilities = rawProb.map(p => ({
                level: p.Level,
                success: p.Success,
                cost: p.Cost,
                sellPrice: p.SellPrice
            }));
        } catch (e) {
            console.error('[GameData] probabilities.json 로드 실패:', e.message);
            this.probabilities = [];
        }

        // Load chat data
        try {
            this.chatData = JSON.parse(fs.readFileSync(CHAT_PATH, 'utf-8'));
        } catch (e) {
            console.error('[GameData] chat.json 로드 실패:', e.message);
            this.chatData = {};
        }

        // Load bot messages
        try {
            this.messages = JSON.parse(fs.readFileSync(MESSAGES_PATH, 'utf-8'));
        } catch (e) {
            console.error('[GameData] bot_messages.json 로드 실패:', e.message);
            this.messages = {};
        }

        // Load weapon lores
        try {
            const enhancementPath = path.join(__dirname, '..', '..', 'resources', 'img', 'enhancement');
            if (fs.existsSync(enhancementPath)) {
                this._loadLoresRecursive(enhancementPath);
                console.log(`[GameData] WeaponLores 데이터 로드 완료: ${Object.keys(this.weaponLores).length}개 무기`);
            }
        } catch (e) {
            console.error('[GameData] 무기 Lore 데이터 로드 실패:', e.message);
        }

        // Load game data (users)
        try {
            if (fs.existsSync(GAMEDATA_PATH)) {
                this.users = JSON.parse(fs.readFileSync(GAMEDATA_PATH, 'utf-8'));
                console.log(`[GameData] ${Object.keys(this.users).length}명의 유저 데이터 로드 완료`);
            }
        } catch (e) {
            console.error('[GameData] gamedata.json 로드 실패:', e.message);
            this.users = {};
        }

        // Auto-save every 5 minutes
        this._saveTimer = setInterval(() => this.saveGameData(), 5 * 60 * 1000);
    }

    _loadLoresRecursive(dir) {
        const files = fs.readdirSync(dir);
        for (const file of files) {
            const fullPath = path.join(dir, file);
            if (fs.statSync(fullPath).isDirectory()) {
                this._loadLoresRecursive(fullPath);
            } else if (file.endsWith('_data.json')) {
                try {
                    let raw = fs.readFileSync(fullPath, 'utf8');
                    raw = raw.replace(/^\uFEFF/, ''); // Strip BOM if present
                    const loreData = JSON.parse(raw);
                    if (loreData && loreData.weaponName) {
                        this.weaponLores[loreData.weaponName] = loreData;
                    }
                } catch (e) {
                    console.error(`[GameData] Failed to parse ${fullPath}:`, e.message);
                }
            }
        }
    }

    /** 유저 데이터 가져오기 (없으면 생성) */
    getUser(userId) {
        const id = String(userId);
        if (!this.users[id]) {
            this.users[id] = createDefaultUser();
        }
        return this.users[id];
    }

    /** 게임 데이터 저장 */
    saveGameData() {
        try {
            if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
            fs.writeFileSync(GAMEDATA_PATH, JSON.stringify(this.users, null, 2), 'utf-8');
            console.log(`[GameData] 저장 완료 (${Object.keys(this.users).length}명)`);
        } catch (e) {
            console.error('[GameData] 저장 실패:', e.message);
        }
    }

    /** 확률 정보 조회 */
    getProbability(level) {
        return this.probabilities.find(p => p.level === level) || null;
    }

    /** 대장장이 대사 랜덤 */
    getChat(level, type) {
        const lvl = String(level);
        if (this.chatData[lvl] && this.chatData[lvl][type]) {
            const arr = this.chatData[lvl][type];
            return arr[Math.floor(Math.random() * arr.length)];
        }
        return '';
    }

    /** 봇 메시지 가져오기 (포맷 지원) */
    getMessage(key, ...args) {
        let msg = this.messages[key] || key;
        args.forEach((arg, i) => {
            msg = msg.replace(new RegExp(`\\{${i}\\}`, 'g'), String(arg));
        });
        return msg;
    }

    destroy() {
        if (this._saveTimer) clearInterval(this._saveTimer);
        this.saveGameData();
    }
}

/* ── 이미지 탐색기 ── */
function getRandomWeaponImage(level, currentWeaponType, gamedata) {
    if (level === 0 || !currentWeaponType) {
        const keys = Object.keys(gamedata.weaponLores);
        currentWeaponType = keys.length > 0 ? keys[Math.floor(Math.random() * keys.length)] : '곡괭이';
    }

    const searchLevel = level === 0 ? 1 : level;
    const enhancementPath = path.join(__dirname, '..', '..', 'resources', 'img', 'enhancement', currentWeaponType);
    
    if (fs.existsSync(enhancementPath)) {
        const files = fs.readdirSync(enhancementPath).filter(f => f.startsWith(`${currentWeaponType}_Lv${searchLevel}_`) && f.endsWith('.png'));
        if (files.length > 0) {
            const file = files[Math.floor(Math.random() * files.length)];
            const parts = path.parse(file).name.split('_');
            const name = parts.length >= 3 ? `${parts[2]} ${parts[0]}` : '알 수 없는 무기';
            return {
                imageName: path.join(currentWeaponType, file),
                name,
                weaponType: currentWeaponType
            };
        }
    }
    
    return { imageName: null, name: '알 수 없는 무기', weaponType: currentWeaponType };
}

function getWeaponLore(weaponType, level, gamedata) {
    if (gamedata.weaponLores[weaponType]) {
        const searchLevel = level === 0 ? 1 : level;
        const stage = gamedata.weaponLores[weaponType].stages?.find(s => s.level === searchLevel);
        if (stage) return stage.lore;
    }
    return '';
}

function getRandomImage(prefix) {
    try {
        const etcPath = path.join(__dirname, '..', '..', 'resources', 'img', 'enhancement', 'Etc');
        if (fs.existsSync(etcPath)) {
            const files = fs.readdirSync(etcPath).filter(f => f.startsWith(prefix) && f.endsWith('.png'));
            if (files.length > 0) {
                return path.join('Etc', files[Math.floor(Math.random() * files.length)]);
            }
        }
    } catch (e) {}
    return null;
}

/* ── 기본 유저 데이터 생성 ── */
function createDefaultUser() {
    return {
        money: 100000,
        sword: { level: 0, name: '이름 없는 검', weaponType: '', imageName: '' },
        maxLevel: 0,
        lastAttendance: null,
        lastGiveMeMoney: null,
        dailyBattleCount: 0,
        dailyBattleDate: null,
        stocks: {},       // { SYMBOL: { amount, avgPrice } }
        totalEnhanceCount: 0,
        totalDestroyCount: 0,
        totalMoneySpent: 0,
    };
}

/* ── 무기 종류 ── */
const WEAPON_TYPES = ['곡괭이', '만월도끼', '장검', '단검', '대검', '창', '활', '지팡이', '망치', '낫'];

function randomWeaponType() {
    return WEAPON_TYPES[Math.floor(Math.random() * WEAPON_TYPES.length)];
}

/* ── 무기 이름 접두어 ── */
const WEAPON_PREFIX = {
    0: ['낡은', '녹슨', '버려진'], 1: ['낡은', '녹슨', '흔한'],
    2: ['평범한', '수리한', '다듬은'], 3: ['괜찮은', '날카로운', '견고한'],
    4: ['튼튼한', '강인한', '빛나는'], 5: ['정교한', '숙련된', '세련된'],
    6: ['묵직한', '강화된', '위엄있는'], 7: ['행운의', '축복받은', '마법의'],
    8: ['푸른빛의', '영롱한', '신비한'], 9: ['웅장한', '전설급', '찬란한'],
    10: ['영웅의', '찬란한', '지배자의'], 11: ['신성한', '심연의', '불멸의'],
    12: ['전설의', '용살자의', '천상의'], 13: ['흉기급', '파멸의', '절망의'],
    14: ['신화급', '세계의', '종말의'], 15: ['천상천하', '우주의', '절대자의'],
};

function generateWeaponName(level, weaponType) {
    const prefixes = WEAPON_PREFIX[level] || WEAPON_PREFIX[0];
    return prefixes[Math.floor(Math.random() * prefixes.length)] + ' ' + weaponType;
}

/* ── 숫자 포맷 ── */
function formatMoney(n) {
    return Number(n).toLocaleString('ko-KR');
}

/* ── 레벨 색상 (Discord embed용 hex) ── */
function getLevelColor(level) {
    if (level >= 14) return 0xFF00FF;
    if (level >= 12) return 0xFF4444;
    if (level >= 10) return 0xFFD700;
    if (level >= 7)  return 0x7C4DFF;
    if (level >= 5)  return 0x00BCD4;
    if (level >= 3)  return 0x4CAF50;
    return 0x9E9E9E;
}

module.exports = {
    GameDataService,
    WEAPON_TYPES,
    randomWeaponType,
    generateWeaponName,
    getRandomWeaponImage,
    getWeaponLore,
    getRandomImage,
    formatMoney,
    getLevelColor,
};
