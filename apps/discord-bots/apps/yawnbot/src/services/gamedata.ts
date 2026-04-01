/**
 * GameDataService — C# GameDataService → Node.js 이식
 * JSON 파일 기반 게임 데이터 관리 (유저 데이터, 확률, 대사, 메시지)
 */
import fs from 'fs';
import path from 'path';
import { PKG_ROOT, enhancementImgDir } from '../paths';

const DATA_DIR = path.join(PKG_ROOT, 'data');
const GAMEDATA_PATH = path.join(DATA_DIR, 'gamedata.json');
const PROBABILITIES_PATH = path.join(DATA_DIR, 'probabilities.json');
const CHAT_PATH = path.join(DATA_DIR, 'chat.json');
const MESSAGES_PATH = path.join(DATA_DIR, 'bot_messages.json');

export interface UserSword {
  level: number;
  name: string;
  weaponType: string;
  imageName: string;
}

export interface UserStockHolding {
  amount: number;
  avgPrice: number;
}

export interface UserData {
  money: number;
  sword: UserSword;
  maxLevel: number;
  lastAttendance: string | null;
  lastGiveMeMoney: string | null;
  dailyBattleCount: number;
  dailyBattleDate: string | null;
  stocks: Record<string, UserStockHolding>;
  totalEnhanceCount: number;
  totalDestroyCount: number;
  totalMoneySpent: number;
}

interface ProbabilityRow {
  level: number;
  success: number;
  cost: number;
  sellPrice: number;
}

export class GameDataService {
  users: Record<string, UserData> = {};
  probabilities: ProbabilityRow[] = [];
  chatData: Record<string, Record<string, string[]>> = {};
  messages: Record<string, string> = {};
  weaponLores: Record<string, { stages?: { level: number; lore: string }[] }> = {};
  private _saveTimer: ReturnType<typeof setInterval> | null = null;

  async initialize(): Promise<void> {
    try {
      const rawProb = JSON.parse(fs.readFileSync(PROBABILITIES_PATH, 'utf-8')) as {
        Level: number;
        Success: number;
        Cost: number;
        SellPrice: number;
      }[];
      this.probabilities = rawProb.map((p) => ({
        level: p.Level,
        success: p.Success,
        cost: p.Cost,
        sellPrice: p.SellPrice,
      }));
    } catch (e: unknown) {
      console.error('[GameData] probabilities.json 로드 실패:', e instanceof Error ? e.message : e);
      this.probabilities = [];
    }

    try {
      this.chatData = JSON.parse(fs.readFileSync(CHAT_PATH, 'utf-8'));
    } catch (e: unknown) {
      console.error('[GameData] chat.json 로드 실패:', e instanceof Error ? e.message : e);
      this.chatData = {};
    }

    try {
      this.messages = JSON.parse(fs.readFileSync(MESSAGES_PATH, 'utf-8'));
    } catch (e: unknown) {
      console.error('[GameData] bot_messages.json 로드 실패:', e instanceof Error ? e.message : e);
      this.messages = {};
    }

    try {
      const enhancementPath = enhancementImgDir();
      if (fs.existsSync(enhancementPath)) {
        this._loadLoresRecursive(enhancementPath);
        console.log(`[GameData] WeaponLores 데이터 로드 완료: ${Object.keys(this.weaponLores).length}개 무기`);
      }
    } catch (e: unknown) {
      console.error('[GameData] 무기 Lore 데이터 로드 실패:', e instanceof Error ? e.message : e);
    }

    try {
      if (fs.existsSync(GAMEDATA_PATH)) {
        this.users = JSON.parse(fs.readFileSync(GAMEDATA_PATH, 'utf-8'));
        console.log(`[GameData] ${Object.keys(this.users).length}명의 유저 데이터 로드 완료`);
      }
    } catch (e: unknown) {
      console.error('[GameData] gamedata.json 로드 실패:', e instanceof Error ? e.message : e);
      this.users = {};
    }

    this._saveTimer = setInterval(() => this.saveGameData(), 5 * 60 * 1000);
  }

  private _loadLoresRecursive(dir: string): void {
    const files = fs.readdirSync(dir);
    for (const file of files) {
      const fullPath = path.join(dir, file);
      if (fs.statSync(fullPath).isDirectory()) {
        this._loadLoresRecursive(fullPath);
      } else if (file.endsWith('_data.json')) {
        try {
          let raw = fs.readFileSync(fullPath, 'utf8');
          raw = raw.replace(/^\uFEFF/, '');
          const loreData = JSON.parse(raw) as { weaponName?: string };
          if (loreData && loreData.weaponName) {
            this.weaponLores[loreData.weaponName] = loreData as { stages?: { level: number; lore: string }[] };
          }
        } catch (e: unknown) {
          console.error(`[GameData] Failed to parse ${fullPath}:`, e instanceof Error ? e.message : e);
        }
      }
    }
  }

  getUser(userId: string | number): UserData {
    const id = String(userId);
    if (!this.users[id]) {
      this.users[id] = createDefaultUser();
    }
    return this.users[id];
  }

  saveGameData(): void {
    try {
      if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
      fs.writeFileSync(GAMEDATA_PATH, JSON.stringify(this.users, null, 2), 'utf-8');
      console.log(`[GameData] 저장 완료 (${Object.keys(this.users).length}명)`);
    } catch (e: unknown) {
      console.error('[GameData] 저장 실패:', e instanceof Error ? e.message : e);
    }
  }

  getProbability(level: number): ProbabilityRow | null {
    return this.probabilities.find((p) => p.level === level) || null;
  }

  getChat(level: number, type: string): string {
    const lvl = String(level);
    if (this.chatData[lvl] && this.chatData[lvl][type]) {
      const arr = this.chatData[lvl][type];
      return arr[Math.floor(Math.random() * arr.length)];
    }
    return '';
  }

  getMessage(key: string, ...args: unknown[]): string {
    let msg = this.messages[key] || key;
    args.forEach((arg, i) => {
      msg = msg.replace(new RegExp(`\\{${i}\\}`, 'g'), String(arg));
    });
    return msg;
  }

  destroy(): void {
    if (this._saveTimer) clearInterval(this._saveTimer);
    this.saveGameData();
  }
}

export function getRandomWeaponImage(
  level: number,
  currentWeaponType: string | null | undefined,
  gamedata: GameDataService,
): { imageName: string | null; name: string; weaponType: string } {
  let wtype = currentWeaponType;
  if (level === 0 || !wtype) {
    const keys = Object.keys(gamedata.weaponLores);
    wtype = keys.length > 0 ? keys[Math.floor(Math.random() * keys.length)] : '곡괭이';
  }

  const searchLevel = level === 0 ? 1 : level;
  const enhancementPath = path.join(enhancementImgDir(), wtype);

  if (fs.existsSync(enhancementPath)) {
    const files = fs
      .readdirSync(enhancementPath)
      .filter((f) => f.startsWith(`${wtype}_Lv${searchLevel}_`) && f.endsWith('.png'));
    if (files.length > 0) {
      const file = files[Math.floor(Math.random() * files.length)];
      const parts = path.parse(file).name.split('_');
      const name = parts.length >= 3 ? `${parts[2]} ${parts[0]}` : '알 수 없는 무기';
      return {
        imageName: path.join(wtype, file),
        name,
        weaponType: wtype,
      };
    }
  }

  return { imageName: null, name: '알 수 없는 무기', weaponType: wtype };
}

export function getWeaponLore(weaponType: string, level: number, gamedata: GameDataService): string {
  if (gamedata.weaponLores[weaponType]) {
    const searchLevel = level === 0 ? 1 : level;
    const stage = gamedata.weaponLores[weaponType].stages?.find((s) => s.level === searchLevel);
    if (stage) return stage.lore;
  }
  return '';
}

export function getRandomImage(prefix: string): string | null {
  try {
    const etcPath = path.join(enhancementImgDir(), 'Etc');
    if (fs.existsSync(etcPath)) {
      const files = fs.readdirSync(etcPath).filter((f) => f.startsWith(prefix) && f.endsWith('.png'));
      if (files.length > 0) {
        return path.join('Etc', files[Math.floor(Math.random() * files.length)]);
      }
    }
  } catch {
    /* ignore */
  }
  return null;
}

function createDefaultUser(): UserData {
  return {
    money: 100000,
    sword: { level: 0, name: '이름 없는 검', weaponType: '', imageName: '' },
    maxLevel: 0,
    lastAttendance: null,
    lastGiveMeMoney: null,
    dailyBattleCount: 0,
    dailyBattleDate: null,
    stocks: {},
    totalEnhanceCount: 0,
    totalDestroyCount: 0,
    totalMoneySpent: 0,
  };
}

export const WEAPON_TYPES = ['곡괭이', '만월도끼', '장검', '단검', '대검', '창', '활', '지팡이', '망치', '낫'];

export function randomWeaponType(): string {
  return WEAPON_TYPES[Math.floor(Math.random() * WEAPON_TYPES.length)];
}

const WEAPON_PREFIX: Record<number, string[]> = {
  0: ['낡은', '녹슨', '버려진'],
  1: ['낡은', '녹슨', '흔한'],
  2: ['평범한', '수리한', '다듬은'],
  3: ['괜찮은', '날카로운', '견고한'],
  4: ['튼튼한', '강인한', '빛나는'],
  5: ['정교한', '숙련된', '세련된'],
  6: ['묵직한', '강화된', '위엄있는'],
  7: ['행운의', '축복받은', '마법의'],
  8: ['푸른빛의', '영롱한', '신비한'],
  9: ['웅장한', '전설급', '찬란한'],
  10: ['영웅의', '찬란한', '지배자의'],
  11: ['신성한', '심연의', '불멸의'],
  12: ['전설의', '용살자의', '천상의'],
  13: ['흉기급', '파멸의', '절망의'],
  14: ['신화급', '세계의', '종말의'],
  15: ['천상천하', '우주의', '절대자의'],
};

export function generateWeaponName(level: number, weaponType: string): string {
  const prefixes = WEAPON_PREFIX[level] || WEAPON_PREFIX[0];
  return prefixes[Math.floor(Math.random() * prefixes.length)] + ' ' + weaponType;
}

export function formatMoney(n: number): string {
  return Number(n).toLocaleString('ko-KR');
}

export function getLevelColor(level: number): number {
  if (level >= 14) return 0xff00ff;
  if (level >= 12) return 0xff4444;
  if (level >= 10) return 0xffd700;
  if (level >= 7) return 0x7c4dff;
  if (level >= 5) return 0x00bcd4;
  if (level >= 3) return 0x4caf50;
  return 0x9e9e9e;
}

