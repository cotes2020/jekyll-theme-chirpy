/**
 * 게임·미니게임·주식·레이드 슬래시 빌더 — deploy-commands.ts 에서 분리.
 */
import { SlashCommandBuilder, Locale } from 'discord.js';

const EN = Locale.EnglishUS;
const enUS = (s: string): Record<string, string> => ({ [EN]: s });

const STOCK_SYMBOLS = [
  { name: '떡락전자 (SAMSUNG)', value: 'SAMSUNG' },
  { name: '화성갈끄니까 (DOGE)', value: 'DOGE' },
  { name: '테슬라 (TESLA)', value: 'TESLA' },
  { name: '사과 (APPLE)', value: 'APPLE' },
  { name: '비트코인 (BITCOIN)', value: 'BITCOIN' },
];

/** 게임 시스템 — 강화, 판매, 정보, 돈, 랭킹, 출첵, 돈내놔 서브커맨드 */
export const gameCommandGroup = () =>
  new SlashCommandBuilder()
    .setName('게임')
    .setDescription('검 강화, 판매, 정보 및 기본 게임 기능')
    .setDescriptionLocalizations(enUS('Sword enhancement, sales, info and game basics'))
    .addSubcommand((sc) =>
      sc
        .setName('강화')
        .setDescription('검을 강화합니다. (확률 존재)')
        .setDescriptionLocalizations(enUS('Enhance your sword (RNG)')),
    )
    .addSubcommand((sc) =>
      sc
        .setName('판매')
        .setDescription('검을 판매하여 돈을 얻습니다.')
        .setDescriptionLocalizations(enUS('Sell your sword for money')),
    )
    .addSubcommand((sc) =>
      sc
        .setName('정보')
        .setDescription('내 검과 재산 정보를 확인합니다.')
        .setDescriptionLocalizations(enUS('View sword and balance')),
    )
    .addSubcommand((sc) =>
      sc
        .setName('돈')
        .setDescription('현재 보유한 돈을 확인합니다.')
        .setDescriptionLocalizations(enUS('Check your balance')),
    )
    .addSubcommand((sc) =>
      sc
        .setName('랭킹')
        .setDescription('전체 유저 랭킹을 확인합니다.')
        .setDescriptionLocalizations(enUS('Leaderboard')),
    )
    .addSubcommand((sc) =>
      sc
        .setName('출첵')
        .setDescription('매일 출석체크 보상을 받습니다.')
        .setDescriptionLocalizations(enUS('Daily attendance reward')),
    )
    .addSubcommand((sc) =>
      sc
        .setName('돈내놔')
        .setDescription('일정 시간마다 랜덤 용돈을 받습니다.')
        .setDescriptionLocalizations(enUS('Random pocket money on cooldown')),
    );

/** 미니게임 — 배틀, 슬롯, 홀짝, 가위바위보 */
export const minigameCommandGroup = () =>
  new SlashCommandBuilder()
    .setName('미니게임')
    .setDescription('배틀, 슬롯, 홀짝, 가위바위보')
    .setDescriptionLocalizations(enUS('Battle, slots, odd/even, rock/paper/scissors'))
    .addSubcommand((sc) =>
      sc
        .setName('배틀')
        .setDescription('다른 유저와 대결합니다.')
        .setDescriptionLocalizations(enUS('Battle another user'))
        .addUserOption((opt) =>
          opt
            .setName('상대')
            .setNameLocalizations(enUS('opponent'))
            .setDescription('대결할 상대를 선택하세요')
            .setDescriptionLocalizations(enUS('Opponent'))
            .setRequired(true),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('슬롯')
        .setDescription('슬롯 머신을 돌립니다.')
        .setDescriptionLocalizations(enUS('Spin the slots'))
        .addIntegerOption((opt) =>
          opt
            .setName('금액')
            .setNameLocalizations(enUS('amount'))
            .setDescription('배팅할 금액')
            .setDescriptionLocalizations(enUS('Bet amount'))
            .setRequired(true)
            .setMinValue(1),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('홀짝')
        .setDescription('홀짝 게임을 합니다.')
        .setDescriptionLocalizations(enUS('Odd or even game'))
        .addStringOption((opt) =>
          opt
            .setName('선택')
            .setNameLocalizations(enUS('pick'))
            .setDescription('홀 또는 짝')
            .setDescriptionLocalizations(enUS('Odd or even'))
            .setRequired(true)
            .addChoices(
              { name: '홀', name_localizations: enUS('Odd'), value: '홀' },
              { name: '짝', name_localizations: enUS('Even'), value: '짝' },
            ),
        )
        .addIntegerOption((opt) =>
          opt
            .setName('금액')
            .setNameLocalizations(enUS('amount'))
            .setDescription('배팅할 금액')
            .setDescriptionLocalizations(enUS('Bet amount'))
            .setRequired(true)
            .setMinValue(1),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('가위바위보')
        .setDescription('가위바위보를 합니다.')
        .setDescriptionLocalizations(enUS('Rock paper scissors'))
        .addStringOption((opt) =>
          opt
            .setName('선택')
            .setNameLocalizations(enUS('pick'))
            .setDescription('가위, 바위, 보')
            .setDescriptionLocalizations(enUS('Rock, paper, or scissors'))
            .setRequired(true)
            .addChoices(
              { name: '가위', name_localizations: enUS('Scissors'), value: '가위' },
              { name: '바위', name_localizations: enUS('Rock'), value: '바위' },
              { name: '보', name_localizations: enUS('Paper'), value: '보' },
            ),
        )
        .addIntegerOption((opt) =>
          opt
            .setName('금액')
            .setNameLocalizations(enUS('amount'))
            .setDescription('배팅할 금액')
            .setDescriptionLocalizations(enUS('Bet amount'))
            .setRequired(true)
            .setMinValue(1),
        ),
    );

/** 주식 — 목록, 차트, 매수, 매도, 내주식 */
export const stockCommandGroup = () =>
  new SlashCommandBuilder()
    .setName('주식')
    .setDescription('주식 시세 조회, 차트, 매수, 매도')
    .setDescriptionLocalizations(enUS('Stock prices, charts, buy, and sell'))
    .addSubcommand((sc) =>
      sc
        .setName('목록')
        .setDescription('현재 상장된 주식 시세를 확인합니다.')
        .setDescriptionLocalizations(enUS('Listed stock prices')),
    )
    .addSubcommand((sc) =>
      sc
        .setName('차트')
        .setDescription('특정 주식의 차트를 확인합니다.')
        .setDescriptionLocalizations(enUS('Stock chart'))
        .addStringOption((opt) =>
          opt
            .setName('종목')
            .setNameLocalizations(enUS('symbol'))
            .setDescription('종목 심볼')
            .setDescriptionLocalizations(enUS('Ticker symbol'))
            .setRequired(true)
            .addChoices(...STOCK_SYMBOLS),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('매수')
        .setDescription('주식을 매수합니다.')
        .setDescriptionLocalizations(enUS('Buy stock'))
        .addStringOption((opt) =>
          opt
            .setName('종목')
            .setNameLocalizations(enUS('symbol'))
            .setDescription('종목 심볼')
            .setDescriptionLocalizations(enUS('Ticker symbol'))
            .setRequired(true)
            .addChoices(...STOCK_SYMBOLS),
        )
        .addIntegerOption((opt) =>
          opt
            .setName('수량')
            .setNameLocalizations(enUS('qty'))
            .setDescription('매수할 수량')
            .setDescriptionLocalizations(enUS('Shares to buy'))
            .setRequired(true)
            .setMinValue(1),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('매도')
        .setDescription('주식을 매도합니다.')
        .setDescriptionLocalizations(enUS('Sell stock'))
        .addStringOption((opt) =>
          opt
            .setName('종목')
            .setNameLocalizations(enUS('symbol'))
            .setDescription('종목 심볼')
            .setDescriptionLocalizations(enUS('Ticker symbol'))
            .setRequired(true)
            .addChoices(...STOCK_SYMBOLS),
        )
        .addIntegerOption((opt) =>
          opt
            .setName('수량')
            .setNameLocalizations(enUS('qty'))
            .setDescription('매도할 수량')
            .setDescriptionLocalizations(enUS('Shares to sell'))
            .setRequired(true)
            .setMinValue(1),
        ),
    )
    .addSubcommand((sc) =>
      sc
        .setName('내주식')
        .setDescription('내 주식 잔고를 확인합니다.')
        .setDescriptionLocalizations(enUS('Your stock holdings')),
    );

/** 레이드 — 정보, 소환, 공격 */
export const raidCommandGroup = () =>
  new SlashCommandBuilder()
    .setName('레이드')
    .setDescription('레이드 정보 확인, 보스 소환, 공격')
    .setDescriptionLocalizations(enUS('Raid info, summon boss, attack'))
    .addSubcommand((sc) =>
      sc
        .setName('정보')
        .setDescription('현재 진행 중인 레이드 정보를 확인합니다.')
        .setDescriptionLocalizations(enUS('Current raid status')),
    )
    .addSubcommand((sc) =>
      sc
        .setName('소환')
        .setDescription('새로운 레이드 보스를 소환합니다.')
        .setDescriptionLocalizations(enUS('Summon a new raid boss')),
    )
    .addSubcommand((sc) =>
      sc
        .setName('공격')
        .setDescription('레이드 보스를 공격합니다.')
        .setDescriptionLocalizations(enUS('Attack the raid boss')),
    );
