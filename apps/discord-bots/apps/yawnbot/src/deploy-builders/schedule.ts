/**
 * /일정 슬래시 빌더 — 일정·기념일·뉴스키워드 통합
 *
 * 서브커맨드 그룹 구조:
 *   /일정 일정  → 추가, 목록, 삭제
 *   /일정 기념일 → 추가, 목록, 삭제
 *   /일정 키워드 → 추가, 목록, 삭제
 */
import { SlashCommandBuilder, Locale } from 'discord.js';

const EN = Locale.EnglishUS;
const enUS = (s: string): Record<string, string> => ({ [EN]: s });

export const scheduleCommand = () =>
  new SlashCommandBuilder()
    .setName('일정')
    .setDescription('일정 · 기념일 · 뉴스 키워드 관리')
    .setDescriptionLocalizations(enUS('Manage schedules, anniversaries, and news keywords'))

    // ── 일정 ─────────────────────────────────────────────────
    .addSubcommandGroup((g) =>
      g
        .setName('일정')
        .setDescription('예정된 일정 추가·조회·삭제')
        .setDescriptionLocalizations(enUS('Add, view, and delete schedule entries'))
        .addSubcommand((sub) =>
          sub
            .setName('추가')
            .setDescription('새 일정을 추가합니다.')
            .setDescriptionLocalizations(enUS('Add a new schedule entry'))
            .addStringOption((opt) =>
              opt
                .setName('제목')
                .setNameLocalizations(enUS('title'))
                .setDescription('일정 제목')
                .setDescriptionLocalizations(enUS('Schedule title'))
                .setRequired(true)
                .setMaxLength(100),
            )
            .addStringOption((opt) =>
              opt
                .setName('날짜시간')
                .setNameLocalizations(enUS('datetime'))
                .setDescription('일시 (KST, 형식: YYYY-MM-DD HH:MM)')
                .setDescriptionLocalizations(enUS('Date and time in KST (format: YYYY-MM-DD HH:MM)'))
                .setRequired(true)
                .setMaxLength(20),
            )
            .addIntegerOption((opt) =>
              opt
                .setName('알림')
                .setNameLocalizations(enUS('notify'))
                .setDescription('몇 분 전에 알릴지 (기본 10분)')
                .setDescriptionLocalizations(enUS('Minutes before the event to notify (default 10)'))
                .setMinValue(0)
                .setMaxValue(1440),
            ),
        )
        .addSubcommand((sub) =>
          sub
            .setName('목록')
            .setDescription('예정된 일정 목록을 봅니다.')
            .setDescriptionLocalizations(enUS('View upcoming schedule entries')),
        )
        .addSubcommand((sub) =>
          sub
            .setName('삭제')
            .setDescription('일정을 삭제합니다.')
            .setDescriptionLocalizations(enUS('Delete a schedule entry'))
            .addStringOption((opt) =>
              opt
                .setName('id')
                .setDescription('삭제할 일정 ID (`/일정 일정 목록` 에서 확인)')
                .setDescriptionLocalizations(enUS('ID of the entry to delete (see /일정 일정 목록)'))
                .setRequired(true)
                .setMaxLength(16),
            ),
        ),
    )

    // ── 기념일 ───────────────────────────────────────────────
    .addSubcommandGroup((g) =>
      g
        .setName('기념일')
        .setDescription('기념일 추가·조회·삭제')
        .setDescriptionLocalizations(enUS('Add, view, and delete anniversaries'))
        .addSubcommand((sub) =>
          sub
            .setName('목록')
            .setNameLocalizations(enUS('list'))
            .setDescription('기념일 목록 조회')
            .setDescriptionLocalizations(enUS('List anniversaries')),
        )
        .addSubcommand((sub) =>
          sub
            .setName('추가')
            .setNameLocalizations(enUS('add'))
            .setDescription('기념일 추가')
            .setDescriptionLocalizations(enUS('Add anniversary'))
            .addStringOption((o) =>
              o
                .setName('이름')
                .setNameLocalizations(enUS('label'))
                .setDescription('기념일 이름')
                .setDescriptionLocalizations(enUS('Label'))
                .setRequired(true),
            )
            .addIntegerOption((o) =>
              o
                .setName('월')
                .setNameLocalizations(enUS('month'))
                .setDescription('월 (1-12)')
                .setDescriptionLocalizations(enUS('Month'))
                .setRequired(true)
                .setMinValue(1)
                .setMaxValue(12),
            )
            .addIntegerOption((o) =>
              o
                .setName('일')
                .setNameLocalizations(enUS('day'))
                .setDescription('일 (1-31)')
                .setDescriptionLocalizations(enUS('Day'))
                .setRequired(true)
                .setMinValue(1)
                .setMaxValue(31),
            )
            .addIntegerOption((o) =>
              o
                .setName('연도')
                .setNameLocalizations(enUS('year'))
                .setDescription('시작 연도 (N주년 계산용)')
                .setDescriptionLocalizations(enUS('Start year for anniversary count')),
            ),
        )
        .addSubcommand((sub) =>
          sub
            .setName('삭제')
            .setNameLocalizations(enUS('delete'))
            .setDescription('기념일 삭제')
            .setDescriptionLocalizations(enUS('Delete anniversary'))
            .addStringOption((o) =>
              o
                .setName('id')
                .setDescription('목록에서 확인한 ID')
                .setDescriptionLocalizations(enUS('ID from list'))
                .setRequired(true),
            ),
        ),
    )

    // ── 뉴스 키워드 ──────────────────────────────────────────
    .addSubcommandGroup((g) =>
      g
        .setName('키워드')
        .setDescription('뉴스 관심사 키워드 추가·조회·삭제')
        .setDescriptionLocalizations(enUS('Add, view, and delete news interest keywords'))
        .addSubcommand((sub) =>
          sub
            .setName('목록')
            .setNameLocalizations(enUS('list'))
            .setDescription('키워드 목록')
            .setDescriptionLocalizations(enUS('List keywords')),
        )
        .addSubcommand((sub) =>
          sub
            .setName('추가')
            .setNameLocalizations(enUS('add'))
            .setDescription('키워드 추가')
            .setDescriptionLocalizations(enUS('Add keyword'))
            .addStringOption((o) =>
              o
                .setName('키워드')
                .setNameLocalizations(enUS('keyword'))
                .setDescription('관심사 키워드')
                .setDescriptionLocalizations(enUS('Interest keyword'))
                .setRequired(true),
            ),
        )
        .addSubcommand((sub) =>
          sub
            .setName('삭제')
            .setNameLocalizations(enUS('delete'))
            .setDescription('키워드 삭제')
            .setDescriptionLocalizations(enUS('Delete keyword'))
            .addStringOption((o) =>
              o
                .setName('id')
                .setDescription('목록에서 확인한 ID')
                .setDescriptionLocalizations(enUS('ID from list'))
                .setRequired(true),
            ),
        ),
    );
