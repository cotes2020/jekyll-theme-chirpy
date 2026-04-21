import { SlashCommandBuilder, Locale } from 'discord.js';

const EN = Locale.EnglishUS;
const enUS = (s: string): Record<string, string> => ({ [EN]: s });

export const scheduleCommand = () =>
  new SlashCommandBuilder()
    .setName('일정')
    .setDescription('일정 관리')
    .setDescriptionLocalizations(enUS('Manage schedules'))
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
            .setDescription('삭제할 일정 ID (`/일정 목록` 에서 확인)')
            .setDescriptionLocalizations(enUS('ID of the entry to delete (see /일정 목록)'))
            .setRequired(true)
            .setMaxLength(16),
        ),
    );
