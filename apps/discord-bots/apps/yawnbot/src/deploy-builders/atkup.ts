/**
 * /atkup 슬래시 빌더 — Unity 무료 에셋 + 긱 뉴스 (atkup-bot 흡수, TASK-YB-003).
 *
 * 서브커맨드:
 *   /atkup unity [force]  — Unity Publisher Sale 무료 에셋 즉시 전송
 *   /atkup news [count]   — Hacker News 상위 글 즉시 전송
 */
import { SlashCommandBuilder, Locale } from 'discord.js';

const EN = Locale.EnglishUS;
const enUS = (s: string): Record<string, string> => ({ [EN]: s });

export const atkupCommandGroup = () =>
  new SlashCommandBuilder()
    .setName('atkup')
    .setDescription('Unity 무료 에셋 알림 · 긱 뉴스 (설정된 알림 채널로 전송)')
    .setDescriptionLocalizations(
      enUS('Unity free asset notifier · GeekNews (sends to configured channel)'),
    )
    .addSubcommand((sub) =>
      sub
        .setName('unity')
        .setDescription('Unity Publisher Sale 무료 에셋을 확인하고 알림 채널에 보냅니다.')
        .setDescriptionLocalizations(
          enUS('Check Unity Publisher Sale free asset and send to notify channel'),
        )
        .addBooleanOption((opt) =>
          opt
            .setName('force')
            .setDescription('같은 쿠폰이어도 강제로 다시 전송합니다.')
            .setDescriptionLocalizations(enUS('Force resend even if coupon was already sent'))
            .setRequired(false),
        ),
    )
    .addSubcommand((sub) =>
      sub
        .setName('news')
        .setDescription('Hacker News 상위 글을 알림 채널에 보냅니다.')
        .setDescriptionLocalizations(enUS('Send Hacker News top stories to notify channel'))
        .addIntegerOption((opt) =>
          opt
            .setName('count')
            .setDescription('글 개수 (5~15)')
            .setDescriptionLocalizations(enUS('Number of stories (5-15)'))
            .setMinValue(5)
            .setMaxValue(15)
            .setRequired(false),
        ),
    );
