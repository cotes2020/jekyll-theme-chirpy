/**
 * /character 슬래시 빌더 — DM/채널별 활성 캐릭터 관리.
 */
import { SlashCommandBuilder, Locale } from 'discord.js';

const EN = Locale.EnglishUS;
const enUS = (s: string): Record<string, string> => ({ [EN]: s });

export const characterCommand = () =>
  new SlashCommandBuilder()
    .setName('character')
    .setDescription('DM/채널별 활성 캐릭터 관리')
    .setDescriptionLocalizations(enUS('Manage active character per DM/channel'))
    .addSubcommand((sub) =>
      sub
        .setName('list')
        .setDescription('등록된 캐릭터 목록과 현재 활성 캐릭터 확인')
        .setDescriptionLocalizations(
          enUS('List registered characters and show current active'),
        ),
    )
    .addSubcommand((sub) =>
      sub
        .setName('switch')
        .setDescription('이 DM/채널의 캐릭터를 전환')
        .setDescriptionLocalizations(enUS('Switch character for this DM/channel'))
        .addStringOption((opt) =>
          opt
            .setName('slug')
            .setDescription('전환할 캐릭터 슬러그 (예: yawn, timeto)')
            .setDescriptionLocalizations(
              enUS('Character slug (e.g. yawn, timeto)'),
            )
            .setRequired(true),
        ),
    )
    .addSubcommand((sub) =>
      sub
        .setName('info')
        .setDescription('캐릭터 카드 상세 정보 (비우면 현재 활성)')
        .setDescriptionLocalizations(
          enUS('Show character card details (empty = current)'),
        )
        .addStringOption((opt) =>
          opt
            .setName('slug')
            .setDescription('조회할 슬러그 (비우면 이 곳의 활성)')
            .setDescriptionLocalizations(
              enUS('Slug (empty = active for this DM/channel)'),
            ),
        ),
    )
    .addSubcommand((sub) =>
      sub
        .setName('reset')
        .setDescription('이 DM/채널 매핑 제거 → default 캐릭터로 복귀')
        .setDescriptionLocalizations(
          enUS('Remove mapping for this DM/channel → fall back to default'),
        ),
    );
