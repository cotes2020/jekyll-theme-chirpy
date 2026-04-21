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
            .setRequired(true)
            .setAutocomplete(true),
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
    )
    .addSubcommand((sub) =>
      sub
        .setName('image-history')
        .setDescription('자동 생성된 씬 이미지 캐시 조회 (최근 10개)')
        .setDescriptionLocalizations(enUS('View auto-generated scene image cache (last 10)')),
    )
    .addSubcommand((sub) =>
      sub
        .setName('image')
        .setDescription('현재 활성 캐릭터의 외형으로 이미지 생성 (상황 비우면 기본 외형)')
        .setDescriptionLocalizations(
          enUS('Generate character image (empty scene = default appearance)'),
        )
        .addStringOption((opt) =>
          opt
            .setName('상황')
            .setNameLocalizations(enUS('scene'))
            .setDescription('상황 · 포즈 · 배경 (비우면 외형만, 영어 권장)')
            .setDescriptionLocalizations(
              enUS('Scene · pose · background (empty=appearance only, English recommended)'),
            )
            .setMaxLength(1500),
        )
        .addStringOption((opt) =>
          opt
            .setName('비율')
            .setNameLocalizations(enUS('aspect'))
            .setDescription('가로세로 비율 (기본 1:1)')
            .setDescriptionLocalizations(enUS('Aspect ratio (default 1:1)'))
            .addChoices(
              { name: '1:1 (정사각)', value: '1:1' },
              { name: '16:9 (가로)', value: '16:9' },
              { name: '9:16 (세로)', value: '9:16' },
              { name: '4:3', value: '4:3' },
              { name: '3:4', value: '3:4' },
            ),
        )
        .addIntegerOption((opt) =>
          opt
            .setName('개수')
            .setNameLocalizations(enUS('count'))
            .setDescription('생성할 이미지 개수 (1~4)')
            .setDescriptionLocalizations(enUS('Number of images (1-4)'))
            .setMinValue(1)
            .setMaxValue(4),
        ),
    );
