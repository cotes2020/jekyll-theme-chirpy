/**
 * /character 슬래시 빌더 — 캐릭터 관리 + 기억 관리 통합
 *
 * 서브커맨드 그룹 구조:
 *   /character 카드 → list, switch, info, reset, image, history
 *   /character 기억 → 확인, 저장, 수정, 핫로그
 */
import { SlashCommandBuilder, Locale } from 'discord.js';

const EN = Locale.EnglishUS;
const enUS = (s: string): Record<string, string> => ({ [EN]: s });

export const characterCommand = () =>
  new SlashCommandBuilder()
    .setName('character')
    .setDescription('캐릭터 · 기억 관리')
    .setDescriptionLocalizations(enUS('Character & memory management'))

    // ── 캐릭터 카드 관리 ─────────────────────────────────────
    .addSubcommandGroup((g) =>
      g
        .setName('카드')
        .setDescription('활성 캐릭터 전환, 정보, 이미지')
        .setDescriptionLocalizations(enUS('Switch, view, and image character'))
        .addSubcommand((sub) =>
          sub
            .setName('list')
            .setDescription('등록된 캐릭터 목록과 현재 활성 캐릭터 확인')
            .setDescriptionLocalizations(enUS('List characters and show current active')),
        )
        .addSubcommand((sub) =>
          sub
            .setName('switch')
            .setDescription('이 DM/채널의 캐릭터를 전환')
            .setDescriptionLocalizations(enUS('Switch character for this DM/channel'))
            .addStringOption((opt) =>
              opt
                .setName('slug')
                .setDescription('전환할 캐릭터 슬러그 (예: yawn)')
                .setDescriptionLocalizations(enUS('Character slug (e.g. yawn)'))
                .setRequired(true)
                .setAutocomplete(true),
            ),
        )
        .addSubcommand((sub) =>
          sub
            .setName('info')
            .setDescription('캐릭터 카드 상세 정보 (비우면 현재 활성)')
            .setDescriptionLocalizations(enUS('Character card details (empty = current)'))
            .addStringOption((opt) =>
              opt
                .setName('slug')
                .setDescription('조회할 슬러그 (비우면 이 곳의 활성)')
                .setDescriptionLocalizations(enUS('Slug (empty = active here)')),
            ),
        )
        .addSubcommand((sub) =>
          sub
            .setName('reset')
            .setDescription('이 DM/채널 매핑 제거 → default 캐릭터로 복귀')
            .setDescriptionLocalizations(enUS('Remove mapping → fall back to default')),
        )
        .addSubcommand((sub) =>
          sub
            .setName('image')
            .setDescription('현재 활성 캐릭터의 외형으로 이미지 생성')
            .setDescriptionLocalizations(enUS('Generate image based on active character appearance'))
            .addStringOption((opt) =>
              opt
                .setName('상황')
                .setNameLocalizations(enUS('scene'))
                .setDescription('상황 · 포즈 · 배경 (비우면 외형만, 영어 권장)')
                .setDescriptionLocalizations(enUS('Scene/pose/background (empty = appearance only)'))
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
        )
        .addSubcommand((sub) =>
          sub
            .setName('history')
            .setDescription('자동 생성된 씬 이미지 캐시 조회 (최근 10개)')
            .setDescriptionLocalizations(enUS('View auto-generated scene image cache (last 10)')),
        ),
    )

    // ── 기억 관리 ────────────────────────────────────────────
    .addSubcommandGroup((g) =>
      g
        .setName('기억')
        .setDescription('캐릭터별 메모리 확인 · 저장 · 수정')
        .setDescriptionLocalizations(enUS('View, save, and edit character memory'))
        .addSubcommand((sub) =>
          sub
            .setName('확인')
            .setNameLocalizations(enUS('view'))
            .setDescription('저장된 나에 대한 정보를 출력합니다.')
            .setDescriptionLocalizations(enUS('View saved information about you')),
        )
        .addSubcommand((sub) =>
          sub
            .setName('저장')
            .setNameLocalizations(enUS('save'))
            .setDescription('지금까지의 대화를 memo 레포에 즉시 커밋합니다.')
            .setDescriptionLocalizations(enUS('Immediately commit conversation to memo repo')),
        )
        .addSubcommand((sub) =>
          sub
            .setName('수정')
            .setNameLocalizations(enUS('edit'))
            .setDescription('user.md를 AI 도움으로 수정합니다.')
            .setDescriptionLocalizations(enUS('Edit user.md with AI assistance'))
            .addStringOption((opt) =>
              opt
                .setName('내용')
                .setNameLocalizations(enUS('content'))
                .setDescription('추가하거나 수정할 사항')
                .setDescriptionLocalizations(enUS('What to add or modify'))
                .setRequired(true),
            ),
        )
        .addSubcommand((sub) =>
          sub
            .setName('핫로그')
            .setNameLocalizations(enUS('hotlog'))
            .setDescription('최근 중요 기억들을 확인합니다.')
            .setDescriptionLocalizations(enUS('View recent important memories')),
        ),
    );
