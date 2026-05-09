/**
 * /atkup unity · /atkup news 슬래시 핸들러 (atkup-bot 흡수, TASK-YB-003).
 */
import { MessageFlags, type ChatInputCommandInteraction } from 'discord.js';
import type { BotContext } from './bot-context';
import { triggerUnityFreeOnce } from '../../services/notifiers/unity-free';
import { triggerGeekNewsOnce } from '../../services/notifiers/geeknews';

export async function handleAtkupUnity(
  ctx: BotContext,
  interaction: ChatInputCommandInteraction,
): Promise<void> {
  const force = interaction.options.getBoolean('force') ?? false;
  await interaction.deferReply({ flags: MessageFlags.Ephemeral });

  const result = await triggerUnityFreeOnce(ctx.client, { force });

  switch (result.status) {
    case 'no_channel':
      await interaction.editReply(
        'YAWNBOT_UNITY_FREE_CHANNEL_ID 가 .env 에 설정되어 있지 않아 전송할 수 없습니다.',
      );
      return;
    case 'fetch_failed':
      await interaction.editReply(`Unity 페이지 가져오기 실패: ${result.error ?? '(원인 미상)'}`);
      return;
    case 'no_data':
      await interaction.editReply('현재 활성화된 무료 에셋 증정이 없는 것 같습니다.');
      return;
    case 'dedup': {
      const coupon = result.info?.couponCode || '(쿠폰 없음)';
      await interaction.editReply(
        `이미 전송했던 쿠폰 코드라서 건너뜁니다: \`${coupon}\` (force 옵션으로 강제 전송 가능)`,
      );
      return;
    }
    case 'channel_unreachable':
      await interaction.editReply(
        '채널을 찾을 수 없거나 메시지를 보낼 수 없습니다 (YAWNBOT_UNITY_FREE_CHANNEL_ID 확인).',
      );
      return;
    case 'sent': {
      const coupon = result.info?.couponCode || '(쿠폰 없음)';
      await interaction.editReply(`Unity 무료 에셋 전송 완료: \`${coupon}\``);
      return;
    }
  }
}

export async function handleAtkupNews(
  ctx: BotContext,
  interaction: ChatInputCommandInteraction,
): Promise<void> {
  const count = interaction.options.getInteger('count') ?? 10;
  await interaction.deferReply({ flags: MessageFlags.Ephemeral });

  let result: Awaited<ReturnType<typeof triggerGeekNewsOnce>>;
  try {
    result = await triggerGeekNewsOnce(ctx.client, count);
  } catch (err: any) {
    await interaction.editReply(`긱 뉴스 가져오기 실패: ${err?.message ?? err}`);
    return;
  }

  switch (result.status) {
    case 'no_channel':
      await interaction.editReply(
        'YAWNBOT_GEEKNEWS_CHANNEL_ID 가 .env 에 설정되어 있지 않아 전송할 수 없습니다.',
      );
      return;
    case 'channel_unreachable':
      await interaction.editReply(
        '채널을 찾을 수 없거나 메시지를 보낼 수 없습니다 (YAWNBOT_GEEKNEWS_CHANNEL_ID 확인).',
      );
      return;
    case 'sent':
      await interaction.editReply(`Hacker News 글 ${result.sent}개를 알림 채널에 보냈습니다.`);
      return;
  }
}
