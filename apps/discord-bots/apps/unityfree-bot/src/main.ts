import path from 'path';
import { config } from 'dotenv';
import { Client, EmbedBuilder, GatewayIntentBits } from 'discord.js';
import type { UnityFreeAssetInfo } from './unity/fetchUnityFreeAssetInfo';
import { fetchUnityFreeAssetInfo } from './unity/fetchUnityFreeAssetInfo';

// cwd와 무관하게 이 패키지 루트의 .env만 사용 (yawnbot/.env와 분리)
config({ path: path.join(__dirname, '..', '.env') });

const TOKEN = process.env.UNITYFREE_DISCORD_TOKEN;
const TARGET_CHANNEL_ID = process.env.UNITYFREE_TARGET_CHANNEL_ID;
const CHECK_INTERVAL_MIN = parseInt(process.env.UNITYFREE_CHECK_INTERVAL_MIN || '60', 10);

if (!TOKEN) {
  console.error('[UnityFreeBot] UNITYFREE_DISCORD_TOKEN이 없습니다. (완전 분리: DISCORD_TOKEN fallback 없음)');
  process.exit(1);
}

if (!TARGET_CHANNEL_ID) {
  console.error('[UnityFreeBot] 알림을 보낼 채널 ID가 없습니다. .env의 UNITYFREE_TARGET_CHANNEL_ID를 설정하세요.');
  process.exit(1);
}

const client = new Client({
  intents: [GatewayIntentBits.Guilds],
});

let lastSentCoupon: string | null = null;

function buildUnityFreeEmbed(info: UnityFreeAssetInfo) {
  const title = info.assetName ? `🎁 이번 주 Unity 무료 에셋: ${info.assetName}` : '🎁 이번 주 Unity 무료 에셋';

  const lines: string[] = [];
  if (info.assetUrl) lines.push(`[에셋 페이지 열기](${info.assetUrl})`);
  if (info.couponCode) lines.push(`✅ **쿠폰 코드**: \`${info.couponCode}\``);
  if (info.promoText) lines.push(`ℹ️ ${info.promoText}`);

  const embed = new EmbedBuilder().setTitle(title).setDescription(lines.join('\n')).setColor(0x00bcd4);
  if (info.assetUrl) embed.setURL(info.assetUrl);
  if (info.imageUrl) embed.setImage(info.imageUrl);
  return embed;
}

async function sendUnityFreeAssetOnce({ force = false }: { force?: boolean } = {}) {
  console.log('[UnityFreeBot] Unity 무료 에셋 정보 확인 중...');

  let info: UnityFreeAssetInfo | null;
  try {
    info = await fetchUnityFreeAssetInfo();
  } catch (err: any) {
    console.error('[UnityFreeBot] Unity 페이지 요청/파싱 실패:', err?.message ?? err);
    return;
  }

  if (!info) {
    console.log('[UnityFreeBot] 현재 활성화된 무료 에셋 증정이 없는 것 같습니다.');
    return;
  }

  if (!force && info.couponCode && lastSentCoupon === info.couponCode) {
    console.log('[UnityFreeBot] 이미 전송했던 쿠폰 코드입니다. 건너뜀:', info.couponCode);
    return;
  }

  const channel = await client.channels.fetch(TARGET_CHANNEL_ID!).catch(() => null);
  if (!channel || !channel.isTextBased()) {
    console.error('[UnityFreeBot] 채널을 찾을 수 없거나 텍스트 채널이 아닙니다:', TARGET_CHANNEL_ID);
    return;
  }

  const embed = buildUnityFreeEmbed(info);
  await channel.send({ embeds: [embed] });
  console.log('[UnityFreeBot] 무료 에셋 알림 전송 완료.');

  if (info.couponCode) {
    lastSentCoupon = info.couponCode;
  }
}

client.once('clientReady', async () => {
  console.log(`\n  🎁  UnityFreeBot`);
  console.log('  ─────────────────────────');
  console.log(`  로그인: ${client.user?.tag}`);
  console.log('');

  await sendUnityFreeAssetOnce();

  const intervalMs = Math.max(5, CHECK_INTERVAL_MIN) * 60 * 1000;
  setInterval(() => {
    void sendUnityFreeAssetOnce();
  }, intervalMs);
});

client.on('interactionCreate', async (interaction) => {
  if (!interaction.isChatInputCommand()) return;
  if (interaction.commandName !== 'unityfree') return;

  const force = interaction.options.getBoolean('force') || false;
  await interaction.deferReply({ ephemeral: true });

  let info: UnityFreeAssetInfo | null;
  try {
    info = await fetchUnityFreeAssetInfo();
  } catch (err: any) {
    await interaction.editReply(`가져오기 실패: ${err?.message ?? err}`);
    return;
  }

  if (!info) {
    await interaction.editReply('현재 활성화된 무료 에셋 증정이 없는 것 같습니다.');
    return;
  }

  const coupon = info.couponCode || '(쿠폰 없음)';
  if (!force && info.couponCode && lastSentCoupon === info.couponCode) {
    await interaction.editReply(`이미 전송했던 쿠폰 코드라서 건너뜁니다: \`${coupon}\` (force=true로 강제 가능)`);
    return;
  }

  const channel = await client.channels.fetch(TARGET_CHANNEL_ID!).catch(() => null);
  if (!channel || !channel.isTextBased()) {
    await interaction.editReply(`채널을 찾을 수 없거나 텍스트 채널이 아닙니다: ${TARGET_CHANNEL_ID}`);
    return;
  }

  const embed = buildUnityFreeEmbed(info);
  await channel.send({ embeds: [embed] });
  if (info.couponCode) lastSentCoupon = info.couponCode;

  await interaction.editReply(`전송 완료: \`${coupon}\``);
});

client.login(TOKEN).catch((err) => {
  console.error('[UnityFreeBot] 로그인 실패:', err);
  process.exit(1);
});

