/**
 * Unity Free Asset Discord Bot
 * - YawnBot과 같은 저장소/폴더를 쓰지만, 토큰/프로세스는 완전히 분리
 */
require('dotenv').config();

const { Client, GatewayIntentBits, EmbedBuilder } = require('discord.js');
const { fetchUnityFreeAssetInfo } = require('./unityFreeFetcher');

const TOKEN = process.env.UNITYFREE_DISCORD_TOKEN || process.env.DISCORD_TOKEN;
const TARGET_CHANNEL_ID = process.env.UNITYFREE_TARGET_CHANNEL_ID || process.env.GITHUB_WEBHOOK_CHANNEL_ID;
const CHECK_INTERVAL_MIN = parseInt(process.env.UNITYFREE_CHECK_INTERVAL_MIN || '60', 10);

if (!TOKEN) {
    console.error('[UnityFreeBot] DISCORD 토큰이 설정되어 있지 않습니다. .env의 UNITYFREE_DISCORD_TOKEN을 확인하세요.');
    process.exit(1);
}

if (!TARGET_CHANNEL_ID) {
    console.error('[UnityFreeBot] 알림을 보낼 채널 ID가 없습니다. .env의 UNITYFREE_TARGET_CHANNEL_ID를 설정하세요.');
    process.exit(1);
}

const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
    ],
});

let lastSentCoupon = null;

async function sendUnityFreeAssetOnce() {
    console.log('[UnityFreeBot] Unity 무료 에셋 정보 확인 중...');
    let info;
    try {
        info = await fetchUnityFreeAssetInfo();
    } catch (err) {
        console.error('[UnityFreeBot] Unity 페이지 요청/파싱 실패:', err.message);
        return;
    }

    if (!info) {
        console.log('[UnityFreeBot] 현재 활성화된 무료 에셋 증정이 없는 것 같습니다.');
        return;
    }

    // 같은 쿠폰 코드는 한 번만 보내기(프로세스 살아 있는 동안)
    if (info.couponCode && lastSentCoupon === info.couponCode) {
        console.log('[UnityFreeBot] 이미 전송했던 쿠폰 코드입니다. 건너뜀:', info.couponCode);
        return;
    }

    const channel = await client.channels.fetch(TARGET_CHANNEL_ID).catch(() => null);
    if (!channel || !channel.isTextBased()) {
        console.error('[UnityFreeBot] 채널을 찾을 수 없거나 텍스트 채널이 아닙니다:', TARGET_CHANNEL_ID);
        return;
    }

    const title = info.assetName
        ? `🎁 이번 주 Unity 무료 에셋: ${info.assetName}`
        : '🎁 이번 주 Unity 무료 에셋';

    const lines = [];
    if (info.assetUrl) lines.push(`[에셋 페이지 열기](${info.assetUrl})`);
    if (info.couponCode) lines.push(`✅ **쿠폰 코드**: \`${info.couponCode}\``);
    if (info.promoText) lines.push(`ℹ️ ${info.promoText}`);

    const embed = new EmbedBuilder()
        .setTitle(title)
        .setDescription(lines.join('\n'))
        .setColor(0x00BCD4);

    if (info.imageUrl) {
        embed.setImage(info.imageUrl);
    }

    await channel.send({ embeds: [embed] });
    console.log('[UnityFreeBot] 무료 에셋 알림 전송 완료.');

    if (info.couponCode) {
        lastSentCoupon = info.couponCode;
    }
}

client.once('clientReady', async () => {
    console.log(`\n  🎁  UnityFreeBot`);
    console.log('  ─────────────────────────');
    console.log(`  로그인: ${client.user.tag}`);
    console.log('');

    // 시작 시 한 번 바로 확인
    await sendUnityFreeAssetOnce();

    // 이후 주기적으로 확인
    const intervalMs = Math.max(5, CHECK_INTERVAL_MIN) * 60 * 1000;
    setInterval(() => {
        void sendUnityFreeAssetOnce();
    }, intervalMs);
});

client.login(TOKEN).catch(err => {
    console.error('[UnityFreeBot] 로그인 실패:', err);
    process.exit(1);
});

