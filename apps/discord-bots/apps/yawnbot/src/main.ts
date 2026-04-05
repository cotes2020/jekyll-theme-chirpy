/**
 * YawnBot — Node.js Discord Bot (game bot)
 * 기존 apps/yawnbot-server/src/index.ts 기반
 */
import './load-env';
import './install-console-timestamps';
import dns from 'node:dns';
import { generateDependencyReport } from '@discordjs/voice';
import sodium from 'libsodium-wrappers';
import { Client, GatewayIntentBits } from 'discord.js';
import { parseCommaSeparatedEnv } from '@discord-bots/common';
import { destroyAllVoiceConnections } from './bot/voice-connection';
import { destroyAllMusicPlayers, setMusicDiscordClient, setMusicPlayFailureReporter } from './bot/music-player';
import type { GenerativeTextClient } from 'karmolab-ai/node';
import { tryCreateGenerativeTextFromEnv } from 'karmolab-ai/node';

import { GameDataService } from './services/gamedata';
import { EnhancementService } from './services/enhancement';
import { StockService } from './services/stock';
import { RaidService } from './services/raid';
import { getImageAttachment } from './bot/attachments';
import { handleMeme } from './bot/meme';
import { handleButtonInteraction } from './bot/buttons';
import { dispatchSlashCommand } from './bot/slash/router';
import { createGithubWebhookApp } from './bot/webhook';
import { startPresenceRotation, stopPresenceRotation } from './bot/presence-rotation';

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
    GatewayIntentBits.GuildVoiceStates,
  ],
});

setMusicPlayFailureReporter(async ({ textChannelId, title, reason }) => {
  try {
    const safeTitle = title.replace(/\*\*/g, '').slice(0, 100);
    const raw = reason.replace(/\s+/g, ' ').trim().slice(0, 180);
    const ch = await client.channels.fetch(textChannelId);
    if (!ch?.isTextBased() || ch.isDMBased()) return;
    await ch.send({
      content: `건너뜀(재생 실패): **${safeTitle}** — ${raw}`.slice(0, 500),
    });
  } catch {
    /* ignore */
  }
});

const gameData = new GameDataService();
const enhancement = new EnhancementService(gameData);
const stock = new StockService(gameData);
const raid = new RaidService(gameData);

const ADMIN_IDS = parseCommaSeparatedEnv(process.env.ADMIN_IDS);

function isAdmin(userId: unknown) {
  return ADMIN_IDS.includes(String(userId));
}

const cursorState = { inFlight: false };

let generativeText: GenerativeTextClient | null = null;
try {
  generativeText = tryCreateGenerativeTextFromEnv();
  if (generativeText) {
    console.log(`[Gemini] AI 초기화 완료 (surface=${generativeText.surface})`);
  }
} catch (e: any) {
  console.warn('[Gemini] 초기화 실패 (선택 기능):', e?.message ?? e);
}

function buildCtx() {
  return {
    client,
    gameData,
    enhancement,
    stock,
    raid,
    getImageAttachment,
    isAdmin,
    generativeText,
    cursorState,
  };
}

client.on('interactionCreate', async (interaction) => {
  const ctx = buildCtx();
  if (interaction.isButton()) {
    await handleButtonInteraction(ctx as any, interaction as any);
    return;
  }
  if (interaction.isChatInputCommand()) {
    await dispatchSlashCommand(ctx as any, interaction as any);
  }
});

client.on('messageCreate', async (message) => {
  if (message.author.bot) return;
  await handleMeme(message as any);
});

const app = createGithubWebhookApp(client as any, gameData as any);

client.once('clientReady', async () => {
  setMusicDiscordClient(client);
  console.log(`\n  ⚔️  YawnBot (Node.js)`);
  console.log(`  ─────────────────────────`);
  console.log(`  로그인: ${client.user?.tag}`);
  console.log(`  서버:   ${client.guilds.cache.size}개`);
  console.log(`  유저:   ${Object.keys(gameData.users).length}명 데이터 로드`);
  console.log('');

  stock.startMarket();

  const webhookChannelIds = parseCommaSeparatedEnv(process.env.GITHUB_WEBHOOK_CHANNEL_ID);
  for (const channelId of webhookChannelIds) {
    const channel = await client.channels.fetch(channelId).catch(() => null);
    if (channel && channel.isTextBased()) {
      const version = process.env.npm_package_version || '1.0.0';
      const greeting = gameData.getMessage('Server_Startup_Greeting', version);
      await channel.send(greeting).catch((e: any) => console.error('[Startup] 인사 메시지 전송 실패:', e?.message ?? e));
    }
  }

  startPresenceRotation(client);
});

async function main() {
  /** Discord 음성 UDP가 IPv6 경로에서만 막히는 환경 완화 (Node 17+) */
  if (typeof dns.setDefaultResultOrder === 'function') {
    dns.setDefaultResultOrder('ipv4first');
    console.log('[voice] DNS: IPv4 우선 (음성 연결 안정화)');
  }
  if (process.env.VOICE_DEBUG === '1') {
    console.log('[voice] VOICE_DEBUG=1 — join 시 [voice] 디버그 로그 출력');
  }

  await gameData.initialize();

  /** 음성 암호화(sodium) 준비 전에 join하면 signalling↔connecting만 반복되는 경우가 많음 */
  console.log('[voice] dependency report:\n' + generateDependencyReport());
  await sodium.ready;
  console.log('[voice] libsodium 준비 완료');

  const WEBHOOK_PORT = process.env.WEBHOOK_PORT || 8080;
  app.listen(WEBHOOK_PORT, () => {
    console.log(`[Webhook] GitHub Webhook 서버 시작: http://0.0.0.0:${WEBHOOK_PORT}/webhook/github`);
  });

  const token = process.env.DISCORD_TOKEN?.trim();
  if (!token) {
    console.error(
        '[YawnBot] DISCORD_TOKEN이 비어 있습니다. apps/yawnbot/.env 에 봇 토큰을 넣으세요. (Discord Developer Portal → 앱 → Bot → Token)',
    );
    process.exit(1);
  }

  try {
    await client.login(token);
  } catch (e: any) {
    if (e?.code === 'TokenInvalid') {
      console.error(
        '[YawnBot] TokenInvalid — 토큰이 만료되었거나 잘못되었습니다. Discord Developer Portal에서 Bot Token을 재발급하고 .env 의 DISCORD_TOKEN을 갱신하세요.',
      );
    }
    throw e;
  }
}

process.on('SIGINT', () => {
  console.log('\n[Shutdown] 종료 중...');
  setMusicDiscordClient(null);
  stopPresenceRotation();
  stock.stopMarket();
  gameData.destroy();
  destroyAllMusicPlayers();
  destroyAllVoiceConnections();
  client.destroy();
  process.exit(0);
});

process.on('SIGTERM', () => {
  setMusicDiscordClient(null);
  stopPresenceRotation();
  stock.stopMarket();
  gameData.destroy();
  destroyAllMusicPlayers();
  destroyAllVoiceConnections();
  client.destroy();
  process.exit(0);
});

main().catch((err) => {
  console.error('[Fatal]', err);
  process.exit(1);
});

