// @ts-nocheck
/**
 * YawnBot — Node.js Discord Bot Server
 * C# YawnBot → discord.js v14 이식
 */
import 'dotenv/config';
import { Client, GatewayIntentBits } from 'discord.js';
import { GoogleGenerativeAI, GenerativeModel } from '@google/generative-ai';

import { GameDataService } from './services/gamedata';
import { EnhancementService } from './services/enhancement';
import { StockService } from './services/stock';
import { RaidService } from './services/raid';
import { getImageAttachment } from './bot/attachments';
import { handleMeme } from './bot/meme';
import { handleButtonInteraction } from './bot/buttons';
import { dispatchSlashCommand } from './bot/slash/router';
import { createGithubWebhookApp } from './bot/webhook';

const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
    ],
});

const gameData = new GameDataService();
const enhancement = new EnhancementService(gameData);
const stock = new StockService(gameData);
const raid = new RaidService(gameData);

const ADMIN_IDS = (process.env.ADMIN_IDS || '').split(',').map(s => s.trim()).filter(Boolean);
function isAdmin(userId) {
    return ADMIN_IDS.includes(String(userId));
}

const cursorState = { inFlight: false };

let geminiModel: GenerativeModel | null = null;
try {
    if (process.env.GEMINI_API_KEY) {
        const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
        geminiModel = genAI.getGenerativeModel({ model: process.env.GEMINI_MODEL || 'gemini-2.0-flash' });
        console.log('[Gemini] AI 모델 초기화 완료');
    }
} catch (e) {
    console.warn('[Gemini] 초기화 실패 (선택 기능):', e instanceof Error ? e.message : e);
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
        geminiModel,
        cursorState,
    };
}

client.on('interactionCreate', async interaction => {
    const ctx = buildCtx();
    if (interaction.isButton()) {
        await handleButtonInteraction(ctx, interaction);
        return;
    }
    if (interaction.isChatInputCommand()) {
        await dispatchSlashCommand(ctx, interaction);
    }
});

client.on('messageCreate', async message => {
    if (message.author.bot) return;
    await handleMeme(message);
});

const app = createGithubWebhookApp(client, gameData);

client.once('clientReady', async () => {
    console.log(`\n  ⚔️  YawnBot (Node.js)`);
    console.log(`  ─────────────────────────`);
    console.log(`  로그인: ${client.user.tag}`);
    console.log(`  서버:   ${client.guilds.cache.size}개`);
    console.log(`  유저:   ${Object.keys(gameData.users).length}명 데이터 로드`);
    console.log('');

    stock.startMarket();

    const channelId = process.env.GITHUB_WEBHOOK_CHANNEL_ID;
    if (channelId) {
        const channel = await client.channels.fetch(channelId).catch(() => null);
        if (channel && channel.isTextBased()) {
            const version = process.env.npm_package_version || '1.0.0';
            const greeting = gameData.getMessage('Server_Startup_Greeting', version);
            await channel.send(greeting).catch(e => console.error('[Startup] 인사 메시지 전송 실패:', e.message));
        }
    }
});

async function main() {
    await gameData.initialize();

    const WEBHOOK_PORT = process.env.WEBHOOK_PORT || 8080;
    app.listen(WEBHOOK_PORT, () => {
        console.log(`[Webhook] GitHub Webhook 서버 시작: http://0.0.0.0:${WEBHOOK_PORT}/webhook/github`);
    });

    await client.login(process.env.DISCORD_TOKEN);
}

process.on('SIGINT', () => {
    console.log('\n[Shutdown] 종료 중...');
    stock.stopMarket();
    gameData.destroy();
    client.destroy();
    process.exit(0);
});

process.on('SIGTERM', () => {
    stock.stopMarket();
    gameData.destroy();
    client.destroy();
    process.exit(0);
});

main().catch(err => {
    console.error('[Fatal]', err);
    process.exit(1);
});
