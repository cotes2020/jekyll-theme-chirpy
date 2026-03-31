// @ts-nocheck
/**
 * YawnBot — Node.js Discord Bot Server
 * C# YawnBot → discord.js v14 이식
 */
import 'dotenv/config';
import {
    Client,
    GatewayIntentBits,
    EmbedBuilder,
    Collection,
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    StringSelectMenuBuilder,
} from 'discord.js';
import type { ChatInputCommandInteraction, Message, StringSelectMenuInteraction } from 'discord.js';
import express from 'express';
import fs from 'fs';
import path from 'path';
import { spawn, execFile } from 'child_process';
import { GoogleGenerativeAI, GenerativeModel } from '@google/generative-ai';

import { GameDataService, formatMoney, getLevelColor, getWeaponLore, getRandomImage } from './services/gamedata';
import { EnhancementService } from './services/enhancement';
import { StockService } from './services/stock';
import { RaidService } from './services/raid';
import { enhancementImgDir, memeImgDir, cursorRunnerScript } from './paths';

const ENHANCEMENT_DIR = enhancementImgDir();

function getImageAttachment(imageRelativePath) {
    if (!imageRelativePath) return null;
    const fullPath = path.join(ENHANCEMENT_DIR, imageRelativePath);
    if (fs.existsSync(fullPath)) {
        const name = path.basename(fullPath);
        return { file: fullPath, name };
    }
    return null;
}

/* ── Discord 클라이언트 ── */
const client = new Client({
    intents: [
        GatewayIntentBits.Guilds,
        GatewayIntentBits.GuildMessages,
        GatewayIntentBits.MessageContent,
    ],
});

/* ── 서비스 초기화 ── */
const gameData = new GameDataService();
const enhancement = new EnhancementService(gameData);
const stock = new StockService(gameData);
const raid = new RaidService(gameData);

/* ── 관리자 ID ── */
const ADMIN_IDS = (process.env.ADMIN_IDS || '').split(',').map(s => s.trim()).filter(Boolean);

function isAdmin(userId) { return ADMIN_IDS.includes(String(userId)); }

/** Cursor 로컬 에이전트 동시 실행 방지 */
let cursorEditInFlight = false;

const CURSOR_RUNNER_PATH = cursorRunnerScript();

function getCursorMaxPromptChars() {
    return parseInt(process.env.CURSOR_MAX_PROMPT_CHARS || '2000', 10);
}

/**
 * Cursor `cursor/ask_question` → 디스코드 셀렉트(최대 25개). 여러 번 연속으로 올 수 있음(순차 처리).
 */
async function discordAnswerCursorQuestion(
    interaction: ChatInputCommandInteraction,
    payload: { rpcId?: unknown; params?: Record<string, unknown> },
) {
    const params = payload.params || {};
    const raw = (params.options ?? params.choices ?? []) as unknown[];
    const lines = Array.isArray(raw) ? raw : [];
    const selectOptions = lines.slice(0, 25).map((o, i) => {
        if (typeof o === 'string') {
            return { label: o.slice(0, 100), value: String(i) };
        }
        const label = String(o.label ?? o.title ?? o.text ?? `선택 ${i + 1}`).slice(0, 100);
        const out = { label, value: String(i) };
        if (o.description != null) out.description = String(o.description).slice(0, 100);
        return out;
    });
    if (selectOptions.length === 0) {
        return { cancelled: true };
    }
    const heading = params.title || params.question || '에이전트 질문';
    const embed = new EmbedBuilder()
        .setTitle('에이전트 질문')
        .setDescription(truncateDiscordDescription(String(heading)))
        .setColor(0x5865F2);
    const customId = `cursor_q_${String(payload.rpcId)}`;
    const menu = new StringSelectMenuBuilder()
        .setCustomId(customId)
        .setPlaceholder('답을 선택하세요')
        .addOptions(selectOptions);
    const row = new ActionRowBuilder().addComponents(menu);
    const reply = await interaction.followUp({
        embeds: [embed],
        components: [row],
        ephemeral: true,
        fetchReply: true,
    });
    const uid = interaction.user.id;
    try {
        const comp = (await reply.awaitMessageComponent({
            filter: (i): i is StringSelectMenuInteraction =>
                i.user.id === uid && i.customId === customId,
            time: 600_000,
        })) as StringSelectMenuInteraction;
        const idx = parseInt(comp.values[0], 10);
        await comp.update({ components: [] });
        return { selectedIndex: idx };
    } catch {
        try {
            await reply.edit({ components: [] });
        } catch {
            /* ignore */
        }
        return { cancelled: true };
    }
}

/**
 * @param {string} cwd
 * @param {string} prompt
 * @param {string} mode
 * @param {(payload: { type: string, rpcId: unknown, params: object }) => Promise<{ selectedIndex?: number, cancelled?: boolean }>} [onQuestion]
 * @returns {Promise<object>}
 */
function runCursorLocalRunner(
    cwd: string,
    prompt: string,
    mode: string,
    onProgress?: (chunk: string) => void,
    onQuestion?: (msg: { type: string; rpcId: unknown; params: object }) => Promise<{
        selectedIndex?: number;
        cancelled?: boolean;
    }>,
) {
    const innerTimeoutMs = parseInt(process.env.CURSOR_TIMEOUT_MS || '600000', 10);
    /** 러너(자식 node)가 안 끝나도 봇이 영원히 멈추지 않도록 여유(ms) */
    const outerGraceMs = parseInt(process.env.CURSOR_RUNNER_GRACE_MS || '120000', 10);
    const hardCapMs = Math.max(60000, innerTimeoutMs + outerGraceMs);

    return new Promise((resolve, reject) => {
        const args = [
            CURSOR_RUNNER_PATH,
            '--cwd',
            cwd,
            '--prompt',
            prompt,
            '--mode',
            mode || 'agent',
            '--timeoutMs',
            String(innerTimeoutMs),
        ];
        const child = spawn(process.execPath, args, {
            env: {
                ...process.env,
                CURSOR_LOCAL_REPO_DIR: process.env.CURSOR_LOCAL_REPO_DIR || cwd,
                CURSOR_GIT_SNAPSHOT: process.env.CURSOR_GIT_SNAPSHOT || 'baseline',
            },
            windowsHide: true,
            stdio: ['pipe', 'pipe', 'pipe'],
        });
        let out = '';
        let err = '';
        const maxOut = 12 * 1024 * 1024;
        let settled = false;
        let resultJson: Record<string, unknown> | null = null;
        let outBuf = '';

        const killTree = () => {
            if (!child.pid) return;
            try {
                if (process.platform === 'win32') {
                    execFile('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true }, () => {});
                } else {
                    child.kill('SIGKILL');
                }
            } catch {
                try {
                    child.kill('SIGKILL');
                } catch {
                    /* ignore */
                }
            }
        };

        const hardTimer = setTimeout(() => {
            if (settled) return;
            settled = true;
            killTree();
            reject(new Error(`러너 시간 초과 (${Math.round(hardCapMs / 1000)}초). Cursor 에이전트가 멈췄거나 종료되지 않았습니다.`));
        }, hardCapMs);

        child.stdout.on('data', chunk => {
            const s = chunk.toString();
            if (out.length < maxOut) out += s;
            outBuf += s;
            // NDJSON parsing: each line is one JSON object
            while (true) {
                const idx = outBuf.indexOf('\n');
                if (idx === -1) break;
                const line = outBuf.slice(0, idx).trim();
                outBuf = outBuf.slice(idx + 1);
                if (!line) continue;
                try {
                    const msg = JSON.parse(line);
                    if (msg && msg.type === 'assistant_chunk' && typeof msg.text === 'string') {
                        if (typeof onProgress === 'function') onProgress(msg.text);
                    } else if (msg && msg.type === 'cursor_question' && typeof onQuestion === 'function') {
                        void (async () => {
                            try {
                                const ans = await onQuestion(msg);
                                const rid = msg.rpcId;
                                const payload =
                                    ans && ans.cancelled === true
                                        ? { rpcId: rid, cancelled: true }
                                        : { rpcId: rid, selectedIndex: ans.selectedIndex };
                                if (child.stdin && !child.stdin.destroyed) {
                                    child.stdin.write(`${JSON.stringify(payload)}\n`);
                                }
                            } catch (e) {
                                console.error('[cursor question]', e.message || e);
                                if (child.stdin && !child.stdin.destroyed) {
                                    child.stdin.write(`${JSON.stringify({ rpcId: msg.rpcId, cancelled: true })}\n`);
                                }
                            }
                        })();
                    } else if (msg && typeof msg.ok === 'boolean') {
                        resultJson = msg;
                    }
                } catch {
                    // ignore non-JSON lines
                }
            }
        });
        child.stderr.on('data', chunk => {
            if (err.length < 256 * 1024) err += chunk.toString();
        });
        child.on('error', e => {
            if (settled) return;
            settled = true;
            clearTimeout(hardTimer);
            reject(e);
        });
        child.on('close', code => {
            if (settled) return;
            settled = true;
            clearTimeout(hardTimer);
            try {
                // 마지막 NDJSON 줄에 \\n이 없으면 outBuf에만 남아 있음
                const tail = outBuf.trim();
                if (tail) {
                    try {
                        const msg = JSON.parse(tail);
                        if (msg && typeof msg.ok === 'boolean') resultJson = msg;
                    } catch {
                        /* ignore */
                    }
                }
                if (resultJson && typeof resultJson.ok === 'boolean') {
                    resolve({ json: resultJson, code, err });
                    return;
                }
                const trimmed = out.trim();
                const lastLine = trimmed.includes('\n') ? trimmed.split('\n').pop().trim() : trimmed;
                const json = JSON.parse(lastLine);
                resolve({ json, code, err });
            } catch (e) {
                reject(new Error(`runner JSON parse: ${e.message}; exit=${code}; stderr=${err.slice(0, 2000)}`));
            }
        });
    });
}

/** Discord embed field 값 상한(1024) 고려 */
function truncateEmbedField(str, max = 1000) {
    if (str == null || str === '') return '';
    const t = String(str);
    if (t.length <= max) return t;
    return `${t.slice(0, max - 1)}…`;
}

/** Discord embed description 상한(4096) */
function truncateDiscordDescription(str, max = 4090) {
    if (str == null || str === '') return '';
    const t = String(str);
    if (t.length <= max) return t;
    return `${t.slice(0, max - 1)}…`;
}

/** 로컬 작업으로 보이는 Git 변경이 있을 때만 diff/status 필드를 보여줌 */
function hasGitWorkingChanges(g) {
    if (!g || !g.isRepo) return false;
    const st = String(g.statusPorcelain || '').trim();
    const ds = String(g.diffStat || '').trim();
    const dp = String(g.diffPreview || '').trim();
    return st.length > 0 || ds.length > 0 || dp.length > 0;
}

/**
 * Cursor 완료 임베드: 에이전트가 cursor/ask_question·cursor/create_plan을 호출했는지 안내
 * @param {{ askQuestionCount?: number, createPlanCount?: number }|null|undefined} s
 */
function formatCursorAcpRpcSummaryField(s: { askQuestionCount?: number; createPlanCount?: number } | null | undefined) {
    const q = typeof s?.askQuestionCount === 'number' ? s.askQuestionCount : 0;
    const p = typeof s?.createPlanCount === 'number' ? s.createPlanCount : 0;
    const lineAsk =
        q === 0
            ? '**cursor/ask_question** · 호출 없음 — 디스코드 선택 메뉴는 이 RPC가 있을 때만 뜹니다.'
            : `**cursor/ask_question** · ${q}회 호출됨 (디스코드에서 선택 연동).`;
    const linePlan =
        p === 0
            ? '**cursor/create_plan** · 호출 없음.'
            : `**cursor/create_plan** · ${p}회 요청됨 (러너 자동 승인, 별도 디스코드 UI 없음).`;
    return truncateEmbedField(`${lineAsk}\n${linePlan}`, 1020);
}

/**
 * @param {'cursor'|'gemini'} kind
 * @param {{ elapsedSec: number, spinner: string, progressMin: number, requestText?: string, modeLabel?: string, liveAssistantText?: string }} s
 */
function buildDeferProgressEmbed(kind, s) {
    const mm = Math.floor(s.elapsedSec / 60);
    const ss = String(s.elapsedSec % 60).padStart(2, '0');
    const reqSnippet = s.requestText ? truncateEmbedField(s.requestText, 350) : '';
    const liveSnippet = s.liveAssistantText ? truncateEmbedField(s.liveAssistantText, 450) : '';

    if (kind === 'cursor') {
        const mode = s.modeLabel ? String(s.modeLabel) : 'agent';
        const parts = [];
        if (reqSnippet) parts.push(`**요청** ${reqSnippet}`);
        if (liveSnippet) parts.push(`**스트림** ${liveSnippet}`);
        const desc = parts.length ? parts.join('\n') : '처리 중…';
        return new EmbedBuilder()
            .setTitle(`${s.spinner} Cursor (${mode})`)
            .setDescription(desc)
            .setColor(0x5865F2)
            .addFields(
                { name: '경과', value: `${mm}:${ss}`, inline: true },
                { name: '한도', value: `~${s.progressMin}분`, inline: true },
            )
            .setFooter({ text: '완료 시 이 메시지가 결과로 바뀝니다' });
    }
    const desc = reqSnippet ? `**질문** ${reqSnippet}` : 'Gemini 호출 중…';
    return new EmbedBuilder()
        .setTitle(`${s.spinner} Gemini`)
        .setDescription(desc)
        .setColor(0x4285F4)
        .addFields({ name: '경과', value: `${mm}:${ss}`, inline: true })
        .setFooter({ text: '완료 시 이 메시지가 결과로 바뀝니다' });
}

/**
 * defer 직후 embed를 주기적으로 갱신 (경과·짧은 요약).
 * @param {'cursor'|'gemini'} kind
 * @param {{ progressMin?: number, requestText?: string, modeLabel?: string }} [extra]
 */
async function startDeferElapsedTicker(interaction, kind, extra = {}) {
    const tickMs = Math.max(1500, parseInt(process.env.DEFER_TICK_MS || '2500', 10));
    const t0 = Date.now();
    const SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    const progressMin = typeof extra.progressMin === 'number' ? extra.progressMin : 10;
    let tick = 0;
    const run = async () => {
        const elapsedSec = Math.floor((Date.now() - t0) / 1000);
        const spinner = SPINNER[tick % SPINNER.length];
        tick += 1;
        const liveAssistantText =
            typeof extra.liveAssistantText === 'function' ? extra.liveAssistantText() : extra.liveAssistantText;
        const embed = buildDeferProgressEmbed(kind, {
            elapsedSec,
            spinner,
            progressMin,
            requestText: extra.requestText || '',
            modeLabel: extra.modeLabel || '',
            liveAssistantText: liveAssistantText || '',
        });
        try {
            await interaction.editReply({ content: null, embeds: [embed] });
        } catch (e: unknown) {
            const code = e && typeof e === 'object' && 'code' in e ? (e as { code?: number }).code : undefined;
            if (code !== 50006) console.error('[defer ticker] editReply 실패:', e instanceof Error ? e.message : e);
        }
    };
    const id = setInterval(run, tickMs);
    await run();
    return () => clearInterval(id);
}

/**
 * editReply는 ‘수정’이라 디스코드가 새 메시지처럼 알림을 주지 않는 경우가 많음.
 * 완료 시 followUp으로 짧은 **새 메시지**를 보내 알림을 받기 쉽게 함.
 * DEFER_COMPLETION_NOTIFY: off | ephemeral | mention (기본 ephemeral)
 */
async function notifyDeferCompletion(interaction, { ok, kind }) {
    const mode = (process.env.DEFER_COMPLETION_NOTIFY || 'ephemeral').toLowerCase().trim();
    if (mode === 'off' || mode === 'false' || mode === '0') return;
    const uid = interaction.user.id;
    const label = kind === 'cursor' ? 'Cursor' : kind === 'gemini' ? 'Gemini' : '작업';
    const line = ok
        ? `✅ **${label}** 작업이 완료되었습니다. 위 응답 메시지를 확인하세요.`
        : `❌ **${label}** 처리가 끝났습니다(실패 또는 오류). 위 메시지를 확인하세요.`;
    try {
        if (mode === 'mention') {
            await interaction.followUp({
                content: `<@${uid}> ${line}`,
                allowedMentions: { users: [uid] },
            });
        } else {
            await interaction.followUp({ content: line, ephemeral: true });
        }
    } catch (e) {
        console.error('[notifyDeferCompletion] followUp 실패:', e.message);
    }
}

/* ── Gemini AI (Optional) ── */
let geminiModel: GenerativeModel | null = null;
try {
    if (process.env.GEMINI_API_KEY) {
        const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
        geminiModel = genAI.getGenerativeModel({ model: process.env.GEMINI_MODEL || 'gemini-2.0-flash' });
        console.log('[Gemini] AI 모델 초기화 완료');
    }
} catch (e: unknown) {
    console.warn('[Gemini] 초기화 실패 (선택 기능):', e instanceof Error ? e.message : e);
}

/* ── Meme 서비스 (메시지에 해당하는 이미지 파일 전송) ── */
const MEME_DIR = memeImgDir();

async function handleMeme(message: Message): Promise<boolean> {
    if (message.author.bot || message.content.startsWith('!') || message.content.startsWith('/')) return false;
    const query = message.content.trim().toLowerCase();
    if (!query) return false;

    try {
        if (!fs.existsSync(MEME_DIR)) return false;
        const files = fs.readdirSync(MEME_DIR);
        const match = files.find(f => path.parse(f).name.toLowerCase() === query);
        if (match) {
            const embed = new EmbedBuilder()
                .setTitle(`🖼️ ${query}`)
                .setImage(`attachment://${match}`)
                .setColor(0xFFD700)
                .setFooter({ text: `Requested by ${message.author.username}`, iconURL: message.author.displayAvatarURL() });
            await message.channel.send({ embeds: [embed], files: [path.join(MEME_DIR, match)] });
            return true;
        }
    } catch (e) { /* ignore */ }
    return false;
}

/* ══════════════════════════════════════
   공용 상호작용 및 커맨드 핸들러
   ══════════════════════════════════════ */
async function showHelpPage(interaction, pageIndex, isUpdate = false) {
    const pages = [
        { title: gameData.getMessage('Help_Basic_Title'), desc: gameData.getMessage('Help_Basic_Desc'), content: gameData.getMessage('Help_Basic_Content') },
        { title: gameData.getMessage('Help_MiniGame_Title'), desc: gameData.getMessage('Help_MiniGame_Desc'), content: gameData.getMessage('Help_MiniGame_Content') },
        { title: gameData.getMessage('Help_Stock_Title'), desc: gameData.getMessage('Help_Stock_Desc'), content: gameData.getMessage('Help_Stock_Content') },
        { title: gameData.getMessage('Help_Raid_Title'), desc: gameData.getMessage('Help_Raid_Desc'), content: gameData.getMessage('Help_Raid_Content') }
    ];
    if (pageIndex < 0) pageIndex = 0;
    if (pageIndex >= pages.length) pageIndex = pages.length - 1;
    const page = pages[pageIndex];

    const embed = new EmbedBuilder()
        .setTitle(gameData.getMessage('General_Help_Title', pageIndex + 1, pages.length))
        .setDescription(page.desc)
        .addFields({ name: page.title, value: page.content })
        .setColor(0x7C4DFF);

    const row = new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId(`help_page:${pageIndex - 1}`).setLabel('이전').setStyle(ButtonStyle.Primary).setDisabled(pageIndex === 0),
        new ButtonBuilder().setCustomId(`help_page:${pageIndex + 1}`).setLabel('다음').setStyle(ButtonStyle.Primary).setDisabled(pageIndex === pages.length - 1)
    );

    const payload = { embeds: [embed], components: [row] };
    if (isUpdate) await interaction.update(payload);
    else await interaction.reply(payload);
}

async function handleEnhance(interaction, userId, userName, isUpdate = false) {
    const r = enhancement.enhance(userId);
    const embed = new EmbedBuilder();
    let attachment = null;

    const rowPrimary = new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId('enhance_retry').setLabel('다시 강화하기').setStyle(ButtonStyle.Primary),
        new ButtonBuilder().setCustomId('sell_sword').setLabel('판매하기').setStyle(ButtonStyle.Secondary)
    );
    const rowFail = new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId('enhance_retry').setLabel('다시 강화하기').setStyle(ButtonStyle.Primary),
        new ButtonBuilder().setCustomId('consolation').setLabel('위로(놀림)').setStyle(ButtonStyle.Secondary)
    );

    let components = [];

    if (r.type === 'max') {
        embed.setTitle(gameData.getMessage('Enhance_MaxLevel_Title'))
            .setDescription(gameData.getMessage('Enhance_MaxLevel_Desc'))
            .setColor(0xFFD700);
    } else if (r.type === 'no_money') {
        embed.setTitle(gameData.getMessage('Enhance_NoMoney_Title'))
            .addFields(
                { name: gameData.getMessage('Enhance_NoMoney_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                { name: gameData.getMessage('Enhance_NoMoney_Balance'), value: `${formatMoney(r.balance)}원`, inline: true },
            ).setColor(0xF44336);
    } else if (r.type === 'great_success') {
        embed.setTitle(gameData.getMessage('Enhance_GreatSuccess_Title'))
            .setDescription(gameData.getMessage('Enhance_GreatSuccess_Desc', `<@${userId}>`, r.sword.name, r.oldLevel, r.newLevel))
            .addFields(
                { name: gameData.getMessage('Enhance_Increase'), value: `+${r.increase}강`, inline: true },
                { name: gameData.getMessage('Enhance_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                { name: gameData.getMessage('Enhance_RemainingBalance'), value: `${formatMoney(r.balance)}원`, inline: true },
                { name: gameData.getMessage('Enhance_Blacksmith_Comment'), value: `*"${r.chat}"*` }
            ).setColor(0xFFD700);
        if (r.lore) embed.addFields({ name: gameData.getMessage('Enhance_Lore'), value: `*${r.lore}*` });
        attachment = getImageAttachment(r.sword.imageName);
        components = [rowPrimary];
    } else if (r.type === 'success') {
        embed.setTitle(gameData.getMessage('Enhance_Success_Title'))
            .setDescription(gameData.getMessage('Enhance_Success_Desc', `<@${userId}>`, r.sword.name, r.oldLevel, r.newLevel))
            .addFields(
                { name: gameData.getMessage('Enhance_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                { name: gameData.getMessage('Enhance_RemainingBalance'), value: `${formatMoney(r.balance)}원`, inline: true },
                { name: gameData.getMessage('Enhance_Blacksmith_Comment'), value: `*"${r.chat}"*` }
            ).setColor(0x4CAF50);
        if (r.lore) embed.addFields({ name: gameData.getMessage('Enhance_Lore'), value: `*${r.lore}*` });
        attachment = getImageAttachment(r.sword.imageName);
        components = [rowPrimary];
    } else if (r.type === 'protected') {
        embed.setTitle(gameData.getMessage('Enhance_Fail_Protected_Title'))
            .setDescription(gameData.getMessage('Enhance_Fail_Protected_Desc', `<@${userId}>`, r.level, r.sword.name))
            .addFields(
                { name: gameData.getMessage('Enhance_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                { name: gameData.getMessage('Enhance_RemainingBalance'), value: `${formatMoney(r.balance)}원`, inline: true },
                { name: gameData.getMessage('Enhance_Blacksmith_Comment'), value: `*"${r.chat}"*` }
            ).setColor(0x00BCD4);
        attachment = getImageAttachment(r.imageOverride);
        components = [rowPrimary];
    } else if (r.type === 'destroy') {
        embed.setTitle(gameData.getMessage('Enhance_Fail_Title'))
            .setDescription(gameData.getMessage('Enhance_Fail_Desc', `<@${userId}>`, r.sword.name))
            .addFields(
                { name: gameData.getMessage('Enhance_Fail_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                { name: gameData.getMessage('Enhance_RemainingBalance'), value: `${formatMoney(r.balance)}원`, inline: true },
                { name: gameData.getMessage('Enhance_Blacksmith_Comment'), value: `*"${r.chat}"*` }
            ).setColor(0xF44336);
        attachment = getImageAttachment(r.imageOverride);
        components = [rowFail];
    }
    
    const payload = { embeds: [embed], components };
    if (attachment) {
        embed.setThumbnail(`attachment://${attachment.name}`);
        payload.files = [attachment.file];
    }
    if (isUpdate) await interaction.update(payload);
    else await interaction.reply(payload);
}

async function handleSell(interaction, userId, isUpdate = false) {
    const r = enhancement.sell(userId);
    const embed = new EmbedBuilder();
    let components = [];
    if (r.type === 'no_sword') {
        embed.setTitle(gameData.getMessage('Sell_NoSword_Title'))
            .setDescription(gameData.getMessage('Sell_NoSword_Desc')).setColor(0xF44336);
    } else {
        embed.setTitle(gameData.getMessage('Sell_Complete_Title'))
            .setDescription(gameData.getMessage('Sell_Complete_Desc', formatMoney(r.finalPrice)))
            .addFields(
                { name: gameData.getMessage('Sell_BasePrice'), value: `${formatMoney(r.basePrice)}원`, inline: true },
                { name: gameData.getMessage('Sell_FinalPrice'), value: `${formatMoney(r.finalPrice)}원`, inline: true },
                { name: gameData.getMessage('Sell_Blacksmith_Eval'), value: `*"${r.comment}"*` },
                { name: gameData.getMessage('Sell_CurrentBalance'), value: `${formatMoney(r.balance)}원` },
            ).setColor(0x4CAF50);
        components = [new ActionRowBuilder().addComponents(new ButtonBuilder().setCustomId('enhance_retry').setLabel('강화하기').setStyle(ButtonStyle.Primary))];
    }
    const payload = { embeds: [embed], components };
    if (isUpdate) await interaction.update(payload);
    else await interaction.reply(payload);
}

client.on('interactionCreate', async interaction => {
    if (interaction.isButton()) {
        const customId = interaction.customId;
        const userId = interaction.user.id;
        const userName = interaction.user.displayName || interaction.user.username;

        try {
            if (customId.startsWith('help_page:')) {
                const pageIndex = parseInt(customId.split(':')[1], 10);
                await showHelpPage(interaction, pageIndex, true);
                return;
            }

            if (customId === 'consolation') {
                const imageName = getRandomImage('위로(놀림)_');
                const wImg = getImageAttachment(imageName);
                const embed = new EmbedBuilder()
                    .setTitle(gameData.getMessage('Consolation_Title'))
                    .setDescription(gameData.getMessage('Consolation_Desc', `<@${userId}>`))
                    .setColor(0xFF00FF);
                let payload = { embeds: [embed] };
                if (wImg) Object.assign(payload, { files: [wImg.file], embeds: [embed.setImage(`attachment://${wImg.name}`)] });

                await interaction.channel.send(payload);
                await interaction.deferUpdate();
                return;
            }

            if (customId === 'enhance_retry') {
                await handleEnhance(interaction, userId, userName, true);
                return;
            }
            if (customId === 'sell_sword') {
                await handleSell(interaction, userId, true);
                return;
            }
        } catch (err) {
            console.error('[Button Error]', err);
            await interaction.reply({ content: '오류가 발생했습니다.', ephemeral: true }).catch(() => {});
        }
        return;
    }

    if (!interaction.isChatInputCommand()) return;

    const userId = interaction.user.id;
    const userName = interaction.user.displayName || interaction.user.username;

    try {
        switch (interaction.commandName) {

        /* ── ping ── */
        case 'ping': {
            const embed = new EmbedBuilder()
                .setTitle(gameData.getMessage('General_Ping_Title'))
                .setDescription(gameData.getMessage('General_Ping_Desc', client.ws.ping))
                .setColor(0x00BCD4);
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 도움말 ── */
        case '도움말': {
            await showHelpPage(interaction, 0);
            break;
        }

        /* ── 강화 ── */
        case '강화': {
            await handleEnhance(interaction, userId, userName);
            break;
        }

        /* ── 판매 ── */
        case '판매': {
            await handleSell(interaction, userId);
            break;
        }

        /* ── 정보 ── */
        case '정보': {
            const user = gameData.getUser(userId);
            if (!user.sword.weaponType || !user.sword.imageName) enhancement.ensureSword(user);

            const swordName = user.sword.name || gameData.getMessage('Info_NoName');
            const embed = new EmbedBuilder()
                .setTitle(gameData.getMessage('Info_Title', userName))
                .addFields(
                    { name: gameData.getMessage('Info_SwordName'), value: `**${swordName}** (+${user.sword.level}강)`, inline: true },
                    { name: gameData.getMessage('Info_MaxLevel'), value: `+${user.maxLevel}강`, inline: true },
                    { name: gameData.getMessage('Info_Balance'), value: `${formatMoney(user.money)}원`, inline: true },
                ).setColor(getLevelColor(user.sword.level));

            const attachment = getImageAttachment(user.sword.imageName);
            const payload = { embeds: [embed] };
            if (attachment) {
                embed.setThumbnail(`attachment://${attachment.name}`);
                payload.files = [attachment.file];
            }
            await interaction.reply(payload);
            break;
        }

        /* ── 돈 ── */
        case '돈': {
            const user = gameData.getUser(userId);
            const embed = new EmbedBuilder()
                .setTitle(gameData.getMessage('Money_Title'))
                .setDescription(gameData.getMessage('Money_Desc', userName, formatMoney(user.money)))
                .setColor(0xFFD700);
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 랭킹 ── */
        case '랭킹': {
            const all = Object.entries(gameData.users)
                .map(([id, u]) => ({ id, money: u.money, level: u.sword?.level || 0 }))
                .sort((a, b) => b.money - a.money)
                .slice(0, 10);
            let desc = '';
            for (let i = 0; i < all.length; i++) {
                const u = all[i];
                const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `${i + 1}.`;
                let name;
                try { const member = await client.users.fetch(u.id); name = member.displayName || member.username; }
                catch { name = gameData.getMessage('Rank_UnknownUser'); }
                desc += `${medal} **${name}** — ${formatMoney(u.money)}원 (+${u.level}강)\n`;
            }
            const embed = new EmbedBuilder()
                .setTitle(gameData.getMessage('Rank_Title'))
                .setDescription(desc || gameData.getMessage('Rank_NoData'))
                .setColor(0xFFD700);
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 출첵 ── */
        case '출첵': {
            const r = enhancement.checkAttendance(userId);
            const embed = new EmbedBuilder();
            if (r.type === 'ok') {
                embed.setTitle(gameData.getMessage('Attend_Complete_Title'))
                    .setDescription(gameData.getMessage('Attend_Complete_Desc', formatMoney(r.reward)))
                    .addFields({ name: gameData.getMessage('Attend_CurrentBalance'), value: `${formatMoney(r.balance)}원` })
                    .setColor(0x4CAF50);
            } else {
                embed.setTitle(gameData.getMessage('Attend_Wait_Title'))
                    .setDescription(gameData.getMessage('Attend_Wait_Desc', r.min, r.sec))
                    .setColor(0xFF9800);
            }
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 돈내놔 ── */
        case '돈내놔': {
            const r = enhancement.giveMeMoney(userId);
            const embed = new EmbedBuilder()
                .setTitle(gameData.getMessage('GMM_Title'))
                .setDescription(gameData.getMessage('GMM_Desc'))
                .addFields(
                    { name: gameData.getMessage('GMM_Amount'), value: `${formatMoney(r.amount)}원`, inline: true },
                    { name: '잔액', value: `${formatMoney(r.balance)}원`, inline: true },
                ).setColor(0xFFD700);
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 배틀 ── */
        case '배틀': {
            const target = interaction.options.getUser('상대');
            if (!target) { await interaction.reply({ content: gameData.getMessage('Battle_NoTarget_Desc'), ephemeral: true }); break; }
            if (target.id === userId) { await interaction.reply({ content: gameData.getMessage('Battle_Self_Desc'), ephemeral: true }); break; }
            if (target.bot) { await interaction.reply({ content: gameData.getMessage('Battle_Bot_Desc'), ephemeral: true }); break; }

            const targetUser = gameData.getUser(target.id);
            if (!targetUser.sword.weaponType) enhancement.ensureSword(targetUser);

            const r = enhancement.battle(userId, target.id);
            const embed = new EmbedBuilder();
            let attachment = null;

            if (r.type === 'limit') {
                embed.setTitle(gameData.getMessage('Battle_Limit_Title'))
                    .setDescription(gameData.getMessage('Battle_Limit_Desc')).setColor(0xF44336);
            } else {
                embed.setTitle(r.type === 'win' ? gameData.getMessage('Battle_Win_Title') : gameData.getMessage('Battle_Lose_Title'))
                    .setDescription(r.type === 'win'
                        ? gameData.getMessage('Battle_Win_Desc', userName, target.displayName || target.username)
                        : gameData.getMessage('Battle_Lose_Desc', userName, target.displayName || target.username))
                    .addFields(
                        { name: gameData.getMessage('Battle_MySword'), value: `+${r.myLevel}강 ${r.mySword}`, inline: true },
                        { name: gameData.getMessage('Battle_TargetSword'), value: `+${r.targetLevel}강 ${r.targetSword}`, inline: true },
                        { name: gameData.getMessage('Battle_Remaining'), value: `${r.remaining}회`, inline: true },
                    ).setColor(r.type === 'win' ? 0x4CAF50 : 0xF44336);
                if (r.reward) embed.addFields({ name: gameData.getMessage('Battle_Reward'), value: `${formatMoney(r.reward)}원` });
                attachment = getImageAttachment(r.battleImage);
            }
            
            const payload = { embeds: [embed] };
            if (attachment) {
                embed.setThumbnail(`attachment://${attachment.name}`);
                payload.files = [attachment.file];
            }
            await interaction.reply(payload);
            break;
        }

        /* ── 슬롯 ── */
        case '슬롯': {
            const bet = interaction.options.getInteger('금액');
            const r = enhancement.slot(userId, bet);
            const embed = new EmbedBuilder();
            if (r.type === 'error') { await interaction.reply({ content: r.msg, ephemeral: true }); break; }
            if (r.type === 'no_money') { await interaction.reply({ content: gameData.getMessage('Game_NoMoney', formatMoney(r.balance)), ephemeral: true }); break; }

            embed.setTitle(gameData.getMessage('Slot_Result_Title'))
                .setDescription(`**[ ${r.symbols.join(' | ')} ]**\n\n${r.payout > 0 ? `🎉 ${r.msg}` : '😢 ' + gameData.getMessage('Slot_Lose')}`)
                .addFields(
                    { name: gameData.getMessage('Slot_Bet'), value: `${formatMoney(r.bet)}원`, inline: true },
                    { name: gameData.getMessage('Slot_Earn'), value: `${formatMoney(r.payout)}원`, inline: true },
                    { name: gameData.getMessage('Slot_Balance'), value: `${formatMoney(r.balance)}원`, inline: true },
                ).setColor(r.payout > 0 ? 0xFFD700 : 0x9E9E9E);
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 홀짝 ── */
        case '홀짝': {
            const choice = interaction.options.getString('선택');
            const bet = interaction.options.getInteger('금액');
            const r = enhancement.oddEven(userId, choice, bet);
            if (r.type === 'error') { await interaction.reply({ content: r.msg, ephemeral: true }); break; }
            if (r.type === 'no_money') { await interaction.reply({ content: gameData.getMessage('Game_NoMoney', formatMoney(r.balance)), ephemeral: true }); break; }

            const DICE = ['⚀','⚁','⚂','⚃','⚄','⚅'];
            const embed = new EmbedBuilder()
                .setTitle(gameData.getMessage('OddEven_Title'))
                .setDescription(`${DICE[r.dice - 1]} ${gameData.getMessage('OddEven_Result', r.result)}\n\n${r.type === 'win' ? gameData.getMessage('OddEven_Win', formatMoney(r.payout)) : gameData.getMessage('OddEven_Lose')}`)
                .addFields(
                    { name: gameData.getMessage('OddEven_Choice'), value: choice, inline: true },
                    { name: gameData.getMessage('OddEven_Bet'), value: `${formatMoney(r.bet)}원`, inline: true },
                    { name: '잔액', value: `${formatMoney(r.balance)}원`, inline: true },
                ).setColor(r.type === 'win' ? 0x4CAF50 : 0xF44336);
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 가위바위보 ── */
        case '가위바위보': {
            const choice = interaction.options.getString('선택');
            const bet = interaction.options.getInteger('금액');
            const r = enhancement.rps(userId, choice, bet);
            if (r.type === 'error') { await interaction.reply({ content: r.msg, ephemeral: true }); break; }
            if (r.type === 'no_money') { await interaction.reply({ content: gameData.getMessage('Game_NoMoney', formatMoney(r.balance)), ephemeral: true }); break; }

            const RPS_EMOJI = { '가위': '✌️', '바위': '✊', '보': '🖐️' };
            const result = r.type === 'win' ? gameData.getMessage('RPS_Win', formatMoney(r.payout))
                : r.type === 'draw' ? gameData.getMessage('RPS_Draw') : gameData.getMessage('RPS_Lose');
            const embed = new EmbedBuilder()
                .setTitle(gameData.getMessage('RPS_Title'))
                .addFields(
                    { name: gameData.getMessage('RPS_User'), value: `${RPS_EMOJI[r.userChoice]} ${r.userChoice}`, inline: true },
                    { name: gameData.getMessage('RPS_Bot'), value: `${RPS_EMOJI[r.botChoice]} ${r.botChoice}`, inline: true },
                    { name: gameData.getMessage('RPS_Result'), value: result },
                    { name: '잔액', value: `${formatMoney(r.balance)}원`, inline: true },
                ).setColor(r.type === 'win' ? 0x4CAF50 : r.type === 'draw' ? 0xFF9800 : 0xF44336);
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 주식목록 ── */
        case '주식목록': {
            const list = stock.getStockList();
            const embed = new EmbedBuilder()
                .setTitle(gameData.getMessage('Stock_List_Title'))
                .setColor(0x00BCD4);
            for (const st of list) {
                const change = st.diff > 0 ? gameData.getMessage('Stock_List_Change_Up', formatMoney(st.diff), st.pct)
                    : st.diff < 0 ? gameData.getMessage('Stock_List_Change_Down', formatMoney(st.diff), st.pct)
                    : gameData.getMessage('Stock_List_Change_None');
                embed.addFields({ name: `${st.name} (${st.symbol})`, value: gameData.getMessage('Stock_List_Format', formatMoney(st.price), change, st.desc) });
            }
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 주식차트 ── */
        case '주식차트': {
            const symbol = interaction.options.getString('종목');
            const url = stock.getChartUrl(symbol);
            if (!url) { await interaction.reply({ content: '존재하지 않는 종목입니다.', ephemeral: true }); break; }
            const embed = new EmbedBuilder()
                .setTitle(`📈 ${symbol.toUpperCase()} 차트`)
                .setImage(url)
                .setColor(0x00BCD4);
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 매수 ── */
        case '매수': {
            const symbol = interaction.options.getString('종목');
            const amount = interaction.options.getInteger('수량');
            const r = stock.buy(userId, symbol, amount);
            if (!r.ok) { await interaction.reply({ content: r.msg, ephemeral: true }); break; }
            const embed = new EmbedBuilder()
                .setTitle('📈 매수 완료')
                .setDescription(gameData.getMessage('Stock_Buy_Success', r.stock, r.amount, formatMoney(r.price), formatMoney(r.totalCost)))
                .addFields({ name: '잔액', value: `${formatMoney(r.balance)}원` })
                .setColor(0x4CAF50);
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 매도 ── */
        case '매도': {
            const symbol = interaction.options.getString('종목');
            const amount = interaction.options.getInteger('수량');
            const r = stock.sell(userId, symbol, amount);
            if (!r.ok) { await interaction.reply({ content: r.msg, ephemeral: true }); break; }
            const profitStr = r.profit > 0 ? `🔺 +${formatMoney(r.profit)}` : r.profit < 0 ? `🔻 ${formatMoney(r.profit)}` : '➖ 0';
            const embed = new EmbedBuilder()
                .setTitle('📉 매도 완료')
                .setDescription(gameData.getMessage('Stock_Sell_Success', r.stock, r.amount, formatMoney(r.price), formatMoney(r.totalIncome), profitStr))
                .addFields({ name: '잔액', value: `${formatMoney(r.balance)}원` })
                .setColor(r.profit >= 0 ? 0x4CAF50 : 0xF44336);
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 내주식 ── */
        case '내주식': {
            const p = stock.getPortfolio(userId);
            const embed = new EmbedBuilder()
                .setTitle(gameData.getMessage('Stock_MyStock_Header', userName))
                .setColor(0x00BCD4);
            if (p.entries.length === 0) {
                embed.setDescription(gameData.getMessage('Stock_MyStock_Empty'));
            } else {
                for (const e of p.entries) {
                    const profitStr = e.profit >= 0 ? `🔺 +${formatMoney(e.profit)}` : `🔻 ${formatMoney(e.profit)}`;
                    embed.addFields({ name: `${e.name} (${e.amount}주)`, value: gameData.getMessage('Stock_MyStock_Item', e.avgPrice, formatMoney(e.currentPrice), profitStr, formatMoney(Math.abs(e.profit)), e.profitPct) });
                }
                embed.setFooter({ text: gameData.getMessage('Stock_MyStock_Footer', formatMoney(p.totalInvested), formatMoney(p.totalAsset), formatMoney(p.totalProfit), p.totalProfitPct) });
            }
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 레이드정보 ── */
        case '레이드정보': {
            const info = raid.getRaidInfo();
            const embed = new EmbedBuilder();
            if (!info) {
                embed.setTitle(gameData.getMessage('Raid_NoRaid_Title'))
                    .setDescription(gameData.getMessage('Raid_NoRaid_Desc')).setColor(0x9E9E9E);
            } else {
                embed.setTitle(gameData.getMessage('Raid_Status_Title', `${info.emoji} ${info.boss}`))
                    .setDescription(gameData.getMessage('Raid_Status_Desc', formatMoney(info.currentHp), formatMoney(info.maxHp), info.hpPct))
                    .addFields({ name: '참가자', value: `${info.participantCount}명` })
                    .setColor(0xFF4444);
            }
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 레이드소환 ── */
        case '레이드소환': {
            const r = raid.spawnRaid();
            const embed = new EmbedBuilder()
                .setTitle(`${r.emoji} 새로운 레이드 보스 등장!`)
                .setDescription(`**${r.boss}**가 나타났습니다!\n체력: ${formatMoney(r.maxHp)} | 보상: ${formatMoney(r.reward)}원`)
                .setColor(0xFF4444);
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 공격 ── */
        case '공격': {
            const r = raid.attack(userId);
            const embed = new EmbedBuilder();
            if (r.type === 'no_raid') {
                embed.setTitle(gameData.getMessage('Raid_NoRaid_Title'))
                    .setDescription('/레이드소환 으로 보스를 소환하세요!').setColor(0x9E9E9E);
            } else if (r.type === 'cleared') {
                embed.setTitle(gameData.getMessage('Raid_Clean_Title', `${r.emoji} ${r.boss}`))
                    .setDescription(gameData.getMessage('Raid_Clean_Desc')).setColor(0xFFD700);
                let ranking = '';
                r.rewards.forEach((rw, i) => {
                    ranking += `${i + 1}위: <@${rw.userId}> - ${formatMoney(rw.damage)} 딜 → ${formatMoney(rw.reward)}원\n`;
                });
                embed.addFields({ name: gameData.getMessage('Raid_Ranking_Title'), value: ranking || '없음' });
            } else {
                embed.setTitle(`⚔️ ${r.emoji} ${r.boss} 공격!`)
                    .setDescription(`**${formatMoney(r.damage)}** 데미지! ${r.crit ? '💥 크리티컬!' : ''}\n\n❤️ ${formatMoney(r.currentHp)} / ${formatMoney(r.maxHp)} (${r.hpPct}%)`)
                    .setColor(0xFF9800);
            }
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── yawn (Gemini AI) ── */
        /* ── cursor-edit (Cursor 로컬 CLI / agent acp) ── */
        case 'cursor-edit': {
            if (!isAdmin(userId)) {
                await interaction.reply({ content: gameData.getMessage('Admin_AccessDenied_Desc'), ephemeral: true });
                break;
            }
            const repoDir = process.env.CURSOR_LOCAL_REPO_DIR;
            if (!repoDir || !String(repoDir).trim()) {
                await interaction.reply({ content: '`.env`에 `CURSOR_LOCAL_REPO_DIR`(작업할 로컬 git 폴더 절대 경로)을 설정하세요.', ephemeral: true });
                break;
            }
            const promptText = interaction.options.getString('prompt');
            const modeOpt = interaction.options.getString('mode');
            const maxChars = getCursorMaxPromptChars();
            if (promptText.length > maxChars) {
                await interaction.reply({ content: `프롬프트가 너무 깁니다. 최대 ${maxChars}자까지입니다.`, ephemeral: true });
                break;
            }
            if (cursorEditInFlight) {
                await interaction.reply({ content: '이미 Cursor 로컬 작업이 실행 중입니다. 잠시 후 다시 시도하세요.', ephemeral: true });
                break;
            }
            await interaction.deferReply();
            const progressMin = Math.ceil(parseInt(process.env.CURSOR_TIMEOUT_MS || '600000', 10) / 60000);
            cursorEditInFlight = true;
            let stopDeferTicker = () => {};
            try {
                let liveAssistant = '';
                stopDeferTicker = await startDeferElapsedTicker(interaction, 'cursor', {
                    progressMin,
                    requestText: promptText,
                    modeLabel: modeOpt || 'agent',
                    liveAssistantText: () => liveAssistant,
                });
                const { json, code, err } = await runCursorLocalRunner(
                    repoDir,
                    promptText,
                    modeOpt || 'agent',
                    chunk => {
                        liveAssistant += chunk;
                        // keep last N chars
                        const maxLive = parseInt(process.env.CURSOR_LIVE_PREVIEW_CHARS || '700', 10);
                        if (liveAssistant.length > maxLive) liveAssistant = liveAssistant.slice(-maxLive);
                    },
                    q => discordAnswerCursorQuestion(interaction, q),
                );
                /** 러너 종료 직후 즉시 틱 중지 — 안 하면 interval이 최종 editReply 직후에 진행 카드로 다시 덮어씀 */
                stopDeferTicker();
                stopDeferTicker = () => {};
                if (!json.ok) {
                    const git = json.git || {};
                    const embed = new EmbedBuilder()
                        .setTitle('Cursor 로컬 실행 실패')
                        .setDescription(truncateDiscordDescription(
                            `**📝 요청**\n\`\`\`\n${truncateEmbedField(promptText, 900)}\n\`\`\`\n\n` +
                            `**오류**\n${String(json.error || 'unknown').slice(0, 2800)}`,
                        ))
                        .setColor(0xF44336)
                        .addFields(
                            { name: 'exit', value: String(code), inline: true },
                            { name: 'git repo', value: git.isRepo ? 'yes' : 'no', inline: true },
                            { name: 'mode', value: `\`${modeOpt || 'agent'}\``, inline: true },
                        );
                    if (hasGitWorkingChanges(git)) {
                        if (String(git.statusPorcelain || '').trim()) {
                            embed.addFields({
                                name: '변경 파일 (status)',
                                value: `\`\`\`\n${String(git.statusPorcelain).slice(0, 900)}\n\`\`\``,
                            });
                        }
                        if (String(git.diffStat || '').trim()) {
                            embed.addFields({ name: 'diff --stat', value: `\`\`\`\n${String(git.diffStat).slice(0, 900)}\n\`\`\`` });
                        }
                        if (String(git.diffPreview || '').trim()) {
                            embed.addFields({ name: 'diffPreview', value: `\`\`\`\n${String(git.diffPreview).slice(0, 900)}\n\`\`\`` });
                        }
                    }
                    if (err) embed.setFooter({ text: err.slice(0, 500) });
                    await interaction.editReply({ content: null, embeds: [embed] });
                    await notifyDeferCompletion(interaction, { ok: false, kind: 'cursor' });
                    break;
                }
                const g = json.git || {};
                const descParts = [];
                if (json.stopReason != null) descParts.push(`**stopReason:** ${json.stopReason}`);
                if (json.assistantPreview) descParts.push(json.assistantPreview.slice(0, 2600));
                const embed = new EmbedBuilder()
                    .setTitle('Cursor 로컬 실행 완료')
                    .setDescription(truncateDiscordDescription(
                        `**📝 요청**\n\`\`\`\n${truncateEmbedField(promptText, 900)}\n\`\`\`\n\n` +
                            `**에이전트 응답**\n${descParts.join('\n\n') || '(응답 본문 없음)'}`,
                    ))
                    .setColor(0x4CAF50)
                    .addFields(
                        { name: '모드', value: `\`${modeOpt || 'agent'}\``, inline: true },
                        { name: '작업 경로', value: `\`${String(json.cwd || repoDir).slice(0, 900)}\`` },
                        {
                            name: '질문·플랜 (ACP)',
                            value: formatCursorAcpRpcSummaryField(json.acpRpcSummary),
                            inline: false,
                        },
                    );
                if (hasGitWorkingChanges(g)) {
                    if (String(g.statusPorcelain || '').trim()) {
                        embed.addFields({
                            name: '변경 파일 (status)',
                            value: `\`\`\`\n${String(g.statusPorcelain).slice(0, 900)}\n\`\`\``,
                        });
                    }
                    if (String(g.diffStat || '').trim()) {
                        embed.addFields({ name: 'diff --stat', value: `\`\`\`\n${String(g.diffStat).slice(0, 900)}\n\`\`\`` });
                    }
                    if (String(g.diffPreview || '').trim()) {
                        embed.addFields({ name: 'diffPreview', value: `\`\`\`\n${String(g.diffPreview).slice(0, 900)}\n\`\`\`` });
                    }
                }
                if (json.stderrTail) embed.addFields({ name: 'agent stderr (tail)', value: `\`\`\`\n${String(json.stderrTail).slice(0, 800)}\n\`\`\`` });
                await interaction.editReply({ content: null, embeds: [embed] });
                await notifyDeferCompletion(interaction, { ok: true, kind: 'cursor' });
            } catch (e) {
                await interaction.editReply({
                    content: null,
                    embeds: [new EmbedBuilder()
                        .setTitle('Cursor 로컬 실행 예외')
                        .setDescription(truncateDiscordDescription(
                            `**📝 요청**\n\`\`\`\n${truncateEmbedField(promptText, 900)}\n\`\`\`\n\n` +
                            `**오류**\n${String(e.message).slice(0, 1800)}`,
                        ))
                        .setColor(0xF44336)],
                });
                await notifyDeferCompletion(interaction, { ok: false, kind: 'cursor' });
            } finally {
                stopDeferTicker();
                cursorEditInFlight = false;
            }
            break;
        }

        case 'yawn': {
            if (!geminiModel) {
                await interaction.reply({ content: 'Gemini API가 설정되지 않았습니다.', ephemeral: true });
                break;
            }
            const prompt = interaction.options.getString('질문');
            await interaction.deferReply();
            let stopGeminiTicker = () => {};
            try {
                stopGeminiTicker = await startDeferElapsedTicker(interaction, 'gemini', { requestText: prompt });
                const result = await geminiModel.generateContent({
                    contents: [{
                        role: 'user',
                        parts: [{ text: `시스템: 너는 'YawnBot'이라는 이름의 활기차고 재치 있는 디스코드 봇이야. 사용자의 질문에 친절하고 유머러스하게 대답해줘.\n\n사용자: ${prompt}` }],
                    }],
                });
                stopGeminiTicker();
                stopGeminiTicker = () => {};
                const response = result.response.text();
                const embed = new EmbedBuilder()
                    .setTitle('YawnBot AI Response')
                    .setDescription(truncateDiscordDescription(
                        `**📝 질문**\n\`\`\`\n${truncateEmbedField(prompt, 800)}\n\`\`\`\n\n` +
                        `**💬 답변**\n${response.slice(0, 3000)}`,
                    ))
                    .setColor(0x4285F4)
                    .setFooter({ text: 'Powered by Google Gemini' })
                    .setTimestamp();
                await interaction.editReply({ content: null, embeds: [embed] });
                await notifyDeferCompletion(interaction, { ok: true, kind: 'gemini' });
            } catch (e) {
                await interaction.editReply({
                    content: null,
                    embeds: [new EmbedBuilder()
                        .setTitle('Gemini 오류')
                        .setDescription(truncateDiscordDescription(
                            `**📝 질문**\n\`\`\`\n${truncateEmbedField(prompt, 800)}\n\`\`\`\n\n` +
                            `**오류**\n${String(e.message).slice(0, 1800)}`,
                        ))
                        .setColor(0xF44336)],
                });
                await notifyDeferCompletion(interaction, { ok: false, kind: 'gemini' });
            } finally {
                stopGeminiTicker();
            }
            break;
        }

        /* ── 관리자 ── */
        case 'admin-reload': {
            if (!isAdmin(userId)) { await interaction.reply({ content: gameData.getMessage('Admin_AccessDenied_Desc'), ephemeral: true }); break; }
            await gameData.initialize();
            const embed = new EmbedBuilder().setTitle(gameData.getMessage('Admin_Reload_Title')).setDescription(gameData.getMessage('Admin_Reload_Desc')).setColor(0x4CAF50);
            await interaction.reply({ embeds: [embed], ephemeral: true });
            break;
        }
        case 'admin-save': {
            if (!isAdmin(userId)) { await interaction.reply({ content: gameData.getMessage('Admin_AccessDenied_Desc'), ephemeral: true }); break; }
            gameData.saveGameData();
            const embed = new EmbedBuilder().setTitle(gameData.getMessage('Admin_Save_Title')).setDescription(gameData.getMessage('Admin_Save_Desc')).setColor(0x4CAF50);
            await interaction.reply({ embeds: [embed], ephemeral: true });
            break;
        }

        default:
            await interaction.reply({ content: '알 수 없는 명령어입니다.', ephemeral: true });
        }
    } catch (err) {
        console.error(`[Error] ${interaction.commandName}:`, err);
        const reply = interaction.replied || interaction.deferred
            ? interaction.editReply : interaction.reply;
        await reply.call(interaction, { content: `오류가 발생했습니다: ${err.message}`, ephemeral: true }).catch(() => {});
    }
});

/* ── 메시지 이벤트 (밈 서비스) ── */
client.on('messageCreate', async message => {
    if (message.author.bot) return;
    await handleMeme(message);
});

/* ── GitHub Webhook (Express) ── */
const app = express();
app.use(express.json());

app.post('/webhook/github', async (req, res) => {
    try {
        const event = req.headers['x-github-event'];
        const payload = req.body;
        console.log(`[Webhook] Received: ${event}`);

        const channelId = process.env.GITHUB_WEBHOOK_CHANNEL_ID;
        if (!channelId) { res.sendStatus(200); return; }

        const channel = await client.channels.fetch(channelId).catch(() => null);
        if (!channel) { res.sendStatus(200); return; }

        const embed = new EmbedBuilder()
            .setAuthor({ name: payload.sender?.login || 'GitHub', iconURL: payload.sender?.avatar_url })
            .setColor(0x4CAF50)
            .setFooter({ text: payload.repository?.full_name || '' })
            .setTimestamp();

        if (event === 'ping') {
            embed.setTitle(gameData.getMessage('Webhook_Ping_Title')).setDescription(gameData.getMessage('Webhook_Ping_Desc'));
        } else if (event === 'push') {
            if (!payload.commits || !payload.commits.length) { res.sendStatus(200); return; }
            embed.setTitle(gameData.getMessage('Webhook_Push_Title', payload.commits.length));
            const desc = payload.commits.slice(0, 5).map(c => `- [\`${c.id.slice(0, 7)}\`](${c.url}) ${c.message}`).join('\n');
            embed.setDescription(desc);
        } else if (event === 'issues') {
            embed.setTitle(gameData.getMessage('Webhook_Issue_Title', payload.issue?.number, payload.action))
                .setDescription(gameData.getMessage('Webhook_Issue_Desc', payload.issue?.title, payload.issue?.html_url))
                .setColor(payload.action === 'opened' ? 0xFF9800 : 0x4285F4);
        } else { res.sendStatus(200); return; }

        await channel.send({ embeds: [embed] });
        res.sendStatus(200);
    } catch (err) {
        console.error('[Webhook] Error:', err.message);
        res.sendStatus(500);
    }
});

/* ── Bot Ready ── */
client.once('clientReady', async () => {
    console.log(`\n  ⚔️  YawnBot (Node.js)`);
    console.log(`  ─────────────────────────`);
    console.log(`  로그인: ${client.user.tag}`);
    console.log(`  서버:   ${client.guilds.cache.size}개`);
    console.log(`  유저:   ${Object.keys(gameData.users).length}명 데이터 로드`);
    console.log('');

    stock.startMarket();

    // 서버 시작 알림 (Webhook Channel)
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

/* ── 시작 ── */
async function main() {
    await gameData.initialize();

    const WEBHOOK_PORT = process.env.WEBHOOK_PORT || 8080;
    app.listen(WEBHOOK_PORT, () => {
        console.log(`[Webhook] GitHub Webhook 서버 시작: http://0.0.0.0:${WEBHOOK_PORT}/webhook/github`);
    });

    await client.login(process.env.DISCORD_TOKEN);
}

/* ── Graceful shutdown ── */
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
