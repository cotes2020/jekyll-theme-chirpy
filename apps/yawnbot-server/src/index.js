/**
 * YawnBot — Node.js Discord Bot Server
 * C# YawnBot → discord.js v14 이식
 */
require('dotenv').config();

const { Client, GatewayIntentBits, EmbedBuilder, Collection } = require('discord.js');
const express = require('express');

const { GameDataService, formatMoney, getLevelColor, getWeaponLore } = require('./services/gamedata');
const { EnhancementService } = require('./services/enhancement');
const { StockService } = require('./services/stock');
const { RaidService } = require('./services/raid');

const fs = require('fs');
const path = require('path');
const ENHANCEMENT_DIR = path.join(__dirname, '..', 'resources', 'img', 'enhancement');

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

/* ── Gemini AI (Optional) ── */
let geminiModel = null;
try {
    if (process.env.GEMINI_API_KEY) {
        const { GoogleGenerativeAI } = require('@google/generative-ai');
        const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
        geminiModel = genAI.getGenerativeModel({ model: process.env.GEMINI_MODEL || 'gemini-2.0-flash' });
        console.log('[Gemini] AI 모델 초기화 완료');
    }
} catch (e) {
    console.warn('[Gemini] 초기화 실패 (선택 기능):', e.message);
}

/* ── Meme 서비스 (메시지에 해당하는 이미지 파일 전송) ── */
const MEME_DIR = path.join(__dirname, '..', 'resources', 'img', 'meme');

async function handleMeme(message) {
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
   커맨드 핸들러
   ══════════════════════════════════════ */
client.on('interactionCreate', async interaction => {
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
            const embed = new EmbedBuilder()
                .setTitle('📚 YawnBot 도움말')
                .setColor(0x7C4DFF)
                .addFields(
                    { name: gameData.getMessage('Help_Basic_Title'), value: gameData.getMessage('Help_Basic_Content') },
                    { name: gameData.getMessage('Help_MiniGame_Title'), value: gameData.getMessage('Help_MiniGame_Content') },
                    { name: gameData.getMessage('Help_Stock_Title'), value: gameData.getMessage('Help_Stock_Content') },
                    { name: gameData.getMessage('Help_Raid_Title'), value: gameData.getMessage('Help_Raid_Content') },
                );
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 강화 ── */
        case '강화': {
            const r = enhancement.enhance(userId);
            const embed = new EmbedBuilder();
            let attachment = null;

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
                    .setDescription(gameData.getMessage('Enhance_GreatSuccess_Desc', userName, r.sword.name, r.oldLevel, r.newLevel))
                    .addFields(
                        { name: gameData.getMessage('Enhance_Increase'), value: `+${r.increase}강`, inline: true },
                        { name: gameData.getMessage('Enhance_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                        { name: gameData.getMessage('Enhance_RemainingBalance'), value: `${formatMoney(r.balance)}원`, inline: true },
                        { name: gameData.getMessage('Enhance_Blacksmith_Comment'), value: `*${r.chat}*` },
                    ).setColor(0xFFD700);
                if (r.lore) embed.addFields({ name: gameData.getMessage('Enhance_Lore'), value: `*${r.lore}*` });
                attachment = getImageAttachment(r.sword.imageName);
            } else if (r.type === 'success') {
                embed.setTitle(gameData.getMessage('Enhance_Success_Title'))
                    .setDescription(gameData.getMessage('Enhance_Success_Desc', userName, r.sword.name, r.oldLevel, r.newLevel))
                    .addFields(
                        { name: gameData.getMessage('Enhance_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                        { name: gameData.getMessage('Enhance_RemainingBalance'), value: `${formatMoney(r.balance)}원`, inline: true },
                        { name: gameData.getMessage('Enhance_Blacksmith_Comment'), value: `*${r.chat}*` },
                    ).setColor(0x4CAF50);
                if (r.lore) embed.addFields({ name: gameData.getMessage('Enhance_Lore'), value: `*${r.lore}*` });
                attachment = getImageAttachment(r.sword.imageName);
            } else if (r.type === 'protected') {
                embed.setTitle(gameData.getMessage('Enhance_Fail_Protected_Title'))
                    .setDescription(gameData.getMessage('Enhance_Fail_Protected_Desc', userName, r.level, r.sword.name))
                    .addFields(
                        { name: gameData.getMessage('Enhance_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                        { name: gameData.getMessage('Enhance_RemainingBalance'), value: `${formatMoney(r.balance)}원`, inline: true },
                        { name: gameData.getMessage('Enhance_Blacksmith_Comment'), value: `*${r.chat}*` },
                    ).setColor(0x00BCD4);
                attachment = getImageAttachment(r.imageOverride);
            } else if (r.type === 'destroy') {
                embed.setTitle(gameData.getMessage('Enhance_Fail_Title'))
                    .setDescription(gameData.getMessage('Enhance_Fail_Desc', userName, r.sword.name))
                    .addFields(
                        { name: gameData.getMessage('Enhance_Fail_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                        { name: gameData.getMessage('Enhance_RemainingBalance'), value: `${formatMoney(r.balance)}원`, inline: true },
                        { name: gameData.getMessage('Enhance_Blacksmith_Comment'), value: `*${r.chat}*` },
                    ).setColor(0xF44336);
                attachment = getImageAttachment(r.imageOverride);
            }
            
            const payload = { embeds: [embed] };
            if (attachment) {
                embed.setThumbnail(`attachment://${attachment.name}`);
                payload.files = [attachment.file];
            }
            await interaction.reply(payload);
            break;
        }

        /* ── 판매 ── */
        case '판매': {
            const r = enhancement.sell(userId);
            const embed = new EmbedBuilder();
            if (r.type === 'no_sword') {
                embed.setTitle(gameData.getMessage('Sell_NoSword_Title'))
                    .setDescription(gameData.getMessage('Sell_NoSword_Desc')).setColor(0xF44336);
            } else {
                embed.setTitle(gameData.getMessage('Sell_Complete_Title'))
                    .setDescription(gameData.getMessage('Sell_Complete_Desc', r.finalPrice))
                    .addFields(
                        { name: gameData.getMessage('Sell_BasePrice'), value: `${formatMoney(r.basePrice)}원`, inline: true },
                        { name: gameData.getMessage('Sell_FinalPrice'), value: `${formatMoney(r.finalPrice)}원`, inline: true },
                        { name: gameData.getMessage('Sell_Blacksmith_Eval'), value: `*${r.comment}*` },
                        { name: gameData.getMessage('Sell_CurrentBalance'), value: `${formatMoney(r.balance)}원` },
                    ).setColor(0x4CAF50);
            }
            await interaction.reply({ embeds: [embed] });
            break;
        }

        /* ── 정보 ── */
        case '정보': {
            const user = gameData.getUser(userId);
            if (!user.sword.weaponType || !user.sword.imageName) enhancement._ensureSword(user);

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
            if (!targetUser.sword.weaponType) enhancement._ensureSword(targetUser);

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
        case 'yawn': {
            if (!geminiModel) {
                await interaction.reply({ content: 'Gemini API가 설정되지 않았습니다.', ephemeral: true });
                break;
            }
            const prompt = interaction.options.getString('질문');
            await interaction.deferReply();
            try {
                const result = await geminiModel.generateContent({
                    contents: [{
                        role: 'user',
                        parts: [{ text: `시스템: 너는 'YawnBot'이라는 이름의 활기차고 재치 있는 디스코드 봇이야. 사용자의 질문에 친절하고 유머러스하게 대답해줘.\n\n사용자: ${prompt}` }],
                    }],
                });
                const response = result.response.text();
                const embed = new EmbedBuilder()
                    .setTitle('YawnBot AI Response')
                    .setDescription(response.slice(0, 4000))
                    .setColor(0x4285F4)
                    .setFooter({ text: 'Powered by Google Gemini' })
                    .setTimestamp();
                await interaction.editReply({ embeds: [embed] });
            } catch (e) {
                await interaction.editReply(`오류가 발생했습니다: ${e.message}`);
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
client.once('ready', async () => {
    console.log(`\n  ⚔️  YawnBot (Node.js)`);
    console.log(`  ─────────────────────────`);
    console.log(`  로그인: ${client.user.tag}`);
    console.log(`  서버:   ${client.guilds.cache.size}개`);
    console.log(`  유저:   ${Object.keys(gameData.users).length}명 데이터 로드`);
    console.log('');

    stock.startMarket();
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
