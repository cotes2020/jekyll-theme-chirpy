// @ts-nocheck
import {
    EmbedBuilder,
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
} from 'discord.js';
import { formatMoney } from '../services/gamedata';

/** 강화/판매/도움말 카드 — 버튼·슬래시 공용 */
export async function showHelpPage(ctx, interaction, pageIndex, isUpdate = false) {
    const { gameData } = ctx;
    const pages = [
        { title: gameData.getMessage('Help_Basic_Title'), desc: gameData.getMessage('Help_Basic_Desc'), content: gameData.getMessage('Help_Basic_Content') },
        { title: gameData.getMessage('Help_MiniGame_Title'), desc: gameData.getMessage('Help_MiniGame_Desc'), content: gameData.getMessage('Help_MiniGame_Content') },
        { title: gameData.getMessage('Help_Stock_Title'), desc: gameData.getMessage('Help_Stock_Desc'), content: gameData.getMessage('Help_Stock_Content') },
        { title: gameData.getMessage('Help_Raid_Title'), desc: gameData.getMessage('Help_Raid_Desc'), content: gameData.getMessage('Help_Raid_Content') },
    ];
    if (pageIndex < 0) pageIndex = 0;
    if (pageIndex >= pages.length) pageIndex = pages.length - 1;
    const page = pages[pageIndex];

    const embed = new EmbedBuilder()
        .setTitle(gameData.getMessage('General_Help_Title', pageIndex + 1, pages.length))
        .setDescription(page.desc)
        .addFields({ name: page.title, value: page.content })
        .setColor(0x7c4dff);

    const row = new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId(`help_page:${pageIndex - 1}`).setLabel('이전').setStyle(ButtonStyle.Primary).setDisabled(pageIndex === 0),
        new ButtonBuilder().setCustomId(`help_page:${pageIndex + 1}`).setLabel('다음').setStyle(ButtonStyle.Primary).setDisabled(pageIndex === pages.length - 1),
    );

    const payload = { embeds: [embed], components: [row] };
    if (isUpdate) await interaction.update(payload);
    else await interaction.reply(payload);
}

export async function handleEnhance(ctx, interaction, userId, userName, isUpdate = false) {
    const { gameData, enhancement, getImageAttachment } = ctx;
    const r = enhancement.enhance(userId);
    const embed = new EmbedBuilder();
    let attachment = null;

    const rowPrimary = new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId('enhance_retry').setLabel('다시 강화하기').setStyle(ButtonStyle.Primary),
        new ButtonBuilder().setCustomId('sell_sword').setLabel('판매하기').setStyle(ButtonStyle.Secondary),
    );
    const rowFail = new ActionRowBuilder().addComponents(
        new ButtonBuilder().setCustomId('enhance_retry').setLabel('다시 강화하기').setStyle(ButtonStyle.Primary),
        new ButtonBuilder().setCustomId('consolation').setLabel('위로(놀림)').setStyle(ButtonStyle.Secondary),
    );

    let components = [];

    if (r.type === 'max') {
        embed.setTitle(gameData.getMessage('Enhance_MaxLevel_Title'))
            .setDescription(gameData.getMessage('Enhance_MaxLevel_Desc'))
            .setColor(0xffd700);
    } else if (r.type === 'no_money') {
        embed.setTitle(gameData.getMessage('Enhance_NoMoney_Title'))
            .addFields(
                { name: gameData.getMessage('Enhance_NoMoney_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                { name: gameData.getMessage('Enhance_NoMoney_Balance'), value: `${formatMoney(r.balance)}원`, inline: true },
            ).setColor(0xf44336);
    } else if (r.type === 'great_success') {
        embed.setTitle(gameData.getMessage('Enhance_GreatSuccess_Title'))
            .setDescription(gameData.getMessage('Enhance_GreatSuccess_Desc', `<@${userId}>`, r.sword.name, r.oldLevel, r.newLevel))
            .addFields(
                { name: gameData.getMessage('Enhance_Increase'), value: `+${r.increase}강`, inline: true },
                { name: gameData.getMessage('Enhance_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                { name: gameData.getMessage('Enhance_RemainingBalance'), value: `${formatMoney(r.balance)}원`, inline: true },
                { name: gameData.getMessage('Enhance_Blacksmith_Comment'), value: `*"${r.chat}"*` },
            ).setColor(0xffd700);
        if (r.lore) embed.addFields({ name: gameData.getMessage('Enhance_Lore'), value: `*${r.lore}*` });
        attachment = getImageAttachment(r.sword.imageName);
        components = [rowPrimary];
    } else if (r.type === 'success') {
        embed.setTitle(gameData.getMessage('Enhance_Success_Title'))
            .setDescription(gameData.getMessage('Enhance_Success_Desc', `<@${userId}>`, r.sword.name, r.oldLevel, r.newLevel))
            .addFields(
                { name: gameData.getMessage('Enhance_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                { name: gameData.getMessage('Enhance_RemainingBalance'), value: `${formatMoney(r.balance)}원`, inline: true },
                { name: gameData.getMessage('Enhance_Blacksmith_Comment'), value: `*"${r.chat}"*` },
            ).setColor(0x4caf50);
        if (r.lore) embed.addFields({ name: gameData.getMessage('Enhance_Lore'), value: `*${r.lore}*` });
        attachment = getImageAttachment(r.sword.imageName);
        components = [rowPrimary];
    } else if (r.type === 'protected') {
        embed.setTitle(gameData.getMessage('Enhance_Fail_Protected_Title'))
            .setDescription(gameData.getMessage('Enhance_Fail_Protected_Desc', `<@${userId}>`, r.level, r.sword.name))
            .addFields(
                { name: gameData.getMessage('Enhance_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                { name: gameData.getMessage('Enhance_RemainingBalance'), value: `${formatMoney(r.balance)}원`, inline: true },
                { name: gameData.getMessage('Enhance_Blacksmith_Comment'), value: `*"${r.chat}"*` },
            ).setColor(0x00bcd4);
        attachment = getImageAttachment(r.imageOverride);
        components = [rowPrimary];
    } else if (r.type === 'destroy') {
        embed.setTitle(gameData.getMessage('Enhance_Fail_Title'))
            .setDescription(gameData.getMessage('Enhance_Fail_Desc', `<@${userId}>`, r.sword.name))
            .addFields(
                { name: gameData.getMessage('Enhance_Fail_Cost'), value: `${formatMoney(r.cost)}원`, inline: true },
                { name: gameData.getMessage('Enhance_RemainingBalance'), value: `${formatMoney(r.balance)}원`, inline: true },
                { name: gameData.getMessage('Enhance_Blacksmith_Comment'), value: `*"${r.chat}"*` },
            ).setColor(0xf44336);
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

export async function handleSell(ctx, interaction, userId, isUpdate = false) {
    const { gameData, enhancement } = ctx;
    const r = enhancement.sell(userId);
    const embed = new EmbedBuilder();
    let components = [];
    if (r.type === 'no_sword') {
        embed.setTitle(gameData.getMessage('Sell_NoSword_Title'))
            .setDescription(gameData.getMessage('Sell_NoSword_Desc')).setColor(0xf44336);
    } else {
        embed.setTitle(gameData.getMessage('Sell_Complete_Title'))
            .setDescription(gameData.getMessage('Sell_Complete_Desc', formatMoney(r.finalPrice)))
            .addFields(
                { name: gameData.getMessage('Sell_BasePrice'), value: `${formatMoney(r.basePrice)}원`, inline: true },
                { name: gameData.getMessage('Sell_FinalPrice'), value: `${formatMoney(r.finalPrice)}원`, inline: true },
                { name: gameData.getMessage('Sell_Blacksmith_Eval'), value: `*"${r.comment}"*` },
                { name: gameData.getMessage('Sell_CurrentBalance'), value: `${formatMoney(r.balance)}원` },
            ).setColor(0x4caf50);
        components = [new ActionRowBuilder().addComponents(new ButtonBuilder().setCustomId('enhance_retry').setLabel('강화하기').setStyle(ButtonStyle.Primary))];
    }
    const payload = { embeds: [embed], components };
    if (isUpdate) await interaction.update(payload);
    else await interaction.reply(payload);
}
