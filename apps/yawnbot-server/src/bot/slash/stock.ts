// @ts-nocheck
import { EmbedBuilder } from 'discord.js';
import { formatMoney } from '../../services/gamedata';

export async function handleStockList(ctx, interaction) {
    const { gameData, stock } = ctx;
    const list = stock.getStockList();
    const embed = new EmbedBuilder()
        .setTitle(gameData.getMessage('Stock_List_Title'))
        .setColor(0x00bcd4);
    for (const st of list) {
        const change = st.diff > 0 ? gameData.getMessage('Stock_List_Change_Up', formatMoney(st.diff), st.pct)
            : st.diff < 0 ? gameData.getMessage('Stock_List_Change_Down', formatMoney(st.diff), st.pct)
            : gameData.getMessage('Stock_List_Change_None');
        embed.addFields({ name: `${st.name} (${st.symbol})`, value: gameData.getMessage('Stock_List_Format', formatMoney(st.price), change, st.desc) });
    }
    await interaction.reply({ embeds: [embed] });
}

export async function handleStockChart(ctx, interaction) {
    const { stock } = ctx;
    const symbol = interaction.options.getString('종목');
    const url = stock.getChartUrl(symbol);
    if (!url) {
        await interaction.reply({ content: '존재하지 않는 종목입니다.', ephemeral: true });
        return;
    }
    const embed = new EmbedBuilder()
        .setTitle(`📈 ${symbol.toUpperCase()} 차트`)
        .setImage(url)
        .setColor(0x00bcd4);
    await interaction.reply({ embeds: [embed] });
}

export async function handleBuy(ctx, interaction, userId) {
    const { gameData, stock } = ctx;
    const symbol = interaction.options.getString('종목');
    const amount = interaction.options.getInteger('수량');
    const r = stock.buy(userId, symbol, amount);
    if (!r.ok) {
        await interaction.reply({ content: r.msg, ephemeral: true });
        return;
    }
    const embed = new EmbedBuilder()
        .setTitle('📈 매수 완료')
        .setDescription(gameData.getMessage('Stock_Buy_Success', r.stock, r.amount, formatMoney(r.price), formatMoney(r.totalCost)))
        .addFields({ name: '잔액', value: `${formatMoney(r.balance)}원` })
        .setColor(0x4caf50);
    await interaction.reply({ embeds: [embed] });
}

export async function handleSellStock(ctx, interaction, userId) {
    const { gameData, stock } = ctx;
    const symbol = interaction.options.getString('종목');
    const amount = interaction.options.getInteger('수량');
    const r = stock.sell(userId, symbol, amount);
    if (!r.ok) {
        await interaction.reply({ content: r.msg, ephemeral: true });
        return;
    }
    const profitStr = r.profit > 0 ? `🔺 +${formatMoney(r.profit)}` : r.profit < 0 ? `🔻 ${formatMoney(r.profit)}` : '➖ 0';
    const embed = new EmbedBuilder()
        .setTitle('📉 매도 완료')
        .setDescription(gameData.getMessage('Stock_Sell_Success', r.stock, r.amount, formatMoney(r.price), formatMoney(r.totalIncome), profitStr))
        .addFields({ name: '잔액', value: `${formatMoney(r.balance)}원` })
        .setColor(r.profit >= 0 ? 0x4caf50 : 0xf44336);
    await interaction.reply({ embeds: [embed] });
}

export async function handleMyStock(ctx, interaction, userId, userName) {
    const { gameData, stock } = ctx;
    const p = stock.getPortfolio(userId);
    const embed = new EmbedBuilder()
        .setTitle(gameData.getMessage('Stock_MyStock_Header', userName))
        .setColor(0x00bcd4);
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
}
