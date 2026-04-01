// @ts-nocheck
import { EmbedBuilder } from 'discord.js';
import { formatMoney } from '../../services/gamedata';

export async function handleRaidInfo(ctx, interaction) {
  const { gameData, raid } = ctx;
  const info = raid.getRaidInfo();
  const embed = new EmbedBuilder();
  if (!info) {
    embed.setTitle(gameData.getMessage('Raid_NoRaid_Title')).setDescription(gameData.getMessage('Raid_NoRaid_Desc')).setColor(0x9e9e9e);
  } else {
    embed
      .setTitle(gameData.getMessage('Raid_Status_Title', `${info.emoji} ${info.boss}`))
      .setDescription(gameData.getMessage('Raid_Status_Desc', formatMoney(info.currentHp), formatMoney(info.maxHp), info.hpPct))
      .addFields({ name: '참가자', value: `${info.participantCount}명` })
      .setColor(0xff4444);
  }
  await interaction.reply({ embeds: [embed] });
}

export async function handleRaidSpawn(ctx, interaction) {
  const { raid } = ctx;
  const r = raid.spawnRaid();
  const embed = new EmbedBuilder()
    .setTitle(`${r.emoji} 새로운 레이드 보스 등장!`)
    .setDescription(`**${r.boss}**가 나타났습니다!\n체력: ${formatMoney(r.maxHp)} | 보상: ${formatMoney(r.reward)}원`)
    .setColor(0xff4444);
  await interaction.reply({ embeds: [embed] });
}

export async function handleRaidAttack(ctx, interaction, userId) {
  const { gameData, raid } = ctx;
  const r = raid.attack(userId);
  const embed = new EmbedBuilder();
  if (r.type === 'no_raid') {
    embed.setTitle(gameData.getMessage('Raid_NoRaid_Title')).setDescription('/레이드소환 으로 보스를 소환하세요!').setColor(0x9e9e9e);
  } else if (r.type === 'cleared') {
    embed.setTitle(gameData.getMessage('Raid_Clean_Title', `${r.emoji} ${r.boss}`)).setDescription(gameData.getMessage('Raid_Clean_Desc')).setColor(0xffd700);
    let ranking = '';
    r.rewards.forEach((rw, i) => {
      ranking += `${i + 1}위: <@${rw.userId}> - ${formatMoney(rw.damage)} 딜 → ${formatMoney(rw.reward)}원\n`;
    });
    embed.addFields({ name: gameData.getMessage('Raid_Ranking_Title'), value: ranking || '없음' });
  } else {
    embed
      .setTitle(`⚔️ ${r.emoji} ${r.boss} 공격!`)
      .setDescription(`**${formatMoney(r.damage)}** 데미지! ${r.crit ? '💥 크리티컬!' : ''}\n\n❤️ ${formatMoney(r.currentHp)} / ${formatMoney(r.maxHp)} (${r.hpPct}%)`)
      .setColor(0xff9800);
  }
  await interaction.reply({ embeds: [embed] });
}

