import path from 'path';
import { config } from 'dotenv';
import { SlashCommandBuilder } from 'discord.js';
import { deployApplicationCommands } from '@discord-bots/common';

config({ path: path.join(__dirname, '..', '.env') });

const commands = [
  new SlashCommandBuilder()
    .setName('atkup')
    .setDescription('ATKUp — Unity 무료 에셋 알림 · 긱 뉴스 (설정된 알림 채널로 전송)')
    .addSubcommand((sub) =>
      sub
        .setName('unity')
        .setDescription('Unity Publisher Sale 무료 에셋을 확인하고 알림 채널에 보냅니다.')
        .addBooleanOption((opt) =>
          opt.setName('force').setDescription('같은 쿠폰이어도 강제로 다시 전송합니다.').setRequired(false),
        ),
    )
    .addSubcommand((sub) =>
      sub
        .setName('news')
        .setDescription('Hacker News 상위 글을 알림 채널에 보냅니다.')
        .addIntegerOption((opt) =>
          opt
            .setName('count')
            .setDescription('글 개수 (5~15)')
            .setMinValue(5)
            .setMaxValue(15)
            .setRequired(false),
        ),
    ),
].map((cmd) => cmd.toJSON());

async function main(): Promise<void> {
  const token = process.env.ATKUP_DISCORD_TOKEN;
  const clientId = process.env.ATKUP_CLIENT_ID;
  if (!token || !clientId) {
    console.error('[ATKUp Deploy] ATKUP_DISCORD_TOKEN 또는 ATKUP_CLIENT_ID가 없습니다.');
    process.exitCode = 1;
    return;
  }
  await deployApplicationCommands({
    token,
    clientId,
    commands,
    logPrefix: '[ATKUp Deploy]',
  });
}

void main();

