import path from 'path';
import { config } from 'dotenv';
import { SlashCommandBuilder } from 'discord.js';
import { deployApplicationCommands } from '@discord-bots/common';

config({ path: path.join(__dirname, '..', '.env') });

const commands = [
  new SlashCommandBuilder()
    .setName('unityfree')
    .setDescription('Unity 무료 에셋 정보를 지금 확인하고 알림을 보냅니다.')
    .addBooleanOption((opt) =>
      opt
        .setName('force')
        .setDescription('같은 쿠폰 코드여도 강제로 다시 전송합니다.')
        .setRequired(false),
    ),
].map((cmd) => cmd.toJSON());

async function main(): Promise<void> {
  const token = process.env.UNITYFREE_DISCORD_TOKEN;
  const clientId = process.env.UNITYFREE_CLIENT_ID;
  if (!token || !clientId) {
    console.error('[UnityFree Deploy] UNITYFREE_DISCORD_TOKEN 또는 UNITYFREE_CLIENT_ID가 없습니다.');
    process.exitCode = 1;
    return;
  }
  await deployApplicationCommands({
    token,
    clientId,
    commands,
    logPrefix: '[UnityFree Deploy]',
  });
}

void main();

