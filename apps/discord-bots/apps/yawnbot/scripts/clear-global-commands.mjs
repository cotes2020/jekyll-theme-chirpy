/**
 * 글로벌 슬래시 커맨드 전체 초기화.
 * 길드 커맨드와 중복 등록된 글로벌 커맨드를 제거할 때 사용.
 * 실행: npm run deploy:clear-global
 */
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'node:module';
import { REST, Routes } from 'discord.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const nodeRequire = createRequire(import.meta.url);
const { loadKarmoLabAIEnv } = nodeRequire('karmolab-ai/node');
const { applyYawnbotDotenvLayers } = nodeRequire(path.join(__dirname, 'load-dotenv-layers.cjs'));

loadKarmoLabAIEnv();
applyYawnbotDotenvLayers(path.join(__dirname, '..'));

const token = process.env.DISCORD_TOKEN;
const clientId = process.env.CLIENT_ID;

if (!token || !clientId) {
  console.error('[ClearGlobal] DISCORD_TOKEN 또는 CLIENT_ID가 없습니다.');
  process.exit(1);
}

const rest = new REST({ version: '10' }).setToken(token);

console.log('[ClearGlobal] 글로벌 커맨드 초기화 중...');
await rest.put(Routes.applicationCommands(clientId), { body: [] });
console.log('[ClearGlobal] 완료. 글로벌 커맨드가 모두 삭제됐습니다.');
