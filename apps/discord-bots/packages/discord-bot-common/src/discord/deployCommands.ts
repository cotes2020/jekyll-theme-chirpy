import { REST, Routes } from 'discord.js';

/** `.env`에서 쉼표로 나열한 ID·값 목록 (공백 허용). 길드·채널·유저 ID 등 공통. */
export function parseCommaSeparatedEnv(raw: string | undefined): string[] {
  if (!raw?.trim()) return [];
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

/** 길드 ID 목록 — `parseCommaSeparatedEnv`와 동일. */
export const parseDiscordGuildIds = parseCommaSeparatedEnv;

export async function deployApplicationCommands(opts: {
  token: string;
  clientId: string;
  commands: unknown[];
  logPrefix?: string;
  /** 전역 등록에 더해 길드에도 등록. 하나 또는 `id1,id2,id3` 형태. */
  guildId?: string;
}): Promise<void> {
  const { token, clientId, commands, logPrefix, guildId } = opts;
  const prefix = logPrefix || '[Deploy]';

  const rest = new REST().setToken(token);

  console.log(`${prefix} ${commands.length}개 커맨드 전역 등록 중...`);
  await rest.put(Routes.applicationCommands(clientId), { body: commands });
  console.log(`${prefix} 전역 등록 완료!`);

  const guildIds = parseCommaSeparatedEnv(guildId);
  for (const gid of guildIds) {
    console.log(`${prefix} ${commands.length}개 커맨드 길드 등록 중... (${gid})`);
    await rest.put(Routes.applicationGuildCommands(clientId, gid), { body: commands });
    console.log(`${prefix} 길드 ${gid} 등록 완료!`);
  }
}

