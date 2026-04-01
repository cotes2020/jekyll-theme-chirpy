import { REST, Routes } from 'discord.js';

export async function deployApplicationCommands(opts: {
  token: string;
  clientId: string;
  commands: unknown[];
  logPrefix?: string;
}): Promise<void> {
  const { token, clientId, commands, logPrefix } = opts;
  const prefix = logPrefix || '[Deploy]';

  const rest = new REST().setToken(token);
  console.log(`${prefix} ${commands.length}개 커맨드 등록 중...`);
  await rest.put(Routes.applicationCommands(clientId), { body: commands });
  console.log(`${prefix} 커맨드 등록 완료!`);
}

