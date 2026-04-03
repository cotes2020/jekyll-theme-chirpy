import { REST, Routes } from 'discord.js';

export async function deployApplicationCommands(opts: {
  token: string;
  clientId: string;
  commands: unknown[];
  logPrefix?: string;
  /** 설정 시 해당 길드에만 등록(보통 수 초 안에 슬래시에 반영). 비우면 전역 등록(최대 ~1시간 지연 가능). */
  guildId?: string;
}): Promise<void> {
  const { token, clientId, commands, logPrefix, guildId } = opts;
  const prefix = logPrefix || '[Deploy]';

  const rest = new REST().setToken(token);
  const route = guildId?.trim()
    ? Routes.applicationGuildCommands(clientId, guildId.trim())
    : Routes.applicationCommands(clientId);
  console.log(`${prefix} ${commands.length}개 커맨드 등록 중...${guildId?.trim() ? ` (길드 ${guildId.trim()})` : ' (전역)'}`);
  await rest.put(route, { body: commands });
  console.log(`${prefix} 커맨드 등록 완료!`);
}

