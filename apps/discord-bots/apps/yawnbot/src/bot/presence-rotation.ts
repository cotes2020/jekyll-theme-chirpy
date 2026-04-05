/**
 * 멤버 목록에 보이는 활동(Playing …) 문구를 주기적으로 바꿉니다.
 * - BOT_PRESENCE_INTERVAL_SEC: 초 단위(기본 3). 0 이하이면 비활성.
 * - BOT_PRESENCE_LINES: | 로 구분한 문구들. 비우면 기본 3줄 순환.
 *   플레이스홀더 {guilds} → 현재 연결 길드 수.
 */
import { ActivityType, type Client } from 'discord.js';

let timer: ReturnType<typeof setInterval> | null = null;

const DISCORD_ACTIVITY_NAME_MAX = 128;

function clampIntervalSec(n: number): number {
  return Math.min(300, Math.max(3, n));
}

function buildLines(client: Client): string[] {
  const raw = process.env.BOT_PRESENCE_LINES?.trim();
  const fromEnv = raw
    ? raw
        .split('|')
        .map((s) => s.trim())
        .filter(Boolean)
    : [];
  const base =
    fromEnv.length > 0
      ? fromEnv
      : ['검 강화 | /도움말', '/music play', '{guilds}개 서버'];
  const guilds = client.guilds.cache.size;
  return base.map((s) =>
    s.replace(/\{guilds\}/gi, String(guilds)).slice(0, DISCORD_ACTIVITY_NAME_MAX),
  );
}

export function startPresenceRotation(client: Client): void {
  stopPresenceRotation();
  const parsed = parseInt(String(process.env.BOT_PRESENCE_INTERVAL_SEC ?? '3').trim(), 10);
  const intervalSec = Number.isFinite(parsed) && parsed > 0 ? clampIntervalSec(parsed) : 0;
  if (intervalSec <= 0) {
    return;
  }

  let index = 0;
  const tick = () => {
    const lines = buildLines(client);
    if (lines.length === 0) return;
    const name = lines[index % lines.length]!;
    index += 1;
    try {
      client.user?.setPresence({
        activities: [{ name, type: ActivityType.Playing }],
        status: 'online',
      });
    } catch {
      /* ignore */
    }
  };

  tick();
  timer = setInterval(tick, intervalSec * 1000);
}

export function stopPresenceRotation(): void {
  if (timer != null) {
    clearInterval(timer);
    timer = null;
  }
}
