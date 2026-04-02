import type { ChatFeedSource, ChatLine, Unsubscribe } from "./types";

const SAMPLE_AUTHORS = [
  "viewer1",
  "다람쥐",
  "stream_fan",
  "guest",
  "mod",
  "봇테스트"
];

const SAMPLE_PARTS = [
  "안녕하세요",
  "굿",
  "ㅋㅋㅋ",
  "오늘 방송",
  "화이팅",
  "클립 각",
  "ㄱㄱ",
  "오",
  "와",
  "채팅 테스트",
  "Lorem",
  "ipsum"
];

function randomId(): string {
  const buf = new Uint8Array(8);
  crypto.getRandomValues(buf);
  return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
}

function randomSample(): string {
  const n = 1 + Math.floor(Math.random() * 3);
  const parts: string[] = [];
  for (let i = 0; i < n; i += 1) {
    parts.push(SAMPLE_PARTS[Math.floor(Math.random() * SAMPLE_PARTS.length)]!);
  }
  return parts.join(" ");
}

/** MVP: 주기적으로 무작위 문자열을 채팅처럼보냄. */
export class RandomChatFeed implements ChatFeedSource {
  private timer: ReturnType<typeof setInterval> | undefined;

  constructor(private readonly intervalMs: number = 1500) {}

  subscribe(onLine: (line: ChatLine) => void): Unsubscribe {
    const tick = (): void => {
      const author =
        SAMPLE_AUTHORS[Math.floor(Math.random() * SAMPLE_AUTHORS.length)]!;
      onLine({
        id: randomId(),
        author,
        text: randomSample(),
        ts: Date.now()
      });
    };
    tick();
    this.timer = setInterval(tick, this.intervalMs);
    return () => {
      if (this.timer !== undefined) {
        clearInterval(this.timer);
        this.timer = undefined;
      }
    };
  }

  destroy(): void {
    if (this.timer !== undefined) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }
}
