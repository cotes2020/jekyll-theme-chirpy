import { afterEach, describe, expect, it, vi } from "vitest";

import { RandomChatFeed } from "./randomChatFeed";

describe("RandomChatFeed", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("첫 틱에서 즉시 한 줄을 내보낸다", () => {
    const feed = new RandomChatFeed(10_000);
    const lines: string[] = [];
    feed.subscribe((l) => lines.push(l.text));
    expect(lines.length).toBe(1);
    feed.destroy();
  });

  it("interval 후 추가 줄이 나온다", () => {
    vi.useFakeTimers();
    const feed = new RandomChatFeed(1000);
    const lines: string[] = [];
    feed.subscribe((l) => lines.push(l.text));
    expect(lines.length).toBe(1);
    vi.advanceTimersByTime(1000);
    expect(lines.length).toBe(2);
    feed.destroy();
  });

  it("unsubscribe 후에는 더 이상 내보내지 않는다", () => {
    vi.useFakeTimers();
    const feed = new RandomChatFeed(500);
    const lines: string[] = [];
    const unsub = feed.subscribe((l) => lines.push(l.text));
    unsub();
    vi.advanceTimersByTime(2000);
    expect(lines.length).toBe(1);
    feed.destroy();
  });
});
