import { describe, expect, it } from "vitest";

import { ChzzkChatFeed, CHZZK_VIEWER_CHAT_CONSTRAINT_KO } from "./chzzkChatFeed";

describe("ChzzkChatFeed", () => {
  it("구독 시 안내 문구 한 줄을 내보낸다", () => {
    const feed = new ChzzkChatFeed();
    const lines: string[] = [];
    feed.subscribe((l) => lines.push(l.text));
    expect(lines).toHaveLength(1);
    expect(lines[0]).toBe(CHZZK_VIEWER_CHAT_CONSTRAINT_KO);
  });
});
