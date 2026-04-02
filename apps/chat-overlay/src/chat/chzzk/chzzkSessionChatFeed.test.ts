import { describe, expect, it } from "vitest";

import {
  extractSessionKeyFromSystemPayload,
  mapChzzkChatPayloadToLine
} from "./chzzkSessionChatFeed";

describe("extractSessionKeyFromSystemPayload", () => {
  it("최상위 sessionKey를 읽는다", () => {
    expect(extractSessionKeyFromSystemPayload({ type: "connected", sessionKey: "sk-1" })).toBe(
      "sk-1"
    );
  });

  it("data.sessionKey를 읽는다", () => {
    expect(
      extractSessionKeyFromSystemPayload({ type: "connected", data: { sessionKey: "sk-2" } })
    ).toBe("sk-2");
  });

  it("없으면 null", () => {
    expect(extractSessionKeyFromSystemPayload({ type: "connected" })).toBeNull();
  });
});

describe("mapChzzkChatPayloadToLine", () => {
  it("nickname·content·messageTime을 ChatLine으로 매핑한다", () => {
    const line = mapChzzkChatPayloadToLine({
      channelId: "ch1",
      content: "hello",
      messageTime: 1700000000000,
      profile: { nickname: "Tester" }
    });
    expect(line).not.toBeNull();
    expect(line!.author).toBe("Tester");
    expect(line!.text).toBe("hello");
    expect(line!.ts).toBe(1700000000000);
    expect(line!.id).toMatch(/^chzzk-ch1-/);
  });

  it("content가 없으면 null", () => {
    expect(mapChzzkChatPayloadToLine({ profile: { nickname: "x" } })).toBeNull();
  });
});
