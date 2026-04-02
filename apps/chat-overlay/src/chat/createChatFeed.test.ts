import { afterEach, describe, expect, it, vi } from "vitest";

import { chatFeedKindFromEnv, parseChatFeedKind } from "./createChatFeed";

describe("parseChatFeedKind", () => {
  it("chzzk만 인식한다", () => {
    expect(parseChatFeedKind("chzzk")).toBe("chzzk");
    expect(parseChatFeedKind(" CHZZK ")).toBe("chzzk");
  });

  it("extension 인식한다", () => {
    expect(parseChatFeedKind("extension")).toBe("extension");
  });

  it("그 외는 random", () => {
    expect(parseChatFeedKind("")).toBe("random");
    expect(parseChatFeedKind("other")).toBe("random");
  });
});

describe("chatFeedKindFromEnv", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("VITE_CHAT_FEED 없으면 random", () => {
    expect(chatFeedKindFromEnv()).toBe("random");
  });

  it("process.env.VITE_CHAT_FEED=chzzk 이면 chzzk", () => {
    vi.stubEnv("VITE_CHAT_FEED", "chzzk");
    expect(chatFeedKindFromEnv()).toBe("chzzk");
  });

  it("process.env.VITE_CHAT_FEED=extension 이면 extension", () => {
    vi.stubEnv("VITE_CHAT_FEED", "extension");
    expect(chatFeedKindFromEnv()).toBe("extension");
  });
});
