import type { ChatFeedSource } from "./types.js";
import { ChzzkChatFeed } from "./chzzk/chzzkChatFeed.js";
import { RandomChatFeed } from "./randomChatFeed.js";

export type ChatFeedKind = "random" | "chzzk";

/** 환경 변수 `CHAT_FEED`(또는 인자)로 소스 선택. 기본은 random. */
export function createChatFeed(kind: ChatFeedKind = "random"): ChatFeedSource {
  if (kind === "chzzk") {
    return new ChzzkChatFeed();
  }
  return new RandomChatFeed();
}

export function chatFeedKindFromEnv(): ChatFeedKind {
  const raw = process.env.CHAT_FEED?.trim().toLowerCase();
  if (raw === "chzzk") {
    return "chzzk";
  }
  return "random";
}
