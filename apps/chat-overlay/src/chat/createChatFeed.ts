import type { ChatFeedSource } from "./types";
import { ChzzkChatFeed } from "./chzzk/chzzkChatFeed";
import { RandomChatFeed } from "./randomChatFeed";

export type ChatFeedKind = "random" | "chzzk";

export function parseChatFeedKind(raw: string | undefined): ChatFeedKind {
  const t = (raw ?? "").trim().toLowerCase();
  if (t === "chzzk") {
    return "chzzk";
  }
  return "random";
}

function rawFeedEnv(): string {
  try {
    const v = import.meta.env.VITE_CHAT_FEED;
    if (v !== undefined && v !== null && String(v).trim() !== "") {
      return String(v).trim();
    }
  } catch {
    /* no import.meta (tests) */
  }
  if (typeof process !== "undefined" && process.env?.VITE_CHAT_FEED) {
    return String(process.env.VITE_CHAT_FEED).trim();
  }
  return "";
}

/** Vite `VITE_CHAT_FEED` 또는 테스트용 `process.env.VITE_CHAT_FEED`. 기본 random. */
export function chatFeedKindFromEnv(): ChatFeedKind {
  return parseChatFeedKind(rawFeedEnv());
}

export function createChatFeed(kind: ChatFeedKind = "random"): ChatFeedSource {
  if (kind === "chzzk") {
    return new ChzzkChatFeed();
  }
  return new RandomChatFeed();
}
