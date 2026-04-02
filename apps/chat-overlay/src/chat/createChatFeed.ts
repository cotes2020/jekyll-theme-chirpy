import type { ChatFeedSource } from "./types";
import { ChzzkChatFeed } from "./chzzk/chzzkChatFeed";
import { ChzzkSessionChatFeed } from "./chzzk/chzzkSessionChatFeed";
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

function rawChzzkAccessToken(): string {
  try {
    const v = import.meta.env.VITE_CHZZK_ACCESS_TOKEN;
    if (v !== undefined && v !== null && String(v).trim() !== "") {
      return String(v).trim();
    }
  } catch {
    /* no import.meta (tests) */
  }
  if (typeof process !== "undefined" && process.env?.VITE_CHZZK_ACCESS_TOKEN) {
    return String(process.env.VITE_CHZZK_ACCESS_TOKEN).trim();
  }
  return "";
}

/** `VITE_CHZZK_SESSION=1` 이고 토큰이 있으면 공식 Session 소켓 피드를 사용. */
function chzzkSessionFeedEnabled(): boolean {
  try {
    return import.meta.env.VITE_CHZZK_SESSION === "1";
  } catch {
    /* no import.meta */
  }
  if (typeof process !== "undefined") {
    return process.env?.VITE_CHZZK_SESSION === "1";
  }
  return false;
}

/** Vite `VITE_CHAT_FEED` 또는 테스트용 `process.env.VITE_CHAT_FEED`. 기본 random. */
export function chatFeedKindFromEnv(): ChatFeedKind {
  return parseChatFeedKind(rawFeedEnv());
}

export function createChatFeed(kind: ChatFeedKind = "random"): ChatFeedSource {
  if (kind === "chzzk") {
    const token = rawChzzkAccessToken();
    if (token !== "" && chzzkSessionFeedEnabled()) {
      return new ChzzkSessionChatFeed({ accessToken: token });
    }
    return new ChzzkChatFeed();
  }
  return new RandomChatFeed();
}
