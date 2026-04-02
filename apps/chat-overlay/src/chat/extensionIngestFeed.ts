import type { ChatFeedSource, ChatLine, Unsubscribe } from "./types";

/**
 * 채팅 줄은 Tauri `extension-ingest` 이벤트(브라우저 확장 POST)로만 채움.
 */
export class ExtensionIngestFeed implements ChatFeedSource {
  subscribe(_onLine: (line: ChatLine) => void): Unsubscribe {
    return () => {};
  }
}
