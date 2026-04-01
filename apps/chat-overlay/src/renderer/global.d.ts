import type { ChatLine } from "../chat/types.js";

export {};

declare global {
  interface Window {
    chatOverlay: {
      onChatLine(cb: (line: ChatLine) => void): () => void;
    };
  }
}
