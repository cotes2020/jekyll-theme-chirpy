import io from "socket.io-client";

import type { ChatFeedSource, ChatLine, Unsubscribe } from "../types";
import { fetchSessionUrlUser, subscribeChatEvent } from "./chzzkApi";

export interface ChzzkSessionChatFeedOptions {
  accessToken: string;
}

/**
 * SYSTEM connected 메시지에서 sessionKey 추출 (문서: type connected + sessionKey).
 */
export function extractSessionKeyFromSystemPayload(payload: unknown): string | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  const o = payload as Record<string, unknown>;
  if (typeof o.sessionKey === "string" && o.sessionKey.trim() !== "") {
    return o.sessionKey.trim();
  }
  const data = o.data;
  if (data && typeof data === "object") {
    const sk = (data as Record<string, unknown>).sessionKey;
    if (typeof sk === "string" && sk.trim() !== "") {
      return sk.trim();
    }
  }
  return null;
}

/**
 * CHAT 이벤트 본문 → ChatLine (문서: profile.nickname, content, messageTime).
 */
export function mapChzzkChatPayloadToLine(payload: unknown): ChatLine | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  const o = payload as Record<string, unknown>;
  const content = o.content;
  if (typeof content !== "string") {
    return null;
  }
  let author = "viewer";
  const profile = o.profile;
  if (profile && typeof profile === "object") {
    const nick = (profile as Record<string, unknown>).nickname;
    if (typeof nick === "string" && nick.trim() !== "") {
      author = nick.trim();
    }
  }
  const ts =
    typeof o.messageTime === "number" && Number.isFinite(o.messageTime)
      ? o.messageTime
      : Date.now();
  const ch = typeof o.channelId === "string" ? o.channelId : "ch";
  const id = `chzzk-${ch}-${ts}-${Math.random().toString(36).slice(2, 10)}`;
  return { id, author, text: content, ts };
}

const SOCKET_OPTIONS: Record<string, unknown> = {
  reconnection: false,
  "force new connection": true,
  "connect timeout": 3000,
  transports: ["websocket"]
};

/** Session API는 채널 ID를 넘겨 고르는 방식이 아니라, 토큰 주인(방송자) 이벤트만 옵니다. */
export const CHZZK_SESSION_SCOPE_HINT_KO =
  "[치지직] 이 피드는 OAuth로 받은 토큰의 계정(방송자) 채팅만 옵니다. " +
  "다른 스트리머 채널은 API로 지정할 수 없습니다. 방송 중일 때만 채팅이 올라옵니다.";

/**
 * 공식 Session API: 세션 URL → Socket.IO → connected의 sessionKey → POST 구독 → CHAT 수신.
 * Access Token은 프로덕션에서 Vite에 넣지 말고 Tauri 등으로 주입할 것 (VITE_는 번들에 노출됨).
 */
export class ChzzkSessionChatFeed implements ChatFeedSource {
  private readonly accessToken: string;
  private socket: SocketIOClient.Socket | null = null;

  constructor(options: ChzzkSessionChatFeedOptions) {
    this.accessToken = options.accessToken;
  }

  subscribe(onLine: (line: ChatLine) => void): Unsubscribe {
    let cancelled = false;
    let subscribeAttempted = false;

    const cleanup = () => {
      cancelled = true;
      if (this.socket) {
        try {
          this.socket.disconnect();
        } catch {
          /* ignore */
        }
        this.socket = null;
      }
    };

    void (async () => {
      try {
        const sessionUrl = await fetchSessionUrlUser(this.accessToken);
        if (cancelled) {
          return;
        }

        onLine({
          id: "chzzk-session-scope",
          author: "system",
          text: CHZZK_SESSION_SCOPE_HINT_KO,
          ts: Date.now()
        });

        const socket = io(sessionUrl, SOCKET_OPTIONS as object);
        this.socket = socket;

        socket.on("connect", () => {
          /* 연결 완료; sessionKey는 SYSTEM connected에서 옴 */
        });

        socket.on("SYSTEM", (data: unknown) => {
          if (cancelled || subscribeAttempted) {
            return;
          }
          const obj = data && typeof data === "object" ? (data as Record<string, unknown>) : null;
          if (!obj || obj.type !== "connected") {
            return;
          }
          const sessionKey = extractSessionKeyFromSystemPayload(data);
          if (!sessionKey) {
            return;
          }
          subscribeAttempted = true;
          void subscribeChatEvent(this.accessToken, sessionKey).catch((e: unknown) => {
            onLine({
              id: "chzzk-subscribe-err",
              author: "system",
              text: `[치지직] 채팅 구독 API 오류: ${e instanceof Error ? e.message : String(e)}`,
              ts: Date.now()
            });
          });
        });

        socket.on("CHAT", (data: unknown) => {
          if (cancelled) {
            return;
          }
          const line = mapChzzkChatPayloadToLine(data);
          if (line) {
            onLine(line);
          }
        });

        socket.on("connect_error", (err: Error) => {
          if (cancelled) {
            return;
          }
          onLine({
            id: "chzzk-socket-err",
            author: "system",
            text: `[치지직] 소켓 연결 오류: ${err?.message ?? String(err)}`,
            ts: Date.now()
          });
        });
      } catch (e: unknown) {
        if (!cancelled) {
          onLine({
            id: "chzzk-bootstrap-err",
            author: "system",
            text: `[치지직] 세션 시작 실패: ${e instanceof Error ? e.message : String(e)}`,
            ts: Date.now()
          });
        }
      }
    })();

    return cleanup;
  }

  destroy(): void {
    if (this.socket) {
      try {
        this.socket.disconnect();
      } catch {
        /* ignore */
      }
      this.socket = null;
    }
  }
}
