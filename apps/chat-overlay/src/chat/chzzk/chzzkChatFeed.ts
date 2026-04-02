import type { ChatFeedSource, ChatLine, Unsubscribe } from "../types";

/**
 * 토큰·`VITE_CHZZK_SESSION` 없이 `chzzk` 피드를 켰을 때의 안내용 피드.
 *
 * 실시간 수신은 공식 Session API(https://chzzk.gitbook.io/chzzk/chzzk-api/session)로:
 * 세션 URL → Socket.IO → `SYSTEM` connected의 sessionKey → POST 구독 → `CHAT`.
 * 해당 경로는 `ChzzkSessionChatFeed` + `VITE_CHZZK_ACCESS_TOKEN` + `VITE_CHZZK_SESSION=1`로 활성화.
 */
export const CHZZK_VIEWER_CHAT_CONSTRAINT_KO =
  "[치지직] 공개 API만으로는 시청 채팅 실시간 수신이 어려울 수 있습니다. " +
  "chzzk.gitbook.io 문서에서 방송자/앱 연동 범위를 확인한 뒤, " +
  "허용된 방법(예: 공식 위젯, 방송 SW 연동)을 검토하세요.";

export class ChzzkChatFeed implements ChatFeedSource {
  subscribe(onLine: (line: ChatLine) => void): Unsubscribe {
    onLine({
      id: "chzzk-info",
      author: "system",
      text: CHZZK_VIEWER_CHAT_CONSTRAINT_KO,
      ts: Date.now()
    });
    return () => {};
  }
}
