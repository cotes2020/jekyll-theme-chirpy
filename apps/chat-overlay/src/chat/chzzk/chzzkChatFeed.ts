import type { ChatFeedSource, ChatLine, Unsubscribe } from "../types.js";

/**
 * 치지직(Chzzk) 공식 오픈 API는 방송·채널 연동 중심이며,
 * 시청자 채팅 실시간 수신용 공개 REST/WebSocket 스펙은 문서상 제공되지 않는 경우가 많습니다.
 * (최신 내용은 https://chzzk.gitbook.io/ 등 공식 GitBook 확인.)
 *
 * 이 구현은 공식으로 허용된 범위를 넘지 않도록, 채팅 대신 안내 메시지만 한 번 내보냅니다.
 * 실시간 채팅이 필요하면 방송자용 도구·공식 위젯·또는 플랫폼이 허용하는 경로를 사용해야 합니다.
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
