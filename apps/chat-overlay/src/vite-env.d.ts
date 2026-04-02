/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_CHAT_FEED?: string;
  /** 개발용만. 번들에 포함되므로 프로덕션에서는 Tauri 등으로 주입할 것. */
  readonly VITE_CHZZK_ACCESS_TOKEN?: string;
  /** `1`이면 공식 Session API + Socket.IO 채팅 피드 사용 */
  readonly VITE_CHZZK_SESSION?: string;
  /**
   * Open API 베이스 URL 오버라이드(끝 슬래시 없음).
   * 예: 프리뷰/배포에서 Nginx 등으로 `/__chzzk_openapi` 프록시를 둔 경우 동일 경로로 설정.
   */
  readonly VITE_CHZZK_OPENAPI_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
