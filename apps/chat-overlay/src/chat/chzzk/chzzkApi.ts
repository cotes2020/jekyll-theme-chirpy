/** 치지직 Open API 공통 베이스 (https://chzzk.gitbook.io/chzzk/chzzk-api/tips) */
export const CHZZK_OPENAPI_BASE = "https://openapi.chzzk.naver.com";

/** 브라우저에서 CORS 회피: Vite dev `server.proxy` 경로와 맞출 것 */
const CHZZK_OPENAPI_DEV_PROXY = "/__chzzk_openapi";

function openapiBase(): string {
  const override = import.meta.env.VITE_CHZZK_OPENAPI_BASE?.trim();
  if (override) {
    return override.replace(/\/$/, "");
  }
  /** `vite preview` 등은 DEV가 아니어도 localhost이면 Vite 프록시를 쓸 수 있음 */
  const host =
    typeof globalThis !== "undefined" && "location" in globalThis
      ? (globalThis as { location: { hostname: string } }).location.hostname
      : "";
  if (import.meta.env.DEV || host === "localhost" || host === "127.0.0.1") {
    return CHZZK_OPENAPI_DEV_PROXY;
  }
  return CHZZK_OPENAPI_BASE;
}

export interface ChzzkApiSuccess<T> {
  code: number;
  message: string | null;
  content: T;
}

export interface ChzzkApiErrorBody {
  code: number;
  message: string;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

/** Tips 문서의 공통 응답(content.url 등) 파싱 */
export function parseSessionUrlFromAuthResponse(json: unknown): string {
  if (!isRecord(json)) {
    throw new Error("세션 URL 응답 형식이 올바르지 않습니다.");
  }
  if (typeof json.code === "number" && json.code !== 200) {
    const msg = typeof json.message === "string" ? json.message : "알 수 없는 오류";
    throw new Error(`세션 생성 실패 (${json.code}): ${msg}`);
  }
  const content = json.content;
  if (!isRecord(content) || typeof content.url !== "string" || content.url.trim() === "") {
    throw new Error("세션 URL이 응답에 없습니다.");
  }
  return content.url.trim();
}

/** 유저 Access Token으로 소켓 연결 URL 요청 (Scope: 채팅 메시지 조회 등) */
export async function fetchSessionUrlUser(accessToken: string): Promise<string> {
  const res = await fetch(`${openapiBase()}/open/v1/sessions/auth`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json"
    }
  });
  const body: unknown = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = isRecord(body) && typeof body.message === "string" ? body.message : res.statusText;
    throw new Error(`세션 URL 요청 실패 (${res.status}): ${msg}`);
  }
  return parseSessionUrlFromAuthResponse(body);
}

/**
 * 연결된 세션에 채팅 이벤트 구독 (문서: Request Param sessionKey — 쿼리로 전달).
 * 소켓이 연결된 뒤 `SYSTEM` connected의 sessionKey로 호출해야 합니다.
 */
export async function subscribeChatEvent(accessToken: string, sessionKey: string): Promise<void> {
  const q = new URLSearchParams({ sessionKey });
  const res = await fetch(
    `${openapiBase()}/open/v1/sessions/events/subscribe/chat?${q.toString()}`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json"
      }
    }
  );
  const body: unknown = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = isRecord(body) && typeof body.message === "string" ? body.message : res.statusText;
    throw new Error(`채팅 구독 실패 (${res.status}): ${msg}`);
  }
  if (isRecord(body) && typeof body.code === "number" && body.code !== 200) {
    const msg = typeof body.message === "string" ? body.message : "구독 거부";
    throw new Error(`채팅 구독 실패: ${msg}`);
  }
}
