/**
 * 챗봇 UI의 API 표면(`studio` | `vertex`)과 `karmolab-ai`의 `GoogleGenerativeSurface` 정렬.
 * 저장 키·옵션 value는 HTML/Toolbox 프리픽과 동일하게 유지합니다.
 */
import type { GoogleGenerativeSurface } from 'karmolab-ai';

export const CB_API_SURFACE_PREF_KEY = 'cb_api_surface';

/** `<select id="cbApiSurfaceSelect">` 의 `value` 및 `Toolbox` 프리픽 값 */
export const ChatbotApiSurfaceUi = {
  studio: 'studio',
  vertex: 'vertex',
} as const;

export type ChatbotApiSurfaceUiValue = (typeof ChatbotApiSurfaceUi)[keyof typeof ChatbotApiSurfaceUi];

function readToolboxPref(key: string): string | null {
  const T = (globalThis as unknown as { Toolbox?: { getPref?: (k: string) => unknown } }).Toolbox;
  if (!T?.getPref) return null;
  const v = T.getPref(key);
  if (typeof v === 'string') return v;
  if (v == null) return null;
  return String(v);
}

export function getChatbotApiSurfaceUi(): ChatbotApiSurfaceUiValue {
  const v = readToolboxPref(CB_API_SURFACE_PREF_KEY);
  return v === ChatbotApiSurfaceUi.vertex ? ChatbotApiSurfaceUi.vertex : ChatbotApiSurfaceUi.studio;
}

/** 패키지 계약(`aiStudio` | `vertex`) — 로그·분기 일관용 */
export function chatbotUiSurfaceToPackage(surface: ChatbotApiSurfaceUiValue): GoogleGenerativeSurface {
  return surface === ChatbotApiSurfaceUi.vertex ? 'vertex' : 'aiStudio';
}
