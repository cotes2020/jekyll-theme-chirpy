/** UI 단축용. 실제 표시는 `chatOptions` 의 hideAuthor / lineDarkBg 로 저장·적용 */
export type PresetId = "default" | "dark_chat_only";

export function isPresetId(s: string): s is PresetId {
  return s === "default" || s === "dark_chat_only";
}
