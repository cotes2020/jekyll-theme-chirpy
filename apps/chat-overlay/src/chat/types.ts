/** 한 줄 채팅(오버레이 표시용). 소스가 바뀌어도 이 형태로 통일. */
export interface ChatLine {
  id: string;
  author: string;
  text: string;
  ts: number;
}

export type Unsubscribe = () => void;

/** 채팅 공급자. 구현체만 교체하면 UI/메인 로직은 유지. */
export interface ChatFeedSource {
  subscribe(onLine: (line: ChatLine) => void): Unsubscribe;
  destroy?(): void;
}
