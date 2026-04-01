declare module 'bootstrap/js/src/tooltip' {
  export default class Tooltip {
    constructor(element: Element, options?: unknown);
    static getInstance(element: Element): Tooltip | null;
    static getOrCreateInstance(element: Element, options?: unknown): Tooltip;
    show(): void;
    hide(): void;
  }
}

declare module 'bootstrap/js/src/toast' {
  export default class Toast {
    constructor(element: Element, options?: unknown);
    static getOrCreateInstance(element: Element, options?: unknown): Toast;
    show(): void;
    hide(): void;
  }
}
