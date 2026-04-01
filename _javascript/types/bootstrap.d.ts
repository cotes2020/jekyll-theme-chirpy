declare module 'bootstrap/js/src/tooltip' {
  export default class Tooltip {
    constructor(element: Element, options?: unknown);
    static getInstance(element: Element): Tooltip | null;
    static getOrCreateInstance(element: Element, options?: unknown): Tooltip;
    show(): void;
    hide(): void;
  }
}
