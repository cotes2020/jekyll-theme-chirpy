import { requiredGlobal, type TocbotGlobal } from './globals';

function getTocbot(): TocbotGlobal {
  return requiredGlobal<TocbotGlobal>('tocbot');
}

/** Resolve window.tocbot only when used (TOC script is not on every layout). */
export const tocbot = {
  init(options: unknown) {
    return getTocbot().init(options);
  },
  refresh(options: unknown) {
    return getTocbot().refresh(options);
  }
};
