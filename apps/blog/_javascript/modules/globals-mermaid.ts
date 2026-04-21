import { requiredGlobal, type MermaidGlobal } from './globals';

function getMermaid(): MermaidGlobal {
  return requiredGlobal<MermaidGlobal>('mermaid');
}

/** Resolve window.mermaid only when used (script not on every layout). */
export const mermaid = {
  initialize(config: unknown) {
    return getMermaid().initialize(config);
  },
  init(config: unknown, selector: string) {
    return getMermaid().init(config, selector);
  }
};
