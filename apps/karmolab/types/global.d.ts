import type {
  KarmoLabImageBatchAPI,
  KarmoLabImageConvertAPI,
  KarmoLabLazyWidgetStub,
  KarmoWorldNamespace,
  RandomGenTopic
} from './karmolab';

export {};

declare global {
  interface Window {
    KarmoLabImageConvert?: KarmoLabImageConvertAPI;
    KarmoLabImageBatch?: KarmoLabImageBatchAPI;
    KarmoWorld?: KarmoWorldNamespace;
    /** tierlist 네임스페이스 — `namespace.js` */
    Tierlist?: Record<string, unknown>;
    RANDOMGEN_TOPICS?: RandomGenTopic[];
    KARMOLAB_WIDGET_LOADER_WAIT?: Promise<unknown>[];
    KARMOLAB_WIDGET_SCRIPT_BASE?: string;
    KARMOLAB_LAZY_META_BY_ID?: Record<string, KarmoLabLazyWidgetStub>;
    KARMOLAB_WIDGETS_BOOT?: string[];
    KARMOLAB_LAZY_META?: KarmoLabLazyWidgetStub[];

    /** imagegen/config.ts */
    ImageGen?: {
      GALLERY_SESSION_KEY: string;
      GALLERY_SESSION_MAX: number;
      PROMPT_HISTORY_KEY: string;
      PROMPT_HISTORY_MAX: number;
    };

    /** apps/karmolab-react-src 내 React 마운트 */
    mountKarmoPlanner?: (rootId: string) => void;
  }

  /** toolbox.js — global lexical binding (not necessarily window.Toolbox) */
  var Toolbox: {
    registerDeferred?: (stub: KarmoLabLazyWidgetStub) => void;
    register: (config: {
      id: string;
      title: string;
      category?: string;
      desc?: string;
      layout?: string;
      icon?: string;
      hidden?: boolean;
      noHero?: boolean;
      tabs: Array<{
        id: string;
        label: string;
        build: (container: HTMLElement) => void;
      }>;
    }) => void;
    initTheme: () => void;
    init: () => void;
    getTools: () => Array<{ id: string }>;
    showToast?: (msg: string, type?: string, detail?: unknown) => void;
  };

}
