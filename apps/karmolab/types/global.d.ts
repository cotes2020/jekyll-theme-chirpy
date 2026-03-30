import type {
  KarmoLabImageBatchAPI,
  KarmoLabImageConvertAPI,
  KarmoLabLazyWidgetStub,
  RandomGenTopic
} from './karmolab';

export {};

declare global {
  interface Window {
    KarmoLabImageConvert?: KarmoLabImageConvertAPI;
    KarmoLabImageBatch?: KarmoLabImageBatchAPI;
    RANDOMGEN_TOPICS?: RandomGenTopic[];
    KARMOLAB_WIDGET_LOADER_WAIT?: Promise<unknown>[];
    KARMOLAB_WIDGET_SCRIPT_BASE?: string;
    KARMOLAB_LAZY_META_BY_ID?: Record<string, KarmoLabLazyWidgetStub>;
    KARMOLAB_WIDGETS_BOOT?: string[];
    KARMOLAB_LAZY_META?: KarmoLabLazyWidgetStub[];
  }

  /** toolbox.js — global lexical binding (not necessarily window.Toolbox) */
  var Toolbox: {
    registerDeferred?: (stub: KarmoLabLazyWidgetStub) => void;
    initTheme: () => void;
    init: () => void;
    getTools: () => Array<{ id: string }>;
  };
}
